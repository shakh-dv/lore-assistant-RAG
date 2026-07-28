import re
from sqlalchemy import select, func, distinct
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.infrastructure.database.models import ArticleChunk, LoreTerm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from app.domain.ports import IVectorStore

# Слово = буквы/цифры/дефис. Дефисные имена («Аль-Муалим») кладём и целиком, и по частям.
_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)

# Короткие слова не берём: на них триграммы дают слишком много ложных совпадений.
_MIN_TERM_LEN = 4

# Порог триграммного сходства. По замерам: опечатки дают 0.38-0.70,
# разные имена — 0.00. 0.35 разделяет их с запасом.
_SIMILARITY_THRESHOLD = 0.35


def normalize_word(word: str) -> str:
    """Единая нормализация для словаря и для запроса: нижний регистр и ё -> е."""
    return word.lower().replace("ё", "е")


def extract_terms(chunk_text: str) -> set:
    """Слова из текста лора, пригодные для словаря опечаток."""
    terms = set()
    for raw in _WORD_RE.findall(chunk_text):
        word = normalize_word(raw)
        if word.isdigit():
            continue
        if len(word) >= _MIN_TERM_LEN:
            terms.add(word[:64])
        # «аль-муалим» кладём ещё и как «аль» + «муалим»
        if "-" in word:
            for part in word.split("-"):
                if len(part) >= _MIN_TERM_LEN and not part.isdigit():
                    terms.add(part[:64])
    return terms

class PostgresVectorStore(IVectorStore):

    def __init__(self, session: AsyncSession):
        # Адаптер не создает подключение к БД сам!
        # Он получает уже готовую асинхронную сессию извне (Dependency Injection).
        self.session = session

    async def save_chunks(self, chunks_data: List[Dict[str, Any]]) -> None:
        orm_chinks = [
            ArticleChunk(
                article_id=chunk["article_id"],
                universe=chunk["universe"],
                chunk_text=chunk["chunk_text"],
                embedding=chunk["embedding"],
                metadata_obj=chunk.get("metadata", {})
            )
            for chunk in chunks_data
        ]

        self.session.add_all(orm_chinks)
        await self.session.flush()

        await self._save_terms(chunks_data)

    async def _save_terms(self, chunks_data: List[Dict[str, Any]]) -> None:
        """Пополняем словарь опечаток словами из этих же чанков."""
        rows = {}
        for chunk in chunks_data:
            for term in extract_terms(chunk["chunk_text"]):
                # ключ — то же, что и в UniqueConstraint
                rows[(chunk["article_id"], term)] = {
                    "article_id": chunk["article_id"],
                    "universe": chunk["universe"],
                    "term": term,
                }

        if not rows:
            return

        # Слово могло прийти из соседнего чанка той же статьи — просто пропускаем дубль
        stmt = pg_insert(LoreTerm).values(list(rows.values()))
        stmt = stmt.on_conflict_do_nothing(constraint="uq_lore_terms_article_term")
        await self.session.execute(stmt)

    async def search_similar(self, query_vector: List[float], universe: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Ищем самые похожие тексты в вектоной базе данных
        """
        stmt = (
            select(ArticleChunk)
            .where(ArticleChunk.universe == universe)
            .order_by(ArticleChunk.embedding.cosine_distance(query_vector))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        chunks = result.scalars().all()

        return [
            {
                "id": chunk.id,
                "article_id": chunk.article_id,
                "chunk_text": chunk.chunk_text,
                "universe": chunk.universe,
            }
            for chunk in chunks
        ]

    async def correct_typos(self, query: str, universe: str) -> str:
        """
        Чиним опечатки в именах по словарю лора — локально, без вызова LLM.
        «Альтарир» -> «Альтаир». Регистр и пунктуация нормализуются попутно.

        Слово, которое в лоре есть, не трогаем никогда: правим только незнакомые.
        Если похожего слова выше порога нет — оставляем как было.
        """
        candidates = {
            normalize_word(m.group(0))
            for m in _WORD_RE.finditer(query)
            if len(normalize_word(m.group(0))) >= _MIN_TERM_LEN
        }
        if not candidates:
            return query

        # Какие слова запроса и так есть в лоре — их не трогаем
        known = set(
            (await self.session.execute(
                select(LoreTerm.term)
                .where(LoreTerm.universe == universe, LoreTerm.term.in_(candidates))
                .distinct()
            )).scalars().all()
        )

        replacements = {}
        for word in candidates - known:
            # ORDER BY <-> — это KNN по GiST-индексу; similarity() задаёт порог явно,
            # чтобы не зависеть от сессионной настройки pg_trgm.similarity_threshold.
            best = (await self.session.execute(
                select(LoreTerm.term)
                .where(
                    LoreTerm.universe == universe,
                    func.similarity(LoreTerm.term, word) >= _SIMILARITY_THRESHOLD,
                )
                .order_by(LoreTerm.term.op("<->")(word))
                .limit(1)
            )).scalar_one_or_none()

            if best and best != word:
                replacements[word] = best

        if not replacements:
            return query

        def swap(match):
            return replacements.get(normalize_word(match.group(0)), match.group(0))

        corrected = _WORD_RE.sub(swap, query)
        print(f"🔤 Опечатки: '{query}' -> '{corrected}' {replacements}")
        return corrected

    async def list_universes(self) -> List[Dict[str, Any]]:
        """
        Отдаём только те вселенные, по которым реально есть чанки,
        вместе с количеством чанков и статей.
        """
        stmt = (
            select(
                ArticleChunk.universe,
                func.count(ArticleChunk.id).label("chunks"),
                func.count(distinct(ArticleChunk.article_id)).label("articles"),
            )
            .group_by(ArticleChunk.universe)
            .order_by(ArticleChunk.universe)
        )
        result = await self.session.execute(stmt)

        return [
            {"id": row.universe, "chunks": row.chunks, "articles": row.articles}
            for row in result.all()
        ]