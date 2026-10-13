import re
from sqlalchemy import select, func, distinct, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.infrastructure.database.models import ArticleChunk, LoreTerm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from app.domain.ports import IVectorStore

_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)

_LATIN_RE = re.compile(r"[A-Za-z]")

# После них заглавная буква — начало предложения, а не имя. «>» — разделитель
# в префиксе «Статья > Биография»: иначе слова заголовков считались бы именами.
_SENTENCE_END = ".!?…:;>\n"

# На коротких словах триграммы дают много ложных совпадений; длинные — склейки.
_MIN_TERM_LEN = 4
_MAX_TERM_LEN = 30

# Опечатки дают сходство 0.38–0.70, разные имена — около 0.
_SIMILARITY_THRESHOLD = 0.35


def normalize_word(word: str) -> str:
    """Одна нормализация для словаря и запроса: нижний регистр, ё -> е."""
    return word.lower().replace("ё", "е")


def _is_proper(raw: str, text: str, start: int) -> bool:
    """
    Имя собственное? Латиница — всегда да. Кириллица — с заглавной, не капсом
    и не в начале предложения.
    """
    if _LATIN_RE.search(raw):
        return True
    if not raw[:1].isupper() or raw.isupper():
        return False

    before = text[:start]
    stripped = before.rstrip()
    if not stripped:
        return False
    # Перенос — тоже граница; проверять до rstrip(), иначе он его съест.
    if "\n" in before[len(stripped):]:
        return False
    # Цифра перед словом — нумерация («2.1 Технологический стек»).
    return not (stripped[-1] in _SENTENCE_END or stripped[-1].isdigit())


def extract_terms(chunk_text: str) -> Dict[str, bool]:
    """
    {слово: имя_собственное} для словаря опечаток. Флаг — ИЛИ по всем
    вхождениям: одно вхождение в середине фразы выдаёт имя.
    """
    terms: Dict[str, bool] = {}

    def add(word: str, proper: bool) -> None:
        # Слова с цифрами — «250px», склейки таблиц; именами не бывают.
        if not _MIN_TERM_LEN <= len(word) <= _MAX_TERM_LEN or any(ch.isdigit() for ch in word):
            return
        terms[word] = terms.get(word, False) or proper

    for match in _WORD_RE.finditer(chunk_text):
        raw = match.group(0)
        word = normalize_word(raw)
        if word.isdigit():
            continue
        proper = _is_proper(raw, chunk_text, match.start())
        add(word, proper)
        # «аль-муалим» — ещё и как «аль» + «муалим»
        if "-" in word:
            raw_parts = raw.split("-")
            for i, part in enumerate(word.split("-")):
                part_raw = raw_parts[i] if i < len(raw_parts) else raw
                add(part, proper and part_raw[:1].isupper())

    return terms

class PostgresVectorStore(IVectorStore):

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_chunks(self, chunks_data: List[Dict[str, Any]]) -> None:
        # Обязательные ключи без .get(): KeyError на первом чанке лучше NOT NULL
        # violation через сотни статей.
        orm_chinks = [
            ArticleChunk(
                article_id=chunk["article_id"],
                universe=chunk["universe"],
                article_title=chunk["article_title"],
                source_url=chunk["source_url"],
                section_path=chunk.get("section_path"),
                chunk_type=chunk.get("chunk_type", "text"),
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
            for term, is_proper in extract_terms(chunk["chunk_text"]).items():
                key = (chunk["article_id"], term)
                row = rows.get(key)
                if row:
                    row["is_proper"] = row["is_proper"] or is_proper
                    continue
                rows[key] = {
                    "article_id": chunk["article_id"],
                    "universe": chunk["universe"],
                    "term": term,
                    "is_proper": is_proper,
                }

        if not rows:
            return

        # Дубль не вставляем, но флаг имени поднимаем: статью могли грузить частями.
        stmt = pg_insert(LoreTerm).values(list(rows.values()))
        stmt = stmt.on_conflict_do_update(
            constraint="uq_lore_terms_article_term",
            set_={"is_proper": LoreTerm.is_proper.op("OR")(stmt.excluded.is_proper)},
        )
        await self.session.execute(stmt)

    async def search_similar(self, query_vector: List[float], universe: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Top-k чанков вселенной по косинусному расстоянию (0 — идентично).
        Метаданные и distance нужны eval и будущему порогу релевантности.
        """
        distance = ArticleChunk.embedding.cosine_distance(query_vector).label("distance")
        stmt = (
            select(ArticleChunk, distance)
            .where(ArticleChunk.universe == universe)
            .order_by(ArticleChunk.embedding.cosine_distance(query_vector))
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()

        return [
            {
                "id": chunk.id,
                "article_id": chunk.article_id,
                "article_title": chunk.article_title,
                "section_path": chunk.section_path,
                "chunk_type": chunk.chunk_type,
                "chunk_text": chunk.chunk_text,
                "universe": chunk.universe,
                "distance": float(dist),
            }
            for chunk, dist in rows
        ]

    async def correct_typos(self, query: str, universe: str) -> str:
        """
        «Альтарир» -> «Альтаир» по словарю лора, без LLM. Знакомые лору слова
        не трогаются; незнакомое без похожего имени выше порога — тоже.
        """
        candidates = {
            normalize_word(m.group(0))
            for m in _WORD_RE.finditer(query)
            if len(normalize_word(m.group(0))) >= _MIN_TERM_LEN
        }
        if not candidates:
            return query

        # Знакомость — по ОСНОВЕ: «обработкой» и «обработки» — не опечатка.
        # Выражение совпадает с индексом ix_lore_terms_universe_stem символ в символ.
        known = set(
            (await self.session.execute(
                text("""
                    SELECT w.word
                    FROM unnest(CAST(:words AS text[])) AS w(word)
                    WHERE EXISTS (
                        SELECT 1 FROM lore_terms t
                        WHERE t.universe = :universe
                          AND COALESCE((ts_lexize('russian_stem', t.term))[1], t.term)
                            = COALESCE((ts_lexize('russian_stem', w.word))[1], w.word)
                    )
                """),
                {"words": list(candidates), "universe": universe},
            )).scalars().all()
        )

        replacements = {}
        for word in candidates - known:
            # «%» — индексный префильтр (порог 0.3 < 0.35, т.е. надмножество): без
            # него слово без похожих гнало KNN по всему индексу. Точный порог —
            # similarity(). is_proper: чиним только имена, не «происходит» -> «проходит».
            best = (await self.session.execute(
                select(LoreTerm.term)
                .where(
                    LoreTerm.universe == universe,
                    LoreTerm.is_proper.is_(True),
                    LoreTerm.term.op("%")(word),
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
        """Вселенные, по которым есть чанки, с числом чанков и статей."""
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