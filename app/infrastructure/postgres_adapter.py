import re
from sqlalchemy import select, func, distinct, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from app.infrastructure.database.models import ArticleChunk, LoreTerm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from app.domain.ports import IVectorStore

# Слово = буквы/цифры/дефис. Дефисные имена («Аль-Муалим») кладём и целиком, и по частям.
_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)

# Латиница в русском лоре — почти всегда название: Mixvel, Sirena, FastAPI.
_LATIN_RE = re.compile(r"[A-Za-z]")

# После этих символов заглавная буква — просто начало предложения.
_SENTENCE_END = ".!?…:;\n"

# Короткие слова не берём: на них триграммы дают слишком много ложных совпадений.
_MIN_TERM_LEN = 4

# Порог триграммного сходства. По замерам: опечатки дают 0.38-0.70,
# разные имена — 0.00. 0.35 разделяет их с запасом.
_SIMILARITY_THRESHOLD = 0.35


def normalize_word(word: str) -> str:
    """Единая нормализация для словаря и для запроса: нижний регистр и ё -> е."""
    return word.lower().replace("ё", "е")


def _is_proper(raw: str, text: str, start: int) -> bool:
    """
    Похоже ли слово на имя собственное или технический термин.

    Латиница — всегда да: в лоре это названия («Mixvel», «error_source»).
    Кириллица — только «Альтаир», но не «АРХИТЕКТУРА» (заголовок раздела)
    и не «Его» в начале предложения: заглавная там ничего не означает.
    """
    if _LATIN_RE.search(raw):
        return True
    if not raw[:1].isupper() or raw.isupper():
        return False

    # Начало предложения: заглавная буква вынужденная, о слове не говорит ничего
    before = text[:start]
    stripped = before.rstrip()
    if not stripped:
        return False
    # Перенос строки — тоже граница: к тексту чанка приклеен заголовок раздела.
    # Проверяем ДО rstrip(), иначе он этот перенос и съест.
    if "\n" in before[len(stripped):]:
        return False
    # «2.1 Технологический стек» — цифра перед словом означает нумерацию раздела,
    # а не предложение, в середине которого стоит имя.
    return not (stripped[-1] in _SENTENCE_END or stripped[-1].isdigit())


def extract_terms(chunk_text: str) -> Dict[str, bool]:
    """
    Слова из текста лора для словаря опечаток: {слово: имя_собственное}.

    Флаг считается по всем вхождениям через ИЛИ — «Альтаир» в начале
    предложения выглядит обычным словом, но хотя бы одно вхождение
    в середине выдаёт в нём имя.
    """
    terms: Dict[str, bool] = {}

    def add(word: str, proper: bool) -> None:
        if len(word) < _MIN_TERM_LEN or word.isdigit():
            return
        key = word[:64]
        terms[key] = terms.get(key, False) or proper

    for match in _WORD_RE.finditer(chunk_text):
        raw = match.group(0)
        word = normalize_word(raw)
        if word.isdigit():
            continue
        proper = _is_proper(raw, chunk_text, match.start())
        add(word, proper)
        # «аль-муалим» кладём ещё и как «аль» + «муалим»
        if "-" in word:
            raw_parts = raw.split("-")
            for i, part in enumerate(word.split("-")):
                # Регистр берём у своей же половинки: в «Аль-Муалим» имя обе
                part_raw = raw_parts[i] if i < len(raw_parts) else raw
                add(part, proper and part_raw[:1].isupper())

    return terms

class PostgresVectorStore(IVectorStore):

    def __init__(self, session: AsyncSession):
        # Адаптер не создает подключение к БД сам!
        # Он получает уже готовую асинхронную сессию извне (Dependency Injection).
        self.session = session

    async def save_chunks(self, chunks_data: List[Dict[str, Any]]) -> None:
        # article_title/source_url — обязательные ключи, без .get(): лучше упасть
        # явным KeyError на первом чанке, чем тихо словить NOT NULL violation
        # в БД через сотни статей.
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
                # ключ — то же, что и в UniqueConstraint
                key = (chunk["article_id"], term)
                row = rows.get(key)
                if row:
                    # Слово из соседнего чанка той же статьи: имя — если хоть где-то имя
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

        # Дубль по (article_id, term) не вставляем заново, но флаг имени поднимаем:
        # статью могли грузить частями, и «Альтаир» в середине предложения
        # мог встретиться только во второй порции.
        stmt = pg_insert(LoreTerm).values(list(rows.values()))
        stmt = stmt.on_conflict_do_update(
            constraint="uq_lore_terms_article_term",
            set_={"is_proper": LoreTerm.is_proper.op("OR")(stmt.excluded.is_proper)},
        )
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

        # Какие слова запроса уже знакомы лору — их не трогаем.
        # Сравниваем по ОСНОВЕ, а не буквально: в словаре лежит «обработки»,
        # в вопросе «обработкой» — это одно слово, а не опечатка. Триграммы так
        # не умеют: у падежей сходство (0.62) выше, чем у настоящей опечатки (0.55).
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
            # ORDER BY <-> — это KNN по GiST-индексу; similarity() задаёт порог явно,
            # чтобы не зависеть от сессионной настройки pg_trgm.similarity_threshold.
            #
            # is_proper — вторая защита: чинить имеет смысл только имена, ради
            # которых словарь и заводился. Без неё любой глагол из лора становится
            # мишенью, и «происходит» уезжает в «проходит».
            best = (await self.session.execute(
                select(LoreTerm.term)
                .where(
                    LoreTerm.universe == universe,
                    LoreTerm.is_proper.is_(True),
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