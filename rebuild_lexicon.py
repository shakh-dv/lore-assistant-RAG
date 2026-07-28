"""
Пересобирает словарь опечаток (lore_terms) из уже загруженных чанков.

Нужен один раз — после добавления таблицы, чтобы подхватить лор,
который был залит раньше. Дальше словарь пополняется сам в save_chunks.

LLM здесь не вызывается вообще: эмбеддинги не трогаем, квота не тратится.
Запуск: uv run rebuild_lexicon.py
"""
import asyncio
from sqlalchemy import select, delete, text

from app.infrastructure.database.connection import async_session_maker, engine
from app.infrastructure.database.models import ArticleChunk, LoreTerm
from app.infrastructure.postgres_adapter import extract_terms


async def rebuild():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

    async with async_session_maker() as session:
        chunks = (await session.execute(
            select(ArticleChunk.article_id, ArticleChunk.universe, ArticleChunk.chunk_text)
        )).all()

        if not chunks:
            print("В базе нет чанков — нечего собирать.")
            return

        # Полная пересборка: сначала сносим старый словарь целиком
        removed = (await session.execute(delete(LoreTerm))).rowcount
        if removed:
            print(f"Старый словарь очищен ({removed} слов).")

        rows = {}
        for article_id, universe, chunk_text in chunks:
            for term in extract_terms(chunk_text):
                rows[(article_id, term)] = LoreTerm(
                    article_id=article_id, universe=universe, term=term
                )

        session.add_all(list(rows.values()))
        await session.commit()

        by_universe = {}
        for (article_id, term), row in rows.items():
            by_universe[row.universe] = by_universe.get(row.universe, 0) + 1

        print(f"✅ Словарь собран из {len(chunks)} чанков: {len(rows)} слов.")
        for universe, count in sorted(by_universe.items()):
            print(f"   {universe}: {count}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(rebuild())
