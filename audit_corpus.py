"""
Аудит корпуса в БД: то, что может тихо испортить RAG, не роняя ни одного
прогона индексатора. Только чтение — ни БД, ни LLM не трогает.

Жёсткие проверки должны давать 0; хотя бы одна ненулевая — exit code 1.
Мягкие — только отчёт с примерами: у них бывают законные срабатывания.

Каждая проверка — след реальной находки аудита 2026-09-28 (подробности —
docs/ARCHITECTURE.md, 5.2). Запуск: uv run audit_corpus.py
"""
import asyncio
import sys

from sqlalchemy import text

from app.domain.wiki_chunker import CHUNKER_VERSION, SPLIT_THRESHOLD_CHARS
from app.infrastructure.database.connection import async_session_maker, engine

UNIVERSE = "AC"

# Тело чанка без первой строки: префикс section_path дописывается ПОСЛЕ нарезки
# и в лимиты не входит.
BODY = "substr(chunk_text, strpos(chunk_text, E'\\n') + 1)"

# (название, SQL -> (количество, примеры)). В SQL :u — вселенная.
HARD = [
    ("Служебные ссылки: «Категория:» и interwiki («en:…») в начале строки", f"""
        SELECT article_title FROM article_chunks
        WHERE universe = :u AND {BODY} ~ '(^|\\n)(Категория:|[a-z]{{2,3}}:[^\\s])'"""),
    ("Magic words (__NOTOC__ и т.п.)", f"""
        SELECT article_title FROM article_chunks WHERE universe = :u AND chunk_text ~ '__[A-Z]+__'"""),
    ("Остатки разметки: <br, {{", f"""
        SELECT article_title FROM article_chunks
        WHERE universe = :u AND ({BODY} ILIKE '%<br%' OR {BODY} LIKE '%{{{{%')"""),
    ("Невидимые символы (мягкий перенос, zero-width, BOM, nbsp)", """
        SELECT article_title FROM article_chunks
        WHERE universe = :u AND chunk_text ~ '[­​-‏﻿ ]'"""),
    (f"Тело text-чанка длиннее SPLIT_THRESHOLD_CHARS ({SPLIT_THRESHOLD_CHARS})", f"""
        SELECT article_title FROM article_chunks
        WHERE universe = :u AND chunk_type = 'text' AND length({BODY}) > {SPLIT_THRESHOLD_CHARS}"""),
    ("Поля-картинки в инфобоксе", """
        SELECT article_title FROM article_chunks
        WHERE universe = :u AND chunk_type = 'infobox' AND chunk_text ~ '(^|\\. |\\n)(Изображение|Image)\\d*: '"""),
    (f"Статьи, залитые не текущей версией чанкера ({CHUNKER_VERSION})", f"""
        SELECT a.title FROM articles a
        WHERE EXISTS (SELECT 1 FROM article_chunks ch WHERE ch.article_id = a.id AND ch.universe = :u)
          AND coalesce(a.metadata_obj->>'chunker_version', '') <> '{CHUNKER_VERSION}'"""),
    ("Векторы: NULL или нулевой длины", """
        SELECT article_title FROM article_chunks
        WHERE universe = :u AND (embedding IS NULL OR vector_norm(embedding) < 1e-6)"""),
    ("Векторы: один и тот же вектор у разных текстов", """
        SELECT min(article_title) FROM article_chunks WHERE universe = :u
        GROUP BY md5(embedding::text) HAVING count(DISTINCT chunk_text) > 1"""),
    ("Денормализация: title/url/revid чанка расходятся со статьёй", """
        SELECT ch.article_title FROM article_chunks ch JOIN articles a ON a.id = ch.article_id
        WHERE ch.universe = :u AND (ch.article_title <> a.title OR ch.source_url <> a.url
              OR ch.metadata_obj->'revid' <> a.metadata_obj->'revid')"""),
    ("Словарь опечаток: термины с цифрами (имена файлов, склейки таблиц)", """
        SELECT term FROM lore_terms WHERE universe = :u AND term ~ '[0-9]'"""),
]

SOFT = [
    # Бывают законные: «МакКрири», «ДиКаприо».
    ("Слипшиеся слова (строчная + Заглавная без пробела)", """
        SELECT article_title FROM article_chunks
        WHERE universe = :u AND chunk_text ~ '[а-яё]{2,}[А-ЯЁ][а-яё]{2,}'"""),
    # Бывают законные: «[[» внутри стилизованного текста переписки в исходнике.
    ("Остатки разметки: [[", f"""
        SELECT article_title FROM article_chunks WHERE universe = :u AND {BODY} LIKE '%[[%'"""),
    ("Короткие чанки (тело < 50 символов)", f"""
        SELECT article_title FROM article_chunks WHERE universe = :u AND length({BODY}) < 50"""),
    # Слова из заголовков секций не должны быть мишенями исправления опечаток
    # (см. _SENTENCE_END). Законно, если слово где-то стоит с заглавной внутри фразы.
    ("Словарь опечаток: общие слова заголовков помечены именами", """
        SELECT DISTINCT term FROM lore_terms WHERE universe = :u AND is_proper
          AND term IN ('биография', 'история', 'описание', 'появления', 'галерея',
                       'интересные', 'факты', 'личность', 'характеристики', 'примечания', 'ссылки')"""),
]


async def _run(session, sql: str) -> list:
    return [row[0] for row in (await session.execute(text(sql), {"u": UNIVERSE})).all()]


async def audit() -> int:
    failed = 0
    async with async_session_maker() as session:
        total = (await session.execute(
            text("SELECT count(DISTINCT article_id), count(*) FROM article_chunks WHERE universe = :u"),
            {"u": UNIVERSE},
        )).one()
        print(f"Вселенная {UNIVERSE}: {total[0]} статей, {total[1]} чанков\n")

        print("Жёсткие проверки (ожидается 0):")
        for name, sql in HARD:
            rows = await _run(session, sql)
            failed += bool(rows)
            examples = f"  напр.: {sorted(set(rows))[:5]}" if rows else ""
            print(f"  {'❌' if rows else '✅'} {name}: {len(rows)}{examples}")

        print("\nМягкие проверки (отчёт, бывают законные срабатывания):")
        for name, sql in SOFT:
            rows = await _run(session, sql)
            examples = f"  напр.: {sorted(set(rows))[:5]}" if rows else ""
            print(f"  • {name}: {len(rows)}{examples}")

    await engine.dispose()
    print(f"\n{'❌ Жёстких проверок не прошло: ' + str(failed) if failed else '✅ Все жёсткие проверки пройдены'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(audit()))
