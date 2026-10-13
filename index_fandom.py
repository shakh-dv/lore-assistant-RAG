"""
Индексатор Fandom-корпуса: обходит статьи через FandomScraper, режет через
WikiChunker, батчит эмбеддинги через GeminiAdapter, льёт в БД как universe='AC'.

Запуск:
    uv run python index_fandom.py --dry-run --limit 5      # chunker без LLM/БД
    uv run python index_fandom.py --limit 20                # реальная заливка, малая выборка
    uv run python index_fandom.py --title "Эцио Аудиторе"    # точечная переиндексация
    uv run python index_fandom.py --title "..." --force      # форс даже при неизменном revid
    uv run python index_fandom.py                            # полный корпус
    uv run python index_fandom.py --resume                   # продолжить с чекпоинта

Статья пропускается, если не изменились ни revid, ни CHUNKER_VERSION.
Чекпоинт, серия ошибок, переиспользование эмбеддингов — ARCHITECTURE.md, 5.4.
"""
import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import select

from app.core.config import settings
from app.domain.wiki_chunker import CHUNKER_VERSION, chunk_article
from app.infrastructure.database.connection import async_session_maker
from app.infrastructure.database.models import Article, ArticleChunk
from app.infrastructure.gemini_adapter import DailyQuotaExceeded, GeminiAdapter
from app.infrastructure.postgres_adapter import PostgresVectorStore
from fandom_scraper import FandomScraper

UNIVERSE = "AC"
BASE_WIKI_URL = "https://assassinscreed.fandom.com/ru/wiki/"
DRY_RUN_OUTPUT = Path("dry_run_output.jsonl")
CHECKPOINT_FILE = Path("index_checkpoint.json")
MAX_CONSECUTIVE_ERRORS = 5
# Переиспользовать векторы можно только той же модели и размерности.
EMBEDDING_MODEL_TAG = f"{settings.EMBEDDING_MODEL}/{settings.VECTOR_DIMENSION}"


def _article_url(title: str) -> str:
    # Так строит URL сам MediaWiki (проверено через prop=info&inprop=url).
    return BASE_WIKI_URL + quote(title.replace(" ", "_"))


async def _reusable_embeddings(session, existing: Article | None) -> dict:
    """
    {chunk_text: вектор} старой копии статьи — тот же текст той же моделью даёт
    тот же вектор. Статья без метки embedding_model считается текущей моделью.
    """
    if existing is None:
        return {}
    if existing.metadata_obj.get("embedding_model", EMBEDDING_MODEL_TAG) != EMBEDDING_MODEL_TAG:
        return {}
    rows = await session.execute(
        select(ArticleChunk.chunk_text, ArticleChunk.embedding)
        .where(ArticleChunk.article_id == existing.id)
    )
    return {chunk_text: embedding for chunk_text, embedding in rows}


async def _process_article(
    title: str,
    scraper: FandomScraper,
    llm_client: GeminiAdapter,
    dry_run: bool,
    force: bool,
) -> str:
    """Возвращает статус: 'ok' / 'skipped-unchanged' / 'skipped-missing' / 'skipped-empty'."""
    article = await scraper.fetch_article(title)
    if article is None:
        print(f"⚠️  {title}: статья недоступна (missingtitle/invalidtitle), пропуск")
        return "skipped-missing"

    source_url = _article_url(article["title"])

    async with async_session_maker() as session:
        existing = await session.scalar(select(Article).where(Article.url == source_url))

        if (
            existing
            and not force
            and existing.metadata_obj.get("revid") == article["revid"]
            and existing.metadata_obj.get("chunker_version") == CHUNKER_VERSION
        ):
            print(f"⏭  {article['title']}: revid и версия чанкера не изменились, пропуск")
            return "skipped-unchanged"

        chunks = chunk_article(article, source_url)
        if not chunks:
            if existing and not dry_run:
                await session.delete(existing)
                await session.commit()
                print(f"🗑  {article['title']}: чанков больше нет, старая копия удалена из БД")
            else:
                print(f"⚠️  {article['title']}: чанкер не дал ни одного чанка, пропуск")
            return "skipped-empty"

        if dry_run:
            print(f"[dry-run] {article['title']}: {len(chunks)} чанков")
            with DRY_RUN_OUTPUT.open("a", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps({
                        "article_title": c["article_title"],
                        "section_path": c["section_path"],
                        "chunk_type": c["chunk_type"],
                        "chunk_text": c["chunk_text"],
                        "metadata": c["metadata"],
                    }, ensure_ascii=False) + "\n")
            return "ok"

        texts = [c["chunk_text"] for c in chunks]
        known = await _reusable_embeddings(session, existing)
        new_texts = [t for t in dict.fromkeys(texts) if t not in known]
        if new_texts:
            known.update(zip(new_texts, await llm_client.generate_embeddings_batch(new_texts)))
        embeddings = [known[t] for t in texts]
        print(f"  🧮 эмбеддингов: {len(new_texts)} новых из {len(texts)}")

        if existing:
            await session.delete(existing)
            await session.flush()

        db_article = Article(
            title=article["title"],
            url=source_url,
            metadata_obj={
                "revid": article["revid"],
                "categories": article["categories"],
                "chunker_version": CHUNKER_VERSION,
                "embedding_model": EMBEDDING_MODEL_TAG,
            },
        )
        session.add(db_article)
        await session.flush()

        chunks_to_save = [
            {**c, "article_id": db_article.id, "universe": UNIVERSE, "embedding": embedding}
            for c, embedding in zip(chunks, embeddings)
        ]
        await PostgresVectorStore(session).save_chunks(chunks_to_save)
        await session.commit()

    print(f"✅ {article['title']}: {len(chunks)} чанков")
    return "ok"


async def index_all(
    limit: int | None, titles: list[str] | None, dry_run: bool, force: bool, resume: bool
) -> None:
    if dry_run and DRY_RUN_OUTPUT.exists():
        DRY_RUN_OUTPUT.unlink()

    llm_client = GeminiAdapter(
        api_key=settings.GEMINI_API_KEY,
        chat_model_name=settings.CHAT_MODEL,
        embedding_model_name=settings.EMBEDDING_MODEL,
        embedding_dimension=settings.VECTOR_DIMENSION,
    )

    stats = {"ok": 0, "skipped-unchanged": 0, "skipped-missing": 0, "skipped-empty": 0, "error": 0}

    async with FandomScraper() as scraper:
        if titles:
            title_iter = _async_iter(titles)
        else:
            resume_from = None
            if resume and CHECKPOINT_FILE.exists():
                resume_from = json.loads(CHECKPOINT_FILE.read_text()).get("title")
                print(f"↩️  Резюме с чекпоинта ({CHECKPOINT_FILE}): «{resume_from}»")
            title_iter = scraper.list_all_titles(resume_from=resume_from)

        i = 0
        error_streak: list[str] = []  # заголовки статей, упавших подряд
        async for title in title_iter:
            if limit is not None and i >= limit:
                break
            i += 1
            if titles is None:
                CHECKPOINT_FILE.write_text(json.dumps({"title": title}, ensure_ascii=False))
            print(f"\n[{i}/{limit or '?'}] {title}")
            try:
                status = await _process_article(title, scraper, llm_client, dry_run, force)
                stats[status] += 1
                error_streak.clear()
            except DailyQuotaExceeded:
                # До сброса квоты не пройдёт ни одна статья — останавливаем весь прогон.
                print(f"\n⛔ Дневной лимит Gemini API исчерпан на статье «{title}» "
                      f"({i} обработано в этом прогоне). Обработанное уже сохранено. "
                      f"Запусти снова после сброса квоты (полночь по Тихоокеанскому "
                      f"времени) с флагом --resume — продолжит с этого места.")
                break
            except Exception as exc:
                # Одна битая статья не роняет прогон; разбор — руками через --title.
                print(f"❌ {title}: {type(exc).__name__}: {exc}")
                stats["error"] += 1
                # Серия подряд — это уже инфраструктура (БД, сеть, Cloudflare):
                # стоп и откат чекпоинта на начало серии.
                error_streak.append(title)
                if len(error_streak) >= MAX_CONSECUTIVE_ERRORS:
                    if titles is None:
                        CHECKPOINT_FILE.write_text(
                            json.dumps({"title": error_streak[0]}, ensure_ascii=False)
                        )
                    print(f"\n⛔ {len(error_streak)} статей подряд упали с ошибкой — похоже "
                          f"на системный сбой (БД, сеть, Cloudflare), а не на сами статьи. "
                          f"Прогон остановлен, чекпоинт откачен на «{error_streak[0]}». "
                          f"Устрани причину и запусти снова с --resume.")
                    break
        else:
            # Список пройден до конца — чекпоинт больше не нужен.
            if not titles and CHECKPOINT_FILE.exists():
                CHECKPOINT_FILE.unlink()

    print(f"\n--- Готово: {stats} ---")
    if dry_run:
        print(f"Чанки записаны в {DRY_RUN_OUTPUT}")


async def _async_iter(items: list[str]):
    for item in items:
        yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Индексатор Fandom-корпуса AC")
    parser.add_argument("--dry-run", action="store_true", help="Только chunker, без LLM/БД")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число статей")
    parser.add_argument("--title", action="append", dest="titles", help="Точечная статья (можно несколько раз)")
    parser.add_argument("--force", action="store_true", help="Reindex даже при неизменном revid")
    parser.add_argument(
        "--resume", action="store_true",
        help=f"Продолжить с чекпоинта в {CHECKPOINT_FILE} вместо списка статей с начала",
    )
    args = parser.parse_args()

    asyncio.run(index_all(
        limit=args.limit, titles=args.titles, dry_run=args.dry_run,
        force=args.force, resume=args.resume,
    ))


if __name__ == "__main__":
    main()
