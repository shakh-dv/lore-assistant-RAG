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

Чекпоинт (index_checkpoint.json) пишется на каждую статью при полном
прогоне (без --title) — точный заголовок статьи (apfrom, см. fandom_scraper.py).
--resume читает его при старте и продолжает ровно с этой статьи, не с начала
списка и не с границы страницы allpages; удаляется сам, когда корпус пройден
целиком без обрыва на дневном лимите.

Сессия БД и commit() — на КАЖДУЮ статью, не одна на весь прогон: падение на
статье №800 из тысяч не должно откатывать 799 уже успешно залитых.
"""
import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import quote

from sqlalchemy import select

from app.core.config import settings
from app.domain.wiki_chunker import chunk_article
from app.infrastructure.database.connection import async_session_maker
from app.infrastructure.database.models import Article
from app.infrastructure.gemini_adapter import DailyQuotaExceeded, GeminiAdapter
from app.infrastructure.postgres_adapter import PostgresVectorStore
from fandom_scraper import FandomScraper

UNIVERSE = "AC"
BASE_WIKI_URL = "https://assassinscreed.fandom.com/ru/wiki/"
DRY_RUN_OUTPUT = Path("dry_run_output.jsonl")
CHECKPOINT_FILE = Path("index_checkpoint.json")


def _article_url(title: str) -> str:
    # Подтверждено эмпирически против настоящего MediaWiki API
    # (action=query&prop=info&inprop=url): пробелы -> "_", остальное — percent-encoding.
    return BASE_WIKI_URL + quote(title.replace(" ", "_"))


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

        if existing and not force and existing.metadata_obj.get("revid") == article["revid"]:
            print(f"⏭  {article['title']}: revid не изменился, пропуск")
            return "skipped-unchanged"

        chunks = chunk_article(article, source_url)
        if not chunks:
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
        embeddings = await llm_client.generate_embeddings_batch(texts)

        if existing:
            await session.delete(existing)
            await session.flush()

        db_article = Article(
            title=article["title"],
            url=source_url,
            metadata_obj={"revid": article["revid"], "categories": article["categories"]},
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
        async for title in title_iter:
            if limit is not None and i >= limit:
                break
            i += 1
            # Пишем заголовок ТЕКУЩЕЙ статьи на каждую статью, не раз в 500:
            # чтобы --resume подхватил актуальную точку, даже если упали в
            # середине обработки. apfrom у MediaWiki включает сам заголовок,
            # так что при следующем запуске эта статья попадётся ещё раз —
            # дёшево (revid-скип), в отличие от старой схемы на apcontinue,
            # которая пересматривала всю страницу целиком (до 500 статей).
            if titles is None:
                CHECKPOINT_FILE.write_text(json.dumps({"title": title}, ensure_ascii=False))
            print(f"\n[{i}/{limit or '?'}] {title}")
            try:
                status = await _process_article(title, scraper, llm_client, dry_run, force)
                stats[status] += 1
            except DailyQuotaExceeded:
                # Дневной лимит free tier — ждать до полуночи по Тихоокеанскому
                # смысла нет внутри этого прогона. Ретраить на оставшихся
                # статьях (их могут быть тысячи) — часы впустую, ни одна не
                # пройдёт до сброса лимита. Останавливаем прогон целиком,
                # не считаем это ошибкой конкретной статьи: revid-скип +
                # чекпоинт (--resume) на следующем запуске продолжат ровно
                # с этого места.
                print(f"\n⛔ Дневной лимит Gemini API исчерпан на статье «{title}» "
                      f"({i} обработано в этом прогоне). Обработанное уже сохранено. "
                      f"Запусти снова после сброса квоты (полночь по Тихоокеанскому "
                      f"времени) с флагом --resume — продолжит с этого места.")
                break
            except Exception as exc:
                # Одна проблемная статья не должна ронять весь прогон на тысячах
                # остальных — логируем и идём дальше. Транзиентные сбои сети/API
                # уже покрыты ретраями внутри FandomScraper/generate_embeddings_batch,
                # так что если сюда долетело исключение — это либо баг чанкера на
                # редком случае, либо что-то реально не так со статьёй; разбор — руками
                # через `--title` позже, не автоматическим повтором здесь.
                print(f"❌ {title}: {type(exc).__name__}: {exc}")
                stats["error"] += 1
        else:
            # Цикл дошёл до конца списка статей без break (не упёрлись в дневной
            # лимит) — корпус пройден целиком, чекпоинту дальше нечего хранить.
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
