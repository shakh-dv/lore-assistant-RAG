"""
Скрапер Fandom (Assassin's Creed Wiki, ru) через MediaWiki API, без парсинга HTML.

1 запрос/сек. maxlag=5 — конвенция MediaWiki: при отставании реплики сервер
просит подождать, а не отдаёт рассинхронизированные данные.
Cloudflare и детали API — CLAUDE.md, «Как забирать данные с Fandom».
"""
import asyncio
import time
from typing import AsyncGenerator, Optional

import httpx

BASE_URL = "https://assassinscreed.fandom.com/ru/api.php"

# MediaWiki просит контакт в User-Agent. Только ASCII — иначе httpx падает.
USER_AGENT = "LoreAssistantRAG/1.0 (student project; +https://github.com/shakh-dv/lore-assistant-RAG-)"

RATE_LIMIT_SECONDS = 1.0
MAX_ATTEMPTS = 5

# Временные — повтор поможет; missingtitle и т.п. повтор не изменит.
_TRANSIENT_ERROR_CODES = {"maxlag", "ratelimited"}
_PERMANENT_ARTICLE_ERRORS = {"missingtitle", "invalidtitle", "nosuchpageid"}


class MediaWikiAPIError(RuntimeError):
    """Ошибка в теле ответа API — MediaWiki отдаёт их с HTTP 200, не только 4xx/5xx."""

    def __init__(self, code: str, info: str):
        self.code = code
        self.info = info
        super().__init__(f"{code}: {info}")


class FandomScraper:
    """Тонкая обёртка над api.php: список статей + содержимое одной статьи."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        user_agent: str = USER_AGENT,
        rate_limit_seconds: float = RATE_LIMIT_SECONDS,
    ):
        self._url = base_url
        self._client = httpx.AsyncClient(headers={"User-Agent": user_agent}, timeout=30.0)
        self._rate_limit = rate_limit_seconds
        self._last_request = 0.0
        # Лок держит rate limit и при будущих параллельных вызовах.
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "FandomScraper":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def _backoff(self, attempt: int, reason: str) -> None:
        wait = min(2 ** attempt, 30)
        print(f"⏳ Повтор {attempt}/{MAX_ATTEMPTS} после сбоя ({reason}), жду {wait}с...")
        await asyncio.sleep(wait)

    async def _get(self, params: dict) -> dict:
        """Запрос с ретраями сетевых сбоев, 5xx/429 и maxlag; остальное — сразу наверх."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            async with self._lock:
                elapsed = time.monotonic() - self._last_request
                if elapsed < self._rate_limit:
                    await asyncio.sleep(self._rate_limit - elapsed)
                try:
                    response = await self._client.get(
                        self._url,
                        params={**params, "format": "json", "formatversion": "2", "maxlag": 5},
                    )
                    self._last_request = time.monotonic()
                    response.raise_for_status()
                    data = response.json()
                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    if attempt == MAX_ATTEMPTS:
                        raise
                    reason = f"{type(exc).__name__}: {exc}"
                    should_retry = True
                else:
                    should_retry = False

            if not should_retry:
                error = data.get("error")
                if error is None:
                    return data
                if error["code"] not in _TRANSIENT_ERROR_CODES:
                    raise MediaWikiAPIError(error["code"], error.get("info", ""))
                if attempt == MAX_ATTEMPTS:
                    raise MediaWikiAPIError(error["code"], error.get("info", ""))
                reason = f"{error['code']}: {error.get('info', '')}"

            await self._backoff(attempt, reason)

        raise RuntimeError("unreachable")  # цикл всегда возвращает или кидает раньше

    async def list_all_titles(
        self, resume_from: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Заголовки статей основного пространства, без редиректов. resume_from —
        заголовок, с которого начать включительно (apfrom), для --resume.
        """
        apcontinue: Optional[str] = None
        apfrom: Optional[str] = resume_from
        fetched = 0
        while True:
            params = {
                "action": "query",
                "list": "allpages",
                "apnamespace": 0,
                "aplimit": 500,
                "apfilterredir": "nonredirects",
            }
            if apcontinue:
                params["apcontinue"] = apcontinue
            elif apfrom:
                params["apfrom"] = apfrom

            data = await self._get(params)
            for page in data["query"]["allpages"]:
                yield page["title"]
            fetched += len(data["query"]["allpages"])
            print(f"📄 Заголовков собрано: {fetched}...")

            apfrom = None  # только для первого запроса
            cont = data.get("continue")
            if not cont:
                break
            apcontinue = cont["apcontinue"]

    async def fetch_article(self, title: str) -> Optional[dict]:
        """
        Wikitext, секции, категории и revid одним вызовом parse. None, если
        статьи нет. redirects=1 — чтобы --title по редиректу дал статью-цель.
        """
        try:
            data = await self._get({
                "action": "parse",
                "page": title,
                "redirects": "1",
                # revid — отдельный prop, без явного запроса MediaWiki его не отдаёт.
                "prop": "wikitext|sections|categories|revid",
            })
        except MediaWikiAPIError as exc:
            if exc.code in _PERMANENT_ARTICLE_ERRORS:
                print(f"⚠️  Пропускаю «{title}»: {exc}")
                return None
            raise

        parse = data["parse"]
        return {
            "title": parse["title"],
            "revid": parse["revid"],
            "wikitext": parse["wikitext"],
            "sections": parse["sections"],
            "categories": [c["category"] for c in parse.get("categories", [])],
        }

    async def close(self) -> None:
        await self._client.aclose()


async def _smoke_test() -> None:
    """Быстрая проверка: первые 5 заголовков, одна статья целиком, одна несуществующая."""
    async with FandomScraper() as scraper:
        titles = []
        async for title in scraper.list_all_titles():
            titles.append(title)
            if len(titles) >= 5:
                break
        print(f"📄 Первые статьи: {titles}")

        article = await scraper.fetch_article(titles[0])
        print(f"✅ {article['title']} (revid={article['revid']})")
        print(f"   Секций: {len(article['sections'])}, категорий: {article['categories']}")
        print(f"   Wikitext: {len(article['wikitext'])} символов")

        missing = await scraper.fetch_article("ЭтойСтраницыТочноНеСуществует12345")
        print(f"✅ Несуществующая статья вернула: {missing!r}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
