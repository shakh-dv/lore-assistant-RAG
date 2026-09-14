"""
Скрапер Fandom (Assassin's Creed Wiki, ru) через MediaWiki API.

Никакого парсинга HTML: у Fandom есть action=parse и action=query,
которые отдают уже структурированные данные (wikitext, дерево секций,
категории). Детали см. в CLAUDE.md, раздел "Как забирать данные с Fandom".

Rate limit — 1 запрос/сек (последовательно, без параллельных вызовов):
Fandom терпимый, но по договорённости не наглеем. maxlag=5 — конвенция
MediaWiki: если реплика БД отстаёт больше чем на 5с, сервер сам просит
подождать (ошибкой в теле или HTTP 503), вместо того чтобы отдать
рассинхронизированные данные под нагрузкой.

Это только слой получения сырых данных. WikiChunker (разбор wikitext
на чанки по секциям) и заливка в БД — следующий шаг, добавляются
отдельно поверх этого модуля.

Важно для WikiChunker (на будущее, не про этот файл): MediaWiki
`sections` не включает lead — текст ДО первого `==Заголовка==`. Это
`wikitext[:sections[0]['byteoffset']]`, отдельный чанк с
section_path=article_title. Часто самая важная часть статьи для
коротких вопросов — забыть про неё легко.
"""
import asyncio
import time
from typing import AsyncGenerator, Optional

import httpx

BASE_URL = "https://assassinscreed.fandom.com/ru/api.php"

# MediaWiki просит ставить в User-Agent контакт того, кто дёргает API.
# HTTP-заголовки — только ASCII, кириллицу сюда класть нельзя (httpx уронит
# UnicodeEncodeError). Контакт — публичный репозиторий проекта: по конвенции
# MediaWiki подходит и URL, личный email в код не кладём.
USER_AGENT = "LoreAssistantRAG/1.0 (student project; +https://github.com/shakh-dv/lore-assistant-RAG-)"

RATE_LIMIT_SECONDS = 1.0
MAX_ATTEMPTS = 5

# maxlag и ratelimited — временные, инфраструктурные: имеет смысл подождать
# и повторить тот же запрос. missingtitle/invalidtitle/nosuchpageid — свойство
# конкретной статьи, повторный запрос даст тот же результат всегда.
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
        # Лок нужен даже без параллельных вызовов сейчас — защищает от того,
        # что кто-то потом решит дёрнуть fetch_article параллельно и молча
        # пробьёт rate limit.
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
        """
        Один логический запрос с ретраями. Ретраится только то, что имеет шанс
        пройти при повторе: сетевые сбои, 5xx/429, maxlag/ratelimited в теле.
        Всё остальное (например missingtitle) поднимается сразу — вызывающий
        код (fetch_article) решает, фатально это для статьи или нет.
        """
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
        Все статьи основного пространства имён (без редиректов и служебных).
        Пагинация — обычный проход по apcontinue из ответа MediaWiki (до 500
        заголовков за страницу).

        resume_from — точный заголовок статьи, с которого начать (включительно,
        через параметр apfrom MediaWiki) — для --resume в index_fandom.py.
        В отличие от apcontinue (даёт точку только на границе страницы allpages,
        до 500 статей), apfrom стартует ровно с нужного заголовка: --resume
        пересматривает не всю текущую страницу заново, а максимум одну статью
        (саму resume_from — apfrom включает её) — дёшево даже без пропуска,
        не то что 500.
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
                # Редирект — не статья, а указатель на другую. Со своим текстом
                # он даёт в базе пустой/дублирующий чанк.
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

            apfrom = None  # apfrom — только для самого первого запроса
            cont = data.get("continue")
            if not cont:
                break
            apcontinue = cont["apcontinue"]

    async def fetch_article(self, title: str) -> Optional[dict]:
        """
        Wikitext + дерево секций + категории одним вызовом parse
        (три prop в одном запросе экономнее для rate limit, чем три запроса).

        None, если статья не существует/невалидна/удалена — это нормальный
        случай на тысячах заголовков (например переименование между
        составлением списка и запросом), не повод ронять весь прогон.

        redirects=1 — без него редирект (например "Эцио Аудиторе" ->
        "Эцио Аудиторе да Фиренце") отдаёт СВОЙ текст ("#перенаправление
        [[...]]", пара десятков символов), а не статью-цель. list_all_titles
        такие страницы уже фильтрует (apfilterredir), но при точечном вызове
        через заголовок (--title) можно попасть на редирект напрямую —
        без этого параметра тихо получили бы мусорный чанк вместо статьи.
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
