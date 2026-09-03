"""
Обёртка над google-genai. Два независимых пула квоты Gemini — эмбеддинги
(embed_content) и генерация ответа (generate_content) — история этого файла
почти целиком про первый: сегодняшний прогон живьём поймал и минутный (RPM),
и дневной (RPD) лимит free tier, generate_answer_stream/rewrite_query их
не задевают вообще, там всё осталось как было.

Путь одного эмбеддинга — generate_embedding (один текст, вопрос в чате) сам
не ходит в API, а зовёт generate_embeddings_batch([text]) — раньше это были
две независимые реализации, из-за чего чат падал сырым traceback'ом, пока
у батч-версии для индексатора уже была вся защита ниже:

  generate_embeddings_batch  — режет список текстов на суб-батчи по ≤80
  -> _embed_sub_batch         — до 5 попыток на суб-батч:
       _throttle_embed_rate   — упреждающая пауза, если за 60с текстов
                                 набежало больше 80 (лимит free tier — 100,
                                 80 — запас на неточность подсчёта). Батч
                                 не больше 80 — не совпадение: если суб-батч
                                 сам по себе больше лимита, троттлинг никогда
                                 не пропустит его (см. assert у констант)
       embed_content          — сам вызов; на 429 смотрим КОД ошибки:
         _has_daily_quota_violation — если это дневной лимит (PerDay,
                                       ~1000/день) — ретраить бессмысленно,
                                       поднимаем DailyQuotaExceeded сразу
         иначе (PerMinute/сеть)     — backoff и ещё попытка

DailyQuotaExceeded — отдельный класс специально для того, чтобы вызывающий
код (index_fandom.py) мог остановить ВЕСЬ прогон, а не просто пропустить
одну статью: после дневного лимита ни одна из оставшихся не пройдёт всё
равно, ретраить их по одной — часы впустую.
"""
import asyncio
import time

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from typing import AsyncGenerator, List

from app.domain.answer_modes import AnswerMode, DEFAULT_MODE

# Точного лимита Gemini на число текстов в одном embed_content не нашли в SDK —
# консервативный дефолт, подстроить по факту первого реального прогона.
# РАВЕН _EMBED_RATE_LIMIT_PER_MINUTE (см. assert ниже, после её определения) —
# иначе суб-батч сам по себе может быть больше минутного окна, и
# _throttle_embed_rate уходит в бесконечную рекурсию: условие
# used + upcoming > лимит остаётся истинным при любом used >= 0, сколько ни жди.
_EMBED_BATCH_SIZE = 80
_EMBED_MAX_ATTEMPTS = 5
_RETRYABLE_CLIENT_CODES = {429}


class DailyQuotaExceeded(RuntimeError):
    """
    Free tier Gemini embed_content — не только RPM (100/мин, самовосстанавливается
    ретраем), но и RPD (~1000/день, сбрасывается в полночь по Тихоокеанскому
    времени). Дневной лимит ретраем не полечить — ждать нечего, quotaId в
    ошибке содержит "PerDay" вместо "PerMinute". Раз обнаружено — не тратим
    время на 5 попыток backoff на КАЖДОЙ из оставшихся статей корпуса, поднимаем
    сразу, чтобы вызывающий код (index_fandom.py) мог остановить весь прогон.
    """


def _has_daily_quota_violation(exc: ClientError) -> bool:
    """
    exc.details — разобранный JSON тела ответа: {"error": {..., "details": [...]}}.
    Ищем среди QuotaFailure.violations запись, чей quotaId — про PerDay, а не
    про PerMinute. Структурный доступ, а не str(exc).find("PerDay") — 429 может
    перечислить сразу несколько нарушенных квот в одном ответе.
    """
    try:
        error_details = exc.details.get("error", {}).get("details", [])
    except AttributeError:
        return False

    for entry in error_details:
        if not str(entry.get("@type", "")).endswith("QuotaFailure"):
            continue
        for violation in entry.get("violations", []):
            if "PerDay" in violation.get("quotaId", ""):
                return True

    return False


# Обнаружено эмпирически на реальном прогоне (429 RESOURCE_EXHAUSTED):
# free tier Gemini embed_content — 100 текстов/мин (quotaId
# EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier). Ретрай
# с backoff и так восстанавливается, но на 2984 статьях будет упираться в
# лимит почти на каждой — тратим время на паузы вместо того, чтобы просто
# не разгоняться. 80, не 100 — запас на неточность подсчёта скользящего окна.
_EMBED_RATE_LIMIT_PER_MINUTE = 80

# Инвариант, не рантайм-проверка: если это когда-нибудь станет неправдой,
# лучше явный AssertionError при импорте, чем часовое зависание в середине
# прогона на 2000-й статье (см. комментарий у _EMBED_BATCH_SIZE выше).
assert _EMBED_BATCH_SIZE <= _EMBED_RATE_LIMIT_PER_MINUTE, (
    "_EMBED_BATCH_SIZE не может быть больше _EMBED_RATE_LIMIT_PER_MINUTE — "
    "иначе _throttle_embed_rate уходит в бесконечную рекурсию на батче, "
    "который сам по себе больше минутного окна"
)

# Промт и температура на каждый режим ответа.
# Правила про коллизии между вселенными тут нет намеренно: поиск фильтрует
# чанки по одной вселенной, так что в контекст двух разных «Яблок» не попадёт.
_STORYTELLER = (
    "Ты — харизматичный рассказчик и знаток лора. Твоя задача — погружать пользователя в историю, "
    "основываясь на предоставленных архивах.\n\n"

    "ТВОИ ПРАВИЛА (ОБЯЗАТЕЛЬНЫ):\n"
    "1. ТВОРЧЕСКАЯ ПОДАЧА: Изучи факты из контекста и расскажи их своими словами. Делай историю увлекательной, живой и литературной. Не копируй текст из базы дословно.\n"
    "2. ГРАНИЦА ВЫМЫСЛА (КРИТИЧНО): Украшать можно только язык — интонацию, ритм, построение фразы. "
    "Категорически нельзя добавлять то, чего нет в контексте: факты, события, имена, места, числа и даты. "
    "Если тянет дописать красивую подробность, которой нет в архиве, — не дописывай.\n"
    "3. ОТСУТСТВИЕ ДАННЫХ: Если контекст не дает ответа на вопрос, не выдумывай его. Изящно ответь: 'К сожалению, об этом история умалчивает' или 'В моих свитках нет таких деталей'.\n\n"

    "СТИЛЬ:\n"
    "- Избегай роботизированных фраз ('В предоставленном контексте сказано', 'Согласно базе данных'). Говори так, будто ты сам помнишь эти события.\n"
    "- Пиши хлестко и без лишней воды."
)

_ARCHIVIST = (
    "Ты — архивариус базы знаний. Отвечаешь строго по предоставленным архивам.\n\n"

    "ТВОИ ПРАВИЛА (ОБЯЗАТЕЛЬНЫ):\n"
    "1. ТОЛЬКО КОНТЕКСТ: Используй исключительно факты из архивов ниже. Внешние знания не привлекай.\n"
    "2. БЕЗ УКРАШАТЕЛЬСТВА: Держись близко к формулировкам источника. Не добавляй эпитетов, "
    "настроения и подробностей, которых в тексте нет. Сухо и по делу.\n"
    "3. ОТСУТСТВИЕ ДАННЫХ: Если ответа в архивах нет, так и напиши: "
    "'В моих архивах нет информации об этом'. Не догадывайся и не достраивай.\n"
    "4. ФОРМА: Короткие абзацы, при перечислении — список. Без вступлений и без выводов от себя."
)

_MODES = {
    AnswerMode.STORYTELLER: (_STORYTELLER, 0.5),
    AnswerMode.ARCHIVIST: (_ARCHIVIST, 0.1),
}


class GeminiAdapter:
    def __init__(self, api_key: str, chat_model_name: str, embedding_model_name: str, embedding_dimension: int):
        # Клиент создаётся лениво: в конструкторе сетевых вызовов нет.
        self.client = genai.Client(api_key=api_key)
        self.chat_model = chat_model_name
        # Префикс 'models/' новый SDK подставляет сам, вручную ничего не клеим.
        self.embedding_model = embedding_model_name
        self.embedding_dimension = embedding_dimension
        # Скользящее окно: (время_вызова, число_текстов) за последние ~60с.
        self._embed_usage: list[tuple[float, int]] = []

    async def _throttle_embed_rate(self, upcoming: int) -> None:
        """Ждём, если после этого вызова уйдём за _EMBED_RATE_LIMIT_PER_MINUTE."""
        now = time.monotonic()
        self._embed_usage = [(t, n) for t, n in self._embed_usage if now - t < 60]
        used = sum(n for _, n in self._embed_usage)

        if used + upcoming > _EMBED_RATE_LIMIT_PER_MINUTE:
            oldest = self._embed_usage[0][0] if self._embed_usage else now
            wait = max(60 - (now - oldest), 1)
            print(f"⏳ Пауза {wait:.0f}с — держим темп ниже {_EMBED_RATE_LIMIT_PER_MINUTE} текстов/мин (free tier: 100)")
            await asyncio.sleep(wait)
            await self._throttle_embed_rate(upcoming)
            return

        self._embed_usage.append((now, upcoming))

    async def generate_embedding(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[float]:
        # Асимметричный поиск: чанки кодируем как DOCUMENT, вопрос — как QUERY.
        # Модель тогда кладёт вопрос рядом с отвечающим абзацем, а не рядом
        # с похожим по формулировке. Дефолт — под загрузчики, их большинство.
        #
        # Переиспользует generate_embeddings_batch (не отдельный embed_content
        # напрямую) — раньше был отдельный вызов без ретрая/троттлинга/детекта
        # квоты вообще, из-за чего чат падал сырым traceback на первом же 429
        # вместо мягкой деградации, которая уже была у индексатора.
        return (await self.generate_embeddings_batch([text], task_type))[0]

    async def generate_embeddings_batch(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """
        Батч-версия generate_embedding — один вызов API на несколько текстов
        вместо цикла поштучных. Порядок результата соответствует порядку texts.
        """
        if not texts:
            return []

        results: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            sub_batch = texts[i:i + _EMBED_BATCH_SIZE]
            results.extend(await self._embed_sub_batch(sub_batch, task_type))
        return results

    async def _embed_sub_batch(self, sub_batch: list[str], task_type: str) -> list[list[float]]:
        for attempt in range(1, _EMBED_MAX_ATTEMPTS + 1):
            await self._throttle_embed_rate(len(sub_batch))
            try:
                result = await self.client.aio.models.embed_content(
                    model=self.embedding_model,
                    contents=sub_batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.embedding_dimension,
                    ),
                )
            except (ServerError, httpx.TransportError) as exc:
                # bare `raise` обязан жить ВНУТРИ except-блока — вне его Python
                # теряет активное исключение (RuntimeError: No active exception
                # to reraise), поймано на тесте перед этой правкой.
                if attempt == _EMBED_MAX_ATTEMPTS:
                    raise
                reason = f"{type(exc).__name__}: {exc}"
            except ClientError as exc:
                # 429 — rate limit, стоит подождать и повторить. Любой другой
                # 4xx (400/401/403/404) — повтор даст тот же результат, поднимаем сразу.
                if exc.code not in _RETRYABLE_CLIENT_CODES:
                    raise
                # PerDay, а не PerMinute — до завтра не подождём, retry бессмыслен.
                # Читаем quotaId ИЗ структуры violations, не грепаем строку целиком —
                # 429 может перечислять сразу несколько квот в одном ответе, и
                # плоский поиск подстроки рискует сработать не на той причине.
                if _has_daily_quota_violation(exc):
                    print(f"⛔ Daily quota violation, сырой ответ: {exc.details}")
                    raise DailyQuotaExceeded(str(exc)) from exc
                if attempt == _EMBED_MAX_ATTEMPTS:
                    raise
                reason = f"{type(exc).__name__}: {exc}"
            else:
                # Рассинхронизация порядка эмбеддингов и чанков — худший вид
                # бага в RAG: ничего не падает, retrieval работает, но выдаёт
                # не те чанки. Лучше явный сбой сразу.
                if len(result.embeddings) != len(sub_batch):
                    raise RuntimeError(
                        f"Gemini вернул {len(result.embeddings)} эмбеддингов "
                        f"на {len(sub_batch)} текстов — рассинхрон"
                    )
                return [e.values for e in result.embeddings]

            wait = min(2 ** attempt, 30)
            print(f"⏳ Эмбеддинг: повтор {attempt}/{_EMBED_MAX_ATTEMPTS} после сбоя ({reason}), жду {wait}с...")
            await asyncio.sleep(wait)

        raise RuntimeError("unreachable")  # цикл всегда возвращает или кидает раньше

    async def generate_answer_stream(
        self, prompt: str, context: str, mode: AnswerMode = DEFAULT_MODE
    ) -> AsyncGenerator[str, None]:
        # Режим приходит из UI. Незнакомое значение не роняет запрос — берём режим по умолчанию.
        system_instruction, temperature = _MODES.get(mode, _MODES[DEFAULT_MODE])

        full_prompt = (
            f"{system_instruction}\n\n"
            f"Контекст:\n{context}\n\n"
            f"Вопрос пользователя: {prompt}\n"
        )

        # generate_content_stream в async-клиенте — корутина, возвращающая асинхронный итератор.
        stream = await self.client.aio.models.generate_content_stream(
            model=self.chat_model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )

        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def rewrite_query(self, user_question: str, history: list = None) -> str:
        """
        Умный пре-процессинг: исправляем опечатки и раскрываем контекст.
        """
        history_text = ""
        if history:
            history_text = "Контекст диалога:\n"
            for msg in history[-3:]: # Берем последние 3 сообщения для экономии
                history_text += f"{msg['role']}: {msg['content']}\n"

        # ВАЖНО: промт не должен обрываться на приманке вида "Исправленный запрос:".
        # Gemini 3 — thinking-модель, и на такой «допиши за меня» она детерминированно
        # возвращает finish_reason=MALFORMED_RESPONSE без текстовых частей.
        # Поэтому задача ставится явно, а формат ответа описан словами.
        prompt = (
            "Перепиши поисковый запрос пользователя для векторной базы знаний.\n"
            "1. Исправь опечатки.\n"
            "2. Замени местоимения ('он', 'его', 'эта') на имя из контекста диалога.\n"
            "3. Сохрани смысл. Не отвечай на запрос, только перепиши его.\n"
            "Верни одну строку — готовый запрос, без кавычек и пояснений.\n\n"
            f"{history_text}\n"
            f"Запрос: {user_question}"
        )

        # Для rewrite (извлечения фактов) используем нулевую температуру! Нам тут креатив не нужен.
        # Rewrite — вспомогательный шаг: если он упал (пустой ответ, фильтр, квота),
        # ищем по исходному вопросу, а не роняем весь запрос пользователя.
        try:
            response = await self.client.aio.models.generate_content(
                model=self.chat_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            # В новом SDK .text возвращает None, если модель не отдала текстовых частей.
            corrected_query = (response.text or "").strip()
        except Exception as e:
            print(f"⚠️  Query Rewriter недоступен ({type(e).__name__}), ищу по оригиналу.")
            return user_question

        if not corrected_query:
            print("⚠️  Query Rewriter вернул пустой ответ, ищу по оригиналу.")
            return user_question

        print(f"🧠 Query Rewriter: '{user_question}' -> '{corrected_query}'")
        return corrected_query
