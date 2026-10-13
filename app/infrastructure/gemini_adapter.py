"""
Обёртка над google-genai: эмбеддинги (с троттлингом, ретраями и детектом
дневной квоты) и генерация ответа. Путь эмбеддинга и лимиты free tier —
ARCHITECTURE.md, 5.3.
"""
import asyncio
import time

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from typing import AsyncGenerator, List

from app.domain.answer_modes import AnswerMode, DEFAULT_MODE

# Не больше _EMBED_RATE_LIMIT_PER_MINUTE (см. assert ниже).
_EMBED_BATCH_SIZE = 80
_EMBED_MAX_ATTEMPTS = 5
_RETRYABLE_CLIENT_CODES = {429}


class DailyQuotaExceeded(RuntimeError):
    """
    Дневная квота (~1000/день) исчерпана — ретраи бессмысленны до полуночи по
    Тихоокеанскому. Отдельный класс, чтобы индексатор остановил весь прогон.
    """


def _has_daily_quota_violation(exc: ClientError) -> bool:
    """
    Есть ли среди QuotaFailure.violations quotaId с PerDay. По структуре, а не
    поиском по строке: один 429 может перечислять несколько квот.
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


# Free tier — 100 текстов/мин; 80 — запас на неточность скользящего окна.
_EMBED_RATE_LIMIT_PER_MINUTE = 80

assert _EMBED_BATCH_SIZE <= _EMBED_RATE_LIMIT_PER_MINUTE, (
    "_EMBED_BATCH_SIZE не может быть больше _EMBED_RATE_LIMIT_PER_MINUTE — "
    "иначе _throttle_embed_rate уходит в бесконечную рекурсию на батче, "
    "который сам по себе больше минутного окна"
)

# Промт и температура на режим ответа. Правила про разные вселенные нет:
# поиск и так фильтрует чанки по одной.
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
        self.client = genai.Client(api_key=api_key)
        self.chat_model = chat_model_name
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
        # Чанки — DOCUMENT, вопрос чата — QUERY: тогда вопрос ложится рядом с
        # отвечающим абзацем, а не с похожим по формулировке.
        return (await self.generate_embeddings_batch([text], task_type))[0]

    async def generate_embeddings_batch(
        self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[list[float]]:
        """Эмбеддинги в порядке texts, суб-батчами по _EMBED_BATCH_SIZE."""
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
                if attempt == _EMBED_MAX_ATTEMPTS:
                    raise
                reason = f"{type(exc).__name__}: {exc}"
            except ClientError as exc:
                # Повторять имеет смысл только 429, остальные 4xx не изменятся.
                if exc.code not in _RETRYABLE_CLIENT_CODES:
                    raise
                if _has_daily_quota_violation(exc):
                    print(f"⛔ Daily quota violation, сырой ответ: {exc.details}")
                    raise DailyQuotaExceeded(str(exc)) from exc
                if attempt == _EMBED_MAX_ATTEMPTS:
                    raise
                reason = f"{type(exc).__name__}: {exc}"
            else:
                # Рассинхрон векторов и чанков ничего не роняет, а поиск тихо врёт.
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
        # Незнакомый режим из UI не роняет запрос.
        system_instruction, temperature = _MODES.get(mode, _MODES[DEFAULT_MODE])

        full_prompt = (
            f"{system_instruction}\n\n"
            f"Контекст:\n{context}\n\n"
            f"Вопрос пользователя: {prompt}\n"
        )

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
        """Переписывает вопрос для поиска: опечатки и местоимения по истории диалога."""
        history_text = ""
        if history:
            history_text = "Контекст диалога:\n"
            for msg in history[-3:]:
                history_text += f"{msg['role']}: {msg['content']}\n"

        # Не обрывать промт на «Исправленный запрос:» — Gemini 3 (thinking-модель)
        # на такое отвечает MALFORMED_RESPONSE без текста.
        prompt = (
            "Перепиши поисковый запрос пользователя для векторной базы знаний.\n"
            "1. Исправь опечатки.\n"
            "2. Замени местоимения ('он', 'его', 'эта') на имя из контекста диалога.\n"
            "3. Сохрани смысл. Не отвечай на запрос, только перепиши его.\n"
            "Верни одну строку — готовый запрос, без кавычек и пояснений.\n\n"
            f"{history_text}\n"
            f"Запрос: {user_question}"
        )

        # Вспомогательный шаг: при любом сбое ищем по исходному вопросу.
        try:
            response = await self.client.aio.models.generate_content(
                model=self.chat_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            corrected_query = (response.text or "").strip()
        except Exception as e:
            print(f"⚠️  Query Rewriter недоступен ({type(e).__name__}), ищу по оригиналу.")
            return user_question

        if not corrected_query:
            print("⚠️  Query Rewriter вернул пустой ответ, ищу по оригиналу.")
            return user_question

        print(f"🧠 Query Rewriter: '{user_question}' -> '{corrected_query}'")
        return corrected_query
