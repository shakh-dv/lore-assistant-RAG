from google import genai
from google.genai import types
from typing import AsyncGenerator, List

from app.domain.answer_modes import AnswerMode, DEFAULT_MODE

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

    async def generate_embedding(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> list[float]:
        # Асимметричный поиск: чанки кодируем как DOCUMENT, вопрос — как QUERY.
        # Модель тогда кладёт вопрос рядом с отвечающим абзацем, а не рядом
        # с похожим по формулировке. Дефолт — под загрузчики, их большинство.
        result = await self.client.aio.models.embed_content(
            model=self.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.embedding_dimension,
            ),
        )
        return result.embeddings[0].values

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
