import re
from typing import AsyncGenerator
from app.domain.answer_modes import AnswerMode, DEFAULT_MODE
from app.domain.ports import ILLMClient, IVectorStore

# Местоимения, ради раскрытия которых имеет смысл платить за вызов LLM.
# Нет местоимения — нет и смысла: опечатки уже починил Postgres.
_PRONOUNS = {
    "он", "она", "оно", "они", "его", "ее", "их", "ему", "ей", "им",
    "него", "нее", "них", "нем", "ней", "этот", "эта", "это", "эти",
    "тот", "та", "те", "там", "туда", "тогда",
}
_WORDS_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _needs_llm_rewrite(question: str, history: list) -> bool:
    """LLM-переписывание нужно только для раскрытия ссылок на предыдущий контекст."""
    if not history:
        return False
    words = {w.lower().replace("ё", "е") for w in _WORDS_RE.findall(question)}
    return bool(words & _PRONOUNS)


class ChatUseCase:
    """
    Главный сценарий приложения: Ответы на вопросы по лору.
    Никаких зависимостей от окнкретных фрейворков или БД!
    Толко чистая бизнес-логика.
    """

    def __init__(self, vector_store: IVectorStore, llm_client: ILLMClient):
        # Внедрение зависимостей (Dependency Injection)
        # Мы не создаем объекты, мы получаем их извне.
        # Это позволяет нам легко тестировать и заменять компоненты
        self.vector_store = vector_store
        self.llm_client = llm_client

    
    async def execute(
        self,
        user_question: str,
        universe: str,
        history: list = None,
        mode: AnswerMode = DEFAULT_MODE,
    ) -> AsyncGenerator[str, None]:
        """
        Метод-дирижер. Управлякт процессом RAG.
        Возвращает ассинхронный генератор для стриминга.
        """

        # Шаг 1: Опечатки чиним в Postgres по словарю лора — быстро, детерминированно
        # и бесплатно. «Альтарир» -> «Альтаир» без всяких домыслов.
        corrected_question = await self.vector_store.correct_typos(user_question, universe)

        # Шаг 2: LLM зовём только там, где Postgres бессилен, — раскрыть «он», «его».
        # На запросе без местоимений это экономит целый вызов модели
        # и убирает риск, что rewrite сам уведёт поиск не туда.
        if _needs_llm_rewrite(corrected_question, history):
            rewritten_question = await self.llm_client.rewrite_query(corrected_question, history)
        else:
            rewritten_question = corrected_question

        # Шаг 3: Эмбеддинг по улучшенному запросу
        query_vector = await self.llm_client.generate_embedding(rewritten_question)

        relevant_chunks = await self.vector_store.search_similar(
            query_vector,
            universe,
            limit=5
        )

        if not relevant_chunks:
            context_text = 'В архивах нет информации по этой теме.'
        else:
            context_pieces = [chunk['chunk_text'] for chunk in relevant_chunks]
            context_text = '\n\n---\n\n'.join(context_pieces)

        # В генерацию отдаём именно переписанный вопрос: он самодостаточный.
        # С исходным ('а сколько ему лет?') модель не знает, о ком речь —
        # истории диалога она не видит, только контекст из базы.
        stream = self.llm_client.generate_answer_stream(
            prompt=rewritten_question,
            context=context_text,
            mode=mode
        )

        async for chunk in stream:
            yield chunk