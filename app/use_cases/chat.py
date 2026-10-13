import re
from typing import AsyncGenerator
from app.domain.answer_modes import AnswerMode, DEFAULT_MODE
from app.domain.ports import ILLMClient, IVectorStore

# LLM-rewrite нужен только чтобы раскрыть местоимение; опечатки чинит Postgres.
_PRONOUNS = {
    "он", "она", "оно", "они", "его", "ее", "их", "ему", "ей", "им",
    "него", "нее", "них", "нем", "ней", "этот", "эта", "это", "эти",
    "тот", "та", "те", "там", "туда", "тогда",
}
_WORDS_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _needs_llm_rewrite(question: str, history: list) -> bool:
    if not history:
        return False
    words = {w.lower().replace("ё", "е") for w in _WORDS_RE.findall(question)}
    return bool(words & _PRONOUNS)


class ChatUseCase:
    """Ответ на вопрос по лору (RAG). Зависит только от портов, не от БД и SDK."""

    def __init__(self, vector_store: IVectorStore, llm_client: ILLMClient):
        self.vector_store = vector_store
        self.llm_client = llm_client

    
    async def execute(
        self,
        user_question: str,
        universe: str,
        history: list = None,
        mode: AnswerMode = DEFAULT_MODE,
    ) -> AsyncGenerator[str, None]:
        """Опечатки -> rewrite -> эмбеддинг -> поиск -> стриминг ответа."""

        corrected_question = await self.vector_store.correct_typos(user_question, universe)

        if _needs_llm_rewrite(corrected_question, history):
            rewritten_question = await self.llm_client.rewrite_query(corrected_question, history)
        else:
            rewritten_question = corrected_question

        # Единственный внешний вызов до стриминга: при сбое (квота) — вежливый
        # ответ вместо traceback. Ловим широко: use case не знает ошибок SDK.
        try:
            query_vector = await self.llm_client.generate_embedding(
                rewritten_question, task_type="RETRIEVAL_QUERY"
            )
        except Exception as exc:
            yield (
                "Не получилось обратиться к архивам — сервис эмбеддингов сейчас "
                "недоступен (возможно, исчерпан дневной лимит API). "
                "Попробуй ещё раз чуть позже."
            )
            print(f"⚠️  Ошибка эмбеддинга вопроса: {type(exc).__name__}: {exc}")
            return

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

        # Переписанный вопрос самодостаточен: историю диалога модель не видит.
        stream = self.llm_client.generate_answer_stream(
            prompt=rewritten_question,
            context=context_text,
            mode=mode
        )

        async for chunk in stream:
            yield chunk