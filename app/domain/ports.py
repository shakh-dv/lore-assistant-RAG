from typing import Protocol, AsyncGenerator, List, Dict, Any

from app.domain.answer_modes import AnswerMode, DEFAULT_MODE

class IVectorStore(Protocol):
    """
    Порт для работы с векторной базой данных.
    Бизнес-логике неважно, Postgres это, Redis или Pinecone.
    """
    async def save_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        ...

    async def search_similar(self, query_vector: List[float], universe: str, limit: int = 5) -> List[Dict[str, Any]]:
        ...

    async def list_universes(self) -> List[Dict[str, Any]]:
        """Вселенные, реально присутствующие в хранилище, с числом чанков."""
        ...

    async def correct_typos(self, query: str, universe: str) -> str:
        """Исправить опечатки в запросе по словарю лора. Без обращения к LLM."""
        ...

class ILLMClient(Protocol):
    """
    Порт для работы с LLM. 
    Бизнес-логике неважно, Gemini это, OpenAI или локальная Llama.
    """
    async def generate_embedding(
        self, text: str, task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[float]:
        """Вектор текста. Для вопроса пользователя нужен task_type="RETRIEVAL_QUERY"."""
        ...

    async def generate_answer_stream(
        self, prompt: str, context: str, mode: AnswerMode = DEFAULT_MODE
    ) -> AsyncGenerator[str, None]:
        ...

    async def rewrite_query(self, user_question: str, history: list = None) -> str:
        ...
    