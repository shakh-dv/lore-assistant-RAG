from functools import lru_cache

from fastapi import Depends

from app.core.config import settings
from app.domain.ports import IVectorStore, ILLMClient
from app.infrastructure.postgres_adapter import PostgresVectorStore
from app.infrastructure.gemini_adapter import GeminiAdapter
from app.use_cases.chat import ChatUseCase
from app.use_cases.universes import ListUniversesUseCase
from app.infrastructure.database.connection import get_async_session


async def get_vector_store(session=Depends(get_async_session)) -> IVectorStore:
    """Провайдер для векторного хранилища"""
    return PostgresVectorStore(session)


@lru_cache(maxsize=1)
def get_llm_client() -> ILLMClient:
    """Один клиент на процесс: genai.Client держит пул соединений httpx."""
    return GeminiAdapter(
        api_key=settings.GEMINI_API_KEY,
        chat_model_name=settings.CHAT_MODEL,
        embedding_model_name=settings.EMBEDDING_MODEL,
        embedding_dimension=settings.VECTOR_DIMENSION
    )


def get_chat_use_case(
    vector_store: IVectorStore = Depends(get_vector_store),
    llm_client: ILLMClient = Depends(get_llm_client)
) -> ChatUseCase:
    """Провайдер ChatUseCase."""
    return ChatUseCase(vector_store=vector_store, llm_client=llm_client)


def get_universes_use_case(
    vector_store: IVectorStore = Depends(get_vector_store)
) -> ListUniversesUseCase:
    """Провайдер для списка доступных вселенных."""
    return ListUniversesUseCase(vector_store=vector_store)
