from functools import lru_cache

from fastapi import Depends

# Импорт конфигурации
from app.core.config import settings

# Импорт интерфейсов (Портов)
from app.domain.ports import IVectorStore, ILLMClient

# Импорт адаптеров
from app.infrastructure.postgres_adapter import PostgresVectorStore
from app.infrastructure.gemini_adapter import GeminiAdapter

# Импорт Use Cases
from app.use_cases.chat import ChatUseCase
from app.use_cases.universes import ListUniversesUseCase

# Подключение к БД
from app.infrastructure.database.connection import get_async_session


async def get_vector_store(session=Depends(get_async_session)) -> IVectorStore:
    """Провайдер для векторного хранилища"""
    return PostgresVectorStore(session)


@lru_cache(maxsize=1)
def get_llm_client() -> ILLMClient:
    """
    Провайдер для клиента Gemini.

    Кэшируем: адаптер stateless, а вот genai.Client внутри держит пул соединений
    httpx. Без кэша он создавался бы заново на каждый запрос — пул не
    переиспользовался бы, и TLS-хендшейк шёл бы с нуля каждый раз.
    """
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
    """
    Провайдер для бизнес-логики.
    FastAPI сам разрезолвит vector_store и llm_client и передаст их сюда.
    """
    return ChatUseCase(vector_store=vector_store, llm_client=llm_client)


def get_universes_use_case(
    vector_store: IVectorStore = Depends(get_vector_store)
) -> ListUniversesUseCase:
    """Провайдер для списка доступных вселенных."""
    return ListUniversesUseCase(vector_store=vector_store)
