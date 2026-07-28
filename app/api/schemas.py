from typing import List, Optional
from pydantic import BaseModel, Field

from app.domain.answer_modes import AnswerMode, DEFAULT_MODE


class ChatMessage(BaseModel):
    """Одно сообщение из истории диалога."""
    role: str = Field(..., description="'user' или 'assistant'")
    content: str = Field(..., description="Текст сообщения")


class ChatRequest(BaseModel):
    """Схема входящего запроса к эндпоинту чата."""
    question: str = Field(..., min_length=1, description="Вопрос пользователя")
    universe: str = Field(..., min_length=1, description="Вселенная/контекст для поиска")
    history: Optional[List[ChatMessage]] = Field(default=None, description="История диалога для раскрытия контекста")
    mode: AnswerMode = Field(default=DEFAULT_MODE, description="Стиль ответа, выбранный в интерфейсе")


class ChatResponse(BaseModel):
    """Схема ответа (используется для документации, реальный ответ — стрим)."""
    answer: str = Field(..., description="Ответ от LLM")


class AnswerModeOut(BaseModel):
    """Доступный стиль ответа — для переключателя в интерфейсе."""
    id: AnswerMode = Field(..., description="Код режима, передаётся в /chat")
    name: str = Field(..., description="Название для кнопки")
    description: str = Field(..., description="Пояснение под кнопкой")
    icon: str = Field(..., description="Имя иконки Phosphor")
    default: bool = Field(..., description="Режим по умолчанию")


class UniverseOut(BaseModel):
    """Вселенная, по которой в базе реально есть данные."""
    id: str = Field(..., description="Код вселенной, передаётся в /chat")
    name: str = Field(..., description="Человекочитаемое название")
    chunks: int = Field(..., description="Сколько чанков загружено")
    articles: int = Field(..., description="Сколько статей загружено")