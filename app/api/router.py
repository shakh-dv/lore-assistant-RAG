from typing import AsyncGenerator, List
from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

# Импортируем только то, что нужно роутеру
from app.api.schemas import ChatRequest, UniverseOut, AnswerModeOut
from app.api.dependencies import get_chat_use_case, get_universes_use_case
from app.domain.answer_modes import list_modes
from app.use_cases.chat import ChatUseCase
from app.use_cases.universes import ListUniversesUseCase

router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.get("/modes", response_model=List[AnswerModeOut])
async def modes_endpoint():
    """
    Стили ответа для переключателя в интерфейсе.
    Список задаётся в app/domain/answer_modes.py — фронт его не хардкодит.
    """
    return list_modes()


@router.get("/universes", response_model=List[UniverseOut])
async def universes_endpoint(
    use_case: ListUniversesUseCase = Depends(get_universes_use_case)
):
    """
    Список вселенных, по которым в базе реально есть данные.
    Пустая база — пустой список.
    """
    return await use_case.execute()

@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    use_case: ChatUseCase = Depends(get_chat_use_case)
):
    """
    Эндпоинт чата.
    """
    # Конвертируем Pydantic модели в простые dict для совместимости с rewrite_query
    history = [msg.model_dump() for msg in request.history] if request.history else None

    stream: AsyncGenerator[str, None] = use_case.execute(
        user_question=request.question,
        universe=request.universe,
        history=history,
        mode=request.mode
    )

    return StreamingResponse(stream, media_type="text/event-stream")