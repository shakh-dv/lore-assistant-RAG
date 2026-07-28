import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.infrastructure.database.models import Base
from app.infrastructure.database.connection import engine
from app.api.router import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        # pg_trgm — для локального исправления опечаток в запросах.
        # Обязательно ДО create_all: на lore_terms висит GiST-индекс с gist_trgm_ops.
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS pg_trgm'))
        await conn.run_sync(Base.metadata.create_all)

    print("🚀 Сервер успешно запущен, база данных инициализирована!")

    yield

    print("🛑 Выключение сервера...")
    await engine.dispose()


app = FastAPI(
    title="Lore Assistant API",
    description="API для ответов на вопросы по лору",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем всё для локального теста
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы (GET, POST, OPTIONS и т.д.)
    allow_headers=["*"],  # Разрешаем все заголовки
)

app.include_router(chat_router)

# Веб-интерфейс. Монтируется последним, чтобы не перехватывать /api/v1/*
app.mount("/", StaticFiles(directory="static", html=True), name="ui")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)