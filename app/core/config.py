from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API ключ для Google Gemini
    GEMINI_API_KEY: str
    
    # URL для подключения к базе данных
    # Пример: postgresql+psycopg://admin:secretpassword@localhost:5432/lore_db
    DATABASE_URL: str
    
    # Модели Gemini
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    # Фиксированное имя, а не -latest: у -latest квота уезжает вместе с моделью
    CHAT_MODEL: str = "gemini-3.1-flash-lite"

    # Размерность вектора (gemini-embedding-001 по умолчанию отдаёт 3072,
    # но поддерживает урезание через output_dimensionality)
    VECTOR_DIMENSION: int = 768

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Создаем синглтон настроек, который будем импортировать в других файлах
settings = Settings()