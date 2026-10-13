from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    
    # postgresql+psycopg://user:password@localhost:5432/lore_db
    DATABASE_URL: str
    
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    # Фиксированное имя, а не -latest: у -latest квота уезжает вместе с моделью
    CHAT_MODEL: str = "gemini-3.1-flash-lite"

    # gemini-embedding-001 отдаёт 3072, урезаем через output_dimensionality.
    VECTOR_DIMENSION: int = 768

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()