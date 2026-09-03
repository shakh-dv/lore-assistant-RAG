import asyncio
from sqlalchemy import select
from app.infrastructure.database.connection import async_session_maker
from app.infrastructure.database.models import Article, ArticleChunk
from app.infrastructure.gemini_adapter import GeminiAdapter
from app.infrastructure.postgres_adapter import PostgresVectorStore
from app.core.config import settings

async def load_data():
    # 1. Подготавливаем данные
    title = "Искандер"
    url = "https://local-lore.internal/wiki/X"
    universe = "Human"
    
    # Текст для загрузки (разбит по абзацам для чанкинга)
    raw_text = """Искандер Маматкулов — весьма противоречивая и эксцентричная личность, известная в своих кругах крайне хаотичным образом жизни. Он имеет репутацию неисправимого бабника, но при этом открыто идентифицирует себя как гея, что делает его любовные похождения запутанными, парадоксальными и предметом постоянных сплетен.
        Его повседневность неразрывно связана с барами и сомнительными заведениями. Искандер страдает от тяжелой алкогольной зависимости; именно алкоголь чаще всего является катализатором его безрассудных поступков и драм.
        Несмотря на деструктивный образ жизни и пагубные привычки, Искандер обладает невероятной природной харизмой. Эта черта позволяет ему притягивать к себе самых разных людей и всегда оставаться в центре внимания.
        Его история — это непрерывная череда случайных связей, шумных вечеринок и абсурдных ситуаций, в которых стираются границы между его зависимостями и межличностными отношениями.

        Место обитания пивнушка, Mangal, стриптиз клуб и бары. Телефон номер и контакты это +998(90)281-96-98. Инстаграм аккаунт @___iskander___m___. 

        Два раза был жена. В разводе.
    """
    
    chunks_text = [chunk.strip() for chunk in raw_text.split('\n') if chunk.strip()]

    # 2. Инициализируем адаптеры
    llm_client = GeminiAdapter(
        api_key=settings.GEMINI_API_KEY,
        chat_model_name=settings.CHAT_MODEL,
        embedding_model_name=settings.EMBEDDING_MODEL,
        embedding_dimension=settings.VECTOR_DIMENSION
    )
    
    async with async_session_maker() as session:
        # 3. Если статья с таким url уже загружена — сносим прошлую версию.
        #    Её чанки удалятся сами: на FK стоит ON DELETE CASCADE.
        #    Без этого повторный запуск падает на уникальном индексе articles.url.
        existing = await session.scalar(select(Article).where(Article.url == url))
        if existing:
            await session.delete(existing)
            await session.flush()
            print(f"♻️  Прошлая версия статьи (id={existing.id}) удалена, перезаливаю.")

        # 4. Создаем статью
        article = Article(title=title, url=url)
        session.add(article)
        await session.flush() # Получаем ID статьи из БД

        vector_store = PostgresVectorStore(session)

        # 5. Превращаем текст в чанки с векторами
        print(f"Генерирую {len(chunks_text)} эмбеддингов одним батчем...")
        embeddings = await llm_client.generate_embeddings_batch(chunks_text)
        print("Готово.")

        chunks_to_save = [
            {
                "article_id": article.id,
                "universe": universe,
                "article_title": title,
                "source_url": url,
                "section_path": None,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {"source": "local", "category": "Люди"}
            }
            for text, embedding in zip(chunks_text, embeddings)
        ]

        # 6. Сохраняем в базу
        await vector_store.save_chunks(chunks_to_save)
        await session.commit()
        print("✅ Лор персонажа Искандер успешно загружен в базу!")

if __name__ == "__main__":
    asyncio.run(load_data())