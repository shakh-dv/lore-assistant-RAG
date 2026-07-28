import asyncio
from sqlalchemy import select
from app.infrastructure.database.connection import async_session_maker
from app.infrastructure.database.models import Article, ArticleChunk
from app.infrastructure.gemini_adapter import GeminiAdapter
from app.infrastructure.postgres_adapter import PostgresVectorStore
from app.core.config import settings

async def load_data():
    # 1. Подготавливаем данные
    title = "Альтаир ибн Ла-Ахад"
    url = "https://assassinscreed.fandom.com/ru/wiki/Альтаир"
    universe = "AC"
    
    # Текст для загрузки (можно разбить по абзацам)
    raw_text = """Альтаир ибн Ла-Ахад — мастер-ассасин, живший в XII веке во время Третьего крестового похода. Он стал легендой Братства, усовершенствовав скрытый клинок и создав Кодекс ассасинов.
Его жизнь изменилась после провала миссии в храме Соломона, когда он нарушил три догмата Кредо. Чтобы искупить вину, Аль-Муалим приказал ему убить девять целей, распространяющих зло в Святой Земле.
В ходе выполнения заданий Альтаир осознал, что цели связаны между собой, а истинным врагом является Робер де Сабле и предатель в собственных рядах.
Альтаир был первым, кто научился использовать Яблоко Эдема, осознав его истинную силу и опасность. Он посвятил свою жизнь защите человечества от влияния артефактов Предтеч."""
    
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
        chunks_to_save = []
        for text in chunks_text:
            print(f"Генерирую вектор для: {text[:30]}...")
            embedding = await llm_client.generate_embedding(text)
            
            chunks_to_save.append({
                "article_id": article.id,
                "universe": universe,
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {"source": "wiki"}
            })
        
        # 6. Сохраняем в базу
        await vector_store.save_chunks(chunks_to_save)
        await session.commit()
        print("✅ Лор успешно загружен в базу!")

if __name__ == "__main__":
    asyncio.run(load_data())