"""
Загрузка базы знаний о платформе GTS из отчёта по практике.

Два шага, потому что между ними текст правится руками:

    uv run python load_gts.py extract   # docx -> gts_chunks.json
    uv run python load_gts.py           # gts_chunks.json -> база

extract вытаскивает абзацы и кладёт их в JSON в двух полях: raw — как в
документе, text — то, что реально пойдёт в базу. Дальше text приводится к
стилю справочника (третье лицо, без автора), а raw остаётся для сверки.
Правка идёт по JSON, а не по docx: результат воспроизводим и виден в diff.
"""
import asyncio
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from sqlalchemy import select
from app.infrastructure.database.connection import async_session_maker
from app.infrastructure.database.models import Article
from app.infrastructure.gemini_adapter import GeminiAdapter
from app.infrastructure.postgres_adapter import PostgresVectorStore
from app.core.config import settings

DOCX_PATH = "отчет_практика.docx"
CHUNKS_PATH = Path("gts_chunks.json")
TITLE = "Платформа GTS (Global Travel Space)"
URL = "local://gts/отчет_практика.docx"
UNIVERSE = "GTS"

# Документ — отчёт по практике, но в базу знаний идёт только фактура о платформе.
# Разделы 5-7 и заключение — рефлексия о самой практике («чему я научился»),
# в ответе про архитектуру Sirena они только мешают. Пустой набор = брать всё.
KEEP_SECTIONS = ("1.", "2.", "3.", "4.")

# Абзац короче этого — обычно пункт списка. Клеим к предыдущему:
# «изучение архитектуры платформы GTS» отдельным вектором бесполезен.
MIN_CHUNK_LEN = 120

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def read_docx_paragraphs(path: str):
    """Абзацы документа как (стиль, текст). docx — это zip с xml внутри."""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))

    for para in root.iter(f"{_W}p"):
        style_node = para.find(f"{_W}pPr/{_W}pStyle")
        style = style_node.get(f"{_W}val", "") if style_node is not None else ""
        # Word рвёт текст на runs по любому изменению шрифта — склеиваем обратно.
        text = "".join(node.text or "" for node in para.iter(f"{_W}t"))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            yield style, text


def extract() -> None:
    """docx -> JSON. Перезаписывает файл: ручные правки в text потеряются."""
    chunks = []
    section = subsection = ""

    for style, text in read_docx_paragraphs(DOCX_PATH):
        if style == "Heading1":
            section, subsection = text, ""
            continue
        if style == "Heading2":
            subsection = text
            continue

        if KEEP_SECTIONS and not section.startswith(KEEP_SECTIONS):
            continue

        header = f"{section} / {subsection}" if subsection else section
        prev = chunks[-1] if chunks else None

        # Короткий пункт списка дописываем в предыдущий чанк того же раздела
        if prev and prev["header"] == header and len(prev["raw"]) and len(text) < MIN_CHUNK_LEN:
            prev["raw"] += " " + text
            prev["text"] = prev["raw"]
            continue

        chunks.append({
            "header": header,
            "section": section,
            "subsection": subsection,
            "raw": text,
            "text": text,
        })

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(len(c["raw"]) for c in chunks)
    print(f"📄 {DOCX_PATH} -> {CHUNKS_PATH}: {len(chunks)} чанков, {total} символов")


async def load_data() -> None:
    if not CHUNKS_PATH.exists():
        sys.exit(f"Нет {CHUNKS_PATH}. Сначала: uv run python load_gts.py extract")

    all_chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    # Пустой text — способ снять чанк с загрузки, не удаляя его из файла:
    # так видно, что абзац рассмотрели и признали бесполезным, а не потеряли.
    chunks = [c for c in all_chunks if c["text"].strip()]
    edited = sum(1 for c in chunks if c["text"] != c["raw"])
    skipped = len(all_chunks) - len(chunks)
    print(f"📦 {CHUNKS_PATH}: {len(chunks)} чанков "
          f"(переписано {edited}, снято с загрузки {skipped})")

    llm_client = GeminiAdapter(
        api_key=settings.GEMINI_API_KEY,
        chat_model_name=settings.CHAT_MODEL,
        embedding_model_name=settings.EMBEDDING_MODEL,
        embedding_dimension=settings.VECTOR_DIMENSION
    )

    async with async_session_maker() as session:
        # Перезалив: старая версия сносится, чанки уходят каскадом по FK.
        existing = await session.scalar(select(Article).where(Article.url == URL))
        if existing:
            await session.delete(existing)
            await session.flush()
            print(f"♻️  Прошлая версия (id={existing.id}) удалена, перезаливаю.")

        article = Article(title=TITLE, url=URL, metadata_obj={"source": "internship_report"})
        session.add(article)
        await session.flush()

        vector_store = PostgresVectorStore(session)

        # Заголовок раздела приклеиваем к тексту: без него «Ответ приводится
        # к единому формату» не привязан ни к чему.
        texts = [f"{chunk['header']}\n{chunk['text']}" for chunk in chunks]
        print(f"Генерирую {len(texts)} эмбеддингов одним батчем...")
        embeddings = await llm_client.generate_embeddings_batch(texts)
        print("Готово.")

        chunks_to_save = [
            {
                "article_id": article.id,
                "universe": UNIVERSE,
                "article_title": TITLE,
                "source_url": URL,
                "section_path": chunk["header"],
                "chunk_text": text,
                "embedding": embedding,
                "metadata": {
                    "source": "internship_report",
                    "section": chunk["section"],
                    "subsection": chunk["subsection"],
                    "raw_text": chunk["raw"],
                },
            }
            for chunk, text, embedding in zip(chunks, texts, embeddings)
        ]

        await vector_store.save_chunks(chunks_to_save)
        await session.commit()
        print(f"✅ Загружено {len(chunks_to_save)} чанков во вселенную {UNIVERSE}!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        extract()
    else:
        asyncio.run(load_data())
