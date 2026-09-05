from sqlalchemy.sql import func
from sqlalchemy import Boolean, Integer
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, ForeignKey, DateTime, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Article(Base):
    __tablename__ = "articles"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    url: Mapped[str] = mapped_column(String, nullable=False, unique=True) # Добавил unique=True
    # УБРАЛ embedding из Article, он тут не нужен!
    metadata_obj: Mapped[dict] = mapped_column(JSONB, default=dict) # Переименовал, чтобы не конфликтовало со встроенным metadata в Base
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), server_onupdate=func.now())
    
    chunks: Mapped[list["ArticleChunk"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )

class ArticleChunk(Base):
    __tablename__ = "article_chunks"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    
    # ВЕРНУЛ universe как отдельную колонку с индексом для скорости!
    universe: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Денормализовано из Article: (1) дописывается в текст перед эмбеддингом —
    # коротким секциям без имени персонажа это сильно помогает retrieval;
    # (2) отдаётся при цитировании ответа без JOIN на articles.
    article_title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(String, nullable=False)

    # "Статья > Раздел > Подраздел" — путь до места в структуре вики, откуда
    # взят чанк. NULL допустим для источников без структуры разделов; сейчас
    # его заполняют оба загрузчика — WikiChunker (Fandom) и load_gts.py.
    section_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # text/infobox/list — сериализованные инфобоксы и обычный текст нужно
    # показывать и обрабатывать по-разному.
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=False, server_default="text")

    # Задел под hierarchical (parent-child) retrieval — пока никто не пишет
    # сюда данные, просто чтобы потом не делать миграцию под саму колонку.
    parent_chunk_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("article_chunks.id", ondelete="SET NULL"), nullable=True
    )

    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768)) # Аннотация list[float] для SQLAlchemy
    # JSONB, не JSON: бинарное хранение и поддержка операторов (@>, ?), если
    # понадобится фильтрация по содержимому. Индекс на неё пока НЕ ставим —
    # ни одного запроса с фильтром по metadata_obj в проекте сейчас нет,
    # GIN добавляется одной командой в момент, когда такой запрос появится.
    metadata_obj: Mapped[dict] = mapped_column(JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), server_onupdate=func.now())

    article: Mapped["Article"] = relationship(back_populates="chunks")
    parent: Mapped[Optional["ArticleChunk"]] = relationship(remote_side="ArticleChunk.id")


class LoreTerm(Base):
    """
    Словарь слов, реально встречающихся в лоре. Нужен, чтобы чинить опечатки
    в запросе локально (pg_trgm), не тратя вызов LLM.

    Привязан к статье каскадом: перезалил статью — её слова ушли сами,
    отдельная чистка не нужна.
    """
    __tablename__ = "lore_terms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    universe: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Хранится в нормализованном виде: нижний регистр, ё -> е
    term: Mapped[str] = mapped_column(String(64), nullable=False)

    # Имя собственное или технический термин — только такие слова годятся
    # в замену для опечатки. Без этого флага корректор «чинит» обычную
    # лексику: «происходит» -> «проходит», «женат» -> «жена».
    is_proper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        # Одно и то же слово из одной статьи храним один раз
        UniqueConstraint("article_id", "term", name="uq_lore_terms_article_term"),
        # ВАЖНО: именно GiST. GIN не умеет KNN-сортировку `ORDER BY term <-> :слово`
        # и молча выродится в Seq Scan — проверено на EXPLAIN.
        Index(
            "ix_lore_terms_term_trgm",
            "term",
            postgresql_using="gist",
            postgresql_ops={"term": "gist_trgm_ops"},
        ),
    )