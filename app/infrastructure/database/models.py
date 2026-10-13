from sqlalchemy.sql import func
from sqlalchemy import Boolean, Integer
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, ForeignKey, DateTime, Index, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Article(Base):
    __tablename__ = "articles"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    url: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    # Не «metadata»: это имя занято в DeclarativeBase.
    metadata_obj: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), server_onupdate=func.now())
    
    chunks: Mapped[list["ArticleChunk"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )

class ArticleChunk(Base):
    __tablename__ = "article_chunks"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    
    universe: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Денормализовано из Article: цитирование без JOIN.
    article_title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str] = mapped_column(String, nullable=False)

    # «Статья > Раздел > Подраздел».
    section_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # text / infobox
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=False, server_default="text")

    # Задел под parent-child retrieval, пока не заполняется.
    parent_chunk_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("article_chunks.id", ondelete="SET NULL"), nullable=True
    )

    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768))
    # Без GIN-индекса: запросов с фильтром по metadata_obj пока нет.
    metadata_obj: Mapped[dict] = mapped_column(JSONB, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), server_onupdate=func.now())

    article: Mapped["Article"] = relationship(back_populates="chunks")
    parent: Mapped[Optional["ArticleChunk"]] = relationship(remote_side="ArticleChunk.id")


class LoreTerm(Base):
    """
    Словарь слов лора для исправления опечаток без LLM (pg_trgm). Каскад от
    статьи: перезалил статью — её слова ушли сами.
    """
    __tablename__ = "lore_terms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), nullable=False)
    universe: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Нормализовано: нижний регистр, ё -> е.
    term: Mapped[str] = mapped_column(String(64), nullable=False)

    # Только имена годятся в замену: иначе «происходит» -> «проходит».
    is_proper: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("article_id", "term", name="uq_lore_terms_article_term"),
        # Именно GiST: GIN не умеет KNN `ORDER BY term <-> :слово` (Seq Scan).
        Index(
            "ix_lore_terms_term_trgm",
            "term",
            postgresql_using="gist",
            postgresql_ops={"term": "gist_trgm_ops"},
        ),
        # Для проверки знакомости по основе в correct_typos. Выражение должно
        # совпадать с запросом символ в символ, иначе индекс не подхватится.
        Index(
            "ix_lore_terms_universe_stem",
            "universe",
            text("COALESCE((ts_lexize('russian_stem', term))[1], term)"),
        ),
    )