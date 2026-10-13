"""lore_terms: индекс по основе слова для correct_typos

Revision ID: b7e3c91d4f02
Revises: a04d5680dde2
Create Date: 2026-09-28 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'b7e3c91d4f02'
down_revision: Union[str, Sequence[str], None] = 'a04d5680dde2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Выражение — ровно как в correct_typos (postgres_adapter.py), иначе
    # планировщик индекс не подхватит. IF NOT EXISTS: на свежей БД его уже
    # создал create_all из модели (см. раздел «Запуск» в ARCHITECTURE.md).
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_lore_terms_universe_stem
        ON lore_terms (universe, (COALESCE((ts_lexize('russian_stem', term))[1], term)))
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_lore_terms_universe_stem")
