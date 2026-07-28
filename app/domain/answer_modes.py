from enum import Enum
from typing import Any, Dict, List


class AnswerMode(str, Enum):
    """
    Стиль ответа. Выбирается пользователем в интерфейсе и не зависит от вселенной:
    сухую справку может хотеться и по выдуманному лору, и наоборот.
    """
    STORYTELLER = "storyteller"
    ARCHIVIST = "archivist"


DEFAULT_MODE = AnswerMode.STORYTELLER

# Описания для интерфейса. Держим рядом с самим перечислением,
# чтобы добавление режима было правкой в одном месте.
MODE_INFO: Dict[AnswerMode, Dict[str, str]] = {
    AnswerMode.STORYTELLER: {
        "name": "Рассказчик",
        "description": "Живо и литературно. Факты те же, подача — художественная.",
        "icon": "ph-book-open-text",
    },
    AnswerMode.ARCHIVIST: {
        "name": "Архивариус",
        "description": "Сухо и по делу, близко к тексту источника. Без приукрашивания.",
        "icon": "ph-archive",
    },
}


def list_modes() -> List[Dict[str, Any]]:
    """Режимы для выпадающего списка в UI."""
    return [
        {"id": mode.value, "default": mode is DEFAULT_MODE, **MODE_INFO[mode]}
        for mode in AnswerMode
    ]
