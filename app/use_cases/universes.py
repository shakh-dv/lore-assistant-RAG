from typing import Any, Dict, List

from app.domain.ports import IVectorStore

# Человекочитаемые названия для кодов вселенных.
# Кода нет в словаре — отдаём сам код, ничего не выдумываем.
UNIVERSE_TITLES: Dict[str, str] = {
    "AC": "Assassin's Creed",
    "Human": "Люди",
}


class ListUniversesUseCase:
    """
    Сценарий: какие вселенные доступны для вопросов.
    Источник правды — само хранилище, а не захардкоженный список.
    """

    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store

    async def execute(self) -> List[Dict[str, Any]]:
        universes = await self.vector_store.list_universes()

        return [
            {
                "id": item["id"],
                "name": UNIVERSE_TITLES.get(item["id"], item["id"]),
                "chunks": item["chunks"],
                "articles": item["articles"],
            }
            for item in universes
        ]
