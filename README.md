# Lore Assistant

RAG-ассистент, отвечающий на вопросы строго по загруженной базе знаний. Вопрос
превращается в вектор, ближайшие куски текста достаются из Postgres и уходят в
Gemini как контекст — модель отвечает только по ним, без домыслов.

Основной корпус — русская вики Assassin's Creed на Fandom, забирается через
MediaWiki API и режется по секциям статей. Архитектура — порты и адаптеры:
бизнес-логика в `app/use_cases` не знает ни про Gemini, ни про Postgres.
Подробный разбор решений, замеры и известные пробелы —
в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Стек

Python 3.12 · FastAPI · SQLAlchemy (async) · Alembic · PostgreSQL 16 + pgvector +
pg_trgm · Gemini (`gemini-embedding-001` для векторов, `gemini-3.1-flash-lite`
для ответов) · httpx + mwparserfromhell для Fandom

## Первый запуск

```bash
docker compose up -d      # Postgres с pgvector, контейнер lore_vector_db, порт 5432
uv sync                   # зависимости
```

В `.env` нужны две переменные:

```
GEMINI_API_KEY=...
DATABASE_URL=postgresql+psycopg://admin:secretpassword@localhost:5432/lore_db
```

Запуск с тунелем в интернет
```bash
cloudflared tunnel --url http://localhost:8000
```

Остальное (`EMBEDDING_MODEL`, `CHAT_MODEL`, `VECTOR_DIMENSION`) имеет значения
по умолчанию в [`app/core/config.py`](app/core/config.py).

### Схема БД

Схема версионируется Alembic'ом, но при старте сервер по-прежнему делает
`create_all` — он создаёт отсутствующие таблицы и расширения, но не добавляет
колонки в уже существующие. Отсюда два пути:

```bash
# База уже есть и создана раньше — накатить миграции:
uv run alembic upgrade head

# База пустая — запустить сервер один раз (create_all создаст актуальную схему),
# остановить его и отметить схему как текущую, чтобы Alembic не пытался
# накатывать поверх неё то, что уже есть:
uv run python main.py           # дождаться «Сервер успешно запущен», Ctrl+C
uv run alembic stamp head
```

Почему так: единственная миграция добавляет колонки в существующую таблицу и на
пустой базе упадёт. Все следующие миграции — обычный `alembic upgrade head`.

### Сервер

```bash
uv run python main.py     # или make run
```

Интерфейс — http://localhost:8000, Swagger — http://localhost:8000/docs.
Список вселенных интерфейс берёт из базы автоматически, а карточки-подсказки
на пустом экране захардкожены во фронте ([`static/index.html`](static/index.html),
объект `SUGGESTIONS`) — для новой вселенной их нужно дописать руками.

## Загрузка данных

База пустая — ассистенту нечего отвечать.

### Fandom — основной источник (вселенная `AC`)

```bash
uv run python index_fandom.py --dry-run --limit 5       # только чанкер, без Gemini и БД → dry_run_output.jsonl
uv run python index_fandom.py --limit 20                # небольшая реальная заливка
uv run python index_fandom.py --title "Эцио Аудиторе"   # одна статья; флаг можно повторять
uv run python index_fandom.py --title "..." --force     # переиндексировать, даже если статья не менялась
uv run python index_fandom.py                           # весь корпус
```

Индексатор инкрементальный: `revid` каждой статьи хранится в базе, при повторном
прогоне неизменившиеся статьи пропускаются без обращения к Gemini. Коммит в БД —
после каждой статьи, поэтому прерывать прогон безопасно: сделанное не откатится.

**Полный корпус — это дни, не минуты.** Бесплатная квота Gemini даёт ~1000
embed-запросов в сутки, это 150–200 статей. Когда квота кончается, индексатор
пишет `⛔ Дневной лимит Gemini API исчерпан` и сам останавливается; на следующий
день его просто запускают снова.

**Запуск в фоне и логи:**

```bash
nohup uv run python -u index_fandom.py > index_fandom.log 2>&1 &
tail -f index_fandom.log          # следить в реальном времени
pgrep -fl index_fandom            # проверить, жив ли процесс
pkill -f index_fandom.py          # остановить (безопасно, см. про коммиты выше)
```

`nohup ... &` — процесс переживёт закрытие терминала. `-u` обязателен: без него
Python буферизует `print` при выводе в файл, и `tail -f` показывает пустоту, пока
буфер не заполнится. Строки лога: `✅ Название: N чанков` — залито,
`⏭ revid не изменился` — пропущено, `⚠️` — статья недоступна или без чанков,
`❌` — ошибка на конкретной статье, прогон продолжается. Файлы `*.log`
в `.gitignore`.

**Cloudflare.** С некоторых IP (у автора — домашний, динамический) `api.php`
отдаёт `403 Forbidden` с JS-challenge, и лог обрывается на первом же запросе к
`action=query&list=allpages`. Это не ошибка кода: ретраи бессильны, cookie из
браузера не переносится. Со статического рабочего IP всё работает. Индексатор
гонять оттуда.

### Вторая вселенная и служебное

| Команда | Что делает |
|---|---|
| `uv run python load_gts.py extract` | `.docx` → `gts_chunks.json` (без обращения к API) |
| `uv run python load_gts.py` | `gts_chunks.json` → база, вселенная `GTS` |
| `uv run python rebuild_lexicon.py` | пересборка словаря опечаток из уже залитых чанков, без трат на Gemini |

`load_gts.py` разбит на два шага намеренно: между ними текст правится руками.
В [`gts_chunks.json`](gts_chunks.json) у каждого чанка два поля — `raw` как в
документе и `text` как уйдёт в базу; пустой `text` снимает чанк с загрузки, не
удаляя его из файла. Исходный `.docx` в репозитории не лежит, поэтому шаг
`extract` не воспроизвести — рабочий путь только второй, из готового JSON.

## Eval — измерить качество, а не почувствовать

Контрольный набор вопросов с известными ответами ([`eval/dataset.yaml`](eval/dataset.yaml))
прогоняется через тот же конвейер, что и чат. Запускать **только как модуль** —
иначе Python не найдёт пакет `app`:

```bash
uv run python -m eval.run_eval --validate                  # схема датасета + все статьи-эталоны есть в БД
uv run python -m eval.run_eval --label my-change           # только retrieval, ~1 мин
uv run python -m eval.run_eval --full --label my-change    # + ответы модели, ~4 мин
uv run python -m eval.run_eval --compare eval/results/A.json eval/results/B.json
uv run python -m eval.run_eval --only-id fact-altair-birthplace   # отладить один вопрос
```

Что считается: `recall@5` — нашёлся ли нужный чанк в пятёрке, `MRR` — насколько
высоко, `facts_score` — есть ли в ответе ожидаемые факты, `refusal_ok` — отказался
ли бот на вопросе не по теме, `unexpected_correction` — не испортил ли корректор
опечаток нормальное слово. Результат каждого прогона — JSON в
[`eval/results/`](eval/results/), они коммитятся: это история качества проекта.
Порядок работы: прогон до правки → правка → прогон после → `--compare`.

## Структура

```
app/
  domain/          порты (протоколы), режимы ответа, чанкер вики — без внешних зависимостей
  use_cases/       ChatUseCase: весь RAG-конвейер
  infrastructure/  адаптеры Gemini и Postgres, модели БД
  api/             роутер FastAPI и схемы
migrations/        Alembic
eval/              датасет, раннер, результаты прогонов
static/index.html  веб-интерфейс, один файл
docs/              техническая опись
index_fandom.py    индексатор Fandom-корпуса
fandom_scraper.py  FandomScraper — MediaWiki API с rate limit и ретраями
load_gts.py        разовый загрузчик вселенной GTS
rebuild_lexicon.py пересборка словаря опечаток
```
