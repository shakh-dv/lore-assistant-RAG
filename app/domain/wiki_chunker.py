"""
Чанкинг wikitext-статей Fandom: секции -> merge коротких -> recursive split
длинных, инфобокс — отдельным чанком. Чистая функция, без БД и сети.

byteoffset секций у этой вики — индекс СИМВОЛОВ, а не байтов (проверено
эмпирически), поэтому wikitext режется посимвольно. Подробно — ARCHITECTURE.md, 5.2.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import mwparserfromhell

# Поднимать при ЛЮБОМ изменении вывода chunk_article: иначе revid-скип
# индексатора оставит в БД чанки старой версии.
CHUNKER_VERSION = 2

# Эвристика вместо токенайзера: count_tokens Gemini — сетевой вызов.
CHARS_PER_TOKEN = 3.0

SPLIT_THRESHOLD_TOKENS = 600
TARGET_MIN_TOKENS, TARGET_MAX_TOKENS = 400, 500
MERGE_THRESHOLD_TOKENS = 100
OVERLAP_RATIO = 0.15

SPLIT_THRESHOLD_CHARS = int(SPLIT_THRESHOLD_TOKENS * CHARS_PER_TOKEN)
TARGET_CHARS = int((TARGET_MIN_TOKENS + TARGET_MAX_TOKENS) / 2 * CHARS_PER_TOKEN)
MERGE_THRESHOLD_CHARS = int(MERGE_THRESHOLD_TOKENS * CHARS_PER_TOKEN)
OVERLAP_CHARS = int(TARGET_CHARS * OVERLAP_RATIO)

_SPLIT_SEPARATORS = ["\n\n", "\n", ". ", " "]
_OVERLAP_SEARCH_WINDOW = 80  # запас символов, где ищем чистую границу для overlap

MIN_INFOBOX_PARAMS = 3
_INFOBOX_NAME_KEYWORDS = ("инфобокс", "infobox", "персонаж", "предмет")

# Без фильтра strip_code() отдаёт из [[Файл:X.png|thumb|250px]] текст «thumb|250px».
_FILE_LINK_PREFIXES = ("файл:", "file:", "изображение:", "image:")

# Категории и межъязыковые ссылки ([[en:Article]]) — служебные, не проза.
_NON_CONTENT_LINK_PREFIXES = _FILE_LINK_PREFIXES + ("категория:", "category:")
_INTERLANG_LINK_RE = re.compile(r"^[a-z]{2,3}:")

# strip_code() удаляет <br> без разделителя («БагдадСамарра»). Регуляркой до
# парсинга — кривые «<br/ >», «<br/ />» парсер тегом не считает.
_BR_RE = re.compile(r"<br[\s/]*>", re.IGNORECASE)

# Мягкий перенос внутри слов, zero-width, BOM: глазами не видно, но для
# эмбеддинга и словаря опечаток это другое слово.
_INVISIBLE_RE = re.compile("[\u00ad\u200b-\u200f\ufeff]")

# __NOTOC__ и прочие magic words strip_code() оставляет как текст.
_MAGIC_WORD_RE = re.compile(r"__[A-Z]+__")

_TABLE_ROW_TAIL_RE = re.compile(r"[ \t]*\|[ \t]*$", re.MULTILINE)

# Подпись вкладки <tabber> («Видео:»). Секция из одних подписей — пустая галерея.
_LABEL_LINE_RE = re.compile(r"^.{1,80}:$")

# Значение — имя файла, а не факт. «Обложка» сюда не входит: там имя художника.
_INFOBOX_IMAGE_KEY_RE = re.compile(r"^(изображение|image|подпись|caption)\d*$", re.IGNORECASE)

_INFOBOX_EMPTY_VALUE_RE = re.compile(r"^[-—–?\s]*$")

# Ключ шаблона («ДР») -> подпись, которую видит читатель вики. Словарь по
# шаблону, а не общий: «ДР» у Персонажа — дата рождения, у Фракции —
# реорганизации. Сверено с исходниками шаблонов, см. ARCHITECTURE.md, 5.2.
_INFOBOX_KEY_LABELS: Dict[str, Dict[str, str]] = {
    "персонаж": {
        "ДР": "Дата рождения", "ДС": "Дата смерти", "МС": "Место смерти",
        "Убит": "Убийца", "База данных": "Записи в базе данных",
    },
    "воспоминание": {
        "Пред": "Предыдущее воспоминание", "След": "Следующее воспоминание",
        "Парал": "Параллельное воспоминание",
        "100%Синх": "Условие полной синхронизации",
    },
    "фракция": {
        "ДФ": "Дата основания", "ДР": "Дата реорганизации", "ДС": "Дата распада",
        "База данных": "Записи в базе данных",
    },
    # Кроме Пред/След/Парал — старые параметры битв, подписи по смыслу значений.
    "событие": {
        "Пред": "Предыдущее событие", "След": "Следующее событие",
        "Парал": "Параллельное событие",
        "prev": "Предыдущее событие", "next": "Следующее событие",
        "сторона1": "Сторона 1", "сторона2": "Сторона 2",
        "у1": "Силы стороны 1", "у2": "Силы стороны 2",
        "командующий1": "Командующий стороны 1", "командующий2": "Командующий стороны 2",
        "жертва1": "Потери стороны 1", "жертва2": "Потери стороны 2",
    },
    "локация": {"Основал": "Основатель", "Дата": "Дата основания"},
    "предметы": {"Создано": "Создатель"},
    "игра": {"Релиз": "Дата релиза"},
    "книга": {
        "Обложка": "Автор обложки", "Языки": "Язык оригинала",
        "Перевод": "Переводчик", "Страниц": "Количество страниц",
    },
    "комикс": {
        "Обложка": "Автор обложки", "Языки": "Язык оригинала",
        "Перевод": "Переводчик", "Страниц": "Количество страниц",
    },
}


def _clean_wikitext(raw: str) -> str:
    """wikitext -> читаемый текст: сноски, служебные ссылки и шаблоны убраны, границы списков и таблиц сохранены."""
    code = mwparserfromhell.parse(_BR_RE.sub("\n", raw))
    # Вложенный <ref> уже удалён вместе со своим <references> — ValueError не ошибка.
    for tag in code.filter_tags(matches=lambda t: str(t.tag) in ("ref", "references")):
        try:
            code.remove(tag)
        except ValueError:
            pass
    for comment in code.filter_comments():
        code.remove(comment)
    for link in code.filter_wikilinks(
        matches=lambda l: (
            str(l.title).strip().lower().startswith(_NON_CONTENT_LINK_PREFIXES)
            or _INTERLANG_LINK_RE.match(str(l.title).strip())
        )
    ):
        code.remove(link)

    # strip_code() склеивает ячейки таблиц. Перенос ПЕРЕД tr, а не после:
    # у первой строки таблицы тега tr («|-») нет.
    for cell in code.filter_tags(matches=lambda t: str(t.tag) in ("td", "th")):
        code.insert_after(cell, " | ")
    for row in code.filter_tags(matches=lambda t: str(t.tag) == "tr"):
        code.insert_before(row, "\n")

    text = code.strip_code(normalize=True, collapse=True)
    text = _INVISIBLE_RE.sub("", text).replace("\xa0", " ")
    text = _MAGIC_WORD_RE.sub("", text)

    # Мини-синтаксис <tabber>: «|-|» — разделитель вкладок, «Label=» — подпись.
    text = text.replace("|-|", "")
    text = _TABLE_ROW_TAIL_RE.sub("", text)
    text = re.sub(r"^(.{1,80})=$", r"\1:", text, flags=re.MULTILINE)

    # Пробелы у переносов не трогаем: правка изменила бы текст ~60% чанков
    # и стоила бы их переэмбеддинга ради косметики.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _classify_template(template) -> Tuple[bool, str]:
    """Это шаблон-инфобокс? Сначала по имени, иначе — по доле именованных параметров."""
    name = str(template.name).strip().lower()
    if any(kw in name for kw in _INFOBOX_NAME_KEYWORDS):
        return True, f"по имени шаблона ({name!r})"

    params = template.params
    if len(params) >= MIN_INFOBOX_PARAMS:
        named = sum(1 for p in params if p.showkey)
        # Отсекает навигационные шаблоны {{Серия|AC1|AC2|AC3}} — у них параметры позиционные.
        if named > len(params) / 2:
            return True, f"по именованным параметрам ({named}/{len(params)})"

    return False, ""


def _clean_infobox_value(raw: str) -> str:
    """Список -> «А, Б, В». Длинный текст (письмо в «Электронной почте») сохраняет переносы."""
    value = _clean_wikitext(raw)
    lines = [line.strip().lstrip("*#:;").strip() for line in value.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) > 1 and all(len(line) <= 80 for line in lines):
        return ", ".join(line.rstrip(",;") for line in lines)
    return "\n".join(lines)


def _extract_infobox(
    wikicode,
) -> Tuple[Optional[Dict[str, str]], Optional[str], Optional[str], Optional[str]]:
    """Первый top-level инфобокс -> (поля, сырой текст, причина классификации, имя шаблона)."""
    for template in wikicode.filter_templates(recursive=False):
        is_infobox, reason = _classify_template(template)
        if not is_infobox:
            continue

        params: Dict[str, str] = {}
        for p in template.params:
            key = str(p.name).strip()
            if _INFOBOX_IMAGE_KEY_RE.match(key):
                continue
            value = _clean_infobox_value(str(p.value))
            if not _INFOBOX_EMPTY_VALUE_RE.match(value):
                params[key] = value

        if params:
            return params, str(template), reason, str(template.name).strip()

    return None, None, None, None


def _infobox_label(template_name: str, key: str, params: Dict[str, str]) -> str:
    # В MediaWiki «Событие» и «событие», «_» и пробел — один шаблон.
    template = template_name.replace("_", " ").strip().lower()
    # У Фракции подпись зависит от типа — так же, как в самом шаблоне.
    if template == "фракция" and key in ("Штаб", "Локации"):
        is_state = params.get("Тип") == "Государство"
        if key == "Штаб":
            return "Столица" if is_state else "Штаб-квартира"
        return "Крупные города" if is_state else "Локации"
    label = _INFOBOX_KEY_LABELS.get(template, {}).get(key, key)
    return label[:1].upper() + label[1:]


def _serialize_infobox(params: Dict[str, str], template_name: str) -> str:
    # rstrip: иначе «1459 г.» + разделитель = «1459 г..».
    return ". ".join(
        f"{_infobox_label(template_name, key, params)}: {value.rstrip('.')}"
        for key, value in params.items()
    ) + "."


def _build_records(article: dict) -> List[Dict[str, Any]]:
    """
    Плоский список секций (lead + H2/H3...) с section_path и сырым текстом.
    Родитель содержит только свой текст до первого подраздела, не текст детей.
    """
    wikitext = article["wikitext"]
    title = article["title"]

    # У секций из шаблонов (порталы) byteoffset=null — резать по ним нечем.
    sections = [s for s in article["sections"] if s.get("byteoffset") is not None]

    lead_end = int(sections[0]["byteoffset"]) if sections else len(wikitext)
    records = [{
        "title": title,
        "toclevel": 0,
        "section_path": title,
        "raw_text": wikitext[:lead_end],
    }]

    starts = [int(s["byteoffset"]) for s in sections]
    ends = starts[1:] + [len(wikitext)]

    stack: List[Tuple[int, str]] = []
    for sec, start, end in zip(sections, starts, ends):
        toclevel = int(sec["toclevel"])
        while stack and stack[-1][0] >= toclevel:
            stack.pop()
        # В заголовке бывает разметка («===<b>Текст</b>===»).
        clean_line = _clean_wikitext(sec["line"]) or sec["line"]
        stack.append((toclevel, clean_line))

        # Строку «==Заголовок==» пропускаем: заголовок уже есть в section_path.
        section_text = wikitext[start:end]
        first_newline = section_text.find("\n")
        section_body = section_text[first_newline + 1:] if first_newline != -1 else ""

        records.append({
            "title": clean_line,
            "toclevel": toclevel,
            "section_path": title + " > " + " > ".join(t for _, t in stack),
            "raw_text": section_body,
        })

    return records


def _merge_short(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Короткие секции сливаются НАЗАД, в предыдущий чанк: подраздел обычно
    продолжает тему предыдущего. Пустые записи и записи из одних подписей
    вкладок пропускаются.
    """
    result: List[Dict[str, Any]] = []
    for rec in records:
        text = rec["clean_text"].strip()
        if not text or all(_LABEL_LINE_RE.match(line) for line in text.splitlines() if line.strip()):
            continue
        rec["clean_text"] = text

        if result and len(text) < MERGE_THRESHOLD_CHARS:
            # section_path у слитого чанка — от предыдущей секции, поэтому свой
            # заголовок пишем в текст: иначе «Интересные факты» читаются как «Смерть».
            heading = rec["title"].rstrip(":")
            result[-1]["clean_text"] += f"\n\n{heading}:\n{text}"
            continue

        result.append(rec)

    return result


def _find_overlap_start(text: str) -> str:
    """Последние OVERLAP_CHARS символов, по возможности с границы строки или предложения."""
    if len(text) <= OVERLAP_CHARS:
        return text

    raw_start = len(text) - OVERLAP_CHARS
    window = text[raw_start:raw_start + _OVERLAP_SEARCH_WINDOW]
    for boundary in ("\n", ". "):
        idx = window.find(boundary)
        if idx != -1:
            return text[raw_start + idx + len(boundary):]

    return text[raw_start:]


def _split_finer(text: str, seps: List[str]) -> List[str]:
    """Режет более мелким разделителем, на последнем — жёстко по символам."""
    if len(seps) > 1:
        return _recursive_split(text, seps[1:])
    step = max(TARGET_CHARS - OVERLAP_CHARS, 1)
    return [text[i:i + TARGET_CHARS] for i in range(0, len(text), step)]


def _recursive_split(text: str, seps: List[str] = _SPLIT_SEPARATORS) -> List[str]:
    """
    Инвариант: ни один кусок на выходе не длиннее SPLIT_THRESHOLD_CHARS.
    """
    if len(text) <= SPLIT_THRESHOLD_CHARS:
        return [text]

    sep = seps[0]
    chunks: List[str] = []
    current = ""

    for part in text.split(sep):
        candidate = current + (sep if current else "") + part
        if len(candidate) <= TARGET_CHARS:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = _find_overlap_start(current) + sep + part
            # part бывает огромным сам по себе — тогда режем мельче вместе с overlap.
            if len(current) > SPLIT_THRESHOLD_CHARS:
                chunks.extend(_split_finer(current, seps))
                current = ""
        else:
            chunks.extend(_split_finer(part, seps))
            current = ""

    if current:
        chunks.append(current)

    return chunks


def chunk_article(article: dict, source_url: str) -> List[Dict[str, Any]]:
    """
    Статья из FandomScraper.fetch_article -> чанки (article_title, section_path,
    chunk_type, chunk_text с префиксом section_path, metadata). article_id и
    embedding добавляет индексатор.
    """
    article_title = article["title"]
    base_metadata = {"categories": article.get("categories", []), "revid": article.get("revid")}
    chunks: List[Dict[str, Any]] = []

    # Шаг 1 — инфобокс, на полном wikitext, до нарезки по секциям.
    wikicode = mwparserfromhell.parse(article["wikitext"])
    infobox_params, infobox_raw, infobox_reason, infobox_template = _extract_infobox(wikicode)
    if infobox_params:
        infobox_text = _serialize_infobox(infobox_params, infobox_template)
        print(f"  📦 {article_title}: инфобокс ({infobox_reason}), {len(infobox_params)} полей")
        chunks.append({
            "article_title": article_title,
            "source_url": source_url,
            "section_path": article_title,
            "chunk_type": "infobox",
            "chunk_text": f"{article_title} (инфобокс)\n{infobox_text}",
            # Сырые ключи шаблона — для отладки; подписи только в chunk_text.
            "metadata": {
                **base_metadata,
                "infobox_params": infobox_params,
                "infobox_template": infobox_template,
            },
        })

    # Шаг 2-3 — секции + очистка; инфобокс вырезается из текста своей секции.
    records = _build_records(article)
    for rec in records:
        raw = rec["raw_text"]
        if infobox_raw and infobox_raw in raw:
            raw = raw.replace(infobox_raw, "", 1)
        rec["clean_text"] = _clean_wikitext(raw)

    # Шаг 4 — merge коротких секций.
    merged = _merge_short(records)

    # Шаг 5-6 — сплиттер + префикс section_path в chunk_text: LLM получает
    # чанки как есть и иначе не узнает, из какого раздела кусок.
    for rec in merged:
        text = rec["clean_text"]
        pieces = _recursive_split(text) if len(text) > SPLIT_THRESHOLD_CHARS else [text]

        for i, piece in enumerate(pieces):
            metadata = dict(base_metadata)
            if len(pieces) > 1:
                metadata["split_index"] = i
                metadata["split_total"] = len(pieces)

            chunks.append({
                "article_title": article_title,
                "source_url": source_url,
                "section_path": rec["section_path"],
                "chunk_type": "text",
                "chunk_text": f"{rec['section_path']}\n{piece}",
                "metadata": metadata,
            })

    return chunks
