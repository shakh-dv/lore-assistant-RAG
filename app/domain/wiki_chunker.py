"""
Structure-aware чанкинг wikitext-статей Fandom по пайплайну из CLAUDE.md:
split по секциям -> merge коротких -> recursive split длинных -> инфобокс отдельно.

Чистая трансформация dict -> list[dict], без обращений к БД/сети/диску —
поэтому domain-слой, не infrastructure.

Вход — dict от FandomScraper.fetch_article (fandom_scraper.py): title, revid,
wikitext, sections, categories. sections — от MediaWiki prop=sections, поле
byteoffset у ЭТОЙ инсталляции Fandom эмпирически ведёт себя как индекс символов
python-строки, не байтов (проверено на живой статье "Альтаир ибн Ла-Ахад" —
посимвольный срез чисто совпадает с началом заголовка секции на 4/4 секциях,
байтовый даёт битые символы). Резать wikitext ПОСИМВОЛЬНО, без .encode()/.decode().
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import mwparserfromhell

# Поднимать при ЛЮБОМ изменении того, что выдаёт chunk_article (очистка,
# нарезка, формат инфобокса). Индексатор хранит версию в articles.metadata_obj
# и перечанковывает статью при расхождении, даже если revid на вики тот же.
# Без этого статьи, залитые старой версией, навсегда остаются с её ошибками:
# так 200 статей, залитых до фильтра категорий/interwiki, жили с «Категория:…»
# в тексте, потому что revid-скип их больше не трогал.
CHUNKER_VERSION = 2

# Приближение для смешанного ru/en корпуса, не точный подсчёт токенов —
# tiktoken/langchain/sentencepiece в проекте нет и не добавляются
# (count_tokens Gemini бьёт по сети, неприемлемо на тысячах статей).
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

# [[Файл:X.png|thumb|250px|left|Подпись]] — mwparserfromhell трактует всё после
# первого "|" как единый "текст" вики-ссылки (не знает семантику File-синтаксиса),
# strip_code() без этого фильтра отдаёт буквально "thumb|250px|left|Подпись".
_FILE_LINK_PREFIXES = ("файл:", "file:", "изображение:", "image:")

# [[Категория:X]] / [[en:Article]] — служебные ссылки в конце КАЖДОЙ статьи
# (категории + межъязыковые версии), не часть прозы. strip_code() не знает их
# семантику и разворачивает как обычную ссылку в текст: "Категория:Частицы
# Эдема", "en:Apples of Eden" — найдено живьём в ~13% чанков корпуса.
_NON_CONTENT_LINK_PREFIXES = _FILE_LINK_PREFIXES + ("категория:", "category:")
# Межъязыковой код — 2-3 строчные латинские буквы перед ":" (en, uk, de...).
# Реальные заголовки статей в таком виде на практике не встречаются.
_INTERLANG_LINK_RE = re.compile(r"^[a-z]{2,3}:")

# strip_code() выбрасывает <br> без разделителя: «[[Багдад]]<br>[[Самарра]]»
# превращалось в «БагдадСамарра» (так склеились списки в 26% инфобоксов).
# Меняем на перенос ДО парсинга — регуляркой, потому что в исходнике
# встречается и кривой «<br/ >» и «<br/ />», которые парсер тегом не признаёт.
# Правка под «<br/ />» (2026-10-13) меняет вывод только у трёх статей, где он
# есть, — их перезалили через --force --title, CHUNKER_VERSION не поднимали:
# иначе ради трёх статей пришлось бы снова проходить весь корпус.
_BR_RE = re.compile(r"<br[\s/]*>", re.IGNORECASE)

# Невидимые символы из исходника: мягкий перенос внутри слов («Во\xadсточ\xadная»),
# zero-width/bidi-метки, BOM. Глазами их не видно, но для эмбеддинга и словаря
# опечаток «Во\xadсточная» и «Восточная» — разные слова.
_INVISIBLE_RE = re.compile("[­​-‏﻿]")

# __NOTOC__, __TOC__ и прочие magic words MediaWiki — strip_code() оставляет
# их как текст; у страниц-галерей это было ЕДИНСТВЕННОЕ содержимое чанка.
_MAGIC_WORD_RE = re.compile(r"__[A-Z]+__")

# Хвостовой разделитель ячейки в конце строки таблицы (см. _clean_wikitext).
_TABLE_ROW_TAIL_RE = re.compile(r"[ \t]*\|[ \t]*$", re.MULTILINE)

# Строка-подпись: «Видео:», «Промо-медиа:» — подписи вкладок <tabber> (см.
# замену «Label=» -> «Label:» в _clean_wikitext). Если в секции остались только
# они (галерея, чьё содержимое — <gallery>, уже вырезано), текста там нет.
_LABEL_LINE_RE = re.compile(r"^.{1,80}:$")

# Поля-картинки инфобокса: значение — имя файла («ACI-Ezio.png», «250px»),
# а не факт. Обложка книги сюда не относится — там имя художника.
_INFOBOX_IMAGE_KEY_RE = re.compile(r"^(изображение|image|подпись|caption)\d*$", re.IGNORECASE)

# Значение-заглушка: в шаблонах событий незаполненные поля стоят как «-».
_INFOBOX_EMPTY_VALUE_RE = re.compile(r"^[-—–?\s]*$")

# Ключи инфобокса — внутренние имена параметров шаблона, а не подписи, которые
# видит читатель вики: «ДР», «ДС», «Пред». Подписи сверены с исходниками
# шаблонов (action=parse&page=Шаблон:…, разметка <data source=…><label>…).
#
# Словарь ПО ШАБЛОНУ, а не общий: «ДР» у Персонажа — «Дата рождения», а у
# Фракции — «Дата реорганизации». Из исходников автоматически не берём: там
# встречаются ошибки самой вики (у Комикса ключ UPC подписан «Следующий») и
# подписи-условия ({{#if:…}}). Ключи, которых здесь нет, остаются как есть.
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
    # Пред/След/Парал — из шаблона. Остальное — старые параметры битв: текущий
    # шаблон их уже не выводит, но в статьях они есть. Подписи — по смыслу
    # значений (у1: «Французская армия (30.000 солдат)»).
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
    """
    wikitext -> читаемый текст. <ref>/комментарии убираются ЦЕЛИКОМ (с
    содержимым, не только тегами) — это сноски/служебное, не часть текста
    статьи. Файловые/картиночные вики-ссылки убираются целиком — их "текст"
    после strip_code это layout-мусор (thumb/250px/left), не проза. strip_code
    разворачивает [[Цель|Текст]]/markup, шаблоны без явной обработки исчезают
    (их и не должно тут остаться — инфобокс вырезается отдельно до этого вызова).
    """
    code = mwparserfromhell.parse(_BR_RE.sub("\n", raw))
    # filter_tags рекурсивный: <references group="DLC"><ref>...</ref><ref>...</ref></references>
    # (групповые именованные сноски) даёт в списке и внешний <references>, и вложенные
    # <ref>. Первое же remove(references) уносит вложенные ref вместе с собой из дерева —
    # remove() на них следом не находит узел и кидает ValueError. Раз узла уже нет,
    # цель (убрать его) и так достигнута — просто пропускаем.
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

    # Таблицы: strip_code() склеивает ячейки без разделителя — таблица
    # характеристик превращалась в «СкоростьМощностьЗащитаСтоимость245150000».
    # " | " после каждой ячейки и перенос ПЕРЕД каждой строкой: перед, а не
    # после — у первой строки таблицы явного «|-» (тега tr) в wikitext нет.
    # Связь «колонка — значение» остаётся по порядку ячеек.
    for cell in code.filter_tags(matches=lambda t: str(t.tag) in ("td", "th")):
        code.insert_after(cell, " | ")
    for row in code.filter_tags(matches=lambda t: str(t.tag) == "tr"):
        code.insert_before(row, "\n")

    text = code.strip_code(normalize=True, collapse=True)
    text = _INVISIBLE_RE.sub("", text).replace("\xa0", " ")
    text = _MAGIC_WORD_RE.sub("", text)

    # <tabber> (вкладки с диалоговыми ветками) — strip_code снимает сами теги,
    # но не знает внутренний mini-синтаксис: "|-|" разделитель веток и "Label="
    # перед репликой остаются обычным текстом. "Label=" -> "Label:" — читаемо
    # и не теряет информацию (это подпись ветки диалога), "|-|" просто мусор.
    text = text.replace("|-|", "")
    text = _TABLE_ROW_TAIL_RE.sub("", text)
    text = re.sub(r"^(.{1,80})=$", r"\1:", text, flags=re.MULTILINE)

    # Пробелы у переносов строк (« текст» у вложенных пунктов списка) сознательно
    # не трогаем: смысла для поиска в них нет, а правка сдвинула бы текст ~60%
    # чанков корпуса и стоила бы их полного переэмбеддинга (проверено dry-run).
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
        # Навигационные шаблоны вида {{Игровая_серия|AC1|AC2|AC3}} тоже проходят
        # порог по количеству, но все параметры позиционные (showkey=False) —
        # без этой проверки они бы ложно считались инфобоксом.
        if named > len(params) / 2:
            return True, f"по именованным параметрам ({named}/{len(params)})"

    return False, ""


def _clean_infobox_value(raw: str) -> str:
    """
    Значение поля инфобокса -> одна строка. Список (пункты по строкам, в
    исходнике через <br> или «*») -> перечисление через запятую: «Семья
    Аудиторе, Ассасины, Итальянское Братство». Длинный текст (письмо в
    инфобоксе «Электронной почты») сохраняет переносы — запятые его бы испортили.
    """
    value = _clean_wikitext(raw)
    lines = [line.strip().lstrip("*#:;").strip() for line in value.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) > 1 and all(len(line) <= 80 for line in lines):
        return ", ".join(line.rstrip(",;") for line in lines)
    return "\n".join(lines)


def _extract_infobox(
    wikicode,
) -> Tuple[Optional[Dict[str, str]], Optional[str], Optional[str], Optional[str]]:
    """
    Первый top-level шаблон, похожий на инфобокс -> (поля, сырой текст шаблона,
    причина классификации, имя шаблона). Поля-картинки и заглушки «-» отброшены.
    """
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
    # Имя шаблона в MediaWiki нечувствительно к регистру первой буквы и к «_»
    # вместо пробела: «Событие» и «событие» — один шаблон.
    template = template_name.replace("_", " ").strip().lower()
    # У Фракции подпись зависит от типа — так же, как в самом шаблоне
    # ({{#ifeq:{{{Тип|}}}|Государство|Столица|Штаб-квартира}}).
    if template == "фракция" and key in ("Штаб", "Локации"):
        is_state = params.get("Тип") == "Государство"
        if key == "Штаб":
            return "Столица" if is_state else "Штаб-квартира"
        return "Крупные города" if is_state else "Локации"
    label = _INFOBOX_KEY_LABELS.get(template, {}).get(key, key)
    # Старые параметры шаблонов бывают со строчной: «дата», «место».
    return label[:1].upper() + label[1:]


def _serialize_infobox(params: Dict[str, str], template_name: str) -> str:
    # rstrip(".") — чтобы «1459 г.» + разделитель не давали «1459 г..».
    return ". ".join(
        f"{_infobox_label(template_name, key, params)}: {value.rstrip('.')}"
        for key, value in params.items()
    ) + "."


def _build_records(article: dict) -> List[Dict[str, Any]]:
    """
    Плоский список секций (lead + H2/H3...) с их section_path и сырым текстом.
    Срез посимвольный (см. докстринг модуля) — byteoffset у сортированных
    sections монотонно растёт, конец записи = начало следующей в исходном
    порядке (не следующей на том же уровне — родитель содержит только свой
    вступительный текст до первого подраздела, без текста детей).
    """
    wikitext = article["wikitext"]
    title = article["title"]

    # Порталы/главная страница вики собираются из вложенных шаблонов —
    # MediaWiki не может указать позицию таких секций в исходном wikitext
    # самой страницы и отдаёт byteoffset=null. Резать по ним нечем, пропускаем;
    # если ВСЕ секции такие — вся статья уйдёт одним lead-чанком, не упадёт.
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
        # sec["line"] отдаётся MediaWiki как есть, включая случайную inline
        # HTML-разметку в самом заголовке (например "===<b>Текст</b>==="
        # в исходнике) — без очистки она утекает прямо в section_path.
        clean_line = _clean_wikitext(sec["line"]) or sec["line"]
        stack.append((toclevel, clean_line))

        # Срез с byteoffset начинается прямо с "==Заголовок==\n" самой секции —
        # без пропуска этой строки strip_code() превратит разметку в обычный
        # текст "Заголовок", который задублируется с section_path-префиксом.
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
    Короткие секции сливаются НАЗАД, в уже принятый предыдущий чанк (не вперёд):
    подраздел обычно продолжает тему предыдущего, а не следующего. section_path
    смерженного чанка остаётся у ранее принятого — он конкретнее. Пустые записи
    (заголовок-контейнер без своего текста, только подразделы) пропускаются —
    их подразделы уже получили свой section_path независимо через stack выше.
    Так же пропускаются записи из одних подписей вкладок («Видео:», «Промо-медиа:»).
    """
    result: List[Dict[str, Any]] = []
    for rec in records:
        text = rec["clean_text"].strip()
        if not text or all(_LABEL_LINE_RE.match(line) for line in text.splitlines() if line.strip()):
            continue
        rec["clean_text"] = text

        if result and len(text) < MERGE_THRESHOLD_CHARS:
            # Заголовок секции срезан в _build_records (он живёт в section_path),
            # а section_path слитого чанка — от предыдущей секции. Без заголовка
            # в тексте «Интересные факты» читались бы как продолжение
            # «Биография > Смерть», в том числе и моделью при ответе.
            heading = rec["title"].rstrip(":")
            result[-1]["clean_text"] += f"\n\n{heading}:\n{text}"
            continue

        result.append(rec)

    return result


def _find_overlap_start(text: str) -> str:
    """
    Последние OVERLAP_CHARS символов text — по возможности начиная с чистой
    границы (перенос строки или конец предложения), а не с середины слова.
    """
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
    """
    text не уменьшить разделителем seps[0] — пробуем более мелкий, а на
    последнем — жёсткая нарезка по символам.
    """
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
            # part может быть сам по себе огромным (абзац диалога на 90 реплик
            # после короткого вступления) — тогда и current с ним огромный.
            # Раньше он уходил в чанки как есть: мельче резали только part,
            # перед которым буфер был пуст (ветка ниже). Режем мельче вместе
            # с overlap, чтобы связка с предыдущим чанком не терялась.
            if len(current) > SPLIT_THRESHOLD_CHARS:
                chunks.extend(_split_finer(current, seps))
                current = ""
        else:
            # part сам по себе уже длиннее TARGET_CHARS — этим разделителем
            # его не уменьшить.
            chunks.extend(_split_finer(part, seps))
            current = ""

    if current:
        chunks.append(current)

    return chunks


def chunk_article(article: dict, source_url: str) -> List[Dict[str, Any]]:
    """
    article — из FandomScraper.fetch_article (title, revid, wikitext, sections,
    categories). Возвращает список dict без article_id/embedding (их добавляет
    индексатор): article_title, source_url, section_path, chunk_type
    ('text'|'infobox'), chunk_text (уже с префиксом section_path — см. ниже),
    metadata (categories, revid, ...).
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
            # infobox_params — с сырыми ключами шаблона (для отладки), подписи
            # только в chunk_text.
            "metadata": {
                **base_metadata,
                "infobox_params": infobox_params,
                "infobox_template": infobox_template,
            },
        })

    # Шаг 2-3 — плоский список секций + очистка (с вырезанием инфобокса из
    # текста секции, где он физически находится).
    records = _build_records(article)
    for rec in records:
        raw = rec["raw_text"]
        if infobox_raw and infobox_raw in raw:
            raw = raw.replace(infobox_raw, "", 1)
        rec["clean_text"] = _clean_wikitext(raw)

    # Шаг 4 — merge коротких секций.
    merged = _merge_short(records)

    # Шаг 5-6 — рекурсивный сплиттер + префикс section_path прямо в chunk_text
    # (ChatUseCase склеивает контекст для LLM буквально из chunk_text без
    # обогащения на лету — без префикса модель не увидит, из какого раздела
    # статьи кусок; load_gts.py уже делает так же).
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
