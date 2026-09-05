"""
Eval-набор: измеряем качество retrieval и ответов на контрольных вопросах
с известными эталонами (eval/dataset.yaml).

Запуск — как модуль, из корня проекта (иначе пакет `app` не импортируется:
`python eval/run_eval.py` положил бы в sys.path папку eval/, а не корень):

    uv run python -m eval.run_eval --validate
    uv run python -m eval.run_eval --label baseline          # retrieval-only (по умолчанию)
    uv run python -m eval.run_eval --full --label baseline   # + генерация ответов
    uv run python -m eval.run_eval --compare eval/results/A.json eval/results/B.json

Главная метрика — recall@k на retrieval: детерминирована (эмбеддинг одного
текста всегда одинаков) и стоит 1 embed-вызов на вопрос. --full добавляет
ответ через продакшен-путь (ChatUseCase.execute, режим ARCHIVIST) и проверяет
факты/отказы — дороже (чат-модель: 15 запросов/мин на free tier) и чуть шумнее
(temperature 0.1), поэтому не по умолчанию.

Retrieval-часть повторяет шаги ChatUseCase.execute явно (опечатки -> условный
rewrite -> эмбеддинг -> поиск), потому что execute отдаёт наружу только текст
ответа, а не найденные чанки. _needs_llm_rewrite импортируется оттуда же —
чтобы условие переписывания было ровно продакшен-ным.
"""
import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from sqlalchemy import distinct, select

from app.core.config import settings
from app.domain.answer_modes import AnswerMode
from app.infrastructure.database.connection import async_session_maker
from app.infrastructure.database.models import ArticleChunk
from app.infrastructure.gemini_adapter import GeminiAdapter
from app.infrastructure.postgres_adapter import PostgresVectorStore, normalize_word
from app.use_cases.chat import ChatUseCase, _needs_llm_rewrite

EVAL_DIR = Path(__file__).parent
DEFAULT_DATASET = EVAL_DIR / "dataset.yaml"
RESULTS_DIR = EVAL_DIR / "results"

TYPES = {"fact", "narrative", "relation", "typo", "followup", "out_of_scope"}
# У этих типов есть эталонный источник и факты; у out_of_scope — нет, там
# проверяется отказ.
TYPES_WITH_SOURCE = TYPES - {"out_of_scope"}

# ARCHIVIST при отсутствии данных отвечает «В моих архивах нет информации об
# этом» (см. _ARCHIVIST в gemini_adapter.py). Ловим по устойчивой части фразы.
_REFUSAL_MARKERS = ("нет информации",)

# ChatUseCase при сбое эмбеддинга не падает, а стримит это сообщение — для
# eval это ошибка прогона, не ответ, иначе исказит facts/refusal.
_CHAT_ERROR_PREFIX = "Не получилось обратиться к архивам"

# gemini-3.1-flash-lite free tier — 15 запросов/мин, у generate_answer_stream
# ретрая нет. 4.5с между вопросами держит ~13/мин с запасом.
CHAT_PAUSE_SEC = 4.5


def _norm(text: Optional[str]) -> str:
    return normalize_word(text or "").strip()


def _fact_present(fact: Any, norm_answer: str) -> bool:
    """
    Факт — строка или список синонимов, засчитывается любой из них.
    Нужно из-за русской морфологии: «друг» и «друзьями» — одно и то же,
    но общей подстроки у них нет, одной строкой оба варианта не покрыть.
    """
    alternatives = fact if isinstance(fact, list) else [fact]
    return any(_norm(alt) in norm_answer for alt in alternatives)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------- валидация

def validate_schema(questions: List[dict]) -> List[str]:
    """Обязательные поля по типу и уникальность id — до любого обращения к БД/API."""
    errors: List[str] = []
    seen_ids = set()

    for q in questions:
        qid = q.get("id")
        if not qid:
            errors.append(f"вопрос без id: {q.get('question')!r}")
            continue
        if qid in seen_ids:
            errors.append(f"{qid}: дубль id")
        seen_ids.add(qid)

        qtype = q.get("type")
        if qtype not in TYPES:
            errors.append(f"{qid}: неизвестный type {qtype!r}, ожидается один из {sorted(TYPES)}")
            continue
        if not q.get("question"):
            errors.append(f"{qid}: пустой question")

        if qtype in TYPES_WITH_SOURCE:
            sources = q.get("expected_sources") or []
            if not sources:
                errors.append(f"{qid}: для type={qtype} нужен непустой expected_sources")
            for src in sources:
                if not isinstance(src, dict) or not src.get("article_title"):
                    errors.append(f"{qid}: элемент expected_sources без article_title: {src!r}")
                elif src.get("chunk_type") not in (None, "text", "infobox"):
                    errors.append(f"{qid}: chunk_type должен быть text или infobox, а не {src.get('chunk_type')!r}")
            if not q.get("expected_facts"):
                errors.append(f"{qid}: для type={qtype} нужен непустой expected_facts")
            for fact in q.get("expected_facts") or []:
                ok_str = isinstance(fact, str) and fact.strip()
                ok_list = isinstance(fact, list) and fact and all(isinstance(a, str) and a.strip() for a in fact)
                if not (ok_str or ok_list):
                    errors.append(f"{qid}: элемент expected_facts должен быть строкой или списком строк: {fact!r}")

        if qtype == "typo" and not q.get("expect_corrected_to"):
            errors.append(f"{qid}: для type=typo нужен expect_corrected_to")

        if qtype == "followup":
            history = q.get("history") or []
            if not any(isinstance(m, dict) and m.get("role") == "user" for m in history):
                errors.append(f"{qid}: для type=followup нужен history хотя бы с одной репликой role=user")
            elif not _needs_llm_rewrite(q.get("question", ""), history):
                errors.append(f"{qid}: в вопросе нет местоимения — rewrite_query не сработает, тест ничего не проверит")

        if qtype == "out_of_scope":
            if q.get("expected_sources") or q.get("expected_facts"):
                errors.append(f"{qid}: у out_of_scope не должно быть expected_sources/expected_facts")
            if not q.get("expect_refusal"):
                errors.append(f"{qid}: для type=out_of_scope нужен expect_refusal: true")

    return errors


async def validate_sources(questions: List[dict], session, universe: str) -> List[str]:
    """
    Каждая статья из expected_sources должна реально быть в БД — иначе промах
    будет выглядеть как проблема retrieval, хотя это опечатка в датасете.
    """
    rows = await session.execute(
        select(distinct(ArticleChunk.article_title)).where(ArticleChunk.universe == universe)
    )
    known = {_norm(title) for (title,) in rows.all()}

    errors: List[str] = []
    for q in questions:
        for src in q.get("expected_sources") or []:
            title = src.get("article_title", "")
            if _norm(title) not in known:
                errors.append(f"{q['id']}: статьи «{title}» нет в БД (universe={universe})")
    return errors


# ---------------------------------------------------------------- оценка

def match(top: List[dict], expected_sources: List[dict]) -> Tuple[bool, Optional[int]]:
    """
    Первый чанк из top, совпавший с любым эталоном: (hit, rank 1..k) или (False, None).

    Эталон — article_title (обязательно) + необязательные уточнения:
    section_contains (подстрока section_path) и chunk_type ("infobox"/"text").
    chunk_type нужен отдельно, потому что у инфобокса section_path равен
    названию статьи — section_contains его от других чанков не отличит.
    """
    for rank, chunk in enumerate(top, 1):
        for src in expected_sources:
            if _norm(chunk.get("article_title")) != _norm(src["article_title"]):
                continue
            section = src.get("section_contains")
            if section and section.lower() not in (chunk.get("section_path") or "").lower():
                continue
            wanted_type = src.get("chunk_type")
            if wanted_type and chunk.get("chunk_type") != wanted_type:
                continue
            return True, rank
    return False, None


def _base_result(q: dict) -> dict:
    return {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "corrected_question": None,
        "rewritten_question": None,
        "hit": None,
        "rank": None,
        "best_distance": None,
        "top5": [],
        "typo_fixed": None,
        "unexpected_correction": None,
        "answer": None,
        "facts_found": None,
        "facts_missing": None,
        "forbidden_found": None,
        "refusal_ok": None,
        "error": None,
    }


async def evaluate_one(
    q: dict,
    vector_store: PostgresVectorStore,
    llm: GeminiAdapter,
    use_case: Optional[ChatUseCase],
    universe: str,
    k: int,
) -> dict:
    r = _base_result(q)
    history = q.get("history")

    # --- retrieval: те же шаги, что в ChatUseCase.execute ---
    corrected = await vector_store.correct_typos(q["question"], universe)
    r["corrected_question"] = corrected
    query = corrected
    if _needs_llm_rewrite(corrected, history):
        query = await llm.rewrite_query(corrected, history)
        r["rewritten_question"] = query

    vec = await llm.generate_embedding(query, task_type="RETRIEVAL_QUERY")
    top = await vector_store.search_similar(vec, universe, limit=k)

    r["best_distance"] = top[0]["distance"] if top else None
    r["top5"] = [
        {
            "article_title": c["article_title"],
            "section_path": c["section_path"],
            "chunk_type": c["chunk_type"],
            "distance": c["distance"],
        }
        for c in top
    ]

    if q["type"] in TYPES_WITH_SOURCE:
        r["hit"], r["rank"] = match(top, q["expected_sources"])
    if q["type"] == "typo":
        r["typo_fixed"] = _norm(q["expect_corrected_to"]) in _norm(corrected)
    else:
        # В вопросе без опечатки корректору нечего менять. Если изменил —
        # ложное срабатывание correct_typos (класс «какйо -> какр»,
        # «Геральт -> гера»): реальное слово подменили похожим именем из лора.
        r["unexpected_correction"] = _norm(corrected) != _norm(q["question"])

    # --- full: ответ через продакшен-путь целиком ---
    if use_case is not None:
        answer = "".join([
            piece async for piece in use_case.execute(
                q["question"], universe, history=history, mode=AnswerMode.ARCHIVIST
            )
        ])
        if answer.startswith(_CHAT_ERROR_PREFIX):
            r["error"] = f"chat: {answer}"
            return r
        r["answer"] = answer
        norm_answer = _norm(answer)

        if q["type"] in TYPES_WITH_SOURCE:
            found = [f for f in q["expected_facts"] if _fact_present(f, norm_answer)]
            r["facts_found"] = found
            r["facts_missing"] = [f for f in q["expected_facts"] if f not in found]
        else:
            refused = any(marker in norm_answer for marker in _REFUSAL_MARKERS)
            leaked = [f for f in q.get("forbidden_facts") or [] if _norm(f) in norm_answer]
            r["forbidden_found"] = leaked
            # Пустой ответ — не отказ: молчание нельзя засчитать как честное «не знаю».
            r["refusal_ok"] = bool(answer.strip()) and refused and not leaked

    return r


# ---------------------------------------------------------------- агрегаты

def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(results: List[dict], full: bool) -> dict:
    def agg(rs: List[dict]) -> dict:
        ok = [r for r in rs if not r["error"]]
        scored = [r for r in ok if r["type"] in TYPES_WITH_SOURCE]
        oos = [r for r in ok if r["type"] == "out_of_scope"]
        typo = [r for r in ok if r["type"] == "typo"]

        out: Dict[str, Any] = {"n": len(rs), "errors": sum(1 for r in rs if r["error"])}
        if scored:
            out["hits"] = sum(1 for r in scored if r["hit"])
            out["recall_at_k"] = round(out["hits"] / len(scored), 4)
            out["mrr"] = _mean([(1 / r["rank"]) if r["rank"] else 0.0 for r in scored])
        if typo:
            out["typo_fixed_rate"] = _mean([1.0 if r["typo_fixed"] else 0.0 for r in typo])
        non_typo = [r for r in ok if r["unexpected_correction"] is not None]
        if non_typo:
            out["unexpected_correction_rate"] = _mean(
                [1.0 if r["unexpected_correction"] else 0.0 for r in non_typo]
            )
        if oos:
            out["mean_best_distance"] = _mean(
                [r["best_distance"] for r in oos if r["best_distance"] is not None]
            )
        if full:
            scores = []
            for r in scored:
                if r["facts_found"] is None:
                    continue
                total = len(r["facts_found"]) + len(r["facts_missing"])
                if total:
                    scores.append(len(r["facts_found"]) / total)
            if scores:
                out["facts_score"] = _mean(scores)
            if oos:
                out["refusal_ok_rate"] = _mean([1.0 if r["refusal_ok"] else 0.0 for r in oos])
        return out

    types_present = sorted({r["type"] for r in results})
    return {
        "overall": agg(results),
        "by_type": {t: agg([r for r in results if r["type"] == t]) for t in types_present},
    }


# ---------------------------------------------------------------- вывод

_AGG_COLUMNS = [
    ("n", "n"),
    ("recall_at_k", "recall@k"),
    ("mrr", "mrr"),
    ("typo_fixed_rate", "typo"),
    ("unexpected_correction_rate", "badfix"),
    ("facts_score", "facts"),
    ("refusal_ok_rate", "refusal"),
    ("mean_best_distance", "oos_dist"),
    ("errors", "err"),
]


def print_summary(payload: dict) -> None:
    meta, agg = payload["meta"], payload["aggregates"]
    print(f"\n=== {meta['label']} | {meta['mode']} | {meta['n_questions']} вопросов | k={meta['k']} ===")
    header = f"{'type':14}" + "".join(f"{label:>9}" for _, label in _AGG_COLUMNS)
    print(header)
    for name, a in [("overall", agg["overall"])] + list(agg["by_type"].items()):
        print(f"{name:14}" + "".join(f"{_fmt(a.get(key)):>9}" for key, _ in _AGG_COLUMNS))

    misses = [
        q for q in payload["questions"]
        if q["type"] in TYPES_WITH_SOURCE and not q["error"] and q["hit"] is False
    ]
    if misses:
        print("\n--- Промахи retrieval (эталона нет в top-k) ---")
        for q in misses:
            top1 = q["top5"][0] if q["top5"] else None
            where = (
                f"top-1 = «{top1['article_title']}» / {top1['section_path']} (dist {top1['distance']:.3f})"
                if top1 else "пусто"
            )
            print(f"  {q['id']}: {where}")

    bad_fixes = [q for q in payload["questions"] if q["unexpected_correction"]]
    if bad_fixes:
        print("\n--- Ложные срабатывания correct_typos (вопрос без опечатки изменён) ---")
        for q in bad_fixes:
            print(f"  {q['id']}: «{q['question']}» -> «{q['corrected_question']}»")

    if meta["mode"] == "full":
        bad_facts = [q for q in payload["questions"] if q["facts_missing"]]
        if bad_facts:
            print("\n--- Ответы с недостающими фактами ---")
            for q in bad_facts:
                print(f"  {q['id']}: не нашли {q['facts_missing']}")
        bad_refusal = [
            q for q in payload["questions"]
            if q["type"] == "out_of_scope" and not q["error"] and not q["refusal_ok"]
        ]
        if bad_refusal:
            print("\n--- out_of_scope без корректного отказа ---")
            for q in bad_refusal:
                leak = f", утекло: {q['forbidden_found']}" if q["forbidden_found"] else ""
                print(f"  {q['id']}: «{(q['answer'] or '')[:100]}»{leak}")

    errors = [q for q in payload["questions"] if q["error"]]
    if errors:
        print("\n--- Ошибки прогона ---")
        for q in errors:
            print(f"  {q['id']}: {q['error']}")

    print(f"\nРезультат: {payload['meta']['result_path']}")


def print_compare(a: dict, b: dict) -> None:
    ma, mb = a["meta"], b["meta"]
    print(f"A: {ma['label']} ({ma['timestamp']}, {ma['mode']}, commit {ma.get('git_commit')})")
    print(f"B: {mb['label']} ({mb['timestamp']}, {mb['mode']}, commit {mb.get('git_commit')})\n")

    scopes = ["overall"] + sorted(set(a["aggregates"]["by_type"]) | set(b["aggregates"]["by_type"]))
    print(f"{'scope':14}{'metric':20}{'A':>9}{'B':>9}{'Δ':>9}")
    for scope in scopes:
        da = a["aggregates"]["overall"] if scope == "overall" else a["aggregates"]["by_type"].get(scope, {})
        db = b["aggregates"]["overall"] if scope == "overall" else b["aggregates"]["by_type"].get(scope, {})
        for key, _ in _AGG_COLUMNS:
            va, vb = da.get(key), db.get(key)
            if va is None and vb is None:
                continue
            delta = "-"
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                d = vb - va
                delta = f"{d:+.2f}" if isinstance(d, float) else f"{d:+d}"
            print(f"{scope:14}{key:20}{_fmt(va):>9}{_fmt(vb):>9}{delta:>9}")

    qa = {q["id"]: q for q in a["questions"]}
    qb = {q["id"]: q for q in b["questions"]}
    only_a, only_b = sorted(set(qa) - set(qb)), sorted(set(qb) - set(qa))
    if only_a:
        print(f"\nТолько в A: {only_a}")
    if only_b:
        print(f"\nТолько в B: {only_b}")

    def top1(q: dict) -> str:
        t = q["top5"][0] if q["top5"] else None
        return f"«{t['article_title']}» (dist {t['distance']:.3f})" if t else "пусто"

    common = [i for i in qa if i in qb]
    regressions = [i for i in common if qa[i]["hit"] is True and qb[i]["hit"] is False]
    improvements = [i for i in common if qa[i]["hit"] is False and qb[i]["hit"] is True]
    rank_changes = [
        i for i in common
        if qa[i]["hit"] and qb[i]["hit"] and qa[i]["rank"] != qb[i]["rank"]
    ]

    if regressions:
        print("\n--- РЕГРЕССИИ (hit -> miss) ---")
        for i in regressions:
            print(f"  {i}: было {top1(qa[i])} -> стало {top1(qb[i])}")
    if improvements:
        print("\n--- Улучшения (miss -> hit) ---")
        for i in improvements:
            print(f"  {i}: было {top1(qa[i])} -> стало rank {qb[i]['rank']}")
    if rank_changes:
        print("\n--- Изменился rank ---")
        for i in rank_changes:
            print(f"  {i}: {qa[i]['rank']} -> {qb[i]['rank']}")

    fix_changes = [
        i for i in common
        if qa[i].get("unexpected_correction") != qb[i].get("unexpected_correction")
    ]
    if fix_changes:
        print("\n--- Изменились ложные срабатывания correct_typos ---")
        for i in fix_changes:
            print(f"  {i}: {qa[i].get('unexpected_correction')} -> {qb[i].get('unexpected_correction')}"
                  f"  («{qb[i]['corrected_question']}»)")

    if ma["mode"] == "full" and mb["mode"] == "full":
        refusal_changes = [
            i for i in common
            if qa[i]["type"] == "out_of_scope" and qa[i]["refusal_ok"] != qb[i]["refusal_ok"]
        ]
        if refusal_changes:
            print("\n--- Изменился refusal_ok ---")
            for i in refusal_changes:
                print(f"  {i}: {qa[i]['refusal_ok']} -> {qb[i]['refusal_ok']}")
        facts_changes = [
            i for i in common
            if qa[i]["facts_missing"] is not None and qa[i]["facts_missing"] != qb[i]["facts_missing"]
        ]
        if facts_changes:
            print("\n--- Изменились недостающие факты ---")
            for i in facts_changes:
                print(f"  {i}: {qa[i]['facts_missing']} -> {qb[i]['facts_missing']}")

    if not (regressions or improvements or rank_changes):
        print("\nПо retrieval различий между прогонами нет.")


# ---------------------------------------------------------------- main

async def run(args: argparse.Namespace) -> None:
    data = yaml.safe_load(Path(args.dataset).read_text(encoding="utf-8"))
    universe = data.get("universe", "AC")
    k = int(data.get("k", 5))
    questions: List[dict] = data["questions"]

    if args.only_type:
        questions = [q for q in questions if q.get("type") == args.only_type]
    if args.only_id:
        questions = [q for q in questions if q.get("id") == args.only_id]
    if not questions:
        sys.exit("Под фильтр не попало ни одного вопроса.")

    errors = validate_schema(questions)
    if errors:
        print("Ошибки схемы датасета:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    llm = GeminiAdapter(
        api_key=settings.GEMINI_API_KEY,
        chat_model_name=settings.CHAT_MODEL,
        embedding_model_name=settings.EMBEDDING_MODEL,
        embedding_dimension=settings.VECTOR_DIMENSION,
    )

    async with async_session_maker() as session:
        vector_store = PostgresVectorStore(session)

        errors = await validate_sources(questions, session, universe)
        if errors:
            print("Статьи из датасета отсутствуют в БД:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        if args.validate:
            print(f"OK: {len(questions)} вопросов, все источники есть в БД (universe={universe}).")
            return

        use_case = ChatUseCase(vector_store, llm) if args.full else None
        results: List[dict] = []
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q['id']}")
            try:
                r = await evaluate_one(q, vector_store, llm, use_case, universe, k)
            except Exception as exc:
                r = _base_result(q)
                r["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  ❌ {r['error']}")
            results.append(r)
            if args.full and i < len(questions):
                await asyncio.sleep(CHAT_PAUSE_SEC)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    result_path = RESULTS_DIR / f"{stamp}_{args.label}.json"
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": args.label,
            "mode": "full" if args.full else "retrieval",
            "git_commit": _git_commit(),
            "dataset": str(Path(args.dataset)),
            "universe": universe,
            "k": k,
            "n_questions": len(questions),
            "result_path": str(result_path),
        },
        "aggregates": aggregate(results, args.full),
        "questions": results,
    }
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval-набор RAG: recall@k, факты, отказы")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--label", default="run", help="Метка прогона — попадает в имя файла результата")
    parser.add_argument("--only-type", choices=sorted(TYPES), help="Прогнать только вопросы этого типа")
    parser.add_argument("--only-id", help="Прогнать только один вопрос по id")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true", help="Только проверить датасет и наличие статей в БД")
    mode.add_argument("--retrieval-only", action="store_true", help="Только retrieval (по умолчанию)")
    mode.add_argument("--full", action="store_true", help="Retrieval + генерация ответа (ARCHIVIST)")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="Сравнить два результата")
    args = parser.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        b = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        print_compare(a, b)
        return

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
