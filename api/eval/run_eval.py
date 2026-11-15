"""Measures the RAG against eval/questions.json.

Retrieval (always):
  - hit rate@1 and @k: share of answerable questions with an expected section in the top 1 / top k
  - MRR: mean of 1/rank of the first expected section (0 when it is not in the top k)
  - document hit rate@k: same as hit rate@k, but any section of the expected document counts
  - threshold sweep on the top-1 similarity, treating "top-1 score >= threshold" as
    "the knowledge base has the answer":
      false positive = unanswerable question above the threshold (the LLM gets unrelated context)
      false negative = answerable question below the threshold (it would be refused)

Answers (--answers, one chat call per question) — citation compliance, not factual accuracy:
  - answerable question: compliant when the answer cites an expected section
  - unanswerable question: compliant when the answer cites nothing (it says it doesn't know)
  It does not check that the cited text supports the answer: a wrong answer that cites an
  expected section passes, and so does an invented answer without citations.
  Generation uses the app's own retrieval (rag.TOP_K sections), saved as answer_sources.

--save writes the results into web/data/evaluation.json under the embedding model's name,
which the /evaluation page of the web app renders.

Run from api/ (uses .env.local and the configured database and embedding model):
    python eval/run_eval.py [--k 5] [--strategy NAME|all] [--answers] [--save]
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_env import load_env_file  # noqa: E402

load_env_file()

import rag  # noqa: E402
import retrieval  # noqa: E402
from openai_client import chat_model, embedding_model  # noqa: E402

QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).resolve().parents[2] / "web" / "data" / "evaluation.json"
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]


class Case(TypedDict):
    question: str
    expected: list[str]


class Retrieved(TypedDict):
    id: str
    score: float


class QuestionResult(TypedDict, total=False):
    question: str
    expected: list[str]
    retrieved: list[Retrieved]
    rank: int | None
    latency_ms: float
    answer: str
    answer_sources: list[str]
    cited: list[str]
    citation_ok: bool


def document_of(section_id: str) -> str:
    return section_id.rsplit("-", 1)[0]  # MON-01-04 -> MON-01


def first_hit_rank(expected: list[str], ids: list[str]) -> int | None:
    return next((rank for rank, section in enumerate(ids, start=1) if section in expected), None)


async def retrieve_all(cases: list[Case], k: int, strategy: retrieval.Strategy) -> list[QuestionResult]:
    """Runs the app's own retrieval for each question and times it (embedding, database and,
    for multi_query/rerank, the extra LLM call)."""
    model = embedding_model()
    results: list[QuestionResult] = []
    async with await rag.open_connection() as connection:
        for case in cases:
            started = time.perf_counter()
            sources = await retrieval.run(strategy, connection, case["question"], model, k)
            latency_ms = (time.perf_counter() - started) * 1000
            ids = [source["id"] for source in sources]
            results.append(QuestionResult(
                question=case["question"],
                expected=case["expected"],
                retrieved=[Retrieved(id=s["id"], score=round(s["score"], 4)) for s in sources],
                rank=first_hit_rank(case["expected"], ids) if case["expected"] else None,
                latency_ms=round(latency_ms),
            ))
    return results


def retrieval_metrics(results: list[QuestionResult]) -> dict:
    latencies = sorted(r["latency_ms"] for r in results)
    answerable = [r for r in results if r["expected"]]
    n = len(answerable)
    doc_hits = [
        any(document_of(s["id"]) in {document_of(e) for e in r["expected"]} for s in r["retrieved"])
        for r in answerable
    ]
    return {
        "answerable": n,
        "hit_rate_at_1": sum(r["rank"] == 1 for r in answerable) / n,
        "hit_rate_at_k": sum(r["rank"] is not None for r in answerable) / n,
        "mrr": sum(1 / r["rank"] for r in answerable if r["rank"]) / n,
        "document_hit_rate_at_k": sum(doc_hits) / n,
        "latency_ms_median": latencies[len(latencies) // 2],
    }


def threshold_metrics(results: list[QuestionResult]) -> dict:
    positives = [r["retrieved"][0]["score"] for r in results if r["expected"]]
    negatives = [r["retrieved"][0]["score"] for r in results if not r["expected"]]
    total = len(positives) + len(negatives)

    def row(threshold: float) -> dict:
        false_positives = sum(score >= threshold for score in negatives)
        false_negatives = sum(score < threshold for score in positives)
        return {
            "threshold": round(threshold, 3),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "error_rate": (false_positives + false_negatives) / total,
        }

    # Best threshold: the midpoint between observed scores with the fewest errors (highest on ties).
    scores = sorted(set(positives + negatives))
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    best = min(candidates, key=lambda t: (row(t)["error_rate"], -t))
    return {
        "answerable_top1": {"min": min(positives), "max": max(positives)},
        "unanswerable_top1": {"min": min(negatives), "max": max(negatives)},
        "rows": [row(t) for t in THRESHOLDS],
        "best": row(best),
    }


async def evaluate_answers(results: list[QuestionResult], strategy: retrieval.Strategy) -> dict:
    for result in results:
        answer = await rag.answer_question(result["question"], strategy)
        cited = [source["id"] for source in answer["sources"] if source["cited"]]
        result["answer"] = answer["answer"]
        result["answer_sources"] = [source["id"] for source in answer["sources"]]
        result["cited"] = cited
        result["citation_ok"] = (
            bool(set(cited) & set(result["expected"])) if result["expected"] else not cited
        )
    wrong = [r for r in results if not r["citation_ok"]]
    return {
        "chat_model": chat_model(),
        "answer_k": rag.TOP_K,
        "citation_error_rate": len(wrong) / len(results),
        "answerable_wrong": sum(1 for r in wrong if r["expected"]),
        "unanswerable_answered": sum(1 for r in wrong if not r["expected"]),
    }


def print_report(report: dict) -> None:
    retrieval, threshold, k = report["retrieval"], report["threshold"], report["k"]
    print(f"# Evaluation — {report['embedding_model']}, strategy {report['strategy']}, {len(report['questions'])} questions")
    print(f"\n## Retrieval ({retrieval['answerable']} answerable questions, k={k})")
    print(f"hit rate@1:          {retrieval['hit_rate_at_1']:.0%}")
    print(f"hit rate@{k}:          {retrieval['hit_rate_at_k']:.0%}")
    print(f"MRR:                 {retrieval['mrr']:.3f}")
    print(f"document hit rate@{k}: {retrieval['document_hit_rate_at_k']:.0%}")
    print(f"median latency:      {retrieval['latency_ms_median']} ms")
    misses = [q for q in report["questions"] if q["expected"] and q["rank"] != 1]
    if misses:
        print("\nNot ranked first:")
        for q in misses:
            got = ", ".join(f"{r['id']} {r['score']:.3f}" for r in q["retrieved"])
            where = f"rank {q['rank']}" if q["rank"] else f"not in top {k}"
            print(f"  - {q['question']}\n    expected {q['expected']} ({where}); got {got}")

    if threshold is not None:
        print("\n## Threshold on the top-1 similarity")
        print(f"answerable top-1:   min {threshold['answerable_top1']['min']:.3f}  max {threshold['answerable_top1']['max']:.3f}")
        print(f"unanswerable top-1: min {threshold['unanswerable_top1']['min']:.3f}  max {threshold['unanswerable_top1']['max']:.3f}")
        print("\nthreshold  FP  FN  error rate")
        for row in threshold["rows"] + [threshold["best"]]:
            marker = "  <- best" if row is threshold["best"] else ""
            print(f"  {row['threshold']:.3f}    {row['false_positives']:>2}  {row['false_negatives']:>2}  {row['error_rate']:.0%}{marker}")

    answers = report.get("answers")
    if answers:
        total = len(report["questions"])
        wrong = answers["answerable_wrong"] + answers["unanswerable_answered"]
        print(f"\n## Answer citations ({answers['chat_model']}, generated from the top {answers['answer_k']})")
        print(f"citation error rate:       {answers['citation_error_rate']:.0%} ({wrong}/{total})")
        print(f"answerable, wrong/no cite: {answers['answerable_wrong']}")
        print(f"unanswerable, answered:    {answers['unanswerable_answered']}")
        for q in report["questions"]:
            if not q["citation_ok"]:
                print(f"  - {q['question']}\n    expected {q['expected'] or 'no citation'}; cited {q['cited'] or 'nothing'}\n    {q['answer']}")


def run_key(report: dict) -> str:
    return f"{report['embedding_model']}|{report['strategy']}"


def save_report(report: dict) -> None:
    """Keeps one entry per (embedding model, strategy), so the page can compare them."""
    data = json.loads(RESULTS_FILE.read_text(encoding="utf-8")) if RESULTS_FILE.exists() else {"runs": {}}
    data["runs"][run_key(report)] = report
    data["default_strategy"] = retrieval.default_strategy()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS_FILE.relative_to(RESULTS_FILE.parents[2])}")


async def evaluate(cases: list[Case], k: int, strategy: retrieval.Strategy, answers: bool) -> dict:
    results = await retrieve_all(cases, k, strategy)
    return {
        "embedding_model": embedding_model(),
        "strategy": strategy,
        "date": date.today().isoformat(),
        "k": k,
        "retrieval": retrieval_metrics(results),
        # Only the vector strategy has a similarity score a threshold can be put on.
        "threshold": threshold_metrics(results) if strategy == "vector" else None,
        "answers": await evaluate_answers(results, strategy) if answers else None,
        "questions": results,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--strategy",
        choices=[*retrieval.STRATEGIES, "all"],
        default=None,
        help="retrieval strategy to evaluate (default: RETRIEVAL_STRATEGY), or all of them",
    )
    parser.add_argument("--answers", action="store_true", help="also evaluate the generated answers")
    parser.add_argument("--save", action="store_true", help=f"write the results to {RESULTS_FILE.name}")
    args = parser.parse_args()

    cases: list[Case] = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    strategies = retrieval.STRATEGIES if args.strategy == "all" else [args.strategy or retrieval.default_strategy()]
    for strategy in strategies:
        report = await evaluate(cases, args.k, strategy, args.answers)
        print_report(report)
        if args.save:
            save_report(report)


if __name__ == "__main__":
    asyncio.run(main())
