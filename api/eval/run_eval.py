"""Measures the RAG against eval/questions.json.

Retrieval (always):
  - hit rate@1 and @k: share of answerable questions with an expected section in the top 1 / top k
  - MRR: mean of 1/rank of the first expected section (0 when it is not in the top k)
  - document hit rate@k: same as hit rate@k, but any section of the expected document counts
  - threshold sweep on the top-1 similarity, treating "top-1 score >= threshold" as
    "the knowledge base has the answer":
      false positive = unanswerable question above the threshold (the LLM gets unrelated context)
      false negative = answerable question below the threshold (it would be refused)

Answers (--answers, one chat call per question):
  - answerable question: correct when the answer cites an expected section
  - unanswerable question: correct when the answer cites nothing (it says it doesn't know)

Run from api/ (uses .env.local and the configured database and embedding model):
    python eval/run_eval.py [--k 5] [--answers]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_env import load_env_file  # noqa: E402

load_env_file()

import db  # noqa: E402
import rag  # noqa: E402
from openai_client import embed, embedding_model  # noqa: E402

QUESTIONS_FILE = Path(__file__).parent / "questions.json"


class Case(TypedDict):
    question: str
    expected: list[str]


class Retrieved(TypedDict):
    case: Case
    ids: list[str]
    scores: list[float]


def document_of(section_id: str) -> str:
    return section_id.rsplit("-", 1)[0]  # MON-01-04 -> MON-01


def first_hit_rank(expected: list[str], ids: list[str]) -> int | None:
    return next((rank for rank, section in enumerate(ids, start=1) if section in expected), None)


async def retrieve_all(cases: list[Case], k: int) -> list[Retrieved]:
    model = embedding_model()
    embeddings = await embed([case["question"] for case in cases], model)  # one batch call
    async with await rag.open_connection() as connection:
        results = []
        for case, embedding in zip(cases, embeddings):
            sources = await db.search(connection, embedding, model, k)
            results.append(Retrieved(
                case=case,
                ids=[source["id"] for source in sources],
                scores=[source["score"] for source in sources],
            ))
    return results


def report_retrieval(results: list[Retrieved], k: int) -> None:
    answerable = [r for r in results if r["case"]["expected"]]
    ranks = [first_hit_rank(r["case"]["expected"], r["ids"]) for r in answerable]
    doc_hits = [
        any(document_of(section) in {document_of(e) for e in r["case"]["expected"]} for section in r["ids"])
        for r in answerable
    ]
    n = len(answerable)
    print(f"\n## Retrieval ({n} answerable questions, k={k})")
    print(f"hit rate@1:          {sum(rank == 1 for rank in ranks) / n:.0%}")
    print(f"hit rate@{k}:          {sum(rank is not None for rank in ranks) / n:.0%}")
    print(f"MRR:                 {sum(1 / rank for rank in ranks if rank) / n:.3f}")
    print(f"document hit rate@{k}: {sum(doc_hits) / n:.0%}")

    misses = [(r, rank) for r, rank in zip(answerable, ranks) if rank != 1]
    if misses:
        print("\nNot ranked first:")
        for r, rank in misses:
            got = ", ".join(f"{i} {s:.3f}" for i, s in zip(r["ids"], r["scores"]))
            where = f"rank {rank}" if rank else f"not in top {k}"
            print(f"  - {r['case']['question']}\n    expected {r['case']['expected']} ({where}); got {got}")


def report_threshold(results: list[Retrieved]) -> None:
    positives = [r["scores"][0] for r in results if r["case"]["expected"]]
    negatives = [r["scores"][0] for r in results if not r["case"]["expected"]]
    print(f"\n## Threshold on the top-1 similarity ({len(positives)} answerable, {len(negatives)} unanswerable)")
    print(f"answerable top-1:   min {min(positives):.3f}  max {max(positives):.3f}")
    print(f"unanswerable top-1: min {min(negatives):.3f}  max {max(negatives):.3f}")

    def errors(threshold: float) -> tuple[int, int]:
        false_positives = sum(score >= threshold for score in negatives)
        false_negatives = sum(score < threshold for score in positives)
        return false_positives, false_negatives

    # Candidate thresholds: midpoints between consecutive observed scores.
    scores = sorted(set(positives + negatives))
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    best = min(candidates, key=lambda t: (sum(errors(t)), -t))
    total = len(positives) + len(negatives)
    print("\nthreshold  FP  FN  error rate")
    for threshold in sorted({0.30, 0.35, 0.40, 0.45, round(best, 3)}):
        fp, fn = errors(threshold)
        marker = "  <- best" if threshold == round(best, 3) else ""
        print(f"  {threshold:.3f}    {fp:>2}  {fn:>2}  {(fp + fn) / total:.0%}{marker}")


async def report_answers(cases: list[Case]) -> None:
    print(f"\n## Answers ({len(cases)} questions, one chat call each)")
    failures = []
    for case in cases:
        result = await rag.answer_question(case["question"])
        cited = [source["id"] for source in result["sources"] if source["cited"]]
        correct = bool(set(cited) & set(case["expected"])) if case["expected"] else not cited
        if not correct:
            failures.append((case, cited, result["answer"]))

    answerable = [c for c in cases if c["expected"]]
    wrong_answerable = sum(1 for case, _, _ in failures if case["expected"])
    wrong_unanswerable = len(failures) - wrong_answerable
    print(f"error rate:               {len(failures) / len(cases):.0%} ({len(failures)}/{len(cases)})")
    print(f"answerable, wrong/no cite: {wrong_answerable}/{len(answerable)}")
    print(f"unanswerable, answered:    {wrong_unanswerable}/{len(cases) - len(answerable)}")
    for case, cited, answer in failures:
        print(f"  - {case['question']}\n    expected {case['expected'] or 'no citation'}; cited {cited or 'nothing'}\n    {answer}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--answers", action="store_true", help="also evaluate the generated answers")
    args = parser.parse_args()

    cases: list[Case] = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]
    print(f"# Evaluation — embedding model {embedding_model()}, {len(cases)} questions")
    results = await retrieve_all(cases, args.k)
    report_retrieval(results, args.k)
    report_threshold(results)
    if args.answers:
        await report_answers(cases)


if __name__ == "__main__":
    asyncio.run(main())
