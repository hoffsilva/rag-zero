"""RAGAS metrics on the answers saved by run_eval.py --answers --save.

run_eval.py measures retrieval ranks and citation compliance with no LLM in the loop. This adds
the RAGAS view of the answers themselves, for each saved run (pipeline and strategy) that has answers:

  - faithfulness: share of the answer's claims supported by the sections it was generated
    from (LLM judge: splits the answer into statements and checks each against the context)
  - answer relevancy: does the answer address the question? (LLM judge writes questions the
    answer would fit; mean cosine similarity between their embeddings and the real question's)
  - context precision / recall by ID: share of the sections sent to the LLM that were expected,
    and share of expected sections that were sent (no LLM; the same set arithmetic as RAGAS's
    ID-based metrics, which are deprecated in 0.4 with no replacement)

Only answerable questions are scored: for the unanswerable ones, the right answer is a refusal,
which run_eval.py's citation check already measures. The judge is the configured chat model, so it
grades answers written by the same model; read the scores as a comparison between runs, not as
absolute truth. With 23 questions, differences of a few points are noise.

Run from api/ after run_eval.py (needs requirements-eval.txt and OPENAI_API_KEY):
    python eval/run_ragas.py [--model text-embedding-3-small]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.indexing import load_chunks

RESULTS_FILE = Path(__file__).resolve().parents[2] / "web" / "data" / "evaluation.json"
CONCURRENCY = 8
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class Judge:
    def __init__(self) -> None:
        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.require_openai_api_key())
        llm = llm_factory(settings.require_chat_model(), client=client)
        embeddings = embedding_factory("openai", model=settings.openai_embedding_model, client=client)
        self.chat_model = settings.require_chat_model()
        self.faithfulness = Faithfulness(llm=llm)
        self.relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)  # type: ignore[arg-type]
        self.texts = {chunk["id"]: chunk["text"] for chunk in load_chunks()}
        self.limit = asyncio.Semaphore(CONCURRENCY)

    async def score(self, question: dict[str, Any]) -> dict[str, Any]:
        sources: list[str] = question["answer_sources"]
        found = set(sources) & set(question["expected"])
        async with self.limit:
            faithfulness = await self.faithfulness.ascore(
                user_input=question["question"],
                response=question["answer"],
                retrieved_contexts=[self.texts[section_id] for section_id in sources],
            )
            relevancy = await self.relevancy.ascore(
                user_input=question["question"], response=question["answer"]
            )
        return {
            "question": question["question"],
            "faithfulness": round(float(faithfulness.value), 3),
            "answer_relevancy": round(float(relevancy.value), 3),
            "context_precision": round(len(found) / len(sources), 3),
            "context_recall": round(len(found) / len(question["expected"]), 3),
        }


async def score_run(judge: Judge, run: dict[str, Any]) -> dict[str, Any]:
    answerable = [q for q in run["questions"] if q["expected"]]
    scores = await asyncio.gather(*(judge.score(q) for q in answerable))
    return {
        "judge_model": judge.chat_model,
        "questions_scored": len(scores),
        "means": {metric: round(mean(s[metric] for s in scores), 3) for metric in METRICS},
        "questions": scores,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="text-embedding-3-small", help="embedding model of the runs")
    args = parser.parse_args()

    data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    judge = Judge()
    print(f"{'pipeline':<10} {'strategy':<12} " + " ".join(f"{m:>17}" for m in METRICS))
    for run in data["runs"].values():
        if run["embedding_model"] != args.model or not run.get("answers"):
            continue
        run["ragas"] = await score_run(judge, run)
        means = run["ragas"]["means"]
        print(
            f"{run['pipeline']:<10} {run['strategy']:<12} " + " ".join(f"{means[m]:>17.3f}" for m in METRICS)
        )
        RESULTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {RESULTS_FILE.relative_to(RESULTS_FILE.parents[2])}")


if __name__ == "__main__":
    asyncio.run(main())
