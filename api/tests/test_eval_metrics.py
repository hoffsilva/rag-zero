import pytest

import run_eval


def question(expected: list[str], retrieved: list[tuple[str, float]]) -> run_eval.QuestionResult:
    ids = [section for section, _ in retrieved]
    return run_eval.QuestionResult(
        question="q",
        expected=expected,
        retrieved=[run_eval.Retrieved(id=section, score=score) for section, score in retrieved],
        rank=run_eval.first_hit_rank(expected, ids) if expected else None,
        latency_ms=10,
    )


def test_first_hit_rank():
    assert run_eval.first_hit_rank(["B"], ["A", "B", "C"]) == 2
    assert run_eval.first_hit_rank(["C", "A"], ["A", "B", "C"]) == 1
    assert run_eval.first_hit_rank(["Z"], ["A", "B"]) is None


def test_retrieval_metrics_ignores_unanswerable_questions():
    results = [
        question(["MON-01-01"], [("MON-01-01", 0.7), ("MON-02-01", 0.5)]),  # rank 1
        question(["MON-01-02"], [("MON-01-01", 0.6), ("MON-01-02", 0.5)]),  # rank 2
        question(["MON-03-01"], [("MON-03-02", 0.5), ("MON-02-01", 0.4)]),  # wrong section, right document
        question([], [("MON-01-04", 0.5)]),  # unanswerable: not counted
    ]

    metrics = run_eval.retrieval_metrics(results)

    assert metrics["answerable"] == 3
    assert metrics["hit_rate_at_1"] == pytest.approx(1 / 3)
    assert metrics["hit_rate_at_k"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 2 + 0) / 3)
    assert metrics["document_hit_rate_at_k"] == pytest.approx(1.0)  # MON-03-02 belongs to MON-03


def test_threshold_counts_false_positives_and_negatives():
    results = [
        question(["A"], [("A", 0.70)]),
        question(["B"], [("B", 0.50)]),
        question(["C"], [("C", 0.40)]),
        question([], [("A", 0.45)]),  # on-topic but unanswerable
        question([], [("A", 0.10)]),  # off-topic
    ]

    metrics = run_eval.threshold_metrics(results)
    rows = {row["threshold"]: row for row in metrics["rows"]}

    # 0.30: both unanswerable questions pass -> 1 false positive (0.45); 0.10 is filtered.
    assert rows[0.30]["false_positives"] == 1 and rows[0.30]["false_negatives"] == 0
    # 0.45: the 0.45 question still passes (>=), the 0.40 answerable one is refused.
    assert rows[0.45]["false_positives"] == 1 and rows[0.45]["false_negatives"] == 1
    assert rows[0.50]["error_rate"] == pytest.approx(1 / 5)
    # Best threshold sits between observed scores with the fewest errors.
    assert metrics["best"]["false_positives"] + metrics["best"]["false_negatives"] == 1
    assert metrics["unanswerable_top1"] == {"min": 0.10, "max": 0.45}
