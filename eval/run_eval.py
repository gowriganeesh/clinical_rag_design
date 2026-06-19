"""Production evaluation harness.

RAGAS scores retrieval quality, DeepEval scores the agent's answer, and a
deterministic check scores abstention on unanswerable and ACL-negative items.
"""
from __future__ import annotations

import json

from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from ragas import EvaluationDataset, evaluate as ragas_evaluate
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall

from eval.judges import deepeval_judge, ragas_judge
from src.agents.orchestrator import respond
from src.config import config
from src.tools.search_tool import retrieve

GOLDEN_PATH = config.data_dir / "golden" / "qa_set.jsonl"
ABSTAIN = "I don't have enough information in the records to answer that."


def load_golden() -> list[dict]:
    with open(GOLDEN_PATH, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run() -> None:
    config.validate_production()
    rows = _pipeline_rows()
    answerable = [row for row in rows if row["answerable"]]
    refusal = [row for row in rows if not row["answerable"]]

    ragas_llm, ragas_emb = ragas_judge()
    ragas_ds = EvaluationDataset.from_list(
        [
            {
                "user_input": row["question"],
                "retrieved_contexts": row["contexts"] or [""],
                "reference": row["expected_answer"],
            }
            for row in answerable
        ]
    )
    retrieval_scores = ragas_evaluate(
        dataset=ragas_ds,
        metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    judge = deepeval_judge()
    clinical_correctness = GEval(
        name="Clinical Correctness",
        criteria=(
            "Determine whether the actual output correctly answers the question using "
            "only the retrieved records, matches the key facts in the expected output, "
            "cites a source, and adds no unsupported clinical claims."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
        threshold=0.7,
    )
    test_cases = [
        LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row["expected_answer"],
            retrieval_context=row["contexts"] or ["(no context retrieved)"],
        )
        for row in answerable
    ]
    deepeval_evaluate(
        test_cases=test_cases,
        metrics=[
            FaithfulnessMetric(threshold=0.8, model=judge),
            AnswerRelevancyMetric(threshold=0.7, model=judge),
            clinical_correctness,
        ],
    )

    correct = sum(1 for row in refusal if _is_abstention(row["answer"]))
    rate = correct / len(refusal) if refusal else None

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Items: {len(rows)} (answerable={len(answerable)}, refusal={len(refusal)})")
    print("\n-- Retrieval quality (RAGAS) --")
    print(retrieval_scores)
    print("\n-- Agent answer quality (DeepEval) -- per-metric table printed above")
    print("\n-- Abstention (deterministic) --")
    if rate is not None:
        print(f"Correct abstentions: {correct}/{len(refusal)} ({rate:.0%})")
        for row in refusal:
            status = "PASS" if _is_abstention(row["answer"]) else "FAIL"
            print(f"  [{status}] {row['id']} ({row['category']}): {row['answer'][:80]}")


def _pipeline_rows() -> list[dict]:
    rows = []
    for golden in load_golden():
        retrieval = retrieve(golden["question"], user_id=golden["user_id"])
        contexts = [chunk["content"] for chunk in retrieval["chunks"]]
        result = respond(golden["question"], user_id=golden["user_id"])
        rows.append(
            {
                **golden,
                "contexts": contexts,
                "retrieved_docs": [chunk["source_doc"] for chunk in retrieval["chunks"]],
                "answer": result["answer"],
                "meta": result["meta"],
            }
        )
    return rows


def _is_abstention(answer: str) -> bool:
    return ABSTAIN.lower() in (answer or "").lower()


if __name__ == "__main__":
    run()
