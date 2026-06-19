"""Generate answerable golden candidates and merge the safety core.

Generated items are candidates only and require clinician review before
promotion to data/golden/qa_set.jsonl.
"""
from __future__ import annotations

import json

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig

from eval.judges import deepeval_judge
from src.config import config

DATA_DIR = config.data_dir
SAFETY_CORE = DATA_DIR / "golden" / "safety_core.jsonl"
OUT = DATA_DIR / "golden" / "qa_set.generated.jsonl"

STYLING = StylingConfig(
    scenario="A clinician querying a specific patient's de-identified medical record.",
    task="Answer clinical questions grounded only in the retrieved patient record, with a source citation.",
    input_format="A natural-language clinical question, paraphrased so it does NOT copy the record's wording.",
    expected_output_format="A concise factual answer grounded in the record, citing the source document.",
)


def load_manifest() -> list[dict]:
    return json.loads((DATA_DIR / "access_control.json").read_text(encoding="utf-8"))["documents"]


def main(max_per_doc: int = 3) -> None:
    config.validate_production()
    synth = Synthesizer(model=deepeval_judge(), styling_config=STYLING)

    generated = []
    for entry in load_manifest():
        text = (DATA_DIR / "documents" / entry["source_doc"]).read_text(encoding="utf-8")
        goldens = synth.generate_goldens_from_contexts(
            contexts=[[text]],
            max_goldens_per_context=max_per_doc,
        )
        authorized_user = next((u for u in entry["acl"] if u.startswith("EXAMINER")), entry["acl"][0])
        for index, golden in enumerate(goldens):
            generated.append(
                {
                    "id": f"gen_{entry['patient_id']}_{index:02d}",
                    "question": golden.input,
                    "user_id": authorized_user,
                    "query_type": "patient",
                    "answerable": True,
                    "expected_answer": golden.expected_output,
                    "ground_truth_docs": [entry["source_doc"]],
                    "category": "generated_answerable",
                }
            )

    safety = _read_safety_core()
    _write_rows(generated + safety)
    print(f"Generated {len(generated)} answerable candidate(s) + {len(safety)} safety-core item(s).")
    print(f"Wrote {OUT}")
    print("REVIEW these with clinician sign-off before promoting to data/golden/qa_set.jsonl.")


def _read_safety_core() -> list[dict]:
    if not SAFETY_CORE.exists():
        return []
    return [json.loads(line) for line in SAFETY_CORE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_rows(rows: list[dict]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
