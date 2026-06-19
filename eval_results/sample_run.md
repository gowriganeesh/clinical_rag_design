# Production Evaluation Report Format

Production evaluation requires Azure OpenAI, Azure AI Search, and reviewed
goldens in `data/golden/qa_set.jsonl`.

Command:

```bash
python -m eval.run_eval
```

The run prints:

```text
================ EVALUATION SUMMARY ================
Items: <total> (answerable=<n>, refusal=<n>)

-- Retrieval quality (RAGAS) --
<LLMContextPrecisionWithReference and LLMContextRecall>

-- Agent answer quality (DeepEval) -- per-metric table printed above

-- Abstention (deterministic) --
Correct abstentions: <correct>/<total> (<rate>)
  [PASS|FAIL] <item id> (<category>): <answer preview>
```
