"""Deterministic PHI scrubbing for incoming queries (Presidio).

A query can contain PHI (e.g. a patient name typed by the user). We strip it
BEFORE the query reaches the router, the cache, the retriever, or the LLM, so
PHI never leaves the boundary. This is deterministic on purpose - a dedicated
NER tool is more reliable and cheaper than asking an LLM to de-identify.

If Presidio / spaCy aren't installed, it degrades to a minimal regex scrub so
the pipeline still runs.
"""
import re

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    _PRESIDIO = True
except Exception:  # pragma: no cover - depends on optional deps + spaCy model
    _PRESIDIO = False

_FALLBACK_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b\+?\d[\d\s-]{7,}\d\b"), "[PHONE]"),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"), "[DATE]"),
]


def scrub_phi(text: str) -> str:
    """Return the query with PHI replaced by placeholder tokens."""
    if _PRESIDIO:
        results = _analyzer.analyze(text=text, language="en")
        return _anonymizer.anonymize(text=text, analyzer_results=results).text
    scrubbed = text
    for pattern, repl in _FALLBACK_PATTERNS:
        scrubbed = pattern.sub(repl, scrubbed)
    return scrubbed
