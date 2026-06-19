"""Sentence-aware chunking.

The chunker packs whole sentences into a token budget and overlaps only by
complete sentences. It avoids cutting clinical statements in half.
"""
from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_text(text: str, max_tokens: int = 500, overlap_sentences: int = 1) -> list[str]:
    """Return text chunks, each ending on a sentence boundary."""

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        tokens = _approx_tokens(sentence)
        if current and current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current))
            if overlap_sentences > 0:
                current = current[-overlap_sentences:]
                current_tokens = sum(_approx_tokens(s) for s in current)
            else:
                current = []
                current_tokens = 0
        current.append(sentence)
        current_tokens += tokens

    if current:
        chunks.append(" ".join(current))
    return chunks
