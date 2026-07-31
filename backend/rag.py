"""File-backed retrieval over the knowledge folder.

Two modes, chosen automatically by corpus size:

* Small knowledge base (under SMALL_KB_CHARS) — the whole thing goes into the
  cached system prompt. Perfect recall, no vocabulary-mismatch failures, and
  prompt caching makes the repeated tokens near-free.
* Larger knowledge base — chunk and retrieve. TF-IDF over word bigrams *and*
  character n-grams, so "park" still matches "parking". Pure lexical matching
  cannot bridge true synonyms ("braces" vs "orthodontics"), which is why every
  result also carries an outline of the available sections: it lets the model
  re-query `search_knowledge_base` with the right words.

`Retriever.search` / `.context_for` / `.full_text` are the only things the rest
of the app depends on, so swapping in dense embeddings later means replacing
this class and nothing else.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import settings

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}

CHUNK_CHARS = 700
CHUNK_OVERLAP = 120

# Below this, skip retrieval entirely and put the whole corpus in the prompt.
# ~20k characters is roughly 5k tokens — cheap once the prefix is cached.
SMALL_KB_CHARS = 20_000


@dataclass
class Chunk:
    source: str
    heading: str
    text: str

    def render(self) -> str:
        prefix = f"[{self.source}"
        if self.heading:
            prefix += f" — {self.heading}"
        return f"{prefix}]\n{self.text}"


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text into (heading, body) sections."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if match:
            if buffer:
                sections.append((heading, "\n".join(buffer).strip()))
                buffer = []
            heading = match.group(2).strip()
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer).strip()))
    return [(h, b) for h, b in sections if b]


def _window(body: str) -> list[str]:
    """Break a section body into overlapping chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > CHUNK_CHARS:
            chunks.append(current)
            current = (current[-CHUNK_OVERLAP:] + "\n\n" + para) if CHUNK_OVERLAP else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    # A single paragraph longer than the window still needs splitting.
    out: list[str] = []
    for chunk in chunks:
        while len(chunk) > CHUNK_CHARS * 2:
            out.append(chunk[: CHUNK_CHARS * 2])
            chunk = chunk[CHUNK_CHARS * 2 - CHUNK_OVERLAP :]
        out.append(chunk)
    return [c.strip() for c in out if c.strip()]


class Retriever:
    """Thread-safe TF-IDF index over the knowledge directory."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory or settings.knowledge_dir
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._documents: list[tuple[str, str]] = []   # (source, full text)
        self._word_vec: TfidfVectorizer | None = None
        self._char_vec: TfidfVectorizer | None = None
        self._word_matrix = None
        self._char_matrix = None
        self.reindex()

    # ------------------------------------------------------------------ build

    def reindex(self) -> dict[str, int]:
        chunks: list[Chunk] = []
        documents: list[tuple[str, str]] = []
        for path in sorted(self.directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            text = _read(path).strip()
            if not text:
                continue
            source = path.relative_to(self.directory).as_posix()
            documents.append((source, text))
            for heading, body in _split_sections(text):
                for piece in _window(body):
                    chunks.append(Chunk(source=source, heading=heading, text=piece))

        word_vec = char_vec = None
        word_matrix = char_matrix = None
        if chunks:
            corpus = [f"{c.heading}\n{c.text}" for c in chunks]
            # Word bigrams catch phrases; character n-grams catch morphology and
            # typos ("park" ~ "parking", "instalment" ~ "instalments").
            word_vec = TfidfVectorizer(
                lowercase=True, sublinear_tf=True, ngram_range=(1, 2),
                stop_words="english", min_df=1,
            )
            char_vec = TfidfVectorizer(
                lowercase=True, sublinear_tf=True, analyzer="char_wb",
                ngram_range=(3, 5), min_df=1,
            )
            try:
                word_matrix = word_vec.fit_transform(corpus)
            except ValueError:
                word_vec, word_matrix = None, None
            try:
                char_matrix = char_vec.fit_transform(corpus)
            except ValueError:
                char_vec, char_matrix = None, None

        with self._lock:
            self._chunks = chunks
            self._documents = documents
            self._word_vec, self._word_matrix = word_vec, word_matrix
            self._char_vec, self._char_matrix = char_vec, char_matrix

        return {"files": len(documents), "chunks": len(chunks)}

    # ----------------------------------------------------------------- query

    def search(self, query: str, k: int = 4, min_score: float = 0.03) -> list[tuple[Chunk, float]]:
        query = (query or "").strip()
        if not query:
            return []
        with self._lock:
            chunks = self._chunks
            word_vec, word_matrix = self._word_vec, self._word_matrix
            char_vec, char_matrix = self._char_vec, self._char_matrix
        if not chunks:
            return []

        scores = np.zeros(len(chunks))
        for vec, matrix, weight in ((word_vec, word_matrix, 0.65),
                                    (char_vec, char_matrix, 0.35)):
            if vec is None or matrix is None:
                continue
            try:
                scores += weight * cosine_similarity(vec.transform([query]), matrix)[0]
            except ValueError:
                continue

        top = np.argsort(scores)[::-1][:k]
        return [(chunks[i], float(scores[i])) for i in top if scores[i] >= min_score]

    def outline(self) -> str:
        """Every section heading, so the model can re-query with the right words."""
        with self._lock:
            seen: list[str] = []
            for chunk in self._chunks:
                label = f"{chunk.source}: {chunk.heading}" if chunk.heading else chunk.source
                if label not in seen:
                    seen.append(label)
        return "\n".join(f"- {s}" for s in seen)

    def context_for(self, query: str, k: int = 4) -> str:
        hits = self.search(query, k=k)
        if not hits:
            return ""
        return "\n\n---\n\n".join(chunk.render() for chunk, _ in hits)

    def full_text(self) -> str:
        """The whole corpus when it is small enough to sit in the cached prompt."""
        with self._lock:
            documents = list(self._documents)
        total = sum(len(text) for _, text in documents)
        if not documents or total > SMALL_KB_CHARS:
            return ""
        return "\n\n".join(f"===== {source} =====\n{text}" for source, text in documents)

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            documents = list(self._documents)
            chunk_count = len(self._chunks)
        total = sum(len(t) for _, t in documents)
        return {
            "files": len(documents),
            "chunks": chunk_count,
            "characters": total,
            "inlined": int(bool(documents) and total <= SMALL_KB_CHARS),
        }

    def sources(self) -> list[str]:
        with self._lock:
            return [source for source, _ in self._documents]


retriever = Retriever()
