"""Answer generators. The service works offline with ExtractiveAnswerer;
set OPENAI_API_KEY to use an OpenAI-compatible chat model instead."""
import os
import re
from typing import Protocol

import httpx

from docqa.bm25 import tokenize
from docqa.chunking import Chunk
from docqa.prompt import build_messages


class Answerer(Protocol):
    def answer(self, question: str, chunks: list[Chunk]) -> str: ...


class ExtractiveAnswerer:
    """No-LLM fallback: returns the sentence from the top chunks that best overlaps the question."""

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        q = set(tokenize(question))
        best, best_score, best_idx = "", 0, 0
        for idx, c in enumerate(chunks, start=1):
            for sent in re.split(r"(?<=[.!?])\s+", c.text):
                score = len(q & set(tokenize(sent)))
                if score > best_score:
                    best, best_score, best_idx = sent.strip(), score, idx
        return f"{best} [{best_idx}]" if best else "I don't know based on the provided documents."


class OpenAICompatibleAnswerer:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, client: httpx.Client | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.client = client or httpx.Client(timeout=30)

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": build_messages(question, chunks), "temperature": 0},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def default_answerer() -> Answerer:
    return OpenAICompatibleAnswerer() if os.environ.get("OPENAI_API_KEY") else ExtractiveAnswerer()
