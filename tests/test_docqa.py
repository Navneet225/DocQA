import json

import httpx
import pytest
from fastapi.testclient import TestClient

from docqa.api import create_app
from docqa.bm25 import BM25, tokenize
from docqa.chunking import Chunk, chunk_text
from docqa.llm import ExtractiveAnswerer, OpenAICompatibleAnswerer
from docqa.prompt import SYSTEM, build_messages
from docqa.service import NO_ANSWER, DocQA


# ---------- chunking ----------
def test_chunks_overlap_and_cover_all_words():
    text = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(text, "a.md", size=80, overlap=20)
    assert len(chunks) == 3
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]  # overlap
    covered = set(" ".join(c.text for c in chunks).split())
    assert covered == set(text.split())
    assert [c.id for c in chunks] == [0, 1, 2]


def test_chunk_short_and_empty_text():
    assert len(chunk_text("just a few words", "a.md")) == 1
    assert chunk_text("", "a.md") == []


def test_chunk_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_text("x y z", "a.md", size=10, overlap=10)


# ---------- bm25 ----------
def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What IS the Leave Policy?") == ["leave", "policy"]


def test_bm25_ranks_relevant_doc_first_and_ignores_unknown_terms():
    docs = ["cats purr and sleep all day", "dogs bark at the mailman", "stock markets fell on tuesday"]
    idx = BM25(docs)
    assert idx.top_k("why do dogs bark", 3)[0][0] == 1
    assert idx.top_k("quantum chromodynamics") == []


def test_bm25_empty_corpus():
    assert BM25([]).top_k("anything") == []


# ---------- prompt ----------
def test_prompt_numbers_sources_and_forbids_outside_knowledge():
    msgs = build_messages("When?", [Chunk(0, "a.md", "alpha text"), Chunk(1, "b.md", "beta text")])
    assert msgs[0] == {"role": "system", "content": SYSTEM}
    user = msgs[1]["content"]
    assert "[1] (a.md)" in user and "[2] (b.md)" in user and "Question: When?" in user
    assert "ONLY" in SYSTEM and "I don't know" in SYSTEM


# ---------- service ----------
class FakeAnswerer:
    def __init__(self):
        self.calls = []

    def answer(self, question, chunks):
        self.calls.append((question, chunks))
        return "fake answer [1]"


def make_service(fake=None):
    qa = DocQA(answerer=fake or FakeAnswerer())
    qa.add_document("Employees receive 24 days of paid annual leave each year.", "leave.md")
    qa.add_document("Passwords must be at least 14 characters long.", "security.md")
    return qa


def test_ask_retrieves_right_source_and_calls_answerer_with_it():
    fake = FakeAnswerer()
    qa = make_service(fake)
    out = qa.ask("How many days of annual leave do I get?")
    assert out["answer"] == "fake answer [1]"
    assert out["sources"][0]["source"] == "leave.md"
    assert fake.calls[0][1][0].source == "leave.md"


def test_ask_refuses_without_calling_llm_when_nothing_relevant():
    fake = FakeAnswerer()
    qa = make_service(fake)
    out = qa.ask("What is the capital of Mongolia?")
    assert out == {"answer": NO_ANSWER, "sources": []}
    assert fake.calls == []


def test_ask_on_empty_index():
    assert DocQA(answerer=FakeAnswerer()).ask("anything here")["answer"] == NO_ANSWER


def test_extractive_answerer_picks_best_sentence_with_citation():
    chunks = [Chunk(0, "a.md", "Intro text. Passwords must be at least 14 characters. Other note.")]
    ans = ExtractiveAnswerer().answer("minimum password length", chunks)
    assert "14 characters" in ans and ans.endswith("[1]")


# ---------- OpenAI-compatible client (mocked transport, no network) ----------
def test_openai_client_sends_expected_request_and_parses_reply():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "  Grounded answer [1]  "}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    a = OpenAICompatibleAnswerer(api_key="k123", model="m", base_url="https://llm.test/v1", client=client)
    out = a.answer("Q?", [Chunk(0, "a.md", "some text")])
    assert out == "Grounded answer [1]"
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k123"
    assert seen["body"]["model"] == "m" and seen["body"]["temperature"] == 0
    assert "[1] (a.md)" in seen["body"]["messages"][1]["content"]


def test_openai_client_raises_on_http_error():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={})))
    a = OpenAICompatibleAnswerer(api_key="bad", client=client)
    with pytest.raises(httpx.HTTPStatusError):
        a.answer("Q?", [Chunk(0, "a.md", "t")])


# ---------- API ----------
def test_api_ingest_then_ask_and_validation():
    c = TestClient(create_app(DocQA(answerer=FakeAnswerer())))
    assert c.get("/health").json() == {"status": "ok", "chunks": 0}
    r = c.post("/ingest", json={"source": "leave.md", "text": "Employees get 24 days of annual leave."})
    assert r.json()["chunks_added"] == 1
    r = c.post("/ask", json={"question": "how many days of annual leave", "k": 2})
    assert r.status_code == 200 and r.json()["sources"][0]["source"] == "leave.md"
    assert c.post("/ask", json={"question": "hi"}).status_code == 422  # too short
    assert c.post("/ask", json={"question": "valid question here", "k": 99}).status_code == 422
    assert c.post("/ingest", json={"source": "", "text": "x"}).status_code == 422
