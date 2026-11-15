# DocQA - Question Answering over Documents (RAG)

A small retrieval-augmented generation (RAG) service. It indexes your documents, retrieves the most relevant passages with BM25, and answers questions grounded in those passages, with citations. It refuses to answer when the documents do not contain the answer.

## How it works
1. **Chunking:** documents are split into overlapping 80-word chunks (`docqa/chunking.py`).
2. **Retrieval:** a BM25 ranker written from scratch (`docqa/bm25.py`) with tokenization, stop-word removal, and light plural stemming.
3. **Refusal check:** if less than 40% of the question's terms appear in the best chunk, the service answers "I don't know" without calling a model.
4. **Answering:** the top chunks go to an answerer:
   - **Offline (default):** an extractive answerer returns the best-matching sentence with a `[n]` citation.
   - **LLM mode:** set `OPENAI_API_KEY` to use an OpenAI-compatible chat model. The prompt (`docqa/prompt.py`) numbers the sources, tells the model to answer only from them, and to cite.
5. **API:** FastAPI endpoints `/ingest`, `/ask`, `/health`.

## Run
Requirements: Python 3.10+.

```bash
pip install -r requirements.txt
DOCS_DIR=data/handbook uvicorn docqa.api:app --reload      # Windows PowerShell: $env:DOCS_DIR="data/handbook"; uvicorn docqa.api:app --reload
```
Then open http://localhost:8000/docs, or:
```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How soon must the primary on-call respond to a page?"}'
```
LLM mode (optional): `export OPENAI_API_KEY=...` (and optionally `OPENAI_MODEL`, `OPENAI_BASE_URL`).

## Tests
```bash
python -m pytest -q
```
14 tests: chunking, BM25 ranking, prompt construction, service logic with a fake answerer, refusal behaviour, API validation, and the OpenAI-compatible client against a mocked HTTP transport (no real network call).

## Evaluation
`python eval/run_eval.py` runs retrieval on `eval/qa.json` over the sample corpus in `data/handbook/` (6 short documents, 12 chunks).

| Metric | Result |
|--------|--------|
| Questions (answerable) | 24 |
| Top-1 retrieval hit rate | 23/24 (96%) |
| Top-3 retrieval hit rate | 24/24 (100%) |
| Out-of-scope questions correctly refused | 6/6 |
| Answerable questions wrongly refused | 3/24 |

**Read these numbers with care:** the corpus and the questions were both written by the author, the set is small, and it is not a benchmark. The 3 wrongly refused questions are vocabulary mismatches that lexical search cannot handle (for example, "doctor's note" vs "medical certificate").

## Limitations
- Lexical retrieval only, so synonyms and paraphrases can be missed
- In-memory index, lost on restart
- Plain-text/Markdown input only (no PDF parsing)
- The extractive answerer returns one sentence, not a synthesized answer

## Possible improvements
- Hybrid retrieval with sentence embeddings
- PDF ingestion and a persistent vector store
- Answer-quality evaluation against a real LLM

## Author
Navneet Singh Yadav - [LinkedIn](https://www.linkedin.com/in/navneet-singh-yadav-439886281/)
