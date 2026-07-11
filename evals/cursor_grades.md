# Manual RAG Pipeline Grades

Graded from `evals/model_response.jsonl` against `corpus/` and `evals/qa_set.jsonl`, using the same criteria as `scripts/eval.py` / RAGAS:

- **Retrieval** — were the retrieved chunks relevant to the question?
- **Faithfulness** — is the answer supported by those chunks?
- **Answer** — does the final response match the ground truth?

**Corpus indexed:** 10 PDFs under `corpus/` (Montúfar, Hornik, OpenVINO white paper, Milvus, etc.). Repo source code (`src/rag/`) is **not** ingested.

**Generator:** Ollama `llama3.2:3b` (pipeline queries). Judge LLM was not run — these are manual scores.

---

## Corpus questions (Q1–12)

| # | Topic | Retrieval | Faithfulness | Answer | Verdict |
|---|-------|-----------|--------------|--------|---------|
| 1 | Zaslavsky shallow regions | Good — Zaslavsky bound in ctx | Yes | Correct formula | **Pass** |
| 2 | Rectifier + maxout | Good | Yes | Correct | **Pass** |
| 3 | Deep rectifier lower bound | Good — Corollary 5 in ctx | Yes | Correct Ω bound | **Pass** |
| 4 | TFD dataset | Good | Yes | Correct | **Pass** |
| 5 | Stone-Weierstrass (Hornik) | Good — Stone-Weierstrass in ctx | **No** — said Kolmogorov / Thm 2.4 | Wrong theorem | **Fail** |
| 6 | Squashing function def. | Good — Definition 2.3 in ctx | **No** — "derivative → 0" vs limit at −∞ | Wrong 3rd condition | **Fail** |
| 7 | NNCF tool | Good — NNCF mentioned repeatedly | N/A | Said "I don't know" | **Fail** (over-refusal) |
| 8 | Four offline RAG stages | Good | Yes | Correct 4 steps | **Pass** |
| 9 | $2,500 out-of-pocket | **Miss** — healthcare example not retrieved | Yes (given ctx) | Correct refusal | **Retrieval fail** |
| 10 | Linear region definition | Good | Yes | Correct | **Pass** |
| 11 | Hornik vs Montúfar | Good — both papers retrieved | Partial | Vague, ends with "I don't know" | **Partial** |
| 12 | Theorem 8 / faithfulness | Good — Theorem 8 in ctx | Partial | Cites theorem but calls it upper bound; weak on RAGAS | **Partial** |

**Corpus score: 7 pass, 2 partial, 3 fail → ~71%**

---

## Refusal questions (Q13–15)

Should answer "I don't know" — information is absent from the corpus.

| # | Topic | Verdict |
|---|-------|---------|
| 13 | Montúfar learning rate | **Pass** — correct refusal |
| 14 | OpenVINO on RTX 4090 | **Fail** — hallucinated GPU setup; corpus only mentions Intel |
| 15 | RRF in Hornik (1989) | **Pass** — correct refusal (verbose, but right) |

**Refusal score: 2/3**

---

## Repo architecture questions (Q16–20)

Expected answers reference `src/rag/` files. Only PDFs are indexed, so these are structurally hard.

| # | Expected | Got | Verdict |
|---|----------|-----|---------|
| 16 | `chunker.py` | LangChain `RecursiveCharacterTextSplitter` | **Fail** |
| 17 | `retrieve.py` / `reranker.py` | Milvus rank fusion | **Fail** |
| 18 | `mcp_server.py` | "I don't know" | **Fail** (fair refusal, wrong for eval set) |
| 19 | `guards.py` | "I don't know" (Claude safety docs retrieved) | **Fail** |
| 20 | `pipeline.py` | LangChain chains | **Fail** |

**Repo score: 0/5**

---

## Overall summary

| Category | Score |
|----------|-------|
| Corpus factual (12) | **71%** (7 pass, 2 partial, 3 fail) |
| Refusal / abstention (3) | **67%** (2/3) |
| Repo architecture (5) | **0%** |
| **Overall (20)** | **~50%** |

---

## Failure modes

1. **Generation errors despite good retrieval** — Q5, Q6, Q7: Montúfar / Hornik / OpenVINO chunks were retrieved but the model picked the wrong fact or refused when the answer was in context.
2. **Hallucination under pressure** — Q14 invented NVIDIA GPU steps from generic OpenVINO `device='GPU'` snippets.
3. **Retrieval gaps** — Q9 missed the $2,500 healthcare example buried in the OpenVINO doc.
4. **Corpus / eval mismatch** — Q16–20 ask about repo files; corpus is PDFs only, so retrieval surfaces Milvus / OpenVINO / LangChain content and the model guesses or refuses.

---

## What would move the needle

- Ingest `src/rag/*.py` (or an `ARCHITECTURE.md`) for Q16–20
- Tune refusal behavior so Q7 doesn't say "I don't know" when NNCF is in context
- Consider a stronger generator than `llama3.2:3b` for factual extraction (Q5, Q6)
