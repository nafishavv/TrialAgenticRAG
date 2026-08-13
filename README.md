# RAGTrial

**RAGTrial** is a Retrieval-Augmented Generation chatbot for public-service information in Kabupaten Batang, Indonesia.

The project explores three RAG approaches — **Naive RAG**, **Enhanced RAG**, and **Agentic RAG** — and compares how different retrieval and orchestration strategies affect question answering.

---

## Overview

Public-service information is spread across handbooks, regional regulations, licensing pages, and agency directories. Most of it is public, but it is hard to search: a citizen usually has to know which document holds the answer before they can find it.

RAG is a good fit for this, since the answers already exist in official documents and should be retrieved and cited rather than memorized by a model. The system takes a question in Bahasa Indonesia, finds the relevant passages, and answers from them.

The more interesting question is *how much machinery the retrieval side actually needs*. Adding hybrid search, reranking, or an agent loop makes a system more capable in principle, but also slower and harder to reason about. Rather than assuming a more complex pipeline is always better, this project builds three architectures on the same data and compares them.

---

## Three RAG Approaches

| Approach | Idea | Who controls the flow |
| --- | --- | --- |
| **Naive RAG** | Retrieve and answer, nothing else | Fixed |
| **Enhanced RAG** | A designed pipeline with stronger retrieval | Developer |
| **Agentic RAG** | An LLM decides how to search | The model, at runtime |

### Naive RAG

A simple baseline. It retrieves the most similar passages and uses them as context for the LLM. No routing, no reranking, no filtering — it always retrieves and always answers.

### Enhanced RAG

A fixed pipeline with extra retrieval components: it first decides whether a question needs retrieval at all, then combines semantic and keyword search, reranks the candidates, and generates an answer. The steps are decided ahead of time by the developer, and every question follows the same path.

### Agentic RAG

An LLM-driven workflow. Each domain is exposed as a search tool, and the model decides which tools to use, how to phrase the query, whether the results are good enough, and whether to search again — or to skip retrieval entirely for greetings and out-of-scope questions.

All three run on the same documents, the same index, and the same models, so the comparison reflects the architecture rather than the setup around it.

---

## Data & Use Case

The system works with public-service information from Kabupaten Batang, covering:

- **Civil registration** — ID cards, family cards, birth and death certificates
- **Government agencies** — directory and contact information for local offices
- **Licensing** — requirements, procedures, and fees
- **Social & regulatory** — regional regulations related to social services

These domains are intentionally different in character. An agency lookup and a question about a regional regulation stress retrieval in very different ways, which makes this a useful setting for comparing retrieval strategies. The pipeline is built so that more public-service domains can be added as the project grows.

---

## Technical Highlights

- **Hybrid retrieval** combining semantic and lexical search.
- **Cross-encoder reranking** to refine retrieved candidates before generation.
- **Intent handling** so the assistant can decline questions outside its scope instead of guessing.
- **Agentic retrieval** with LLM-driven tool selection and self-correction when results look weak.
- **LangGraph orchestration** for the agentic workflow.
- **Modular pipeline** where retrieval components can be swapped or configured independently.
- **Query tracing** that records what was retrieved and which decisions were made on every query, so runs can be inspected afterwards.
- **Web and CLI interfaces** sharing the same underlying session layer.

---

## Evaluation

The project includes an evaluation framework used to compare the three architectures on the same test questions. It looks at:

- **Retrieval quality** — whether the right passages are found
- **Answer quality** — faithfulness to the retrieved context and relevance to the question
- **Refusal behavior** — how each architecture handles out-of-scope questions
- **Routing and intent** — which domain the system decides to search, and whether it should search at all
- **Efficiency** — latency broken down by stage, and how many model calls each approach needs

Results are compared per domain and per question type, and individual retrieval components can be turned on and off to see what each one contributes. Experiments are still running as part of the thesis, so no final numbers are reported here.

---

## Tech Stack

- **Python 3.12**
- **LangChain** — retrieval, tools, prompting
- **LangGraph** — agentic workflow
- **ChromaDB** — vector store
- **Gemini** — LLM and embeddings
- **BAAI/bge-reranker** — cross-encoder reranking
- **FastAPI** — backend, with a lightweight static frontend
- **uv** — environment and dependency management

---

## Running the Project

Requires Python 3.12+ and a Gemini API key.

```bash
uv sync && uv pip install -e .
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Build the index from the source documents:

```bash
uv run python scripts/preprocess.py --source all
uv run python scripts/build_vectorstore.py --source all
```

Ask a question from the CLI, choosing the architecture with `--mode`:

```bash
uv run python scripts/ask.py "Apa syarat KTP elektronik?"
uv run python scripts/ask.py "Alamat Disdukcapil?" --mode naive
uv run python scripts/ask.py "Urus pindah domisili, ke dinas mana?" --mode agentic

uv run python scripts/ask.py --chat          # multi-turn
```

Or run the web app at `http://127.0.0.1:8000`:

```bash
uv run python scripts/serve.py
```

To run the evaluation across all three architectures:

```bash
uv run python -m eval.run_eval --systems naive enhanced agentic
```

---

## Documentation

| Document | What it covers |
| --- | --- |
| [docs/ARCHITECTURE_REVIEW.md](docs/ARCHITECTURE_REVIEW.md) | Technical audit of the system: pipelines, capability assessment, gaps vs the state of the art |
| [architecture_mapping.md](architecture_mapping.md) | Concept → source-file map, per-file responsibility, end-to-end pipeline traces |
| [docs/CAPABILITIES.md](docs/CAPABILITIES.md) | Per-capability inventory of what is built vs planned, and the reasoning behind each keep/cut |
| [docs/EVAL_REPORT.md](docs/EVAL_REPORT.md) | Full evaluation results, breakdowns, anomaly analysis and limitations |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Prioritized remaining work, plus the ablation-study design |
| [docs/DATA_REPORT.md](docs/DATA_REPORT.md) | Data sources, preprocessing, chunking strategy and corpus statistics |
| [docs/INDEXING.md](docs/INDEXING.md) | The indexing (pre-retrieval) stage, with verified index numbers |
| [docs/TESTSET_CURATION.md](docs/TESTSET_CURATION.md) | How the evaluation test set was selected and curated |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Running the app locally and deploying it publicly |
| [docs/archive/](docs/archive/) | Superseded planning documents and refactor history, kept for provenance |

---

## Project Status

**Active development.**

The three RAG architectures, the public-service data pipeline, the web and CLI interfaces, and the evaluation framework are implemented and working. The project is currently being refined and evaluated as part of an undergraduate thesis, so some parts of the pipeline may still change.
