#!/usr/bin/env python3
"""Score the RAG pipeline against a fixed QA set using RAGAS."""

import argparse
import json
import os
import statistics
import sys
import time
import types
import warnings

# Ragas pulls in langchain_community vertexai modules that aren't always installed.
for mod_name, class_name in [
    ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
    ("langchain_community.llms.vertexai", "VertexAI"),
]:
    if mod_name not in sys.modules:
        stub = types.ModuleType(mod_name)
        setattr(stub, class_name, type(class_name, (object,), {}))
        sys.modules[mod_name] = stub

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
for _var in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_ENDPOINT", "LANGCHAIN_PROJECT"):
    os.environ.pop(_var, None)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

sys.path.insert(0, os.path.abspath("src"))

from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from rag.pipeline import query
from ragas import RunConfig, evaluate
from ragas.metrics import context_precision, faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def build_evaluator(evaluator: str, gemini_model: str):
    if evaluator == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY or GOOGLE_API_KEY must be set")

        llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            temperature=0.0,
            api_key=api_key,
            max_retries=3,
            timeout=60,
        )
        # One worker at a time; free-tier Gemini keys hit 429s quickly otherwise.
        run_config = RunConfig(max_workers=1, timeout=300, max_wait=30)
    else:
        llm = ChatOpenAI(
            model="llama3.2:3b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model_kwargs={"response_format": {"type": "json_object"}},
            temperature=0.0,
        )
        run_config = RunConfig(max_workers=1, timeout=120)

    return llm, run_config


def run(qa_path: str, evaluator: str, limit: int | None, gemini_model: str) -> None:
    with open(qa_path) as f:
        qa = [json.loads(line) for line in f if line.strip()]
    if limit:
        qa = qa[:limit]

    rows = []
    latencies = []

    print(f"running {len(qa)} queries...")
    for item in qa:
        t0 = time.perf_counter()
        out = query(item["q"])
        latencies.append((time.perf_counter() - t0) * 1000)
        rows.append({
            "question": item["q"],
            "answer": out["answer"],
            "contexts": [c["text"] for c in out["citations"]],
            "ground_truth": item.get("ground_truth", ""),
        })

    llm, run_config = build_evaluator(evaluator, gemini_model)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"scoring with ragas ({evaluator})...")
    metrics = [faithfulness, AnswerRelevancy(strictness=1), context_precision]
    scores = evaluate(
        dataset=Dataset.from_list(rows),
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    print("\n--- results ---")
    print(scores)
    if latencies:
        p95 = sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)]
        print(f"latency ms  p50={statistics.median(latencies):.0f}  p95={p95:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    parser.add_argument("--evaluator", choices=["local", "gemini"], default="local")
    parser.add_argument("--qa-path", default="evals/qa_set.jsonl")
    parser.add_argument("--limit", type=int, help="score only the first N questions")
    parser.add_argument("--gemini-model", default="gemini-2.0-flash")
    args = parser.parse_args()
    run(args.qa_path, args.evaluator, args.limit, args.gemini_model)


if __name__ == "__main__":
    main()
