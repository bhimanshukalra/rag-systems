"""Retrieval-quality eval harness.

Usage:
    uv run python tests/eval/run_eval.py <label> [--compare-to <baseline_label>]

This makes live calls to Pinecone and the embedding model (no LLM/Groq
calls -- it evaluates retrieval only, not generation), so it needs real
credentials in .env. Not part of the pytest suite for that reason.
"""

import argparse
import json
import time
from pathlib import Path

from agentic_rag.config import get_settings
from agentic_rag.indexing.embeddings import get_embeddings
from agentic_rag.indexing.vector_store import get_vector_store

EVAL_DIR = Path(__file__).parent
GOLDEN_QA_PATH = EVAL_DIR / "golden_qa.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

RECALL_KS = (1, 4, 10)

# Fetch far more raw chunks than any RECALL_KS value before deduping to
# distinct source documents. A single well-matched, chunk-heavy document
# (e.g. an 80-chunk PDF) can otherwise fill the entire top-k on its own,
# leaving no room to observe recall for anything beyond rank 1.
RAW_CHUNK_POOL = 40


def load_golden_qa() -> list[dict]:
    with GOLDEN_QA_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def ranked_distinct_sources(documents) -> list[str]:
    seen: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.append(source)
    return seen


def score_question(retrieved_sources: list[str], expected_sources: list[str]):
    expected = set(expected_sources)
    hit_rank = next(
        (i + 1 for i, source in enumerate(retrieved_sources) if source in expected),
        None,
    )
    recalls = {k: int(hit_rank is not None and hit_rank <= k) for k in RECALL_KS}
    reciprocal_rank = 1 / hit_rank if hit_rank else 0.0
    return recalls, reciprocal_rank, hit_rank


def run(label: str) -> dict:
    settings = get_settings()
    embeddings = get_embeddings(settings.embedding_model)
    vector_store = get_vector_store(
        embeddings,
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
    )

    questions = load_golden_qa()
    per_question = []
    totals = dict.fromkeys(RECALL_KS, 0)
    mrr_sum = 0.0

    for item in questions:
        documents = vector_store.similarity_search(item["question"], k=RAW_CHUNK_POOL)
        retrieved_sources = ranked_distinct_sources(documents)
        recalls, reciprocal_rank, hit_rank = score_question(
            retrieved_sources, item["expected_sources"]
        )

        for k, hit in recalls.items():
            totals[k] += hit
        mrr_sum += reciprocal_rank

        per_question.append(
            {
                "question": item["question"],
                "category": item.get("category"),
                "expected_sources": item["expected_sources"],
                "retrieved_sources": retrieved_sources,
                "hit_rank": hit_rank,
                "reciprocal_rank": reciprocal_rank,
                "recalls": recalls,
            }
        )

    n = len(questions)
    summary = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_questions": n,
        "raw_chunk_pool": RAW_CHUNK_POOL,
        "recall_at_k": {str(k): totals[k] / n for k in RECALL_KS},
        "mrr": mrr_sum / n,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"{label}.json"
    output_path.write_text(
        json.dumps({"summary": summary, "per_question": per_question}, indent=2)
    )

    print_report(summary, output_path)
    return summary


def print_report(summary: dict, output_path: Path) -> None:
    print(
        f"Eval run: {summary['label']} "
        f"({summary['num_questions']} questions, raw pool {summary['raw_chunk_pool']})"
    )
    for k in RECALL_KS:
        print(f"  recall@{k}: {summary['recall_at_k'][str(k)]:.2f}")
    print(f"  MRR: {summary['mrr']:.3f}")
    print(f"Saved to {output_path}")


def compare(label: str, baseline_label: str) -> None:
    current = json.loads((RESULTS_DIR / f"{label}.json").read_text())["summary"]
    baseline = json.loads((RESULTS_DIR / f"{baseline_label}.json").read_text())["summary"]

    print(f"\nComparing {label} vs baseline {baseline_label}:")
    for k in RECALL_KS:
        cur = current["recall_at_k"][str(k)]
        base = baseline["recall_at_k"][str(k)]
        print(f"  recall@{k}: {base:.2f} -> {cur:.2f} ({cur - base:+.2f})")
    print(f"  MRR: {baseline['mrr']:.3f} -> {current['mrr']:.3f} ({current['mrr'] - baseline['mrr']:+.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="Name for this eval run, e.g. phase0_dense_only")
    parser.add_argument("--compare-to", help="Label of a previous run to compare against")
    args = parser.parse_args()

    run(args.label)

    if args.compare_to:
        compare(args.label, args.compare_to)


if __name__ == "__main__":
    main()
