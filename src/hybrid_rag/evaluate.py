import json
from pathlib import Path


def retrieval_metrics(sources, expected):
    """Metriche a livello documento; due chunk della stessa fonte non valgono due hit."""
    relevant = set(expected)
    recall = len(set(sources) & relevant) / len(relevant)
    reciprocal_rank = next((1 / rank for rank, source in enumerate(sources, start=1) if source in relevant), 0)
    return recall, reciprocal_rank


def evaluate(index, dataset_path, top_k=3):
    if top_k < 1:
        raise ValueError("top_k deve essere positivo.")
    cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not cases or any(not case["sources"] for case in cases):
        raise ValueError("Servono domande con almeno una fonte attesa.")
    details = []
    totals = {mode: [0.0, 0.0] for mode in ["bm25", "semantic", "hybrid"]}
    for case in cases:
        rankings = index.rankings(case["question"], candidates=max(10, top_k))
        found = {}
        for mode, ranking in rankings.items():
            sources = [index.chunks[chunk_id]["source"] for chunk_id in ranking[:top_k]]
            recall, reciprocal_rank = retrieval_metrics(sources, case["sources"])
            totals[mode][0] += recall
            totals[mode][1] += reciprocal_rank
            found[mode] = sources
        details.append({"question": case["question"], "expected": case["sources"], "retrieved": found})
    return {
        "questions": len(cases),
        "top_k": top_k,
        "metrics": {mode: {"recall": round(values[0] / len(cases), 4), "mrr": round(values[1] / len(cases), 4)} for mode, values in totals.items()},
        "details": details,
    }

