"""Prova un vero modello Ollama e salva risposte e citazioni da esaminare."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.request import urlopen

from hybrid_rag.generate import GENERATION_OPTIONS, SYSTEM_PROMPT, build_prompt, generate_answer
from hybrid_rag.search import SearchIndex


def citation_check(answer, results):
    numbers = sorted({int(number) for number in re.findall(r"\[(\d+)\]", answer)})
    invalid = [number for number in numbers if not 1 <= number <= len(results)]
    sources = sorted({results[number - 1]["source"] for number in numbers if 1 <= number <= len(results)})
    return {"numbers": numbers, "invalid_numbers": invalid, "cited_sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--db", default="storage/index.db")
    parser.add_argument("--dataset", default="data/generation-evaluation.json")
    parser.add_argument("--output", default="reports/generation.json")
    args = parser.parse_args()
    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    with urlopen("http://127.0.0.1:11434/api/version", timeout=10) as response:
        ollama_version = json.load(response)["version"]
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
        models = json.load(response)["models"]
    model = next(model for model in models if model["name"] == args.model)
    report = {
        "model": args.model,
        "model_digest": model["digest"],
        "ollama_version": ollama_version,
        "options": GENERATION_OPTIONS,
        "system_prompt": SYSTEM_PROMPT,
        "think": False,
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "method": "hybrid", "top_k": 3,
        "note": "Citation numbers are checked automatically. Whether a cited passage supports a claim requires separate review.",
        "cases": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    index = SearchIndex(args.db)
    try:
        for case in cases:
            results = index.search(case["question"], top_k=3)
            start = time.perf_counter()
            answer = generate_answer(case["question"], results, args.model)
            elapsed = round(time.perf_counter() - start, 2)
            citations = citation_check(answer, results)
            report["cases"].append({
                **case,
                "prompt": build_prompt(case["question"], results),
                "answer": answer,
                "generation_seconds": elapsed,
                "citations": citations,
                "uses_refusal_phrase": "Non trovo questa informazione nei documenti" in answer,
                "context": results,
            })
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{case['id']}: {elapsed}s; citations={citations['numbers']}; invalid={citations['invalid_numbers']}", flush=True)
    finally:
        index.close()


if __name__ == "__main__":
    main()
