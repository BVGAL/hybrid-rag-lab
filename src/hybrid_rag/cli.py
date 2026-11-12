import argparse
import json
import sqlite3
from pathlib import Path
from urllib.error import URLError

from .evaluate import evaluate
from .generate import generate_answer
from .search import SearchIndex, build_index, create_embedder


def main():
    parser = argparse.ArgumentParser(description="RAG locale con ricerca BM25, semantica e ibrida")
    parser.add_argument("--db", default="storage/index.db")
    commands = parser.add_subparsers(dest="command", required=True)
    indexing = commands.add_parser("index", help="Indicizza una cartella di file .md e .txt")
    indexing.add_argument("folder")
    for command in ["search", "ask"]:
        subparser = commands.add_parser(command)
        subparser.add_argument("question")
        subparser.add_argument("--top-k", type=int, default=3)
        subparser.add_argument("--mode", choices=["bm25", "semantic", "hybrid"], default="hybrid")
        if command == "ask":
            subparser.add_argument("--model", required=True, help="Nome di un modello già disponibile in Ollama")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--dataset", default="data/evaluation.json")
    evaluation.add_argument("--top-k", type=int, default=3)
    evaluation.add_argument("--output", help="Salva anche il report JSON")
    args = parser.parse_args()

    index = None
    try:
        if args.command == "index":
            count = build_index(args.folder, args.db, create_embedder())
            print(f"Indicizzati {count} chunk in {args.db}")
            return
        index = SearchIndex(args.db)
        if args.command == "evaluate":
            report = evaluate(index, args.dataset, args.top_k)
            text = json.dumps(report, ensure_ascii=False, indent=2)
            print(text)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n", encoding="utf-8")
            return
        results = index.search(args.question, args.top_k, args.mode)
        if args.command == "ask":
            print(generate_answer(args.question, results, args.model))
            print("\nFonti recuperate (da confrontare con le citazioni della risposta):")
        if not results:
            print("Nessun risultato.")
        for number, result in enumerate(results, start=1):
            print(f"\n[{number}] {result['source']} · parte {result['part']}")
            print(f"Posizione BM25: {result['bm25_rank']} | semantica: {result['semantic_rank']}")
            print(result["text"])
    except (ValueError, OSError, sqlite3.Error, URLError) as error:
        parser.exit(1, f"Errore: {error}\n")
    finally:
        if index is not None:
            index.close()
