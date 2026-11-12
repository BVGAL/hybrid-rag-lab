import re
import sqlite3
from pathlib import Path

import numpy as np

from .ingest import load_chunks

MODEL_NAME = "intfloat/multilingual-e5-small"


def create_embedder():
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    # E5-small viene registrato esplicitamente in questa versione di FastEmbed.
    if MODEL_NAME not in {model["model"] for model in TextEmbedding.list_supported_models()}:
        TextEmbedding.add_custom_model(
            model=MODEL_NAME, pooling=PoolingType.MEAN, normalization=True,
            sources=ModelSource(hf=MODEL_NAME), dim=384, model_file="onnx/model.onnx",
        )
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=".cache/models", threads=2)


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def build_index(folder, db_path, embedder):
    chunks = load_chunks(folder)
    # Il modello E5 richiede questi prefissi anche per testi in italiano.
    vectors = normalize(list(embedder.passage_embed(["passage: " + chunk["text"] for chunk in chunks])))
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        # Testi e vettori vengono sostituiti nella stessa transazione.
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS settings (model TEXT)")
            connection.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, source TEXT, part INTEGER, text TEXT, vector BLOB)")
            connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='unicode61')")
            connection.execute("DELETE FROM settings")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM chunks_fts")
            connection.execute("INSERT INTO settings VALUES (?)", (MODEL_NAME,))
            for chunk_id, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
                connection.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                                   (chunk_id, chunk["source"], chunk["part"], chunk["text"], vector.tobytes()))
                connection.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (chunk_id, chunk["text"]))
    finally:
        connection.close()
    return len(chunks)


def fuse_rankings(rankings, k=60):
    """RRF combina le posizioni: non somma BM25 e similarità coseno."""
    scores = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + position)
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))


class SearchIndex:
    def __init__(self, db_path, embedder=None):
        db_path = Path(db_path)
        if not db_path.is_file():
            raise ValueError("Indice assente. Esegui prima: rag index data/demo")
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.embedder = embedder
        model = self.connection.execute("SELECT model FROM settings").fetchone()[0]
        if model != MODEL_NAME:
            self.close()
            raise ValueError("Il modello è cambiato: ricostruisci l'indice.")
        self.chunks = {row["id"]: dict(row) for row in self.connection.execute("SELECT * FROM chunks ORDER BY id")}
        self.ids = list(self.chunks)
        self.vectors = np.stack([np.frombuffer(chunk["vector"], dtype=np.float32) for chunk in self.chunks.values()])

    def close(self):
        self.connection.close()

    def rankings(self, question, candidates=10, mode="hybrid"):
        if mode not in {"bm25", "semantic", "hybrid"}:
            raise ValueError("Modalità ammesse: bm25, semantic, hybrid.")
        terms = re.findall(r"\w+", question, flags=re.UNICODE)
        if not terms:
            raise ValueError("Scrivi una domanda contenente almeno una parola.")
        if candidates < 1:
            raise ValueError("Il numero di candidati deve essere positivo.")
        lexical, semantic = [], []
        if mode != "semantic":
            # Le parole sono quotate: la domanda non viene interpretata come sintassi FTS.
            query = " OR ".join(f'"{term}"' for term in terms)
            rows = self.connection.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts), rowid LIMIT ?",
                (query, candidates),
            )
            lexical = [row[0] for row in rows]
        if mode != "bm25":
            if self.embedder is None:
                self.embedder = create_embedder()
            vector = normalize(list(self.embedder.query_embed("query: " + question))[0])
            similarities = self.vectors @ vector
            semantic = [self.ids[i] for i in np.argsort(-similarities, kind="stable")[:candidates]]
        hybrid = fuse_rankings([lexical, semantic])
        return {"bm25": lexical, "semantic": semantic, "hybrid": hybrid}

    def search(self, question, top_k=3, mode="hybrid"):
        if top_k < 1:
            raise ValueError("top_k deve essere positivo.")
        rankings = self.rankings(question, candidates=max(10, top_k), mode=mode)
        results = []
        for chunk_id in rankings[mode][:top_k]:
            chunk = self.chunks[chunk_id]
            results.append({
                "id": chunk_id,
                "source": chunk["source"],
                "part": chunk["part"],
                "text": chunk["text"],
                "bm25_rank": rankings["bm25"].index(chunk_id) + 1 if chunk_id in rankings["bm25"] else None,
                "semantic_rank": rankings["semantic"].index(chunk_id) + 1 if chunk_id in rankings["semantic"] else None,
            })
        return results

