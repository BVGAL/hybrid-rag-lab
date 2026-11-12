import io
import json

import numpy as np
import pytest

from hybrid_rag.evaluate import evaluate, retrieval_metrics
from hybrid_rag.generate import build_prompt, generate_answer
from hybrid_rag.ingest import load_chunks, split_text
from hybrid_rag.search import SearchIndex, build_index, fuse_rankings


class DemoEmbedder:
    """Vettori controllati per testare la pipeline senza scaricare un modello."""

    def passage_embed(self, texts):
        assert all(text.startswith("passage: ") for text in texts)
        return [np.array([1, 0]) if "VPN" in text else np.array([0, 1]) for text in texts]

    def query_embed(self, text):
        assert text.startswith("query: ")
        return [np.array([1, 0])]


@pytest.fixture
def index(tmp_path):
    documents = tmp_path / "docs"
    documents.mkdir()
    (documents / "rete.md").write_text("VPN per la rete aziendale", encoding="utf-8")
    (documents / "ferie.txt").write_text("Ferie nel portale PeopleDesk", encoding="utf-8")
    db = tmp_path / "index.db"
    build_index(documents, db, DemoEmbedder())
    instance = SearchIndex(db, DemoEmbedder())
    yield instance
    instance.close()


def test_chunks_keep_overlap_and_tail():
    chunks = list(split_text("uno due tre quattro cinque sei sette", size=4, overlap=1))
    assert chunks == ["uno due tre quattro", "quattro cinque sei sette"]
    assert list(split_text("")) == []
    with pytest.raises(ValueError):
        list(split_text("test", size=4, overlap=4))


def test_relative_sources_and_supported_files(tmp_path):
    (tmp_path / "it").mkdir()
    (tmp_path / "it" / "rete.md").write_text("VPN", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("secret", encoding="utf-8")
    assert load_chunks(tmp_path) == [{"source": "it/rete.md", "part": 1, "text": "VPN"}]


def test_semantic_search_finds_paraphrase_without_keyword_overlap(index):
    assert index.search("collegamento remoto", mode="bm25") == []
    result = index.search("collegamento remoto", mode="hybrid")[0]
    assert result["source"] == "rete.md"
    assert result["bm25_rank"] is None
    assert result["semantic_rank"] == 1


def test_bm25_treats_fts_operators_as_literal_words(index):
    assert index.search('VPN - OR " NOT : *', mode="bm25")[0]["source"] == "rete.md"
    with pytest.raises(ValueError):
        index.search("!!!")
    with pytest.raises(ValueError):
        index.search("VPN", top_k=0)


def test_rrf_rewards_agreement_without_losing_other_candidates():
    assert fuse_rankings([[1, 2, 3], [2, 4, 1]]) == [2, 1, 4, 3]
    assert fuse_rankings([[], [3, 2]]) == [3, 2]


def test_reindex_replaces_deleted_documents_and_vectors(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    old = docs / "old.md"
    old.write_text("VPN obsoleta", encoding="utf-8")
    db = tmp_path / "index.db"
    build_index(docs, db, DemoEmbedder())
    old.unlink()
    (docs / "new.md").write_text("Ferie aggiornate", encoding="utf-8")
    build_index(docs, db, DemoEmbedder())
    index = SearchIndex(db, DemoEmbedder())
    try:
        assert index.search("VPN", mode="bm25") == []
        assert index.search("ferie")[0]["source"] == "new.md"
        assert len(index.ids) == 1
    finally:
        index.close()


def test_empty_reindex_preserves_previous_index(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    path = docs / "rete.md"
    path.write_text("VPN", encoding="utf-8")
    db = tmp_path / "index.db"
    build_index(docs, db, DemoEmbedder())
    path.unlink()
    with pytest.raises(ValueError):
        build_index(docs, db, DemoEmbedder())
    index = SearchIndex(db, DemoEmbedder())
    try:
        assert index.search("VPN", mode="bm25")[0]["source"] == "rete.md"
    finally:
        index.close()


def test_metrics_do_not_count_duplicate_sources_twice():
    assert retrieval_metrics(["a.md", "a.md", "other.md"], ["a.md", "b.md"]) == (0.5, 1)
    assert retrieval_metrics(["other.md", "a.md"], ["a.md"]) == (1, 0.5)
    assert retrieval_metrics([], ["a.md"]) == (0, 0)


def test_evaluation_compares_all_three_methods(index, tmp_path):
    dataset = tmp_path / "eval.json"
    dataset.write_text(json.dumps([{"question": "VPN", "sources": ["rete.md"]}]), encoding="utf-8")
    report = evaluate(index, dataset, top_k=1)
    assert set(report["metrics"]) == {"bm25", "semantic", "hybrid"}
    assert all(metrics == {"recall": 1.0, "mrr": 1.0} for metrics in report["metrics"].values())


def test_generation_sends_numbered_sources_to_local_ollama(monkeypatch):
    results = [{"source": "rete.md", "part": 2, "text": "Aprire Aurora Connect."}]
    assert "[1] rete.md (parte 2)" in build_prompt("Come entro?", results)

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://127.0.0.1:11434/api/generate"
        payload = json.loads(request.data)
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert "Aprire Aurora Connect." in payload["prompt"]
        assert "Non inventare" in payload["system"]
        return io.BytesIO(json.dumps({"response": "Apri Aurora Connect [1]."}).encode())

    monkeypatch.setattr("hybrid_rag.generate.urlopen", fake_urlopen)
    assert generate_answer("Come entro?", results, "test-model") == "Apri Aurora Connect [1]."
    assert generate_answer("Come entro?", [], "test-model").startswith("Non trovo")
