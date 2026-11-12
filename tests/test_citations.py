import importlib.util
from pathlib import Path


path = Path(__file__).parents[1] / "scripts" / "evaluate_answers.py"
spec = importlib.util.spec_from_file_location("evaluate_answers", path)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_invalid_citation_numbers_are_not_mapped_to_sources():
    results = [{"source": "vpn.md"}, {"source": "account_mfa.md"}]
    check = evaluation.citation_check("VPN [1], account [2], fake [0] [9].", results)
    assert check["numbers"] == [0, 1, 2, 9]
    assert check["invalid_numbers"] == [0, 9]
    assert check["cited_sources"] == ["account_mfa.md", "vpn.md"]


def test_repeated_citations_and_refusal():
    check = evaluation.citation_check("Prima [1]. Poi [1].", [{"source": "vpn.md"}])
    assert check["numbers"] == [1]
    assert evaluation.citation_check("Non trovo questa informazione nei documenti.", []) == {
        "numbers": [], "invalid_numbers": [], "cited_sources": [],
    }
