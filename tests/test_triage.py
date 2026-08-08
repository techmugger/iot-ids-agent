"""
Tests the Orchestration & Triage Engine's routing (Section 3.2) with the LLM
intent-classification call mocked out, so this runs without API keys.

Run with: pytest tests/test_triage.py -v
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import triage


def test_security_check_routes_to_ids_tool_directly():
    with patch.object(triage, "classify_intent", return_value="security_check"):
        result = triage.handle_user_input("Is my device secure right now?")
    assert result["path"] == "security_check"
    assert "is_intrusion" in result["result"]


def test_direct_factual_routes_to_search():
    with patch.object(triage, "classify_intent", return_value="direct_factual"):
        result = triage.handle_user_input("What is the capital of France?")
    assert result["path"] == "direct_factual"


def test_ambiguous_classification_falls_back_to_complex_task():
    """classify_intent's fail-safe: unknown label -> complex_task, never a
    silently-wrong fast path."""
    with patch.object(triage, "call_groq_json", return_value={"intent": "nonsense"}):
        intent = triage.classify_intent("some ambiguous input")
    assert intent == "complex_task"


if __name__ == "__main__":
    test_security_check_routes_to_ids_tool_directly()
    test_direct_factual_routes_to_search()
    test_ambiguous_classification_falls_back_to_complex_task()
    print("All triage tests passed.")
