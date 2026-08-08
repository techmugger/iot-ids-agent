"""
Run with: pytest tests/test_ids_tool.py -v
No API keys required -- only exercises the trained XGBoost artifacts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.tools import iot_intrusion_checker


def test_synthetic_mode_returns_valid_schema():
    result = iot_intrusion_checker({"mode": "synthetic"})
    assert "is_intrusion" in result
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["is_intrusion"], bool)


def test_replay_mode_returns_valid_schema_and_ground_truth():
    result = iot_intrusion_checker({"mode": "replay"})
    assert "is_intrusion" in result
    assert "score" in result
    assert "ground_truth_label" in result
    assert result["ground_truth_label"] in (0, 1)


def test_replay_mode_advances_index():
    r1 = iot_intrusion_checker({"mode": "replay"})
    r2 = iot_intrusion_checker({"mode": "replay"})
    assert r1["replay_source_index"] != r2["replay_source_index"]


if __name__ == "__main__":
    test_synthetic_mode_returns_valid_schema()
    test_replay_mode_returns_valid_schema_and_ground_truth()
    test_replay_mode_advances_index()
    print("All tests passed.")
