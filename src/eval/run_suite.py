"""
Quantitative evaluation harness, reproducing Sections 4.7 (task completion /
latency) and 4.8 (Tool Selection Accuracy) of the base paper.

Run:
    python -m src.eval.run_suite

Requires the FastAPI app running (uvicorn src.agent.app:app) OR set
USE_IN_PROCESS=True below to call handle_user_input() directly without a server.
"""
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.agent.triage import handle_user_input

RESULTS_PATH = Path(__file__).resolve().parents[2] / "eval_results.json"


@dataclass
class TestQuery:
    query: str
    category: str  # "direct_factual" | "security_check" | "complex_task"
    expected_tools: Optional[List[str]] = None  # ground-truth tool sequence, for TSA


# ---------------------------------------------------------------------------
# 50-query stratified test suite: 15 direct factual, 10 security checks,
# 25 complex multi-step tasks (Section 4.7's own text; note the paper's
# ABSTRACT says "2 complex" while Section 4.7 says "25" -- we follow the
# detailed breakdown in 4.7 as it's more specific, and flag this
# inconsistency explicitly in the report).
# ---------------------------------------------------------------------------
TEST_SUITE: List[TestQuery] = [
    # --- 15 direct factual queries ---
    TestQuery("Who is the CEO of OpenAI?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What is the capital of Japan?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What year was the Eiffel Tower built?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What is XGBoost?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("Define zero-day exploit.", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What is the latest version of Python?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("Who invented the transformer architecture?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What does IoT stand for?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What is a DDoS attack?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What is the NIST cybersecurity framework?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What's the weather like today?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What is LangGraph?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("Who created the Mirai botnet?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What is the UNSW-NB15 dataset?", "direct_factual", ["duckduckgo_search"]),
    TestQuery("What does CVE stand for?", "direct_factual", ["duckduckgo_search"]),

    # --- 10 direct IoT security checks ---
    TestQuery("Is my device secure right now?", "security_check", ["iot_intrusion_checker"]),
    TestQuery("I want to make sure nothing suspicious is happening on my network.", "security_check", ["iot_intrusion_checker"]),
    TestQuery("Run a threat assessment.", "security_check", ["iot_intrusion_checker"]),
    TestQuery("Can you verify the integrity of my IoT setup?", "security_check", ["iot_intrusion_checker"]),
    TestQuery("Am I being attacked?", "security_check", ["iot_intrusion_checker"]),
    TestQuery("Check if there are any anomalies in my network traffic.", "security_check", ["iot_intrusion_checker"]),
    TestQuery("Scan my devices for intrusions.", "security_check", ["iot_intrusion_checker"]),
    TestQuery("Is there any malicious activity on my network?", "security_check", ["iot_intrusion_checker"]),
    TestQuery("Please check my device for intrusion.", "security_check", ["iot_intrusion_checker"]),
    TestQuery("Give me a security status update.", "security_check", ["iot_intrusion_checker"]),

    # --- 25 complex multi-step tasks (subset shown; extend as needed) ---
    TestQuery(
        "Please schedule a meeting titled 'Engineering Sync' on September 15th, 2025, "
        "at the current time for 1 hour with abc123@gmail.com and xyz456@gmail.com "
        "in the Engineering Conference Room.",
        "complex_task",
        ["extract_calendar_event_data", "google_calendar_event"],
    ),
    TestQuery(
        "Find three recent articles about zero-day exploits in IoT firmware, summarize "
        "them into one paragraph, and email it to threatintel789@gmail.com with subject "
        "'Recent IoT Zero-Day Intelligence Briefing'.",
        "complex_task",
        ["duckduckgo_search", "send_gmail_message"],
    ),
    TestQuery(
        "Check my device for intrusion. If found, email the IT team at itoffice@gmail.com "
        "with subject 'Emergency Meeting'. Otherwise schedule a meeting titled "
        "'Movie Marathon Planning' on September 19th 2025 at 7 PM for 1 hour in the Conference Room.",
        "complex_task",
        ["iot_intrusion_checker", "google_calendar_event"],
    ),
    # NOTE: extend this list to 25 total complex-task entries covering your
    # own scenarios; the harness below works with any length.
]


def compute_tsa(logged_results: List[dict]) -> dict:
    """Tool Selection Accuracy, Section 4.8.1, Eq. (3)."""
    correct, total = 0, 0
    mismatches = []
    for entry in logged_results:
        if entry["category"] != "complex_task" or not entry.get("expected_tools"):
            continue
        actual_tools = [r["tool"] for r in entry.get("results", [])]
        for expected, actual in zip(entry["expected_tools"], actual_tools):
            total += 1
            if expected == actual:
                correct += 1
            else:
                mismatches.append({"query": entry["query"], "expected": expected, "actual": actual})
    tsa = (correct / total * 100) if total else None
    return {"tsa_percent": tsa, "correct": correct, "total": total, "mismatches": mismatches}


def run_suite():
    logged = []
    direct_latencies, complex_latencies = [], []
    completed, failed = 0, 0

    for tq in TEST_SUITE:
        t0 = time.time()
        try:
            resp = handle_user_input(tq.query)
            elapsed = time.time() - t0

            if tq.category == "complex_task":
                complex_latencies.append(elapsed)
                is_complete = resp.get("completed", False)
            else:
                direct_latencies.append(elapsed)
                is_complete = True  # direct paths either return or raise

            completed += int(is_complete)
            failed += int(not is_complete)

            logged.append({
                "query": tq.query,
                "category": tq.category,
                "expected_tools": tq.expected_tools,
                "completed": is_complete,
                "results": resp.get("results", []),
                "latency_s": round(elapsed, 3),
            })
        except Exception as e:
            failed += 1
            logged.append({
                "query": tq.query, "category": tq.category,
                "expected_tools": tq.expected_tools,
                "completed": False, "error": str(e),
            })

    n = len(TEST_SUITE)
    summary = {
        "n_queries": n,
        "task_completion_rate_percent": round(completed / n * 100, 2),
        "median_latency_direct_s": round(statistics.median(direct_latencies), 3) if direct_latencies else None,
        "median_latency_complex_s": round(statistics.median(complex_latencies), 3) if complex_latencies else None,
        "tool_selection_accuracy": compute_tsa(logged),
        "per_query": logged,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_query"}, indent=2))
    print(f"\nFull per-query log written to {RESULTS_PATH}")
    return summary


if __name__ == "__main__":
    run_suite()
