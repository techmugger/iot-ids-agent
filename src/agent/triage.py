"""
Orchestration and Triage Engine (Section 3.2).

Routes each incoming prompt down one of three paths BEFORE the expensive
agentic loop is ever invoked:
  1. Direct factual query        -> duckduckgo_search directly
  2. Security-check intent       -> iot_intrusion_checker directly (<200ms target)
  3. Complex/multi-step task     -> full LangGraph agentic core

The intent classifier itself is a single fast Groq call (Llama-3-8B), NOT
keyword matching -- Section 3.2.2 stresses this handles semantically diverse
phrasings ("Is my device secure right now?", "Run a threat assessment", etc.)
without relying on specific trigger words.
"""
import time
import uuid
from typing import Dict, Any

from src.agent.tools import TOOL_REGISTRY
from src.agent.graph import run_complex_task
from src.agent.llm_clients import call_groq_json
INTENT_CLASSIFIER_PROMPT = """Classify the user's message into exactly one category:

- "direct_factual": a short, question-like request for general information
  (e.g. "Who is the CEO of OpenAI?")
- "security_check": the user wants to check device/network security, verify
  intrusions, or monitor threats -- REGARDLESS of exact phrasing. Examples:
  "Is my device secure right now?", "Am I being attacked?", "Run a threat assessment",
  "Can you verify the integrity of my IoT setup?"
- "complex_task": anything requiring multiple steps, planning, or combining
  several tools/actions (e.g. scheduling + emailing, conditional logic,
  multi-source research).

User message: "{message}"

Return ONLY valid JSON: {{"intent": "direct_factual" | "security_check" | "complex_task"}}
"""


def classify_intent(message: str) -> str:
    output = call_groq_json(
        system_prompt="You are a precise intent classifier. Return only JSON.",
        user_prompt=INTENT_CLASSIFIER_PROMPT.format(message=message),
    )
    intent = output.get("intent")
    if intent not in ("direct_factual", "security_check", "complex_task"):
        # fail-safe: ambiguous classification routes to the full agentic loop
        # rather than silently guessing a fast-path tool
        return "complex_task"
    return intent


FINAL_SUMMARY_PROMPT = """Given the user's original goal and the results of the steps taken to
accomplish it, write a short, friendly, human-readable summary (2-4 sentences) of what was done.
If completed is false, explain what was accomplished so far and that the task did not fully finish.

User goal: {user_goal}
Completed: {completed}
Results: {results}

Return ONLY valid JSON: {{"summary": "..."}}
"""


def generate_final_summary(user_goal: str, results: list, completed: bool) -> str:
    """Section 3.5: Final Response Generation -- synthesizes tool results into
    a human-readable summary instead of leaving the UI to show raw JSON."""
    try:
        output = call_groq_json(
            system_prompt="You write concise, friendly task summaries. Return only JSON.",
            user_prompt=FINAL_SUMMARY_PROMPT.format(
                user_goal=user_goal, completed=completed, results=results
            ),
        )
        return output.get("summary", "Task processed.")
    except Exception:
        return "Task processed (summary generation unavailable)."


def _risk_assessment(score: float, is_intrusion: bool) -> Dict[str, str]:
    """
    Translates a raw model probability into a risk tier + plain-language
    explanation, so the UI can show something more useful than a bare number.
    Tier bounds are informational only -- the actual safe/attack decision
    still comes from the model's own 0.5 threshold (is_intrusion).
    """
    if score < 0.2:
        tier = "Very Low Risk"
    elif score < 0.5:
        tier = "Low Risk"
    elif score < 0.8:
        tier = "Elevated Risk"
    else:
        tier = "High Risk"

    if is_intrusion:
        explanation = (
            f"The IDS model classified this network traffic sample as malicious, "
            f"with a {score*100:.1f}% probability of intrusion. This pattern is "
            f"consistent with known attack signatures (e.g. abnormal packet volume, "
            f"unusual connection state, or reconnaissance/exploit-like behavior) learned "
            f"from the UNSW-NB15 training data. Recommended action: isolate the "
            f"affected device and review recent connection logs."
        )
    else:
        explanation = (
            f"The IDS model classified this network traffic sample as normal, with "
            f"only a {score*100:.1f}% probability of intrusion. No patterns matching "
            f"known attack signatures (DoS, exploits, reconnaissance, fuzzing, etc.) "
            f"were detected in this sample."
        )

    return {"risk_tier": tier, "explanation": explanation}


def handle_user_input(message: str, session_id: str = None) -> Dict[str, Any]:
    session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
    t0 = time.time()

    intent = classify_intent(message)

    if intent == "direct_factual":
        result = TOOL_REGISTRY["duckduckgo_search"]["fn"]({"query": message})
        return {
            "session_id": session_id,
            "path": "direct_factual",
            "result": result,
            "latency_s": round(time.time() - t0, 3),
        }

    if intent == "security_check":
        result = TOOL_REGISTRY["iot_intrusion_checker"]["fn"]({"mode": "replay"})
        risk = _risk_assessment(result["score"], result["is_intrusion"])
        summary = (
            f"Alert! Intrusion Found -- {risk['risk_tier']} (score: {result['score']})"
            if result["is_intrusion"]
            else f"Environment is safe -- {risk['risk_tier']} (score: {result['score']})"
        )
        return {
            "session_id": session_id,
            "path": "security_check",
            "result": result,
            "risk_tier": risk["risk_tier"],
            "explanation": risk["explanation"],
            "concise_summary": summary,
            "latency_s": round(time.time() - t0, 3),
        }

    # complex_task -> full agentic core
    final_state = run_complex_task(user_goal=message, session_id=session_id)
    concise_summary = generate_final_summary(
        user_goal=message, results=final_state["results"], completed=final_state["goal_achieved"]
    )
    return {
        "session_id": session_id,
        "path": "complex_task",
        "completed": final_state["goal_achieved"],
        "steps": final_state["steps"],
        "results": final_state["results"],
        "iterations": final_state["iteration_count"],
        "concise_summary": concise_summary,
        "latency_s": round(time.time() - t0, 3),
    }
