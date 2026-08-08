"""
Minimal LLM client wrappers. Kept separate from nodes.py so the prompting
logic can be unit-tested / mocked without hitting real APIs.

Requires: pip install groq mistralai
"""
import json
import re
from typing import Dict, Any

from src.agent.config import GROQ_API_KEY, PLANNER_MODEL, MISTRAL_API_KEY, EVALUATOR_MODEL


def _extract_json(text: str) -> Dict[str, Any]:
    """Planner/Executor outputs must be valid JSON (Section 3.2.3 / 3.4.1)."""
    text = text.strip()
    # strip markdown code fences if present
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def call_groq_json(system_prompt: str, user_prompt: str, retries: int = 1) -> Dict[str, Any]:
    """
    Calls Llama-3-8B via Groq API. System prompt and user input are kept in
    SEPARATE messages (Section 3.4.1 defense #1) — never concatenated.
    """
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    last_err = None
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        raw = resp.choices[0].message.content
        try:
            return _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            # Section 3.2.3: on schema failure, ask the Planner to correct itself
            user_prompt = (
                f"Your previous output was not valid JSON and raised: {e}. "
                f"Original output was:\n{raw}\n"
                f"Return ONLY corrected valid JSON, no prose, no markdown fences."
            )
    raise ValueError(f"Groq JSON output failed after {retries+1} attempts: {last_err}")


def call_mistral_yesno(system_prompt: str, user_prompt: str) -> str:
    """
    Calls Mistral Medium for the Evaluator node. Constrained to reply
    ONLY 'YES' or 'NO' (Section 3.2.3, evaluator step).

    NOTE: mistralai's SDK moved the Mistral client class from
    `mistralai.Mistral` (v1.x) to `mistralai.client.Mistral` (v2.x, released
    March 2026). We try both import paths so this works regardless of which
    version pip installed.
    """
    try:
        from mistralai import Mistral  # SDK v1.x
    except ImportError:
        from mistralai.client import Mistral  # SDK v2.x+

    client = Mistral(api_key=MISTRAL_API_KEY)
    resp = client.chat.complete(
        model=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content.strip().upper()
    return "YES" if raw.startswith("YES") else "NO"
