"""
Agent-side configuration. All secrets come from environment variables —
see .env.example at the project root. Never hardcode API keys here.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

# Load variables from a .env file at the project root into the process
# environment (python-dotenv). Without this, os.environ.get() below would
# never see GROQ_API_KEY / MISTRAL_API_KEY even if they're in .env.
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ---- LLM roles (Section 3.1 of the paper) ----
# Llama-3-8B via Groq: Planner + Executor (fast planning/tool selection, <0.2s)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "llama-3.1-8b-instant")

# Mistral Medium: Evaluator only (goal-state assessment, isolated context)
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
EVALUATOR_MODEL = os.environ.get("EVALUATOR_MODEL", "mistral-medium-latest")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is empty. Check that .env exists at the project root "
        "and contains a line like GROQ_API_KEY=gsk_... (no quotes, no spaces "
        "around the '='), then restart the server."
    )
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is empty. Check that .env exists at the project root "
        "and contains a line like MISTRAL_API_KEY=... (no quotes, no spaces "
        "around the '='), then restart the server."
    )

# ---- Loop guard (Section 5.1) ----
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 8))

# ---- Tool registry allow-list (Section 3.4.2) ----
ALLOWED_TOOLS = {
    "duckduckgo_search",
    "iot_intrusion_checker",
    "google_calendar_event",
    "send_gmail_message",
    "extract_calendar_event_data",
}

# ---- Per-session rate limiting (Section 3.4.2) ----
MAX_TOOL_CALLS_PER_SESSION = int(os.environ.get("MAX_TOOL_CALLS_PER_SESSION", 30))

# ---- IDS replay simulation defaults (Section 3.3.2) ----
REPLAY_INTERVAL_MS = int(os.environ.get("REPLAY_INTERVAL_MS", 50))
IDS_ARTIFACT_PREFIX = os.environ.get("IDS_ARTIFACT_PREFIX", "unsw")  # matches primary model
