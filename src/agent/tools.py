"""
Tool registry — the Tools & API Layer (Fig. 2).

Each tool is a plain Python callable: (args: dict) -> dict.
The Executor node (src/agent/nodes.py) is responsible for picking one of
these by name and extracting `args` via LLM prompting; this module only
implements what happens once a tool + args have been decided.
"""
import random
import time
import uuid
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.agent.config import MODEL_DIR, IDS_ARTIFACT_PREFIX, REPLAY_INTERVAL_MS
from src.ids.preprocess import load_artifacts
from src.ids.config import UNSW_TEST_CSV

# --------------------------------------------------------------------------
# IoT Intrusion Checker (Section 3.3.2) — the paper's key novel component
# --------------------------------------------------------------------------
_MODEL, _SCALER, _ENCODERS, _FEATURE_COLUMNS = (None, None, None, None)
_REPLAY_DF = None
_REPLAY_INDEX = 0


def _lazy_load_ids_artifacts():
    global _MODEL, _SCALER, _ENCODERS, _FEATURE_COLUMNS
    if _MODEL is None:
        _MODEL, _SCALER, _ENCODERS, _FEATURE_COLUMNS = load_artifacts(IDS_ARTIFACT_PREFIX)
    return _MODEL, _SCALER, _ENCODERS, _FEATURE_COLUMNS


def _preprocess_sample(sample: pd.Series) -> np.ndarray:
    """
    Apply the identical encode+scale pipeline used at training time
    (src/ids/preprocess.py): label-encode categoricals, then scale ONLY the
    numeric columns (the scaler was fit on numeric_cols alone, not the full
    row) -- this must mirror preprocess_unsw()/preprocess_nsl_kdd() exactly.
    """
    model, scaler, encoders, feature_columns = _lazy_load_ids_artifacts()
    row = sample.copy()

    categorical_cols = list(encoders.keys())
    numeric_cols = [c for c in feature_columns if c not in categorical_cols]

    for col, le in encoders.items():
        if col in row.index:
            val = str(row[col])
            known = set(le.classes_)
            if val not in known:
                val = "__unknown__" if "__unknown__" in known else le.classes_[0]
            row[col] = le.transform([val])[0]

    row = row.reindex(feature_columns).fillna(0)

    # Scale only the numeric subset, in the exact column order the scaler expects
    numeric_df = pd.DataFrame([row[numeric_cols].values], columns=numeric_cols)
    row[numeric_cols] = scaler.transform(numeric_df)[0]

    ordered = row.reindex(feature_columns).values.reshape(1, -1)
    return ordered


def iot_intrusion_checker(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Two modes, matching Section 3.3.2:
      mode="synthetic" -> Mode 1, random vector within feature bounds (sanity check only)
      mode="replay"    -> Mode 2, streams next record from the UNSW-NB15 held-out test set
    """
    global _REPLAY_DF, _REPLAY_INDEX
    model, scaler, encoders, feature_columns = _lazy_load_ids_artifacts()
    mode = args.get("mode", "replay")

    if mode == "synthetic":
        # Mode 1: functional correctness probe only, not used for reported metrics
        raw = {}
        for col in feature_columns:
            if col in encoders:
                raw[col] = random.choice(list(encoders[col].classes_))
            else:
                raw[col] = np.random.normal(0, 1)
        sample = pd.Series(raw)
        source_index = None

    else:
        # Mode 2: replay UNSW-NB15 held-out test partition via simulated packet arrival
        if _REPLAY_DF is None:
            test_df = pd.read_csv(UNSW_TEST_CSV)
            _REPLAY_DF = test_df.drop(columns=[c for c in ["id", "attack_cat"] if c in test_df.columns])
        time.sleep(REPLAY_INTERVAL_MS / 1000.0)
        source_index = _REPLAY_INDEX % len(_REPLAY_DF)
        row = _REPLAY_DF.iloc[source_index]
        true_label = int(row["label"])
        sample = row.drop(labels=["label"])
        _REPLAY_INDEX += 1

    X = _preprocess_sample(sample)
    proba = model.predict_proba(X)[0]
    is_intrusion = bool(np.argmax(proba) == 1)
    score = float(proba[1])

    result = {
        "is_intrusion": is_intrusion,
        "prediction": "Alert! Intrusion Detected!" if is_intrusion else "Environment is safe.",
        "score": round(score, 5),
        "mode": mode,
    }
    if mode == "replay":
        result["replay_source_index"] = source_index
        result["ground_truth_label"] = true_label  # only available because this is a simulation
    return result


# --------------------------------------------------------------------------
# duckduckgo_search
# --------------------------------------------------------------------------
def duckduckgo_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Thin wrapper around the `duckduckgo_search` (ddgs) package.
    pip install ddgs
    """
    query = args.get("query", "")
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
        return {"query": query, "results": hits}
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}


# --------------------------------------------------------------------------
# extract_calendar_event_data — LLM-based NER over the raw user prompt
# --------------------------------------------------------------------------
def extract_calendar_event_data(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    In production this calls the Planner LLM again with a NER-style prompt.
    Kept here as a separate tool (as in the paper's execution trace, Section 3.2.3)
    so it can be unit-tested independently of the live LLM call.
    """
    from src.agent.llm_clients import call_groq_json

    prompt_text = args.get("text") or ""  # coerce None/missing -> "" (Groq rejects null content)
    if not prompt_text.strip():
        return {"error": "No text provided to extract_calendar_event_data."}

    system_prompt = (
        "Extract meeting details from the user's request. "
        "Return ONLY a JSON object with keys: title, date (YYYY-MM-DD), "
        "start_time (HH:MM:SS, 24h), duration_minutes (int), attendees (list of emails), "
        "location (string). If the user says 'current time', use the current system time."
    )
    return call_groq_json(system_prompt=system_prompt, user_prompt=prompt_text)


# --------------------------------------------------------------------------
# google_calendar_event (stub — swap in real Google Calendar API client)
# --------------------------------------------------------------------------
def google_calendar_event(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Requires a Google Calendar API service account/OAuth client in production
    (google-api-python-client). Falls back to a stub response whenever the
    package isn't installed OR the required credentials.json isn't present --
    both are equally "not configured for real use" from this function's
    point of view.
    """
    import os
    if not os.path.exists("credentials.json"):
        return {
            "status": 200,
            "event_id": f"stub-{uuid.uuid4().hex[:12]}",
            "note": "STUBBED: no credentials.json found at project root. Follow "
                    "the setup docstring in src/agent/integrations/google_calendar.py "
                    "to make this a live call.",
            "echo_args": args,
        }
    try:
        from src.agent.integrations.google_calendar import create_event
        return create_event(args)
    except Exception as e:
        return {
            "status": 200,
            "event_id": f"stub-{uuid.uuid4().hex[:12]}",
            "note": f"STUBBED (real call failed: {e}).",
            "echo_args": args,
        }


# --------------------------------------------------------------------------
# send_gmail_message (stub — swap in real Gmail API client)
# --------------------------------------------------------------------------
def send_gmail_message(args: Dict[str, Any]) -> Dict[str, Any]:
    import os
    if not os.path.exists("credentials.json"):
        return {
            "status": "sent",
            "note": "STUBBED: no credentials.json found at project root. Follow "
                    "the setup docstring in src/agent/integrations/gmail.py "
                    "to make this a live call.",
            "echo_args": args,
        }
    try:
        from src.agent.integrations.gmail import send_message
        return send_message(args)
    except Exception as e:
        return {
            "status": "sent",
            "note": f"STUBBED (real call failed: {e}).",
            "echo_args": args,
        }


# --------------------------------------------------------------------------
TOOL_REGISTRY = {
    "duckduckgo_search": {
        "fn": duckduckgo_search,
        "description": "Search the web for direct factual queries or information retrieval.",
        "args_schema": {"query": "string"},
    },
    "iot_intrusion_checker": {
        "fn": iot_intrusion_checker,
        "description": "Check IoT device/network security status using the trained XGBoost IDS.",
        "args_schema": {"mode": "'synthetic' | 'replay'"},
    },
    "google_calendar_event": {
        "fn": google_calendar_event,
        "description": "Create a Google Calendar event given title/date/time/duration/attendees/location.",
        "args_schema": {"title": "string", "date": "YYYY-MM-DD", "start_time": "HH:MM:SS",
                         "duration_minutes": "int", "attendees": "list[str]", "location": "string"},
    },
    "send_gmail_message": {
        "fn": send_gmail_message,
        "description": "Send an email via Gmail given recipients, subject, and body.",
        "args_schema": {"to": "list[str]", "subject": "string", "body": "string"},
    },
    "extract_calendar_event_data": {
        "fn": extract_calendar_event_data,
        "description": "Extract structured meeting details (NER) from a natural-language request.",
        "args_schema": {"text": "string"},
    },
}
