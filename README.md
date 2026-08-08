# IoT Intrusion Detection Centric Agentic AI

Reproduction/implementation of:
> Chirag, Rawat, Singal — *"Design and implementation of IoT intrusion detection
> centric agentic AI with auxiliary conversational capabilities"*,
> Pervasive and Mobile Computing 123 (2026) 102271.

Two tracks, deliberately built differently:

- **Track A — IDS (`notebooks/`, `src/ids/`)**: XGBoost intrusion classifier
  trained on UNSW-NB15 (primary) and NSL-KDD (secondary benchmark). Exploratory,
  so it lives in a Jupyter notebook.
- **Track B — Agentic chatbot (`src/agent/`)**: LangGraph-based multi-LLM
  orchestration (Planner/Executor = Llama-3-8B via Groq, Evaluator = Mistral
  Medium) with a triage layer that routes security checks straight to the IDS
  model. Needs env vars, persistent state, and a running server — not
  notebook-friendly, so it's a proper Python package run from the terminal.

## Project layout

```
iot-ids-agent/
├── data/                          # UNSW-NB15 + NSL-KDD raw CSVs (you provide/download)
├── models/                        # trained artifacts (.pkl) — produced by notebook/train.py
├── notebooks/
│   └── 01_ids_training_unsw_nsl.ipynb
├── src/
│   ├── ids/
│   │   ├── config.py               # hyperparameters, paths
│   │   ├── preprocess.py           # encode/scale pipelines for both datasets
│   │   └── train.py                # CLI trainer: python -m src.ids.train --dataset unsw
│   ├── agent/
│   │   ├── config.py                # env-var driven settings
│   │   ├── state.py                 # AgentState (Table 1 in the paper)
│   │   ├── tools.py                 # tool registry incl. iot_intrusion_checker
│   │   ├── llm_clients.py           # Groq / Mistral wrappers
│   │   ├── nodes.py                 # planner / executor / evaluator nodes
│   │   ├── graph.py                 # LangGraph StateGraph wiring
│   │   ├── triage.py                # orchestration & triage engine (fast paths)
│   │   ├── app.py                   # FastAPI server
│   │   └── integrations/
│   │       ├── google_calendar.py   # real Google Calendar client (optional)
│   │       └── gmail.py             # real Gmail client (optional)
│   └── eval/
│       └── run_suite.py             # 50-query completion-rate/latency/TSA harness
├── tests/
│   ├── test_ids_tool.py             # no API keys required
│   └── test_triage.py               # LLM calls mocked, no API keys required
├── requirements.txt
└── .env.example
```

## Step-by-step setup (terminal)

### 1. Environment

```bash
cd iot-ids-agent
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then fill in GROQ_API_KEY / MISTRAL_API_KEY
```

Get free API keys: Groq at https://console.groq.com, Mistral at
https://console.mistral.ai. Optional: Google Calendar/Gmail credentials
(see docstrings in `src/agent/integrations/`) if you want real calendar/email
actions instead of the built-in stubs (the agent runs fine without them —
`google_calendar_event`/`send_gmail_message` fall back to stub responses
automatically).

### 2. Get the data

Place these files under `data/`:
- `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` — from the UNSW
  research portal (https://research.unsw.edu.au/projects/unsw-nb15-dataset)
  or Kaggle.
- `KDDTrain+.txt`, `KDDTest+.txt` — from the official NSL-KDD distribution.

### 3. Track A — train the IDS (notebook)

```bash
jupyter notebook notebooks/01_ids_training_unsw_nsl.ipynb
```

Run all cells. This performs EDA, trains XGBoost on both the official and
pooled-resplit UNSW-NB15 partitions (see "Reproducibility note" cell — the
paper's stated training-set class counts don't match the official partition;
the pooled-resplit version is what actually reproduces their ~95% figure),
trains on NSL-KDD, and saves all artifacts to `models/`.

Equivalently, from the terminal without the notebook:
```bash
python -m src.ids.train --dataset unsw_resplit   # primary model, ~94.9% acc
python -m src.ids.train --dataset unsw           # official-split variant, ~89.8% acc
python -m src.ids.train --dataset nsl_kdd        # secondary benchmark, ~80.0% acc
```

### 4. Track B — run the agentic chatbot

```bash
uvicorn src.agent.app:app --reload --port 8000
```

In another terminal:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Is my device secure right now?"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Schedule a meeting titled Engineering Sync on 2025-09-15 at 14:00 for 1 hour with abc@gmail.com in the Conference Room"}'
```

### 5. Tests (no API keys needed)

```bash
pytest tests/ -v
```

### 6. Full evaluation suite (Section 4.7/4.8 reproduction — needs API keys)

```bash
python -m src.eval.run_suite
```

Prints task completion rate, median latency (direct vs. complex), and Tool
Selection Accuracy; writes the full per-query log to `eval_results.json`.

## Known reproducibility findings (useful for your report)

1. **UNSW-NB15 class counts**: Section 3.3.1 states the training set has
   "93,000 normal / 82,341 attack" — this doesn't match the *official*
   UNSW-NB15 train partition (56,000/119,341). It matches the *combined*
   dataset's normal count instead, and 82,341 is suspiciously close to the
   official test set size (82,332). We reproduce both interpretations; the
   pooled+resplit version lands at ~94.9% accuracy, very close to the
   paper's reported 95.04%, supporting this as what was actually done.
2. **NSL-KDD gap**: our official-split reproduction gets ~80% vs. the
   paper's 92.10%. This is a well-documented NSL-KDD property (its test set
   deliberately contains attack subtypes absent from training). Pooling+
   resplitting NSL-KDD (which "fixed" the UNSW-NB15 gap) instead produces an
   unrealistic ~99.7%, which is data leakage — NSL-KDD exists specifically to
   remove the near-duplicate records that caused this exact inflation in the
   original KDD-99. We report the honest official-split number.
3. **Query-count inconsistency**: the paper's abstract says the 50-query test
   suite includes "2 complex multi-step tasks," while Section 4.7 says "25."
   We follow Section 4.7's more detailed breakdown.

## Extending this project

- Add more entries to `TEST_SUITE` in `src/eval/run_suite.py` to reach the
  full 25 complex-task queries with your own ground-truth `expected_tools`.
- Swap `src/agent/integrations/google_calendar.py` and `gmail.py` from stub
  to live by following the OAuth setup in their docstrings.
- Extend `iot_intrusion_checker`'s replay mode to a real Scapy-based packet
  capture daemon instead of CSV replay (paper's Section 6, future work item 1).
- Multi-class attack-type classification (paper's future work item 4) —
  swap the binary XGBoost head for a multi-class one and re-run
  `src/ids/train.py` against `attack_cat` instead of `label`.
