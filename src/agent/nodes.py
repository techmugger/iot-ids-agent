"""
The three LangGraph nodes of the Agentic Core (Section 3.2.3, Fig. 2).

  planner_node   -> Llama-3-8B decomposes user_goal into atomic steps (JSON list)
  executor_node  -> Llama-3-8B selects a tool + extracts args for the current step,
                     invokes it, appends result
  evaluator_node -> Mistral Medium issues a strict YES/NO on goal completion
"""
import logging
from typing import Dict, Any

from src.agent.state import AgentState
from src.agent.tools import TOOL_REGISTRY
from src.agent.config import ALLOWED_TOOLS, MAX_TOOL_CALLS_PER_SESSION, MAX_ITERATIONS
from src.agent.llm_clients import call_groq_json, call_mistral_yesno

logger = logging.getLogger(__name__)

_session_tool_call_counts: Dict[str, int] = {}


# --------------------------------------------------------------------------
# Step 1 — Planner Node (Step Generation)
# --------------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """You are the Planner in an agentic AI system.
Decompose the user's goal into an ordered JSON list of atomic, tool-usable steps.
Each step must be a short, clear instruction that maps to exactly one tool from this registry:

{tool_descriptions}

Return ONLY valid JSON in this exact shape, no prose, no markdown fences:
{{"steps": ["step 1 text", "step 2 text", ...]}}
"""

# Used when the Evaluator has already said "NO" once -- instead of throwing
# away everything and starting a fresh plan (which was causing 200+ second,
# 8-iteration runs on simple 2-step tasks), we ask ONLY for what's still
# missing and APPEND it to the existing step list. This preserves progress
# across replanning cycles instead of redoing completed work.
CONTINUATION_SYSTEM_PROMPT = """You are the Planner in an agentic AI system, continuing work on a
task that is not yet complete. Given the original goal and the results of steps already executed,
determine ONLY the additional steps still needed to fully accomplish the goal. Do NOT repeat steps
whose results already appear below -- only plan what is genuinely still missing.

Available tools:
{tool_descriptions}

If the goal is actually already fully accomplished by the results shown, return an empty steps list.

Return ONLY valid JSON in this exact shape, no prose, no markdown fences:
{{"steps": ["remaining step 1 text", ...]}}
"""


def _tool_descriptions_block() -> str:
    lines = []
    for name, spec in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {spec['description']}")
    return "\n".join(lines)


def planner_node(state: AgentState) -> AgentState:
    is_continuation = len(state["results"]) > 0

    if is_continuation:
        # Replanning after a "NO" verdict -- ask for remaining work only,
        # and APPEND to the existing steps rather than replacing them.
        # current_step_index already equals len(state["steps"]) at this point
        # (the executor only reaches the evaluator once every prior step has
        # run), so appending lets execution continue seamlessly without
        # resetting progress.
        system_prompt = CONTINUATION_SYSTEM_PROMPT.format(tool_descriptions=_tool_descriptions_block())
        user_prompt = (
            f"Original goal: {state['user_goal']}\n\n"
            f"Steps already executed and their results: {state['results']}\n\n"
            f"What remaining steps (if any) are still needed?"
        )
        output = call_groq_json(system_prompt=system_prompt, user_prompt=user_prompt)
        new_steps = output.get("steps", [])
        if not isinstance(new_steps, list):
            new_steps = []

        state["steps"] = state["steps"] + new_steps
        # current_step_index is intentionally NOT reset here.
    else:
        system_prompt = PLANNER_SYSTEM_PROMPT.format(tool_descriptions=_tool_descriptions_block())
        # NOTE: user_goal is passed as the USER message only (never merged into
        # the system prompt) -- Section 3.4.1 prompt-injection defense.
        output = call_groq_json(system_prompt=system_prompt, user_prompt=state["user_goal"])

        steps = output.get("steps", [])
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Planner returned no valid steps: {output}")

        state["steps"] = steps
        state["current_step_index"] = 0

    logger.info("Planner now has %d total step(s) for session %s (continuation=%s)",
                len(state["steps"]), state["session_id"], is_continuation)
    return state


# --------------------------------------------------------------------------
# Step 2 — Executor Node (Tool Selection + Invocation)
# --------------------------------------------------------------------------
TOOL_SELECTION_PROMPT = """Given this step instruction, pick the single best tool from the registry below.
Step: "{step_text}"

Available tools:
{tool_descriptions}

Return ONLY valid JSON: {{"tool": "<exact_tool_name>"}}
"""

ARG_EXTRACTION_PROMPT = """Extract the arguments needed to call the tool "{tool_name}" from this step text.
Step: "{step_text}"
Expected argument schema: {args_schema}

Results from PREVIOUS steps in this plan (use these as the source of truth for
any values already extracted -- e.g. if an earlier step already extracted a
meeting title/date/time/attendees/location, REUSE those exact values here,
don't invent placeholders and don't leave them null):
{prior_results}

Return ONLY valid JSON matching the schema (best-effort fill using the prior
results above; use null only for genuinely unknown fields), no prose.
"""


def executor_node(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    call_count = _session_tool_call_counts.get(session_id, 0)
    if call_count >= MAX_TOOL_CALLS_PER_SESSION:
        raise RuntimeError(
            f"Session {session_id} exceeded MAX_TOOL_CALLS_PER_SESSION="
            f"{MAX_TOOL_CALLS_PER_SESSION} -- possible runaway/compromised plan."
        )

    idx = state["current_step_index"]
    step_text = state["steps"][idx]

    # --- Tool Selection ---
    sel_prompt = TOOL_SELECTION_PROMPT.format(
        step_text=step_text, tool_descriptions=_tool_descriptions_block()
    )
    selection = call_groq_json(system_prompt="You select tools precisely and return only JSON.",
                                user_prompt=sel_prompt)
    tool_name = selection.get("tool")

    # Section 3.4.2: allow-list check -- unlisted tool halts planning immediately
    if tool_name not in ALLOWED_TOOLS or tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Executor selected disallowed/unknown tool: {tool_name!r}")

    tool_spec = TOOL_REGISTRY[tool_name]

    if tool_name == "extract_calendar_event_data":
        # This tool's whole job is NER over free text -- there's nothing to
        # "extract" as an argument first, the ENTIRE original user goal (not
        # just the abbreviated step text) is the input, since attendee
        # emails/date/location usually live in the original message.
        args = {"text": state["user_goal"]}
    else:
        # --- Argument Extraction (for all other tools) ---
        # Pass prior step results as context so the model can reuse already-
        # extracted values (title/date/attendees/etc.) instead of inventing
        # placeholders or leaving fields null.
        prior_results = state["results"][-5:]  # last 5 is plenty of context, keeps prompt small
        arg_prompt = ARG_EXTRACTION_PROMPT.format(
            tool_name=tool_name, step_text=step_text, args_schema=tool_spec["args_schema"],
            prior_results=prior_results,
        )
        args = call_groq_json(system_prompt="You extract structured tool arguments and return only JSON.",
                               user_prompt=arg_prompt)

    # --- Invocation ---
    result_output = tool_spec["fn"](args)
    _session_tool_call_counts[session_id] = call_count + 1

    state["results"].append({
        "step": step_text,
        "tool": tool_name,
        "args": args,
        "result": result_output,
    })
    state["current_step_index"] = idx + 1
    logger.info("Executor ran tool=%s for step[%d] session=%s", tool_name, idx, session_id)
    return state


# --------------------------------------------------------------------------
# Step 3 — Evaluator Node (Goal State Assessment)
# --------------------------------------------------------------------------
EVALUATOR_SYSTEM_PROMPT = """You are the Evaluator in an agentic AI system. You independently verify
whether the user's original goal has been fully accomplished, based ONLY on the goal and the
tool results provided. You do not see the plan or reasoning that produced these results.
Reply with EXACTLY one word: YES or NO. Nothing else.
"""


def evaluator_node(state: AgentState) -> AgentState:
    # Section 3.4.4: evaluator gets ONLY goal + results, not the plan itself
    user_prompt = (
        f"User goal: {state['user_goal']}\n\n"
        f"Tool results so far: {state['results']}\n\n"
        f"Has the goal been fully accomplished?"
    )
    verdict = call_mistral_yesno(system_prompt=EVALUATOR_SYSTEM_PROMPT, user_prompt=user_prompt)

    state["iteration_count"] += 1
    state["goal_achieved"] = (verdict == "YES")
    logger.info("Evaluator verdict=%s iteration=%d session=%s",
                verdict, state["iteration_count"], state["session_id"])
    return state


def should_continue(state: AgentState) -> str:
    """Conditional edge from evaluator_node (Section 3.2.3)."""
    if state["goal_achieved"] or state["iteration_count"] >= MAX_ITERATIONS:
        return "end"
    return "planner"
