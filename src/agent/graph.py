"""
LangGraph graph definition — the Agentic Core (Section 3.2.3).

Nodes: planner_node -> executor_node -> evaluator_node
Edges:
  planner_node -> executor_node          (unconditional; planning always precedes execution)
  evaluator_node -> END | planner_node   (conditional on should_continue)

Entry point: planner_node
Checkpointing: LangGraph's MemorySaver, persists AgentState at each node boundary.

Requires: pip install langgraph
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes import planner_node, executor_node, evaluator_node, should_continue


def _has_more_steps(state: AgentState) -> str:
    """
    Conditional edge from executor_node: keep executing steps in the CURRENT
    plan until current_step_index reaches the end of state['steps'], THEN
    hand off to the evaluator. Without this, only the first step of any
    multi-step plan ever runs before evaluation (Section 3.2.3's step-by-step
    mechanics implies sequential execution of the full step list per cycle).
    """
    if state["current_step_index"] < len(state["steps"]):
        return "executor"
    return "evaluator"


def build_agentic_core():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        _has_more_steps,
        {"executor": "executor", "evaluator": "evaluator"},
    )
    graph.add_conditional_edges(
        "evaluator",
        should_continue,
        {"planner": "planner", "end": END},
    )

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# Module-level singleton so the compiled graph (with its checkpointer) is
# reused across requests instead of rebuilt every call.
AGENTIC_CORE = build_agentic_core()


def run_complex_task(user_goal: str, session_id: str) -> AgentState:
    from src.agent.state import new_agent_state

    initial_state = new_agent_state(user_goal=user_goal, session_id=session_id)
    config = {"configurable": {"thread_id": session_id}}
    final_state = AGENTIC_CORE.invoke(initial_state, config=config)
    return final_state
