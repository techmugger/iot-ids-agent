"""
AgentState — mirrors Table 1 of the base paper exactly.
Persisted at every LangGraph node boundary via MemorySaver checkpointing.
"""
from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    user_goal: str                # Original user input
    steps: List[str]              # Ordered list of atomic steps from Planner
    current_step_index: int       # Pointer to step being executed
    results: List[Dict[str, Any]] # Tool outputs for each completed step
    iteration_count: int          # Loop guard counter
    goal_achieved: bool           # Terminal state flag
    session_id: str               # Session identifier for persistence layer


def new_agent_state(user_goal: str, session_id: str) -> AgentState:
    return AgentState(
        user_goal=user_goal,
        steps=[],
        current_step_index=0,
        results=[],
        iteration_count=0,
        goal_achieved=False,
        session_id=session_id,
    )
