"""
State management utilities for performance monitoring.

Provides metrics tracking and garbage collection detection for AgentState.
Part of Performance Optimization Phase 1.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import AgentState


@dataclass
class StateMetrics:
    """Metrics about current state size.
    
    Used to monitor state growth and trigger optimizations.
    """
    message_count: int
    total_message_chars: int
    extracted_data_keys: int
    extracted_data_bytes: int
    visited_urls_count: int
    files_accessed_count: int


def compute_state_metrics(state: "AgentState") -> StateMetrics:
    """Compute size metrics for a state object.
    
    Args:
        state: Current agent state
        
    Returns:
        StateMetrics with size information
    """
    messages = state.get("messages", [])
    extracted = state.get("extracted_data", {})
    
    total_chars = 0
    for msg in messages:
        if hasattr(msg, 'content'):
            content = msg.content
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Multi-part content (text + images)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total_chars += len(part.get("text", ""))
    
    return StateMetrics(
        message_count=len(messages),
        total_message_chars=total_chars,
        extracted_data_keys=len(extracted),
        extracted_data_bytes=sum(len(str(v)) for v in extracted.values()),
        visited_urls_count=len(state.get("visited_urls", [])),
        files_accessed_count=len(state.get("files_accessed", [])),
    )


def should_gc_state(state: "AgentState", threshold_mb: float = 5.0) -> bool:
    """Check if state needs garbage collection.
    
    Returns True if total state content exceeds threshold.
    This can be used to trigger aggressive truncation.
    
    Args:
        state: Current agent state
        threshold_mb: Size threshold in megabytes
        
    Returns:
        True if state is too large
    """
    metrics = compute_state_metrics(state)
    total_bytes = metrics.total_message_chars + metrics.extracted_data_bytes
    return (total_bytes / 1_000_000) > threshold_mb


def log_state_metrics(state: "AgentState", step: int) -> None:
    """Log state metrics for debugging.
    
    Emits a GUI event with current state size info.
    
    Args:
        state: Current agent state
        step: Current step number
    """
    import json
    
    metrics = compute_state_metrics(state)
    event = {
        "type": "state_metrics",
        "step": step,
        "message_count": metrics.message_count,
        "message_kb": round(metrics.total_message_chars / 1024, 1),
        "extracted_keys": metrics.extracted_data_keys,
        "extracted_kb": round(metrics.extracted_data_bytes / 1024, 1),
    }
    print(f"__GUI_EVENT__{json.dumps(event)}")
