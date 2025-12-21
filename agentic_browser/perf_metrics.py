"""
Performance metrics tracking for agent execution.

Lightweight monitoring for debugging step timing and identifying bottlenecks.
Part of Performance Optimization Phase 6.
"""

import time
import json
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager


@dataclass
class StepMetrics:
    """Metrics for a single step."""
    step_num: int
    duration_ms: float
    llm_time_ms: float = 0
    db_time_ms: float = 0
    browser_time_ms: float = 0
    message_count: int = 0


class PerfTracker:
    """Tracks performance metrics across steps.
    
    Singleton pattern - use PerfTracker.get() to access.
    """
    
    _instance: Optional["PerfTracker"] = None
    
    def __init__(self):
        self.steps: list[StepMetrics] = []
        self._current_step: Optional[StepMetrics] = None
    
    @classmethod
    def get(cls) -> "PerfTracker":
        """Get the global PerfTracker instance."""
        if cls._instance is None:
            cls._instance = PerfTracker()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the tracker (for new sessions)."""
        cls._instance = None
    
    @contextmanager
    def track_step(self, step_num: int):
        """Context manager to track a step's duration.
        
        Usage:
            with tracker.track_step(5) as metrics:
                metrics.message_count = len(messages)
                # ... do work ...
        """
        start = time.perf_counter()
        self._current_step = StepMetrics(step_num=step_num, duration_ms=0)
        try:
            yield self._current_step
        finally:
            self._current_step.duration_ms = (time.perf_counter() - start) * 1000
            self.steps.append(self._current_step)
            self._emit_metrics(self._current_step)
            self._current_step = None
    
    def _emit_metrics(self, metrics: StepMetrics) -> None:
        """Emit metrics as GUI event for real-time monitoring."""
        try:
            event = {
                "type": "perf_metrics",
                "step": metrics.step_num,
                "duration_ms": round(metrics.duration_ms, 1),
                "message_count": metrics.message_count,
            }
            print(f"__GUI_EVENT__{json.dumps(event)}")
        except Exception:
            pass
    
    def get_summary(self) -> dict:
        """Get summary statistics for all tracked steps."""
        if not self.steps:
            return {}
        
        durations = [s.duration_ms for s in self.steps]
        return {
            "total_steps": len(self.steps),
            "avg_step_ms": round(sum(durations) / len(durations), 1),
            "max_step_ms": round(max(durations), 1),
            "min_step_ms": round(min(durations), 1),
            "slowest_step": max(range(len(self.steps)), key=lambda i: durations[i]),
            "total_time_s": round(sum(durations) / 1000, 2),
        }
    
    def log_summary(self) -> None:
        """Print a summary of performance metrics."""
        summary = self.get_summary()
        if summary:
            print(f"\n[PERF] Summary: {summary['total_steps']} steps, "
                  f"avg {summary['avg_step_ms']}ms, max {summary['max_step_ms']}ms, "
                  f"total {summary['total_time_s']}s")
