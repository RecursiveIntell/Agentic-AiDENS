"""
Tool for recalling information from past agent runs.

Allows the agent to search through session history to find successful strategies,
avoid repeating mistakes, and retrieve data from previous tasks.
"""

from typing import Any, Optional
from dataclasses import dataclass

from .memory import SessionStore
from ..tools import ToolResult


@dataclass
class SearchRunsRequest:
    """Request to search past runs."""
    query: str
    limit: int = 5
    success_only: bool = True


@dataclass
class GetRunDetailsRequest:
    """Request to get details of a specific run."""
    run_id: str


class RecallTool:
    """Tool for accessing past run history from SessionStore."""
    
    def __init__(self, session_store: Optional[SessionStore] = None):
        """Initialize the recall tool.
        
        Args:
            session_store: The session store instance. If None, creates a new one.
        """
        self.session_store = session_store or SessionStore()
        
    def execute(self, action: str, args: dict[str, Any]) -> ToolResult:
        """Execute a recall action.
        
        Args:
            action: Action name (search_runs, get_run_details)
            args: Action arguments
            
        Returns:
            ToolResult with the requested data
        """
        if action == "search_runs":
            return self.search_runs(
                args.get("query", ""),
                args.get("limit", 5),
                args.get("success_only", True),
            )
        elif action == "get_run_details":
            return self.get_run_details(args.get("run_id", ""))
        elif action == "search_strategies":
            return self.search_strategies(
                args.get("query", ""),
                args.get("limit", 3)
            )
        else:
            return ToolResult(
                success=False,
                message=f"Unknown recall action: {action}. Valid actions: search_runs, get_run_details, search_strategies"
            )

    def search_strategies(self, query: str, limit: int = 3) -> ToolResult:
        """Search for high-level strategies from the strategy bank.
        
        Args:
            query: Topic or goal to find strategies for
            limit: Max strategies to return
            
        Returns:
            ToolResult with list of strategies
        """
        try:
            strategies = self.session_store.search_strategies(query, limit)
            
            if not strategies:
                return ToolResult(
                    success=True,
                    data=[],
                    message=f"No crystallized strategies found for '{query}'. Try search_runs to find raw examples."
                )
                
            summary = [f"Found {len(strategies)} strategies:"]
            for s in strategies:
                stars = "⭐" * min(5, s.get('usage_count', 1))
                summary.append(f"\n--- STRATEGY: {s['name']} {stars} ---")
                summary.append(f"Used {s['usage_count']} times")
                summary.append(f"Description: {s['description']}")
                
            return ToolResult(
                success=True,
                data=strategies,
                message="\n".join(summary)
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error searching strategies: {str(e)}"
            )

    def search_runs(self, query: str, limit: int = 5, success_only: bool = True) -> ToolResult:
        """Search for past runs matching a query."""
        if not query:
            return ToolResult(success=False, message="Missing required argument: query")
        
        try:
            # Pass success_only to SQL level
            sessions = self.session_store.search_sessions(query, limit=limit, success_only=success_only)
            
            # Format results
            results = []
            for s in sessions:
                results.append({
                    "id": s["id"],
                    "goal": s["goal"],
                    "status": s.get("status", "unknown"),
                    "date": s["created_at"],
                    "final_answer": (s.get("final_answer") or "")[:200] + "..." 
                })
            
            if not results:
                return ToolResult(
                    success=True, 
                    message=f"No matching runs found for '{query}'.",
                    data=[]
                )
                
            formatted = "\n".join(
                f"- [{r['date']}] {r['goal']} (ID: {r['id']})\n  Status: {r['status']}\n  Answer: {r['final_answer']}"
                for r in results
            )
            
            return ToolResult(
                success=True,
                message=f"Found matching runs:\n\n{formatted}",
                data=results
            )
            
        except Exception as e:
            return ToolResult(success=False, message=f"Error searching runs: {e}")

    def get_run_details(self, run_id: str) -> ToolResult:
        """Get detailed steps for a specific run."""
        if not run_id:
            return ToolResult(success=False, message="Missing required argument: run_id")
            
        try:
            # Get session metadata
            session = self.session_store.get_session(run_id)
            if not session:
                return ToolResult(success=False, message=f"Run ID {run_id} not found.")
                
            # Get steps
            steps = self.session_store.get_session_steps(run_id)
            
            # Format output
            lines = [
                f"Run ID: {run_id}",
                f"Goal: {session['goal']}",
                f"Status: {session.get('completed') and 'Completed' or 'Incomplete'}",
                f"Final Answer: {session.get('final_answer')}",
                "\nSteps:"
            ]
            
            for s in steps:
                action_desc = s.get('action') or "Thinking"
                result_desc = (s.get('result') or "")[:100]
                if len(result_desc) >= 100: result_desc += "..."
                
                lines.append(f"{s['step_number']}. [{s['agent']}] {action_desc} -> {result_desc}")
                
            return ToolResult(
                success=True,
                message="\n".join(lines),
                data={"session": session, "steps": steps}
            )
            
        except Exception as e:
            return ToolResult(success=False, message=f"Error getting run details: {e}")
