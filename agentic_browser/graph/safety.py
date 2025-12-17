"""
Safety integration for LangGraph with human-in-the-loop interrupts.

Wraps the existing SafetyClassifier for graph-compatible safety checks.
"""

from typing import Literal
from langgraph.graph import END

from ..safety import SafetyClassifier, RiskLevel
from .state import AgentState


class GraphSafetyChecker:
    """Safety checker adapted for LangGraph execution.
    
    Integrates with existing SafetyClassifier and provides
    interrupt points for human approval.
    """
    
    def __init__(self):
        """Initialize the safety checker."""
        self.classifier = SafetyClassifier()
    
    def check_action(
        self, 
        action: str, 
        args: dict, 
        domain: str = "browser",
    ) -> tuple[RiskLevel, str, bool]:
        """Check an action for safety.
        
        Args:
            action: Action name (goto, click, os_exec, etc.)
            args: Action arguments
            domain: Current domain (browser, os)
            
        Returns:
            Tuple of (risk_level, reason, requires_approval)
        """
        result = self.classifier.classify(action, args)
        
        # HIGH risk always requires approval
        # MEDIUM risk requires approval unless auto-approved
        requires_approval = result.risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM)
        
        return result.risk_level, result.reason, requires_approval
    
    def requires_double_confirm(self, action: str, args: dict) -> bool:
        """Check if action requires double confirmation.
        
        Used for destructive OS commands like rm -rf.
        
        Args:
            action: Action name
            args: Action arguments
            
        Returns:
            True if double confirmation needed
        """
        return self.classifier.requires_double_confirm(action, args)


def should_interrupt(state: AgentState) -> bool:
    """Check if execution should be interrupted for approval.
    
    Args:
        state: Current graph state
        
    Returns:
        True if human approval is needed
    """
    pending = state.get("pending_approval")
    if not pending:
        return False
    
    # Check if this action was already approved
    action_key = f"{pending.get('action')}:{hash(str(pending.get('args')))}"
    if action_key in state.get("approved_actions", []):
        return False
    
    return True


def safety_gate(state: AgentState) -> Literal["continue", "interrupt", "reject"]:
    """Conditional edge function for safety gating.
    
    Returns:
        - "continue": Action is safe, proceed
        - "interrupt": Needs human approval
        - "reject": Action is blocked (e.g., denylist)
    """
    pending = state.get("pending_approval")
    if not pending:
        return "continue"
    
    checker = GraphSafetyChecker()
    
    action = pending.get("action", "")
    args = pending.get("args", {})
    domain = state.get("current_domain", "browser")
    
    risk_level, reason, requires_approval = checker.check_action(action, args, domain)
    
    # Check for blocked actions
    if checker.classifier.is_blocked(action, args):
        return "reject"
    
    if requires_approval:
        return "interrupt"
    
    return "continue"


def process_approval(state: AgentState, approved: bool) -> AgentState:
    """Process a human approval decision.
    
    Args:
        state: Current graph state
        approved: Whether the action was approved
        
    Returns:
        Updated state
    """
    pending = state.get("pending_approval")
    if not pending:
        return state
    
    if approved:
        # Add to approved actions
        action_key = f"{pending.get('action')}:{hash(str(pending.get('args')))}"
        approved_actions = state.get("approved_actions", []) + [action_key]
        
        return {
            **state,
            "pending_approval": None,
            "approved_actions": approved_actions,
        }
    else:
        # Rejected - clear pending and set error
        return {
            **state,
            "pending_approval": None,
            "error": f"User rejected action: {pending.get('action')}",
        }
