"""
Policy Engine for Agentic Browser.

Provides centralized action authorization with:
- Command normalization and denylist patterns
- Risk classification extending SafetyClassifier
- Dry-run mode for previewing actions without execution
- Structured approval workflow
"""

import re
import shlex
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from .safety import RiskLevel, SafetyClassifier
from .tool_schemas import (
    ActionRequest,
    ActionDomain,
    RunCommandRequest,
    WriteFileRequest,
    DeleteFileRequest,
    validate_action_args,
)


class ApprovalRequirement(str, Enum):
    """Type of approval required for an action."""
    NONE = "none"
    SINGLE = "single"  # Standard approval
    DOUBLE = "double"  # Double confirmation for dangerous ops


class ActionDecision(BaseModel):
    """Result of policy evaluation for an action."""
    
    allowed: bool = Field(description="Whether action is allowed to proceed")
    risk_level: RiskLevel = Field(description="Classified risk level")
    rationale: str = Field(description="Human-readable explanation")
    requires_approval: ApprovalRequirement = Field(
        default=ApprovalRequirement.NONE,
        description="Type of approval required"
    )
    blocked_reason: Optional[str] = Field(
        default=None,
        description="If blocked, why it was blocked"
    )
    
    # Dry-run output
    would_execute: Optional[str] = Field(
        default=None,
        description="What would be executed (for dry-run)"
    )
    dry_run_output: Optional[str] = Field(
        default=None,
        description="Simulated output (for dry-run)"
    )
    
    def to_explanation(self) -> str:
        """Generate human-readable explanation for CLI/GUI."""
        lines = [
            f"Risk Level: {self.risk_level.value.upper()}",
            f"Rationale: {self.rationale}",
        ]
        
        if self.requires_approval != ApprovalRequirement.NONE:
            lines.append(f"Approval Required: {self.requires_approval.value}")
        
        if self.blocked_reason:
            lines.append(f"BLOCKED: {self.blocked_reason}")
        
        if self.would_execute:
            lines.append(f"Would Execute: {self.would_execute}")
        
        return "\n".join(lines)


class PolicyEngine:
    """Centralized policy engine for action authorization.
    
    Extends SafetyClassifier with:
    - Hard denylist for dangerous patterns
    - Command normalization (shell strings → argv)
    - Dry-run simulation
    - Structured approval workflow
    """
    
    # DENYLIST: These patterns are NEVER allowed, even with approval
    HARD_DENYLIST = [
        # System destruction
        r"sudo\s+rm\s+(-[rf]+\s+)*\s*/\s*$",  # sudo rm -rf /
        r"rm\s+(-[rf]+\s+)*\s*/\s*$",          # rm -rf /
        r"dd\s+.*\s+of=/dev/(sd[a-z]|nvme|vd)", # dd to disk device
        r"mkfs\.\w+\s+/dev/(sd[a-z]|nvme|vd)",  # mkfs to disk
        
        # Critical system files
        r">\s*/etc/(passwd|shadow|sudoers)",    # Overwrite auth files
        r"rm\s+(-[rf]+\s+)*/etc/(passwd|shadow|sudoers)",
        
        # Boot destruction
        r"rm\s+(-[rf]+\s+)*/boot",
        r"dd\s+.*\s+of=/boot",
        
        # Fork bombs and resource exhaustion
        r":\(\)\s*\{\s*:\|:",                   # Classic fork bomb
        r"while\s+true;\s*do\s+fork",
    ]
    
    # Additional high-risk patterns (require double confirm)
    DOUBLE_CONFIRM_PATTERNS = [
        r"\brm\s+(-[rf]+\s+)+",     # rm with -r or -f flags
        r"\bsudo\s+",               # Any sudo command
        r"\bdd\b",                  # Any dd command
        r"\bmkfs\b",                # Any mkfs command
        r"\bshutdown\b",            # Shutdown
        r"\breboot\b",              # Reboot
        r"\bpoweroff\b",            # Power off
        r"\bsystemctl\s+(stop|disable|mask)\b",  # Dangerous systemctl
        r"\bchmod\s+-R\b",          # Recursive chmod
        r"\bchown\s+-R\b",          # Recursive chown
        r"\bkill\s+-9\b",           # Force kill
        r"\bkillall\b",             # Kill all
    ]
    
    # Medium risk patterns (require single approval unless auto-approved)
    MEDIUM_RISK_PATTERNS = [
        r"\bmv\s+",                 # Move files
        r"\bcp\s+",                 # Copy files
        r"\brm\s+",                 # Remove (without -rf)
        r">\s*[^|]",                # Redirect to file
        r"\btee\s+",                # Tee to file
        r"\bsed\s+-i\b",            # In-place sed
        r"\bchmod\s+",              # chmod (non-recursive)
        r"\bchown\s+",              # chown (non-recursive)
        r"\bmkdir\s+",              # Create directory
        r"\btouch\s+",              # Create file
        r"\bln\s+",                 # Create link
    ]
    
    # Protected paths that require extra scrutiny
    PROTECTED_PATHS = [
        "/etc", "/usr", "/bin", "/sbin", "/boot", 
        "/var", "/lib", "/lib64", "/opt", "/root",
        "/sys", "/proc", "/dev",
    ]
    
    def __init__(self, auto_approve_low: bool = True, auto_approve_medium: bool = False):
        """Initialize the policy engine.
        
        Args:
            auto_approve_low: Auto-approve LOW risk actions
            auto_approve_medium: Auto-approve MEDIUM risk actions
        """
        self.classifier = SafetyClassifier()
        self.auto_approve_low = auto_approve_low
        self.auto_approve_medium = auto_approve_medium
    
    def evaluate(
        self,
        action: str,
        args: dict[str, Any],
        domain: str = "os",
        dry_run: bool = False,
        current_url: str = "",
    ) -> ActionDecision:
        """Evaluate an action request against policy.
        
        Args:
            action: Action name (os_exec, goto, click, etc.)
            args: Action arguments
            domain: Execution domain (os, browser)
            dry_run: If True, simulate and return what would happen
            current_url: Current browser URL (for browser actions)
            
        Returns:
            ActionDecision with evaluation result
        """
        # First, check against hard denylist
        blocked = self._check_denylist(action, args)
        if blocked:
            return ActionDecision(
                allowed=False,
                risk_level=RiskLevel.HIGH,
                rationale="Action matches blocked pattern",
                blocked_reason=blocked,
                requires_approval=ApprovalRequirement.NONE,
            )
        
        # Get base risk classification from SafetyClassifier
        # Note: SafetyClassifier may not handle argv lists properly,
        # so we also do our own pattern matching below
        base_risk = self.classifier.classify_action(
            action, args, current_url=current_url
        )
        
        # Override risk if our patterns say it's higher
        risk_level = self._classify_with_patterns(action, args, base_risk)
        
        # Determine approval requirement
        approval = self._determine_approval(action, args, risk_level)
        
        # Build rationale
        rationale = self._build_rationale(action, args, risk_level)
        
        # For dry-run, add what would be executed
        would_execute = None
        dry_run_output = None
        if dry_run:
            would_execute, dry_run_output = self._simulate_action(action, args)
        
        # Determine if allowed based on approval and auto-approve settings
        allowed = self._is_allowed(risk_level, approval)
        
        return ActionDecision(
            allowed=allowed,
            risk_level=risk_level,
            rationale=rationale,
            requires_approval=approval,
            would_execute=would_execute,
            dry_run_output=dry_run_output,
        )
    
    def evaluate_typed(
        self,
        request: ActionRequest,
        dry_run: bool = False,
        current_url: str = "",
    ) -> ActionDecision:
        """Evaluate a typed ActionRequest against policy.
        
        Args:
            request: Typed action request
            dry_run: If True, simulate and return what would happen
            current_url: Current browser URL
            
        Returns:
            ActionDecision with evaluation result
        """
        return self.evaluate(
            action=request.action,
            args=request.to_legacy_args(),
            domain=request.domain.value,
            dry_run=dry_run,
            current_url=current_url,
        )
    
    def normalize_command(self, cmd: Union[str, list[str]]) -> list[str]:
        """Normalize a command to argv format.
        
        Refuses shell strings with dangerous patterns and converts
        safe shell strings to argv lists.
        
        Args:
            cmd: Command as string or list
            
        Returns:
            Normalized argv list
            
        Raises:
            ValueError: If command contains dangerous patterns
        """
        if isinstance(cmd, list):
            return cmd
        
        # Check for shell operators that are dangerous
        dangerous_shell_patterns = [
            r"\$\(",       # Command substitution
            r"`",          # Backtick substitution
            r"\|",         # Pipe
            r"&&",         # And
            r"\|\|",       # Or
            r";",          # Command separator
            r"\n",         # Newline
        ]
        
        for pattern in dangerous_shell_patterns:
            if re.search(pattern, cmd):
                raise ValueError(
                    f"Shell command contains dangerous pattern: {pattern}. "
                    "Use argv list format instead."
                )
        
        # Safe to split
        try:
            return shlex.split(cmd)
        except ValueError as e:
            raise ValueError(f"Invalid command syntax: {e}")
    
    def _check_denylist(self, action: str, args: dict[str, Any]) -> Optional[str]:
        """Check if action matches hard denylist.
        
        Returns:
            Blocked reason if denied, None if OK
        """
        if action not in ("os_exec", "os_write_file", "os_delete_file"):
            return None
        
        # Get the command/path to check
        if action == "os_exec":
            # Check both cmd and argv
            cmd = args.get("cmd", "")
            argv = args.get("argv", [])
            check_str = cmd or " ".join(argv)
        else:
            check_str = args.get("path", "")
        
        # Check against hard denylist
        for pattern in self.HARD_DENYLIST:
            if re.search(pattern, check_str, re.IGNORECASE):
                return f"Matches blocked pattern: {pattern}"
        
        return None
    
    def _classify_with_patterns(
        self,
        action: str,
        args: dict[str, Any],
        base_risk: RiskLevel,
    ) -> RiskLevel:
        """Classify risk using internal patterns, upgrading if needed.
        
        SafetyClassifier may not handle argv lists properly.
        This method checks our own patterns for os_exec commands
        and upgrades the risk level if necessary.
        
        Args:
            action: Action name
            args: Action arguments
            base_risk: Risk from SafetyClassifier
            
        Returns:
            Final risk level (never lower than base_risk)
        """
        if action != "os_exec":
            return base_risk
        
        # Get command string to check
        cmd = args.get("cmd", "")
        argv = args.get("argv", [])
        check_str = cmd or " ".join(argv)
        
        if not check_str:
            return base_risk
        
        # Check for HIGH risk patterns
        for pattern in self.DOUBLE_CONFIRM_PATTERNS:
            if re.search(pattern, check_str, re.IGNORECASE):
                return RiskLevel.HIGH
        
        # Check for MEDIUM risk patterns
        for pattern in self.MEDIUM_RISK_PATTERNS:
            if re.search(pattern, check_str, re.IGNORECASE):
                # Only upgrade if base was LOW
                if base_risk == RiskLevel.LOW:
                    return RiskLevel.MEDIUM
        
        return base_risk
    
    def _determine_approval(
        self,
        action: str,
        args: dict[str, Any],
        risk_level: RiskLevel,
    ) -> ApprovalRequirement:
        """Determine what approval is required."""
        # HIGH risk always needs at least single approval
        if risk_level == RiskLevel.HIGH:
            # Check for double-confirm patterns
            if self._matches_double_confirm(action, args):
                return ApprovalRequirement.DOUBLE
            return ApprovalRequirement.SINGLE
        
        # MEDIUM risk needs approval unless auto-approved
        if risk_level == RiskLevel.MEDIUM:
            if self.auto_approve_medium:
                return ApprovalRequirement.NONE
            return ApprovalRequirement.SINGLE
        
        # LOW risk - auto-approve
        return ApprovalRequirement.NONE
    
    def _matches_double_confirm(self, action: str, args: dict[str, Any]) -> bool:
        """Check if action requires double confirmation."""
        if action != "os_exec":
            return False
        
        cmd = args.get("cmd", "")
        argv = args.get("argv", [])
        check_str = cmd or " ".join(argv)
        
        for pattern in self.DOUBLE_CONFIRM_PATTERNS:
            if re.search(pattern, check_str, re.IGNORECASE):
                return True
        
        return False
    
    def _build_rationale(
        self,
        action: str,
        args: dict[str, Any],
        risk_level: RiskLevel,
    ) -> str:
        """Build human-readable rationale for the decision."""
        reasons = []
        
        if action == "os_exec":
            cmd = args.get("cmd", "")
            argv = args.get("argv", [])
            cmd_str = cmd or " ".join(argv)
            
            if risk_level == RiskLevel.HIGH:
                # Identify the specific dangerous pattern
                for pattern in self.DOUBLE_CONFIRM_PATTERNS:
                    if re.search(pattern, cmd_str, re.IGNORECASE):
                        reasons.append(f"Matches high-risk pattern: {pattern}")
                        break
                else:
                    reasons.append("Command classified as high-risk")
            elif risk_level == RiskLevel.MEDIUM:
                for pattern in self.MEDIUM_RISK_PATTERNS:
                    if re.search(pattern, cmd_str, re.IGNORECASE):
                        reasons.append(f"Command modifies files/system: {pattern}")
                        break
            else:
                reasons.append("Read-only or informational command")
        
        elif action in ("os_write_file", "os_delete_file"):
            path = args.get("path", "")
            for protected in self.PROTECTED_PATHS:
                if path.startswith(protected):
                    reasons.append(f"Path in protected area: {protected}")
                    break
            else:
                if risk_level == RiskLevel.MEDIUM:
                    reasons.append("File modification in user-writable area")
                else:
                    reasons.append("File operation in safe area")
        
        elif action.startswith(("goto", "click", "type")):
            reasons.append(f"Browser action: {action}")
        
        else:
            reasons.append(f"Action: {action}")
        
        return "; ".join(reasons) if reasons else f"{action} action"
    
    def _is_allowed(
        self,
        risk_level: RiskLevel,
        approval: ApprovalRequirement,
    ) -> bool:
        """Determine if action is allowed based on approval requirements."""
        if approval == ApprovalRequirement.NONE:
            return True
        
        # If approval is required, we can't auto-approve
        # The caller must handle the approval flow
        if risk_level == RiskLevel.LOW and self.auto_approve_low:
            return True
        
        if risk_level == RiskLevel.MEDIUM and self.auto_approve_medium:
            return True
        
        # Requires explicit approval
        return False
    
    def _simulate_action(
        self,
        action: str,
        args: dict[str, Any],
    ) -> tuple[str, str]:
        """Simulate what an action would do.
        
        Returns:
            Tuple of (would_execute description, simulated output)
        """
        if action == "os_exec":
            cmd = args.get("cmd", "")
            argv = args.get("argv", [])
            cmd_str = cmd or " ".join(argv)
            return (
                f"Execute command: {cmd_str}",
                "[DRY RUN] Command would be executed. No actual execution.",
            )
        
        elif action == "os_list_dir":
            path = args.get("path", "")
            return (
                f"List directory: {path}",
                f"[DRY RUN] Would list contents of {path}",
            )
        
        elif action == "os_read_file":
            path = args.get("path", "")
            return (
                f"Read file: {path}",
                f"[DRY RUN] Would read contents of {path}",
            )
        
        elif action == "os_write_file":
            path = args.get("path", "")
            mode = args.get("mode", "overwrite")
            content_len = len(args.get("content", ""))
            return (
                f"Write file: {path} ({mode}, {content_len} bytes)",
                f"[DRY RUN] Would write {content_len} bytes to {path}",
            )
        
        elif action == "os_delete_file":
            path = args.get("path", "")
            return (
                f"Delete file: {path}",
                f"[DRY RUN] Would permanently delete {path}",
            )
        
        elif action == "goto":
            url = args.get("url", "")
            return (
                f"Navigate to: {url}",
                f"[DRY RUN] Would navigate browser to {url}",
            )
        
        elif action == "click":
            selector = args.get("selector", "")
            return (
                f"Click element: {selector}",
                f"[DRY RUN] Would click element matching {selector}",
            )
        
        elif action == "type":
            selector = args.get("selector", "")
            text = args.get("text", "")
            # Mask potentially sensitive input
            masked_text = text[:3] + "***" if len(text) > 3 else "***"
            return (
                f"Type into: {selector}",
                f"[DRY RUN] Would type '{masked_text}' into {selector}",
            )
        
        else:
            return (
                f"Execute action: {action}",
                f"[DRY RUN] Would execute {action} with args",
            )


def create_policy_engine(
    auto_approve_low: bool = True,
    auto_approve_medium: bool = False,
) -> PolicyEngine:
    """Factory function to create a PolicyEngine.
    
    Args:
        auto_approve_low: Auto-approve LOW risk actions
        auto_approve_medium: Auto-approve MEDIUM risk actions
        
    Returns:
        Configured PolicyEngine instance
    """
    return PolicyEngine(
        auto_approve_low=auto_approve_low,
        auto_approve_medium=auto_approve_medium,
    )
