"""
Tests for Policy Engine.

Tests denylist patterns, risk classification, dry-run mode, and approval workflow.
"""

import pytest

from agentic_browser.policy_engine import (
    PolicyEngine,
    ActionDecision,
    ApprovalRequirement,
    create_policy_engine,
)
from agentic_browser.safety import RiskLevel


class TestDenylistPatterns:
    """Tests for hard denylist patterns."""
    
    @pytest.fixture
    def engine(self):
        return PolicyEngine()
    
    def test_rm_rf_root_blocked(self, engine):
        """Test rm -rf / is blocked."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "rm -rf /"}
        )
        assert decision.allowed is False
        assert decision.blocked_reason is not None
        assert "blocked" in decision.blocked_reason.lower()
    
    def test_sudo_rm_rf_root_blocked(self, engine):
        """Test sudo rm -rf / is blocked."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "sudo rm -rf /"}
        )
        assert decision.allowed is False
    
    def test_dd_to_disk_blocked(self, engine):
        """Test dd to disk device is blocked."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "dd if=/dev/zero of=/dev/sda"}
        )
        assert decision.allowed is False
    
    def test_mkfs_to_disk_blocked(self, engine):
        """Test mkfs to disk device is blocked."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "mkfs.ext4 /dev/sda1"}
        )
        assert decision.allowed is False
    
    def test_overwrite_passwd_blocked(self, engine):
        """Test overwriting /etc/passwd is blocked."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "echo 'x' > /etc/passwd"}
        )
        assert decision.allowed is False
    
    def test_safe_command_not_blocked(self, engine):
        """Test safe commands are not blocked."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "ls -la"}
        )
        # Not blocked, but may require approval based on risk
        assert decision.blocked_reason is None


class TestDoubleConfirmPatterns:
    """Tests for double confirmation patterns."""
    
    @pytest.fixture
    def engine(self):
        return PolicyEngine()
    
    def test_rm_rf_requires_double_confirm(self, engine):
        """Test rm -rf requires double confirmation."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "rm -rf /home/user/folder"}
        )
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.requires_approval == ApprovalRequirement.DOUBLE
    
    def test_sudo_requires_double_confirm(self, engine):
        """Test sudo commands require double confirmation."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "sudo apt update"}
        )
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.requires_approval == ApprovalRequirement.DOUBLE
    
    def test_dd_requires_double_confirm(self, engine):
        """Test dd requires double confirmation (when not blocked)."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "dd if=/dev/zero of=/home/user/file.img bs=1M count=100"}
        )
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.requires_approval == ApprovalRequirement.DOUBLE
    
    def test_shutdown_requires_double_confirm(self, engine):
        """Test shutdown requires double confirmation."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "shutdown now"}
        )
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.requires_approval == ApprovalRequirement.DOUBLE
    
    def test_chmod_recursive_requires_double_confirm(self, engine):
        """Test recursive chmod requires double confirmation."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "chmod -R 777 /home/user"}
        )
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.requires_approval == ApprovalRequirement.DOUBLE


class TestMediumRiskPatterns:
    """Tests for medium risk patterns."""
    
    @pytest.fixture
    def engine(self):
        return PolicyEngine(auto_approve_medium=False)
    
    def test_mv_medium_risk(self, engine):
        """Test mv is medium risk."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "mv file1.txt file2.txt"}
        )
        assert decision.risk_level == RiskLevel.MEDIUM
        assert decision.requires_approval == ApprovalRequirement.SINGLE
    
    def test_mkdir_medium_risk(self, engine):
        """Test mkdir is medium risk."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "mkdir /home/user/new_folder"}
        )
        assert decision.risk_level == RiskLevel.MEDIUM
    
    def test_redirect_medium_risk(self, engine):
        """Test file redirection is medium risk."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "echo 'data' > /home/user/file.txt"}
        )
        assert decision.risk_level == RiskLevel.MEDIUM


class TestLowRiskPatterns:
    """Tests for low risk patterns."""
    
    @pytest.fixture
    def engine(self):
        return PolicyEngine()
    
    def test_ls_low_risk(self, engine):
        """Test ls is low risk."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "ls -la"}
        )
        assert decision.risk_level == RiskLevel.LOW
        assert decision.requires_approval == ApprovalRequirement.NONE
        assert decision.allowed is True
    
    def test_cat_low_risk(self, engine):
        """Test cat is low risk."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "cat /etc/os-release"}
        )
        assert decision.risk_level == RiskLevel.LOW
    
    def test_grep_low_risk(self, engine):
        """Test grep is low risk."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "grep -r 'pattern' ."}
        )
        assert decision.risk_level == RiskLevel.LOW


class TestDryRunMode:
    """Tests for dry-run simulation."""
    
    @pytest.fixture
    def engine(self):
        return PolicyEngine()
    
    def test_dry_run_command(self, engine):
        """Test dry-run for command."""
        decision = engine.evaluate(
            "os_exec",
            {"cmd": "ls -la /home"},
            dry_run=True
        )
        assert decision.would_execute is not None
        assert "ls -la /home" in decision.would_execute
        assert decision.dry_run_output is not None
        assert "DRY RUN" in decision.dry_run_output
    
    def test_dry_run_write_file(self, engine):
        """Test dry-run for write file."""
        decision = engine.evaluate(
            "os_write_file",
            {"path": "/home/user/test.txt", "content": "hello world", "mode": "overwrite"},
            dry_run=True
        )
        assert decision.would_execute is not None
        assert "/home/user/test.txt" in decision.would_execute
        assert "11 bytes" in decision.would_execute
    
    def test_dry_run_goto(self, engine):
        """Test dry-run for browser navigation."""
        decision = engine.evaluate(
            "goto",
            {"url": "https://example.com"},
            domain="browser",
            dry_run=True
        )
        assert decision.would_execute is not None
        assert "example.com" in decision.would_execute
    
    def test_dry_run_type_masks_input(self, engine):
        """Test dry-run masks sensitive input."""
        decision = engine.evaluate(
            "type",
            {"selector": "#password", "text": "mysecretpassword123"},
            domain="browser",
            dry_run=True
        )
        assert decision.dry_run_output is not None
        assert "mysecretpassword123" not in decision.dry_run_output
        assert "***" in decision.dry_run_output


class TestCommandNormalization:
    """Tests for command normalization."""
    
    @pytest.fixture
    def engine(self):
        return PolicyEngine()
    
    def test_normalize_list_unchanged(self, engine):
        """Test list is returned unchanged."""
        argv = engine.normalize_command(["ls", "-la"])
        assert argv == ["ls", "-la"]
    
    def test_normalize_safe_string(self, engine):
        """Test safe string is split."""
        argv = engine.normalize_command("ls -la /home")
        assert argv == ["ls", "-la", "/home"]
    
    def test_normalize_rejects_pipe(self, engine):
        """Test pipe is rejected."""
        with pytest.raises(ValueError, match="dangerous pattern"):
            engine.normalize_command("cat file | grep pattern")
    
    def test_normalize_rejects_command_substitution(self, engine):
        """Test command substitution is rejected."""
        with pytest.raises(ValueError, match="dangerous pattern"):
            engine.normalize_command("echo $(cat /etc/passwd)")
    
    def test_normalize_rejects_semicolon(self, engine):
        """Test command separator is rejected."""
        with pytest.raises(ValueError, match="dangerous pattern"):
            engine.normalize_command("ls; rm -rf /")
    
    def test_normalize_rejects_and(self, engine):
        """Test && is rejected."""
        with pytest.raises(ValueError, match="dangerous pattern"):
            engine.normalize_command("true && rm -rf /")


class TestAutoApproveSettings:
    """Tests for auto-approve settings."""
    
    def test_auto_approve_low_default(self):
        """Test low risk is auto-approved by default."""
        engine = PolicyEngine(auto_approve_low=True)
        decision = engine.evaluate("os_exec", {"cmd": "ls"})
        assert decision.allowed is True
    
    def test_no_auto_approve_low(self):
        """Test low risk requires approval when disabled."""
        engine = PolicyEngine(auto_approve_low=False)
        decision = engine.evaluate("os_exec", {"cmd": "ls"})
        # Low risk doesn't need approval even without auto-approve
        assert decision.allowed is True
    
    def test_no_auto_approve_medium_default(self):
        """Test medium risk not auto-approved by default."""
        engine = PolicyEngine(auto_approve_medium=False)
        decision = engine.evaluate("os_exec", {"cmd": "mv a b"})
        assert decision.allowed is False
        assert decision.requires_approval == ApprovalRequirement.SINGLE
    
    def test_auto_approve_medium_enabled(self):
        """Test medium risk auto-approved when enabled."""
        engine = PolicyEngine(auto_approve_medium=True)
        decision = engine.evaluate("os_exec", {"cmd": "mv a b"})
        assert decision.allowed is True


class TestExplanation:
    """Tests for explanation generation."""
    
    @pytest.fixture
    def engine(self):
        return PolicyEngine()
    
    def test_explanation_contains_risk_level(self, engine):
        """Test explanation contains risk level."""
        decision = engine.evaluate("os_exec", {"cmd": "rm -rf /home/folder"})
        explanation = decision.to_explanation()
        assert "HIGH" in explanation
    
    def test_explanation_contains_rationale(self, engine):
        """Test explanation contains rationale."""
        decision = engine.evaluate("os_exec", {"cmd": "ls -la"})
        explanation = decision.to_explanation()
        assert "Rationale" in explanation
    
    def test_explanation_shows_blocked_reason(self, engine):
        """Test explanation shows blocked reason."""
        decision = engine.evaluate("os_exec", {"cmd": "rm -rf /"})
        explanation = decision.to_explanation()
        assert "BLOCKED" in explanation


class TestArgvSupport:
    """Tests for argv list support."""
    
    @pytest.fixture
    def engine(self):
        return PolicyEngine()
    
    def test_argv_list_evaluated(self, engine):
        """Test argv list is properly evaluated."""
        decision = engine.evaluate(
            "os_exec",
            {"argv": ["ls", "-la", "/home"]}
        )
        assert decision.blocked_reason is None
        assert decision.risk_level == RiskLevel.LOW
    
    def test_argv_dangerous_evaluated(self, engine):
        """Test dangerous argv list is properly evaluated."""
        decision = engine.evaluate(
            "os_exec",
            {"argv": ["rm", "-rf", "/home/user/folder"]}
        )
        assert decision.risk_level == RiskLevel.HIGH
        assert decision.requires_approval == ApprovalRequirement.DOUBLE


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_policy_engine_default(self):
        """Test factory creates engine with defaults."""
        engine = create_policy_engine()
        assert engine.auto_approve_low is True
        assert engine.auto_approve_medium is False
    
    def test_create_policy_engine_custom(self):
        """Test factory creates engine with custom settings."""
        engine = create_policy_engine(
            auto_approve_low=False,
            auto_approve_medium=True
        )
        assert engine.auto_approve_low is False
        assert engine.auto_approve_medium is True
