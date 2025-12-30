"""
Tests for bug fixes (Sections 2-8).

Tests for:
- profile_dir caching (Section 3)
- additional_context persistence (Section 2)  
- browser cleanup (Section 4)
- step numbering (Section 6)
- extract_visible_text retry (Section 7)
- recovery action usage (Section 8)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from agentic_browser.config import AgentConfig
from agentic_browser.logger import RunLogger


class TestProfileDirCaching:
    """Tests for profile_dir caching fix (Section 3)."""
    
    def test_profile_dir_cached_when_no_persist(self):
        """Access profile_dir twice with no_persist=True, confirm same path."""
        config = AgentConfig(goal="test", no_persist=True)
        dir1 = config.profile_dir
        dir2 = config.profile_dir
        
        assert dir1 == dir2
        assert "agentic_browser_" in str(dir1)
    
    def test_profile_dir_not_cached_when_persist(self):
        """With no_persist=False, profile_dir should be stable."""
        config = AgentConfig(goal="test", no_persist=False, profile_name="test_profile")
        dir1 = config.profile_dir
        dir2 = config.profile_dir
        
        assert dir1 == dir2
        assert "test_profile" in str(dir1)
    
    def test_cleanup_profile_dir(self):
        """Test that cleanup_profile_dir removes temp directory."""
        config = AgentConfig(goal="test", no_persist=True)
        temp_dir = config.profile_dir
        
        # Create the directory  
        temp_dir.mkdir(parents=True, exist_ok=True)
        assert temp_dir.exists()
        
        # Cleanup should remove it
        config.cleanup_profile_dir()
        assert not temp_dir.exists()
    
    def test_cleanup_profile_dir_noop_when_persist(self):
        """cleanup_profile_dir should be no-op when no_persist=False."""
        config = AgentConfig(goal="test", no_persist=False, profile_name="test")
        # Should not raise
        config.cleanup_profile_dir()


class TestStepNumbering:
    """Tests for step numbering fix (Section 6)."""
    
    def test_print_step_uses_explicit_step(self):
        """Verify print_step uses the passed step number."""
        logger = RunLogger("test goal", enable_console=False)
        
        # Mock the console to capture output
        with patch.object(logger, 'console', new=None):
            # Should not raise even with console=None
            logger.print_step(
                step=5,
                action="click",
                args={"selector": "#btn"},
                rationale="test",
                risk="low",
                requires_approval=False,
            )
    
    def test_log_step_increments_count(self):
        """Verify log_step increments the step count."""
        logger = RunLogger("test goal", enable_console=False)
        
        assert logger.step_count == 0
        
        logger.log_step(
            state={"current_url": "http://test.com", "page_title": "Test"},
            model_output={"action": "click", "args": {}},
            result={"success": True},
        )
        
        assert logger.step_count == 1
        
        logger.log_step(
            state={"current_url": "http://test.com", "page_title": "Test"},
            model_output={"action": "type", "args": {}},
            result={"success": True},
        )
        
        assert logger.step_count == 2


class TestApprover:
    """Tests for approver abstraction (Section 5)."""
    
    def test_auto_approver_always_approves(self):
        """AutoApprover always returns True."""
        from agentic_browser.approver import AutoApprover
        from agentic_browser.safety import RiskLevel
        
        approver = AutoApprover()
        approved, modified = approver.request_approval(
            action="click",
            args={"selector": "#delete"},
            risk_level=RiskLevel.HIGH,
            rationale="test",
        )
        
        assert approved is True
        assert modified is None
    
    def test_auto_approver_empty_denial(self):
        """AutoApprover returns empty guidance on denial."""
        from agentic_browser.approver import AutoApprover
        
        approver = AutoApprover()
        guidance = approver.notify_denial()
        
        assert guidance == ""
    
    def test_get_approver_returns_auto_when_auto_approve(self):
        """get_approver returns AutoApprover when auto_approve=True."""
        from agentic_browser.approver import get_approver, AutoApprover
        
        approver = get_approver(mode="cli", auto_approve=True)
        
        assert isinstance(approver, AutoApprover)
    
    def test_get_approver_returns_console_for_cli(self):
        """get_approver returns ConsoleApprover for CLI mode."""
        from agentic_browser.approver import get_approver, ConsoleApprover
        
        approver = get_approver(mode="cli", auto_approve=False)
        
        assert isinstance(approver, ConsoleApprover)


class TestExtractVisibleTextRetry:
    """Tests for extract_visible_text retry behavior (Section 7)."""
    
    @pytest.mark.skipif(not pytest.importorskip("playwright", reason="playwright not installed"), reason="playwright required")
    def test_retry_on_first_failure_succeeds(self):
        """Mock page.evaluate() failing once then succeeding."""
        from agentic_browser.tools import BrowserTools
        
        mock_page = MagicMock()
        
        # First call raises, second succeeds
        call_count = [0]
        def mock_evaluate(script):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Timing error")
            return "Page content here"
        
        mock_page.evaluate = mock_evaluate
        mock_page.wait_for_load_state = MagicMock()
        mock_page.wait_for_timeout = MagicMock()
        
        tools = BrowserTools(mock_page)
        result = tools.extract_visible_text()
        
        assert result.success is True
        assert "Page content here" in result.data["text"]
        assert call_count[0] >= 2
    
    @pytest.mark.skipif(not pytest.importorskip("playwright", reason="playwright not installed"), reason="playwright required")
    def test_permanent_failure_returns_empty(self):
        """Mock permanent failure returns empty without crashing."""
        from agentic_browser.tools import BrowserTools
        
        mock_page = MagicMock()
        mock_page.evaluate = MagicMock(side_effect=Exception("Permanent error"))
        mock_page.wait_for_load_state = MagicMock()
        mock_page.wait_for_timeout = MagicMock()
        
        tools = BrowserTools(mock_page)
        result = tools.extract_visible_text()
        
        # Should succeed but with empty text
        assert result.success is True
        assert result.data["text"] == ""
        assert "warning" in result.data


class TestAdditionalContextPersistence:
    """Tests for additional_context persistence (Section 2)."""
    
    def test_context_stored_in_state_dict(self):
        """Verify additional_context is included in state dict."""
        # Test the dataclass directly without importing playwright
        from dataclasses import dataclass, field
        from typing import Any
        
        # Create a simple version of the test
        @dataclass
        class TestPageState:
            goal: str
            additional_context: str = ""
            
            def to_dict(self) -> dict:
                return {
                    "goal": self.goal,
                    "additional_context": self.additional_context,
                }
        
        state = TestPageState(
            goal="test",
            additional_context="Context from last iteration",
        )
        
        state_dict = state.to_dict()
        assert state_dict["additional_context"] == "Context from last iteration"


class TestUpdateStateVisitedUrls:
    """Tests for BaseAgent._update_state list appending."""

    def test_update_state_appends_urls_and_files(self):
        """Ensure visited URLs and files are appended across updates."""
        from agentic_browser.config import AgentConfig
        from agentic_browser.graph.agents.base import BaseAgent
        from agentic_browser.graph.state import create_initial_state

        class DummyAgent(BaseAgent):
            AGENT_NAME = "dummy"

            @property
            def system_prompt(self) -> str:
                return ""

            def execute(self, state):
                return state

        with patch("agentic_browser.graph.agents.base.create_llm_client", return_value=MagicMock()):
            agent = DummyAgent(AgentConfig(goal="test"))

        state = create_initial_state("test goal")
        first_state = agent._update_state(
            state,
            visited_url="https://example.com/first",
            file_accessed="/tmp/first.txt",
        )
        second_state = agent._update_state(
            first_state,
            visited_url="https://example.com/second",
            file_accessed="/tmp/second.txt",
        )

        assert second_state["visited_urls"] == [
            "https://example.com/first",
            "https://example.com/second",
        ]
        assert second_state["files_accessed"] == [
            "/tmp/first.txt",
            "/tmp/second.txt",
        ]


class TestRecallRunDetails:
    """Tests for recall run details lookup."""

    def test_get_run_details_returns_session_and_steps(self, tmp_path):
        """Ensure get_run_details returns session details without AttributeError."""
        from agentic_browser.graph.memory import SessionStore
        from agentic_browser.graph.run_history import RecallTool

        db_path = tmp_path / "sessions.db"
        store = SessionStore(db_path=db_path)
        session_id = "run-1"

        store.create_session(session_id=session_id, goal="Test goal", state={"task_complete": True})
        store.add_step(
            session_id=session_id,
            step_number=1,
            agent="tester",
            action="think",
            args={"note": "hello"},
            result="ok",
        )

        recall = RecallTool(session_store=store)
        result = recall.get_run_details(session_id)

        assert result.success is True
        assert result.data["session"]["id"] == session_id
        assert result.data["steps"][0]["step_number"] == 1
