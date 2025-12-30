"""
Tests for JSON parsing functionality.
"""

import json
import pytest

from agentic_browser.config import AgentConfig
from agentic_browser.graph.agents.automation_agent import AutomationAgentNode
from agentic_browser.graph.agents.browser_agent import BrowserAgentNode
from agentic_browser.graph.agents.code_agent import CodeAgentNode
from agentic_browser.graph.agents.data_agent import DataAgentNode
from agentic_browser.graph.agents.media_agent import MediaAgentNode
from agentic_browser.graph.agents.network_agent import NetworkAgentNode
from agentic_browser.graph.agents.os_agent import OSAgentNode
from agentic_browser.graph.agents.package_agent import PackageAgentNode
from agentic_browser.graph.agents.planner_agent import PlannerAgentNode
from agentic_browser.graph.agents.research_agent import ResearchAgentNode
from agentic_browser.graph.agents.retrospective_agent import RetrospectiveAgent
from agentic_browser.graph.agents.sysadmin_agent import SysAdminAgentNode
from agentic_browser.graph.agents.workflow_agent import WorkflowAgentNode
from agentic_browser.llm_client import (
    parse_json_with_recovery,
    ActionResponse,
    LLMClient,
)
from agentic_browser.utils import extract_json_from_response


class TestExtractJsonFromResponse:
    """Tests for JSON extraction from various response formats."""
    
    def test_raw_json(self):
        """Test extracting raw JSON."""
        response = '{"action": "goto", "args": {"url": "https://example.com"}}'
        result = extract_json_from_response(response)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["action"] == "goto"
    
    def test_markdown_code_block(self):
        """Test extracting JSON from markdown code block."""
        response = '''Here's the action:
```json
{"action": "click", "args": {"selector": "#submit"}, "rationale": "test", "risk": "low"}
```
That should work.'''
        result = extract_json_from_response(response)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["action"] == "click"
    
    def test_markdown_without_json_tag(self):
        """Test extracting JSON from code block without json tag."""
        response = '''```
{"action": "type", "args": {"selector": "input", "text": "hello"}}
```'''
        result = extract_json_from_response(response)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["action"] == "type"
    
    def test_json_with_prose(self):
        """Test extracting JSON surrounded by prose."""
        response = '''I think we should navigate to the page.
{"action": "goto", "args": {"url": "https://test.com"}, "rationale": "navigate", "risk": "low"}
This will load the page.'''
        result = extract_json_from_response(response)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["args"]["url"] == "https://test.com"
    
    def test_no_json_returns_none(self):
        """Test that non-JSON response returns None."""
        response = "This is just plain text with no JSON."
        result = extract_json_from_response(response)
        # Should still find nothing valid
        assert result is None or result == ""


class TestParseJsonWithRecovery:
    """Tests for JSON parsing with recovery strategies."""
    
    def test_valid_json(self):
        """Test parsing valid JSON directly."""
        response = '{"action": "scroll", "args": {"amount": 500}}'
        result = parse_json_with_recovery(response)
        assert result["action"] == "scroll"
        assert result["args"]["amount"] == 500
    
    def test_trailing_comma_fix(self):
        """Test fixing trailing commas."""
        response = '{"action": "back", "args": {},}'
        result = parse_json_with_recovery(response)
        assert result["action"] == "back"
    
    def test_markdown_wrapped(self):
        """Test parsing markdown-wrapped JSON."""
        response = '''```json
{"action": "done", "args": {}, "final_answer": "Task complete"}
```'''
        result = parse_json_with_recovery(response)
        assert result["action"] == "done"
        assert result["final_answer"] == "Task complete"
    
    def test_json_with_prefix(self):
        """Test parsing JSON with leading text."""
        response = 'The next action is: {"action": "screenshot", "args": {"label": "test"}}'
        result = parse_json_with_recovery(response)
        assert result["action"] == "screenshot"
    
    def test_invalid_json_raises(self):
        """Test that completely invalid JSON raises an error."""
        response = "This is not JSON at all, just random text without braces"
        with pytest.raises(json.JSONDecodeError):
            parse_json_with_recovery(response)


class TestActionResponse:
    """Tests for ActionResponse validation."""
    
    def test_valid_action(self):
        """Test creating a valid action response."""
        data = {
            "action": "goto",
            "args": {"url": "https://example.com"},
            "rationale": "Navigate to target",
            "risk": "low",
            "requires_approval": False,
        }
        response = ActionResponse(**data)
        assert response.action == "goto"
        assert response.risk == "low"
    
    def test_invalid_action_type(self):
        """Test that invalid action type fails validation."""
        data = {
            "action": "invalid_action",
            "args": {},
            "rationale": "test",
            "risk": "low",
        }
        with pytest.raises(Exception):  # Pydantic ValidationError
            ActionResponse(**data)
    
    def test_invalid_risk_level(self):
        """Test that invalid risk level fails validation."""
        data = {
            "action": "click",
            "args": {"selector": "#btn"},
            "rationale": "test",
            "risk": "extreme",  # Invalid
        }
        with pytest.raises(Exception):  # Pydantic ValidationError
            ActionResponse(**data)
    
    def test_done_with_final_answer(self):
        """Test done action with final answer."""
        data = {
            "action": "done",
            "args": {},
            "rationale": "Task complete",
            "risk": "low",
            "final_answer": "The page title is 'Example Domain'",
        }
        response = ActionResponse(**data)
        assert response.action == "done"
        assert response.final_answer == "The page title is 'Example Domain'"
    
    def test_defaults(self):
        """Test that defaults are applied."""
        data = {
            "action": "scroll",
            "args": {"amount": 100},
            "rationale": "Scroll down",
        }
        response = ActionResponse(**data)
        assert response.risk == "low"
        assert response.requires_approval is False
        assert response.final_answer is None


class TestRetryLogic:
    """Tests for JSON retry logic (mocked)."""
    
    def test_retry_count_respected(self):
        """Test that retry count is respected in parsing logic."""
        # This tests the conceptual retry - actual LLM calls would need mocking
        attempts = []
        
        def mock_parse(response: str) -> dict:
            attempts.append(1)
            if len(attempts) < 3:
                raise json.JSONDecodeError("Invalid", response, 0)
            return {"action": "done", "args": {}, "rationale": "test", "risk": "low"}
        
        # Simulate retry logic
        max_retries = 2
        result = None
        for attempt in range(max_retries + 1):
            try:
                result = mock_parse("test")
                break
            except json.JSONDecodeError:
                if attempt == max_retries:
                    raise
        
        assert result is not None
        assert len(attempts) == 3


@pytest.mark.parametrize(
    "agent_cls,allowed_actions,expected_action",
    [
        (
            BrowserAgentNode,
            {"goto", "click", "type", "press", "scroll", "extract_visible_text", "download_image", "done", "back"},
            "scroll",
        ),
        (
            ResearchAgentNode,
            {"goto", "click", "back", "extract_visible_text", "scroll", "done"},
            "scroll",
        ),
        (
            OSAgentNode,
            {"os_exec", "os_list_dir", "os_read_file", "os_write_file", "done"},
            "done",
        ),
        (
            CodeAgentNode,
            {"os_list_dir", "os_read_file", "os_exec", "done"},
            "done",
        ),
        (
            AutomationAgentNode,
            {"notify_desktop", "delay", "at_schedule", "cron_show", "reminder_list", "countdown", "done"},
            "done",
        ),
        (
            DataAgentNode,
            {
                "state_to_file",
                "json_to_csv",
                "csv_to_json",
                "json_query",
                "csv_summary",
                "text_search",
                "text_stats",
                "compress",
                "extract",
                "done",
            },
            "done",
        ),
        (
            MediaAgentNode,
            {
                "ffmpeg_convert",
                "ffmpeg_extract_audio",
                "ffmpeg_trim",
                "ffmpeg_info",
                "image_resize",
                "image_convert",
                "image_compress",
                "done",
            },
            "done",
        ),
        (
            NetworkAgentNode,
            {
                "ping",
                "dns_lookup",
                "traceroute",
                "port_check",
                "netstat_listen",
                "http_get",
                "http_post",
                "ssl_check",
                "done",
            },
            "done",
        ),
        (
            PackageAgentNode,
            {
                "venv_create",
                "venv_info",
                "pip_install",
                "pip_list",
                "pip_search",
                "apt_search",
                "git_clone",
                "requirements_parse",
                "done",
            },
            "done",
        ),
        (
            SysAdminAgentNode,
            {
                "service_status",
                "service_control",
                "process_list",
                "disk_usage",
                "memory_usage",
                "cpu_usage",
                "journal_logs",
                "uptime",
                "done",
            },
            "done",
        ),
        (
            WorkflowAgentNode,
            {"trigger_webhook", "list_webhooks", "check_health", "done"},
            "done",
        ),
    ],
)
def test_empty_response_fallback_action_schema(agent_cls, allowed_actions, expected_action):
    config = AgentConfig(goal="test fallback")
    agent = agent_cls(config)

    payload = agent._empty_response_fallback("empty response")

    assert isinstance(payload, dict)
    assert payload.get("action") in allowed_actions
    assert payload.get("action") == expected_action
    assert isinstance(payload.get("args"), dict)

    if expected_action == "scroll":
        assert payload["args"].get("amount")
    if expected_action == "done":
        assert payload["args"].get("summary")


def test_empty_response_fallback_planner_schema():
    config = AgentConfig(goal="test fallback")
    agent = PlannerAgentNode(config)

    payload = agent._empty_response_fallback("empty response")

    assert isinstance(payload, dict)
    assert payload.get("goal_analysis")
    assert isinstance(payload.get("steps"), list)
    assert payload.get("estimated_agents")


def test_empty_response_fallback_retrospective_schema():
    config = AgentConfig(goal="test fallback")
    agent = RetrospectiveAgent(config)

    payload = agent._empty_response_fallback("empty response")

    assert isinstance(payload, dict)
    assert isinstance(payload.get("strategies"), list)
    assert isinstance(payload.get("apocalypse"), list)
    assert payload.get("summary")
