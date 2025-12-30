"""Tests for BaseAgent safe_invoke fallback behavior."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from agentic_browser.config import AgentConfig
from agentic_browser.graph.agents.code_agent import CodeAgentNode


def test_safe_invoke_fallback_for_code_agent():
    """Ensure fallback JSON is parsable and valid for non-browser agents."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="   ")

    with patch("agentic_browser.graph.agents.base.create_llm_client", return_value=mock_llm):
        agent = CodeAgentNode(AgentConfig(goal="test"), os_tools=MagicMock())

    response = agent.safe_invoke([])
    data = json.loads(response.content)

    assert data["action"] == "done"
    assert "summary" in data.get("args", {})
