import agentic_browser.graph.run_history as run_history
import agentic_browser.graph.supervisor as supervisor_module
import agentic_browser.graph.tool_registry as tool_registry
from agentic_browser.graph.agents import planner_agent as planner_module


def test_supervisor_node_handles_missing_tools(monkeypatch):
    monkeypatch.setattr(tool_registry, "get_tools", lambda session_id: None)

    def fake_route(self, state):
        return {
            **state,
            "config_goal": self.config.goal,
            "config_model": self.config.model,
            "config_endpoint": self.config.model_endpoint,
        }

    monkeypatch.setattr(supervisor_module.Supervisor, "route", fake_route)

    state = {"session_id": "missing-tools", "goal": "Test goal"}
    result = supervisor_module.supervisor_node(state)

    assert result["config_goal"] == "Test goal"
    assert result["config_model"]
    assert result["config_endpoint"]


def test_planner_node_handles_missing_tools(monkeypatch):
    monkeypatch.setattr(tool_registry, "get_tools", lambda session_id: None)

    class DummyRecallTool:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(run_history, "RecallTool", DummyRecallTool)

    def fake_execute(self, state):
        return {
            **state,
            "config_goal": self.config.goal,
            "config_model": self.config.model,
            "config_endpoint": self.config.model_endpoint,
        }

    monkeypatch.setattr(planner_module.PlannerAgentNode, "execute", fake_execute)

    state = {"session_id": "missing-tools", "goal": "Plan for the test"}
    result = planner_module.planner_agent_node(state)

    assert result["config_goal"] == "Plan for the test"
    assert result["config_model"]
    assert result["config_endpoint"]
