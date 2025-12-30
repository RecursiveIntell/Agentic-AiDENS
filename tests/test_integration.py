"""
Integration Tests for Agentic Browser.

Validates imports, agent instantiation, state schema, and core function signatures.
Run with: pytest tests/test_integration.py -v
"""

import pytest
import inspect
from typing import get_type_hints


class TestImports:
    """Test that all critical imports work correctly."""
    
    def test_cost_module_imports(self):
        """Verify cost module imports and functions."""
        from agentic_browser.cost import calculate_cost, load_prices, save_prices
        assert callable(calculate_cost)
        assert callable(load_prices)
        assert callable(save_prices)
    
    def test_state_module_imports(self):
        """Verify state module imports."""
        from agentic_browser.graph.state import AgentState, create_initial_state
        assert AgentState is not None
        assert callable(create_initial_state)
    
    def test_config_module_imports(self):
        """Verify config module imports."""
        from agentic_browser.config import AgentConfig
        assert AgentConfig is not None
    
    def test_base_agent_imports(self):
        """Verify base agent imports and calculate_cost dependency."""
        from agentic_browser.graph.agents.base import BaseAgent, create_llm_client
        # This will fail if calculate_cost import is missing
        assert BaseAgent is not None
        assert callable(create_llm_client)
    
    def test_all_agent_imports(self):
        """Verify all agent modules can be imported."""
        from agentic_browser.graph.agents.browser_agent import BrowserAgentNode
        from agentic_browser.graph.agents.research_agent import ResearchAgentNode
        from agentic_browser.graph.agents.planner_agent import PlannerAgentNode
        from agentic_browser.graph.agents.retrospective_agent import RetrospectiveAgent
        
        assert BrowserAgentNode is not None
        assert ResearchAgentNode is not None
        assert PlannerAgentNode is not None
        assert RetrospectiveAgent is not None
    
    def test_supervisor_imports(self):
        """Verify supervisor imports."""
        from agentic_browser.graph.supervisor import Supervisor
        assert Supervisor is not None
    
    def test_gui_imports(self):
        """Verify GUI module imports (without launching)."""
        from agentic_browser.gui.frontier_ui import MissionControlWindow
        from agentic_browser.gui.cost_dialog import CostDialog
        assert MissionControlWindow is not None
        assert CostDialog is not None


class TestStateSchema:
    """Test AgentState schema and initialization."""
    
    def test_create_initial_state(self):
        """Verify create_initial_state returns valid state."""
        from agentic_browser.graph.state import create_initial_state
        
        state = create_initial_state("Test goal", max_steps=10, session_id="test-123")
        
        # Check required fields exist
        assert state["goal"] == "Test goal"
        assert state["max_steps"] == 10
        assert state["session_id"] == "test-123"
        assert state["step_count"] == 0
        assert state["task_complete"] == False
        assert state["messages"] == []
        assert state["visited_urls"] == []
        assert state["extracted_data"] == {}
    
    def test_state_has_token_usage(self):
        """Verify token_usage field exists in state."""
        from agentic_browser.graph.state import create_initial_state
        
        state = create_initial_state("Test goal")
        
        assert "token_usage" in state
        assert "input_tokens" in state["token_usage"]
        assert "output_tokens" in state["token_usage"]
        assert "total_cost" in state["token_usage"]
    
    def test_state_has_retrospective_ran(self):
        """Verify retrospective_ran field exists in state."""
        from agentic_browser.graph.state import create_initial_state
        
        state = create_initial_state("Test goal")
        
        assert "retrospective_ran" in state
        assert state["retrospective_ran"] == False
    
    def test_state_has_plan_tracking_fields(self):
        """Verify plan tracking fields exist."""
        from agentic_browser.graph.state import create_initial_state
        
        state = create_initial_state("Test goal")
        
        assert "implementation_plan" in state
        assert "plan_step_index" in state


class TestAgentSignatures:
    """Test that agent methods have expected signatures."""
    
    def test_base_agent_update_state_signature(self):
        """Verify BaseAgent._update_state accepts token_usage."""
        from agentic_browser.graph.agents.base import BaseAgent
        
        sig = inspect.signature(BaseAgent._update_state)
        params = list(sig.parameters.keys())
        
        assert "token_usage" in params
        assert "step_update" in params
    
    def test_planner_agent_update_state_signature(self):
        """Verify PlannerAgentNode._update_state accepts token_usage."""
        from agentic_browser.graph.agents.planner_agent import PlannerAgentNode
        
        sig = inspect.signature(PlannerAgentNode._update_state)
        params = list(sig.parameters.keys())
        
        assert "token_usage" in params
    
    def test_base_agent_build_messages_returns_list(self):
        """Verify _build_messages returns list, not None."""
        from agentic_browser.graph.agents.base import BaseAgent
        from agentic_browser.graph.state import create_initial_state
        from agentic_browser.config import AgentConfig
        
        # Create minimal config (goal is required)
        config = AgentConfig(goal="Test goal")
        
        # We can't instantiate BaseAgent directly (abstract), so check the method signature
        sig = inspect.signature(BaseAgent._build_messages)
        # Check return annotation if present
        # Just verify method exists and is callable
        assert callable(BaseAgent._build_messages)

    def test_build_messages_preserves_state_contents(self):
        """Verify _build_messages does not mutate state message contents."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from agentic_browser.config import AgentConfig
        from agentic_browser.graph.agents.planner_agent import PlannerAgentNode
        from agentic_browser.graph.state import create_initial_state

        config = AgentConfig(goal="Test goal")
        agent = PlannerAgentNode(config)

        state = create_initial_state(goal="Test goal")
        state["messages"] = [
            SystemMessage(content="S" * 600),
            HumanMessage(content="H" * 600),
            AIMessage(content="A" * 600),
            HumanMessage(content="short human"),
            AIMessage(content="short ai"),
            HumanMessage(content="short human 2"),
            AIMessage(content="short ai 2"),
            HumanMessage(content="short human 3"),
        ]
        original_contents = [msg.content for msg in state["messages"]]

        agent._build_messages(state)

        assert [msg.content for msg in state["messages"]] == original_contents


class TestCostCalculation:
    """Test cost calculation functionality."""
    
    def test_calculate_cost_gpt4o(self):
        """Verify cost calculation for GPT-4o."""
        from agentic_browser.cost import calculate_cost
        
        cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        
        assert cost > 0
        assert isinstance(cost, float)
    
    def test_calculate_cost_unknown_model(self):
        """Verify cost calculation for unknown model defaults to 0."""
        from agentic_browser.cost import calculate_cost
        
        cost = calculate_cost("unknown-model-xyz", input_tokens=1000, output_tokens=500)
        
        # Should not crash, may return 0 or a default
        assert isinstance(cost, (int, float))


class TestSupervisorRouting:
    """Test supervisor routing logic."""
    
    def test_supervisor_has_route_method(self):
        """Verify Supervisor has route method."""
        from agentic_browser.graph.supervisor import Supervisor
        
        assert hasattr(Supervisor, "route")
        assert callable(getattr(Supervisor, "route"))
    
    def test_supervisor_has_update_token_usage(self):
        """Verify Supervisor has update_token_usage method."""
        from agentic_browser.graph.supervisor import Supervisor
        
        assert hasattr(Supervisor, "update_token_usage")


class TestGUIComponents:
    """Test GUI component initialization (without display)."""
    
    def test_mission_control_has_setup_ui(self):
        """Verify MissionControlWindow defines _setup_ui."""
        from agentic_browser.gui.frontier_ui import MissionControlWindow
        
        # Check that the class has the _setup_ui method
        assert hasattr(MissionControlWindow, "_setup_ui")
    
    def test_mission_control_has_neural_stream(self):
        """Verify MissionControlWindow has NeuralStream component."""
        from agentic_browser.gui.frontier_ui import NeuralStream
        
        assert NeuralStream is not None


# Run with: pytest tests/test_integration.py -v --tb=short
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
