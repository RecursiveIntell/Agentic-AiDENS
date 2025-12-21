"""
Main LangGraph construction for multi-agent system.

Wires together supervisor and specialized agents into a StateGraph.
"""

from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, create_initial_state
from .supervisor import supervisor_node, route_to_agent
from .agents.browser_agent import browser_agent_node
from .agents.os_agent import os_agent_node
from .agents.research_agent import research_agent_node
from .agents.code_agent import code_agent_node
from .agents.data_agent import data_agent_node
from .agents.network_agent import network_agent_node
from .agents.sysadmin_agent import sysadmin_agent_node
from .agents.media_agent import media_agent_node
from .agents.package_agent import package_agent_node
from .agents.automation_agent import automation_agent_node
from .agents.planner_agent import planner_agent_node
from .agents.workflow_agent import workflow_agent_node
from .agents.retrospective_agent import retrospective_agent_node


def build_agent_graph(checkpointer: MemorySaver | None = None):
    """Build the multi-agent StateGraph.
    
    Creates a graph with:
    - supervisor: Routes tasks and synthesizes results
    - browser: Web navigation agent
    - os: Filesystem/shell agent
    - research: Multi-source research agent
    - code: Code analysis agent
    - data: Data transformation agent
    - network: Network diagnostics agent
    - sysadmin: System administration agent
    - media: Media processing agent
    - package: Package & dev environment agent
    - automation: Scheduling & reminders agent
    - workflow: n8n workflow integration agent
    
    Args:
        checkpointer: Optional memory saver for persistence
        
    Returns:
        Compiled StateGraph
    """
    # Create the graph
    graph = StateGraph(AgentState)
    
    # Add nodes - original agents
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("browser", browser_agent_node)
    graph.add_node("os", os_agent_node)
    graph.add_node("research", research_agent_node)
    graph.add_node("code", code_agent_node)
    
    # Add nodes - new agents
    graph.add_node("data", data_agent_node)
    graph.add_node("network", network_agent_node)
    graph.add_node("sysadmin", sysadmin_agent_node)
    graph.add_node("media", media_agent_node)
    graph.add_node("package", package_agent_node)
    graph.add_node("automation", automation_agent_node)
    graph.add_node("workflow", workflow_agent_node)
    graph.add_node("retrospective", retrospective_agent_node)
    
    # Planning-First: Planner runs first
    graph.add_node("planner", planner_agent_node)
    
    # Set entry point to PLANNER (Planning-First Architecture)
    graph.set_entry_point("planner")
    
    # Planner always goes to supervisor after creating plan
    graph.add_edge("planner", "supervisor")
    
    # Add conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "browser": "browser",
            "os": "os",
            "research": "research",
            "code": "code",
            "data": "data",
            "network": "network",
            "sysadmin": "sysadmin",
            "media": "media",
            "package": "package",
            "automation": "automation",
            "workflow": "workflow",
            "retrospective": "retrospective",
            "__end__": END,
        }
    )
    
    # Function to check if task is complete
    def check_task_complete(state: AgentState) -> str:
        if state.get("task_complete"):
            return "__end__"
        return "supervisor"
    
    # Worker agents: return to supervisor OR end if task complete
    # Original agents
    graph.add_conditional_edges("browser", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("os", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("research", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("code", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    
    # New agents
    graph.add_conditional_edges("data", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("network", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("sysadmin", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("media", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("package", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("automation", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("workflow", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    graph.add_conditional_edges("retrospective", check_task_complete, {"supervisor": "supervisor", "__end__": END})
    
    # Compile with optional checkpointer
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


class MultiAgentRunner:
    """High-level interface for running the multi-agent graph.
    
    Manages tool initialization and graph execution.
    """
    
    def __init__(
        self,
        config,
        browser_manager=None,
        os_tools=None,
        enable_checkpointing: bool = False,
    ):
        """Initialize the multi-agent runner.
        
        Args:
            config: AgentConfig instance
            browser_manager: Optional LazyBrowserManager for on-demand browser
            os_tools: Optional OSTools instance
            enable_checkpointing: Enable session persistence
        """
        import uuid
        from .tool_registry import ToolRegistry
        
        self.config = config
        self.browser_manager = browser_manager
        self.os_tools = os_tools
        self.enable_checkpointing = enable_checkpointing
        
        # Initialize persistence
        from .memory import SessionStore
        self.session_store = SessionStore() if enable_checkpointing else None
        
        # Initialize RecallTool
        from .run_history import RecallTool
        self.recall_tool = RecallTool(self.session_store) if self.session_store else None
        
        # Generate session ID and register tools
        self.session_id = str(uuid.uuid4())
        self._registry = ToolRegistry.get_instance()
        self._registry.register(
            self.session_id,
            config=config,
            browser_manager=browser_manager,
            os_tools=os_tools,
            recall_tool=self.recall_tool,
        )
        
        # Build graph
        # CRITICAL: MemorySaver causes hangs - checkpoints after every super-step
        # causing exponential memory growth. Disable by default.
        # See: https://langchain.com/ docs on checkpoint overhead
        checkpointer = None  # Disabled - was causing step 13+ hangs
        self.graph = build_agent_graph(checkpointer)
    
    def set_browser_manager(self, browser_manager) -> None:
        """Set browser manager after initialization."""
        self.browser_manager = browser_manager
        self._registry.update_browser_manager(self.session_id, browser_manager)
    
    def set_os_tools(self, os_tools) -> None:
        """Set OS tools after initialization."""
        self.os_tools = os_tools
        self._registry.update_os_tools(self.session_id, os_tools)
    
    def run(self, goal: str, max_steps: int = 30) -> dict[str, Any]:
        """Run the multi-agent system on a goal.
        
        Args:
            goal: User's goal/request
            max_steps: Maximum steps allowed
            
        Returns:
            Final state dict
        """
        # Create initial state with session_id for tool lookup
        initial_state = create_initial_state(
            goal=goal,
            max_steps=max_steps,
            session_id=self.session_id,
        )
        
        # Run the graph with thread_id for checkpointer
        try:
            final_state = self.graph.invoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": self.session_id,
                    }
                },
            )
            return final_state
        except Exception as e:
            error_msg = str(e).lower()
            # Handle empty response errors - model returned nothing
            empty_patterns = ["empty", "must contain", "output text", "tool calls", "cannot both be empty"]
            if any(p in error_msg for p in empty_patterns):
                print(f"[GRAPH] ⚠️ Model returned empty response: {e}")
                return {
                    **initial_state,
                    "task_complete": True,
                    "final_answer": "Model returned empty response - please try a different model or restart LM Studio",
                    "error": str(e),
                }
            raise
    
    def stream(self, goal: str, max_steps: int = 30):
        """Stream execution steps for real-time updates.
        
        Args:
            goal: User's goal/request
            max_steps: Maximum steps allowed
            
        Yields:
            State updates as they occur
        """
        # Create initial state with session_id for tool lookup
        initial_state = create_initial_state(
            goal=goal,
            max_steps=max_steps,
            session_id=self.session_id,
        )
        
        try:
            # Create session in DB
            if self.session_store:
                self.session_store.create_session(
                    self.session_id, 
                    goal, 
                    {**initial_state, "session_id": self.session_id}
                )
            
            # We need an LLM client.
            from .agents.base import create_llm_client
            
            # Get config from registry
            for event in self.graph.stream(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": self.session_id,
                    },
                    "recursion_limit": 50,  # Allow more agent interactions
                },
            ):
                # PERSISTENCE: Update session store (THROTTLED for performance)
                if self.session_store:
                    for node_name, node_state in event.items():
                        # THROTTLE: Only persist every 3 steps or on completion
                        step_count = node_state.get("step_count", 0)
                        is_finished = node_state.get("task_complete", False)
                        
                        if is_finished or step_count % 3 == 0 or step_count == 1:
                            self.session_store.update_session(self.session_id, node_state)
                        
                        # Log step if applicable
                        if "messages" in node_state and len(node_state["messages"]) > 0:
                            last_msg = node_state["messages"][-1]
                            agent_name = node_name
                            step_num = node_state.get("step_count", 0)
                            
                            # VECTOR DB: Persist message to separate table
                            role = type(last_msg).__name__.replace("Message", "").lower()  # SystemMessage -> system
                            content = str(last_msg.content) if hasattr(last_msg, 'content') else str(last_msg)
                            try:
                                self.session_store.add_message(self.session_id, role, content, step_num)
                            except Exception as e:
                                print(f"[SESSION] Message save failed: {e}")
                            
                            # Log tool calls or text
                            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                                for tc in last_msg.tool_calls:
                                    self.session_store.add_step(
                                        self.session_id,
                                        step_num,
                                        agent_name,
                                        tc.get("name"),
                                        tc.get("args"),
                                        None # Result comes later
                                    )
                            else:
                                content = str(last_msg.content)
                                # Log simplified thought
                                self.session_store.add_step(
                                    self.session_id,
                                    step_num,
                                    agent_name,
                                    "think",
                                    {"text": content[:200]},
                                    None
                                )


                
                yield event
            
            if self.session_store:
                self.session_store.close()
                
        except Exception as e:
            error_msg = str(e).lower()
            if self.session_store:
                 # Update session with error
                 try:
                     self.session_store.update_session(self.session_id, {"error": str(e), "task_complete": True})
                 except:
                     pass
            


            # Handle empty response errors from Anthropic/OpenAI
            if "empty" in error_msg or "must contain" in error_msg:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Graph stream error (empty response): {e}")
                # Yield a final state with error message
                yield {
                    "supervisor": {
                        "task_complete": True,
                        "final_answer": "Model returned an empty response. Please try again.",
                        "error": str(e),
                    }
                }
            else:
                raise
    
    def cleanup(self) -> None:
        """Unregister tools from registry."""
        self._registry.unregister(self.session_id)
    
    def get_result(self, state: dict) -> str:
        """Extract final answer from state.
        
        Args:
            state: Final state dict
            
        Returns:
            Final answer string
        """
        if state.get("final_answer"):
            return state["final_answer"]
        
        if state.get("extracted_data"):
            import json
            return f"Collected data: {json.dumps(state['extracted_data'], indent=2)}"
        
        # Fall through to check for results in messages
        messages = state.get("messages", [])
        if messages:
            return str(messages[-1].content) if hasattr(messages[-1], 'content') else str(messages[-1])
        return "No result"
    

