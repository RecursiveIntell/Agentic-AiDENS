"""
Research Agent for multi-source information gathering.

Coordinates browser actions to research topics from multiple sources.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class ResearchAgentNode(BaseAgent):
    """Specialized agent for research tasks.
    
    Coordinates browser operations to gather information from
    multiple sources and synthesize findings.
    """
    
    AGENT_NAME = "research"
    MAX_STEPS_PER_INVOCATION = 8
    
    SYSTEM_PROMPT = """You are a RESEARCH agent. Find information using web search.

MANDATORY WORKFLOW:
1. FIRST: goto "https://duckduckgo.com"
2. SECOND: type {"selector": "input[name='q']", "text": "your search query"} and press Enter
3. THIRD: Extract search results to find URLs
4. FOURTH: Visit 1-2 real sites from search results
5. FINALLY: Call "done" with synthesized findings

Available actions:
- goto: { "url": "https://..." } - Navigate to URL
- type: { "selector": "input[name='q']", "text": "search query" } - Type in search box
- press: { "key": "Enter" } - Submit search
- extract_visible_text: { "max_chars": 5000 } - Extract page content
- done: { "summary": "YOUR COMPREHENSIVE REPORT" } - Finish with report

CRITICAL RULES:
1. START at DuckDuckGo - DO NOT make up URLs like "petdiaryapp.com"!
2. Only visit URLs you SEE in search results
3. If you get ERR_NAME_NOT_RESOLVED or 404, call "done" with what you have
4. After 3-5 actions, you MUST call "done" with a proper summary
5. The "summary" in done should be a FULL REPORT, not raw data

ERROR HANDLING:
- If a URL fails, skip it and synthesize from what you have
- Don't retry failed URLs - just complete with available info

Respond with JSON:
{
  "action": "goto|type|press|extract_visible_text|done",
  "args": { ... },
  "rationale": "brief reason"
}"""

    def __init__(self, config, browser_tools=None):
        """Initialize research agent.
        
        Args:
            config: Agent configuration
            browser_tools: Browser tools for web access
        """
        super().__init__(config)
        self._browser_tools = browser_tools
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def set_browser_tools(self, browser_tools) -> None:
        """Set browser tools after initialization."""
        self._browser_tools = browser_tools
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute research workflow.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with research findings
        """
        if not self._browser_tools:
            return self._update_state(
                state,
                error="Browser tools not available for research",
            )
        
        # Get current page state
        try:
            page_state = self._browser_tools.get_page_state()
        except Exception:
            page_state = {}
        
        # Build context showing progress
        sources_visited = len([u for u in state['visited_urls'] if u and 'duckduckgo' not in u])
        
        task_context = f"""
RESEARCH TASK: {state['goal']}

Sources visited so far: {sources_visited}/3
Visited URLs: {chr(10).join(state['visited_urls'][-5:]) or '(none)'}

Current page: {page_state.get('title', 'Unknown')}
URL: {page_state.get('url', 'about:blank')}

Visible content (truncated):
{page_state.get('visible_text', '')[:2000]}

Data collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}

REMINDER: Visit 2-3 actual sources, then synthesize with "done".
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.llm.invoke(messages)
            action_data = self._parse_action(response.content)
            
            if action_data.get("action") == "done":
                # Research agent CAN mark complete - it's typically the final step
                summary = action_data.get("args", {}).get("summary", "Research completed")
                return self._update_state(
                    state,
                    message=AIMessage(content=response.content),
                    task_complete=True,
                    final_answer=summary,
                    extracted_data={"research_findings": summary},
                )
            
            # Execute browser action
            result = self._browser_tools.execute(
                action_data.get("action", ""),
                action_data.get("args", {}),
            )
            
            visited = None
            if action_data.get("action") == "goto":
                visited = action_data.get("args", {}).get("url")
            
            extracted = None
            if action_data.get("action") == "extract_visible_text" and result.success:
                key = f"research_source_{sources_visited + 1}"
                extracted = {key: result.data[:1000] if result.data else result.message[:500]}
            
            return self._update_state(
                state,
                message=AIMessage(content=response.content),
                visited_url=visited,
                extracted_data=extracted,
                error=result.message if not result.success else None,
            )
            
        except Exception as e:
            return self._update_state(
                state,
                error=f"Research agent error: {str(e)}",
            )
    
    def _parse_action(self, response: str) -> dict:
        """Parse LLM response into action dict."""
        try:
            content = response.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {"action": "done", "args": {"summary": "Failed to parse action"}}


def research_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for research agent."""
    from ..tool_registry import get_tools
    
    # Get tools from registry using session_id
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
        browser_tools = tools.browser_tools
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig()
        browser_tools = None
    
    agent = ResearchAgentNode(agent_config, browser_tools)
    return agent.execute(state)


