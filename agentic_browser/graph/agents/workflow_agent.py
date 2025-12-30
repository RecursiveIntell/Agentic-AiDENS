"""
n8n Workflow Agent for external integrations.

Handles communication with n8n webhooks and API.
"""

import json
import logging
import requests
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState

logger = logging.getLogger("agentic_browser.agents.workflow")

class WorkflowAgentNode(BaseAgent):
    """Specialized agent for n8n workflow integration.
    
    Acts as a bridge between the agent system and n8n workflows,
    translating high-level intents into webhook calls.
    """
    
    AGENT_NAME = "workflow"
    MAX_STEPS_PER_INVOCATION = 5
    
    SYSTEM_PROMPT = """You are a WORKFLOW agent. You integrate with n8n to automate external tasks.

Available actions:
- trigger_webhook: { "name": "email_summary", "data": { "subject": "Daily Report", "body": "..." } }
- list_webhooks: {}  # Show available configured webhooks
- check_health: {}   # Check n8n connectivity
- done: { "summary": "what was triggered" }

NOTES:
1. You function as a bridge to external tools via n8n.
2. If a specific webhook isn't listed, check list_webhooks or ask the user to configure it.
3. Handle errors gracefully - if n8n is down, report it without crashing.

Respond with JSON:
{
  "action": "trigger_webhook|list_webhooks|...",
  "args": { ... },
  "rationale": "brief reason"
}"""

    def __init__(self, config):
        """Initialize workflow agent."""
        super().__init__(config)
        
        # Load n8n settings here to ensure we have latest config
        # We access the raw settings store to get the n8n config
        try:
            from ...settings_store import get_settings
            self.settings = get_settings()
        except ImportError:
            self.settings = None

    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _empty_response_fallback(self, error: str | None = None) -> dict:
        summary = "Model returned empty response"
        if error:
            summary = f"{summary}: {error}"
        return {
            "action": "done",
            "args": {"summary": summary},
            "rationale": "Fallback response",
        }
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute workflow task."""
        # 1. State Context construction
        task_context = f"""
WORKFLOW TASK: {state['goal']}

n8n Configured: {'Yes' if self._is_configured() else 'No (Missing URL)'}
Available Webhooks: {list(self.settings.n8n_webhooks.keys()) if self._is_configured() else []}

Data collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            # 2. Planning/Decision (LLM)
            response = self.safe_invoke(messages)
            action_data = self._parse_action(response.content)
            
            action = action_data.get("action", "")
            args = action_data.get("args", {})
            
            # 3. Execution
            if action == "done":
                summary = args.get("summary", "Workflow task completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"workflow_result": summary},
                )
            
            result = self._execute_action(action, args)
            
            # 4. Result Processing
            tool_msg = HumanMessage(content=f"Tool '{action}' output:\n{result['message'][:2000]}")
            
            extracted = None
            if result['success'] and result.get('data'):
                extracted = {f"workflow_{action}": str(result['data'])[:1500]}
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                extracted_data=extracted,
                error=result['message'] if not result['success'] else None,
            )
            
        except Exception as e:
            logger.exception("Workflow agent error")
            return self._update_state(
                state,
                error=f"Workflow agent error: {str(e)}",
            )
    
    def _execute_action(self, action: str, args: dict) -> dict:
        """Execute a workflow action with broad error handling."""
        handlers = {
            "trigger_webhook": self._trigger_webhook,
            "list_webhooks": self._list_webhooks,
            "check_health": self._check_health,
        }
        
        handler = handlers.get(action)
        if not handler:
            return {"success": False, "message": f"Unknown action: {action}"}
        
        try:
            return handler(args)
        except Exception as e:
            return {"success": False, "message": f"Error executing {action}: {str(e)}"}

    def _trigger_webhook(self, args: dict) -> dict:
        """Trigger a specific n8n webhook."""
        if not self._is_configured():
            return {
                "success": False, 
                "message": "n8n is not configured. Please set 'n8n_url' in settings."
            }
            
        name = args.get("name")
        data = args.get("data", {})
        
        # 1. Resolve URL
        url = self.settings.n8n_webhooks.get(name)
        if not url:
            # Fallback: Check if user provided a raw URL in 'name' (edge case)
            if name and (name.startswith("http://") or name.startswith("https://")):
                url = name
            else:
                 return {
                    "success": False, 
                    "message": f"Webhook '{name}' not found. Available: {list(self.settings.n8n_webhooks.keys())}"
                }
        
        # 2. Execute Request with Timeout
        try:
            logger.info(f"Triggering webhook {name} at {url}")
            response = requests.post(
                url, 
                json=data, 
                timeout=10,  # 10s default timeout
                headers={"Content-Type": "application/json"}
            )
            
            # 3. Handle Response
            # Even 4xx/5xx responses might contain useful n8n error info
            success = 200 <= response.status_code < 300
            
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"raw_text": response.text}
                if success and not response.text:
                   # Empty success is common for webhooks (204 No Content)
                   response_data = {"status": "triggered"}

            return {
                "success": success,
                "message": f"Webhook triggered. Status: {response.status_code}. Response: {str(response_data)[:200]}",
                "data": response_data
            }
            
        except requests.Timeout:
            # Edge Case: Long running workflow
            # We assume it triggered but timed out waiting for response
            return {
                "success": True, # Soft success for async triggers
                "message": "Webhook triggered but timed out waiting for response ( >10s). Workflow likely running asynchronously.",
                "data": {"status": "async_timeout"}
            }
        except requests.RequestException as e:
            return {"success": False, "message": f"Connection error: {str(e)}"}

    def _list_webhooks(self, args: dict) -> dict:
        """List configured webhooks."""
        if not self._is_configured():
             return {
                "success": False, 
                "message": "n8n is not configured."
            }
            
        return {
            "success": True,
            "message": f"Configured webhooks: {list(self.settings.n8n_webhooks.keys())}",
            "data": {"webhooks": self.settings.n8n_webhooks}
        }
        
    def _check_health(self, args: dict) -> dict:
        """Check n8n instance health."""
        if not self._is_configured():
             return {
                "success": False, 
                "message": "n8n is not configured."
            }
            
        # Try hitting base URL or a health endpoint if known
        # Usually n8n exposes /healthz
        base_url = self.settings.n8n_url.rstrip("/")
        health_url = f"{base_url}/healthz"
        
        try:
            resp = requests.get(health_url, timeout=5)
            return {
                "success": resp.status_code == 200,
                "message": f"Health check: {resp.status_code}. {resp.text}",
                "data": {"status_code": resp.status_code}
            }
        except Exception as e:
             return {"success": False, "message": f"Health check failed: {str(e)}"}

    def _is_configured(self) -> bool:
        """Check if minimal n8n config is present."""
        if not self.settings: return False
        # Need at least a URL or defined webhooks
        return bool(self.settings.n8n_url or self.settings.n8n_webhooks)

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
            return {"action": "done", "args": {"summary": "Unable to parse response"}}


def workflow_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for workflow agent."""
    from ..tool_registry import get_tools
    
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig(goal=state['goal'])
    
    agent = WorkflowAgentNode(agent_config)
    return agent.execute(state)
