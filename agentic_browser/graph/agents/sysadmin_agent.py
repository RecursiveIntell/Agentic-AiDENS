"""
System Admin Agent for system monitoring and service management.

Handles systemctl, process management, disk/memory stats, and logs.
"""

import json
import subprocess
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class SysAdminAgentNode(BaseAgent):
    """Specialized agent for system administration tasks.
    
    Provides system monitoring and management capabilities
    with strict safety controls.
    """
    
    AGENT_NAME = "sysadmin"
    MAX_STEPS_PER_INVOCATION = 10
    
    # Services that cannot be managed
    PROTECTED_SERVICES = {
        "systemd", "dbus", "NetworkManager", "gdm", "sddm", "lightdm",
        "sshd", "systemd-journald", "systemd-logind", "polkit",
    }
    
    SYSTEM_PROMPT = """You are a SYSTEM ADMIN agent. You monitor and manage Linux systems.

Available actions:
- service_status: { "service": "nginx" }
- service_control: { "service": "docker", "action": "start|stop|restart" }  # HIGH RISK
- process_list: { "filter": "python" }  # Optional filter
- disk_usage: { "path": "/" }
- memory_usage: {}
- cpu_usage: {}
- journal_logs: { "unit": "nginx", "lines": 50 }
- uptime: {}
- done: { "summary": "system status report" }

SAFETY RULES:
1. NEVER manage critical system services (systemd, dbus, NetworkManager, sshd, gdm)
2. Service start/stop/restart requires HIGH risk confirmation
3. Cannot kill system processes (PID < 1000)
4. Read-only operations (status, logs, usage) are safe

BLOCKED ACTIONS (will be rejected):
- Modifying systemd, dbus, NetworkManager, display managers
- Killing processes with PID < 1000
- sudo commands that modify system files

Respond with JSON:
{
  "action": "service_status|process_list|...|done",
  "args": { ... },
  "rationale": "brief reason",
  "risk": "low|medium|high"
}"""

    def __init__(self, config):
        """Initialize sysadmin agent."""
        super().__init__(config)
    
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
        """Execute system administration task."""
        task_context = f"""
SYSADMIN TASK: {state['goal']}

Data collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}

REMINDER: You cannot manage protected services: {', '.join(list(self.PROTECTED_SERVICES)[:5])}...
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.safe_invoke(messages)
            action_data = self._parse_action(response.content)
            
            action = action_data.get("action", "")
            args = action_data.get("args", {})
            
            if action == "done":
                summary = args.get("summary", "System admin task completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"sysadmin_result": summary},
                )
            
            # Execute the sysadmin action
            result = self._execute_action(action, args)
            
            tool_msg = HumanMessage(content=f"Tool '{action}' output:\n{result['message'][:2000]}")
            
            extracted = None
            if result['success'] and result.get('data'):
                extracted = {f"sysadmin_{action}": str(result['data'])[:1500]}
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                extracted_data=extracted,
                error=result['message'] if not result['success'] else None,
            )
            
        except Exception as e:
            return self._update_state(
                state,
                error=f"SysAdmin agent error: {str(e)}",
            )
    
    def _execute_action(self, action: str, args: dict) -> dict:
        """Execute a sysadmin action."""
        handlers = {
            "service_status": self._service_status,
            "service_control": self._service_control,
            "process_list": self._process_list,
            "disk_usage": self._disk_usage,
            "memory_usage": self._memory_usage,
            "cpu_usage": self._cpu_usage,
            "journal_logs": self._journal_logs,
            "uptime": self._uptime,
        }
        
        handler = handlers.get(action)
        if not handler:
            return {"success": False, "message": f"Unknown action: {action}"}
        
        try:
            return handler(args)
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _service_status(self, args: dict) -> dict:
        """Get service status."""
        service = args.get("service", "")
        
        if not service:
            return {"success": False, "message": "Service name is required"}
        
        cmd = ["systemctl", "status", service, "--no-pager"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        # Parse status
        is_active = "active (running)" in result.stdout.lower()
        
        return {
            "success": True,
            "message": result.stdout[:1500],
            "data": {"service": service, "active": is_active}
        }
    
    def _service_control(self, args: dict) -> dict:
        """Control a service (start/stop/restart)."""
        service = args.get("service", "")
        action = args.get("action", "")
        
        if not service or not action:
            return {"success": False, "message": "Service and action are required"}
        
        if action not in ("start", "stop", "restart"):
            return {"success": False, "message": "Action must be start, stop, or restart"}
        
        # Safety check
        if service.lower() in (s.lower() for s in self.PROTECTED_SERVICES):
            return {"success": False, "message": f"BLOCKED: Cannot manage protected service '{service}'"}
        
        # This requires sudo, which will prompt for password
        cmd = ["systemctl", action, service]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            # Try with sudo (will fail without tty, but log intent)
            return {
                "success": False,
                "message": f"Service control requires sudo. Run manually: sudo systemctl {action} {service}",
                "data": {"command": f"sudo systemctl {action} {service}"}
            }
        
        return {
            "success": True,
            "message": f"Service {service} {action}ed successfully",
            "data": {"service": service, "action": action}
        }
    
    def _process_list(self, args: dict) -> dict:
        """List processes."""
        filter_str = args.get("filter", "")
        
        cmd = ["ps", "aux", "--sort=-%cpu"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        lines = result.stdout.split('\n')
        
        if filter_str:
            # Filter lines containing the filter string
            filtered = [lines[0]]  # Keep header
            filtered.extend([l for l in lines[1:] if filter_str.lower() in l.lower()])
            lines = filtered
        
        return {
            "success": True,
            "message": '\n'.join(lines[:30]),  # Top 30 processes
            "data": {"count": len(lines) - 1, "filter": filter_str or "none"}
        }
    
    def _disk_usage(self, args: dict) -> dict:
        """Get disk usage."""
        path = args.get("path", "/")
        
        cmd = ["df", "-h", path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        return {
            "success": True,
            "message": result.stdout,
            "data": {"path": path}
        }
    
    def _memory_usage(self, args: dict) -> dict:
        """Get memory usage."""
        cmd = ["free", "-h"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        return {
            "success": True,
            "message": result.stdout,
            "data": {}
        }
    
    def _cpu_usage(self, args: dict) -> dict:
        """Get CPU usage."""
        cmd = ["top", "-bn1", "-o", "%CPU"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        # Get just the summary and top processes
        lines = result.stdout.split('\n')[:15]
        
        return {
            "success": True,
            "message": '\n'.join(lines),
            "data": {}
        }
    
    def _journal_logs(self, args: dict) -> dict:
        """Get systemd journal logs."""
        unit = args.get("unit", "")
        lines = min(args.get("lines", 50), 200)  # Max 200 lines
        
        cmd = ["journalctl", "-n", str(lines), "--no-pager"]
        if unit:
            cmd.extend(["-u", unit])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        return {
            "success": True,
            "message": result.stdout[:3000],
            "data": {"unit": unit or "all", "lines": lines}
        }
    
    def _uptime(self, args: dict) -> dict:
        """Get system uptime."""
        cmd = ["uptime"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        return {
            "success": True,
            "message": result.stdout.strip(),
            "data": {}
        }
    
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


def sysadmin_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for sysadmin agent."""
    from ..tool_registry import get_tools
    
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig(goal=state['goal'])
    
    agent = SysAdminAgentNode(agent_config)
    return agent.execute(state)
