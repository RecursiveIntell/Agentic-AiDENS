"""
Network & API Agent for network diagnostics and HTTP testing.

Handles ping, DNS lookup, port scanning, and HTTP requests.
"""

import json
import socket
import subprocess
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class NetworkAgentNode(BaseAgent):
    """Specialized agent for network diagnostics and API testing.
    
    Provides network troubleshooting and HTTP testing capabilities
    with appropriate safety controls.
    """
    
    AGENT_NAME = "network"
    MAX_STEPS_PER_INVOCATION = 10
    
    SYSTEM_PROMPT = """You are a NETWORK agent. You diagnose network issues and test APIs.

Available actions:
- ping: { "host": "google.com", "count": 4 }
- dns_lookup: { "domain": "google.com", "type": "A" }  # A, AAAA, MX, TXT, NS
- traceroute: { "host": "example.com" }
- port_check: { "host": "localhost", "port": 80 }
- netstat_listen: {}  # Show listening ports
- http_get: { "url": "https://api.example.com/status" }
- http_post: { "url": "...", "body": {...}, "headers": {...} }
- ssl_check: { "domain": "example.com" }
- done: { "summary": "network analysis results" }

SAFETY RULES:
1. Only port scan localhost or explicitly owned servers
2. Don't send credentials in HTTP requests without user confirmation
3. Rate limit external requests (max 10 per session)
4. Never access internal/private IP ranges without confirmation

Respond with JSON:
{
  "action": "ping|dns_lookup|...|done",
  "args": { ... },
  "rationale": "brief reason"
}"""

    def __init__(self, config):
        """Initialize network agent."""
        super().__init__(config)
        self._request_count = 0
        self._max_requests = 10
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute network diagnostics."""
        task_context = f"""
NETWORK TASK: {state['goal']}

Requests made this session: {self._request_count}/{self._max_requests}

Data collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.safe_invoke(messages)
            action_data = self._parse_action(response.content)
            
            action = action_data.get("action", "")
            args = action_data.get("args", {})
            
            if action == "done":
                summary = args.get("summary", "Network analysis completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"network_result": summary},
                )
            
            # Execute the network action
            result = self._execute_action(action, args)
            
            tool_msg = HumanMessage(content=f"Tool '{action}' output:\n{result['message'][:2000]}")
            
            extracted = None
            if result['success'] and result.get('data'):
                extracted = {f"network_{action}": str(result['data'])[:1500]}
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                extracted_data=extracted,
                error=result['message'] if not result['success'] else None,
            )
            
        except Exception as e:
            return self._update_state(
                state,
                error=f"Network agent error: {str(e)}",
            )
    
    def _execute_action(self, action: str, args: dict) -> dict:
        """Execute a network action."""
        handlers = {
            "ping": self._ping,
            "dns_lookup": self._dns_lookup,
            "traceroute": self._traceroute,
            "port_check": self._port_check,
            "netstat_listen": self._netstat_listen,
            "http_get": self._http_get,
            "http_post": self._http_post,
            "ssl_check": self._ssl_check,
        }
        
        handler = handlers.get(action)
        if not handler:
            return {"success": False, "message": f"Unknown action: {action}"}
        
        try:
            return handler(args)
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _ping(self, args: dict) -> dict:
        """Ping a host."""
        host = args.get("host", "")
        count = min(args.get("count", 4), 10)  # Max 10 pings
        
        if not host:
            return {"success": False, "message": "Host is required"}
        
        cmd = ["ping", "-c", str(count), host]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        return {
            "success": result.returncode == 0,
            "message": result.stdout if result.returncode == 0 else result.stderr,
            "data": {"host": host, "reachable": result.returncode == 0}
        }
    
    def _dns_lookup(self, args: dict) -> dict:
        """DNS lookup."""
        domain = args.get("domain", "")
        record_type = args.get("type", "A")
        
        if not domain:
            return {"success": False, "message": "Domain is required"}
        
        cmd = ["dig", "+short", domain, record_type]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        records = [r for r in result.stdout.strip().split('\n') if r]
        
        return {
            "success": True,
            "message": f"DNS {record_type} records for {domain}: {records}",
            "data": {"domain": domain, "type": record_type, "records": records}
        }
    
    def _traceroute(self, args: dict) -> dict:
        """Traceroute to host."""
        host = args.get("host", "")
        
        if not host:
            return {"success": False, "message": "Host is required"}
        
        cmd = ["traceroute", "-m", "15", host]  # Max 15 hops
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        return {
            "success": True,
            "message": result.stdout[:2000],
            "data": {"host": host}
        }
    
    def _port_check(self, args: dict) -> dict:
        """Check if a port is open."""
        host = args.get("host", "localhost")
        port = args.get("port", 80)
        
        # Safety: Only allow localhost or common public hosts
        if host not in ("localhost", "127.0.0.1") and not host.endswith(('.com', '.org', '.net', '.io')):
            return {"success": False, "message": "Port check only allowed on localhost or public domains"}
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            is_open = result == 0
            return {
                "success": True,
                "message": f"Port {port} on {host} is {'OPEN' if is_open else 'CLOSED'}",
                "data": {"host": host, "port": port, "open": is_open}
            }
        except socket.error as e:
            return {"success": False, "message": f"Socket error: {e}"}
    
    def _netstat_listen(self, args: dict) -> dict:
        """Show listening ports."""
        cmd = ["ss", "-tlnp"]  # TCP, listening, numeric, show process
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        return {
            "success": True,
            "message": result.stdout[:2000],
            "data": {"output": result.stdout}
        }
    
    def _http_get(self, args: dict) -> dict:
        """HTTP GET request."""
        url = args.get("url", "")
        
        if not url:
            return {"success": False, "message": "URL is required"}
        
        if self._request_count >= self._max_requests:
            return {"success": False, "message": "Request limit reached"}
        
        self._request_count += 1
        
        cmd = ["curl", "-s", "-w", "\n%{http_code}", "-L", "--max-time", "10", url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        lines = result.stdout.strip().split('\n')
        status_code = lines[-1] if lines else "0"
        body = '\n'.join(lines[:-1])[:1500]
        
        return {
            "success": status_code.startswith('2'),
            "message": f"HTTP {status_code}: {body[:500]}",
            "data": {"status": status_code, "body": body, "url": url}
        }
    
    def _http_post(self, args: dict) -> dict:
        """HTTP POST request."""
        url = args.get("url", "")
        body = args.get("body", {})
        headers = args.get("headers", {})
        
        if not url:
            return {"success": False, "message": "URL is required"}
        
        if self._request_count >= self._max_requests:
            return {"success": False, "message": "Request limit reached"}
        
        self._request_count += 1
        
        cmd = ["curl", "-s", "-X", "POST", "-w", "\n%{http_code}", "-L", "--max-time", "10"]
        cmd.extend(["-H", "Content-Type: application/json"])
        
        for k, v in headers.items():
            cmd.extend(["-H", f"{k}: {v}"])
        
        if body:
            cmd.extend(["-d", json.dumps(body)])
        
        cmd.append(url)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        lines = result.stdout.strip().split('\n')
        status_code = lines[-1] if lines else "0"
        response_body = '\n'.join(lines[:-1])[:1500]
        
        return {
            "success": status_code.startswith('2'),
            "message": f"HTTP {status_code}: {response_body[:500]}",
            "data": {"status": status_code, "body": response_body, "url": url}
        }
    
    def _ssl_check(self, args: dict) -> dict:
        """Check SSL certificate."""
        domain = args.get("domain", "")
        
        if not domain:
            return {"success": False, "message": "Domain is required"}
        
        cmd = ["openssl", "s_client", "-connect", f"{domain}:443", "-servername", domain]
        
        # Use timeout and prevent hanging
        result = subprocess.run(
            cmd,
            input="",
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Parse certificate info
        output = result.stdout + result.stderr
        
        # Extract expiry
        expiry = "Unknown"
        if "notAfter" in output:
            for line in output.split('\n'):
                if "notAfter" in line:
                    expiry = line.split('=')[-1].strip()
                    break
        
        return {
            "success": "Verify return code: 0" in output,
            "message": f"SSL certificate for {domain} expires: {expiry}",
            "data": {"domain": domain, "expiry": expiry, "valid": "Verify return code: 0" in output}
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


def network_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for network agent."""
    from ..tool_registry import get_tools
    
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig(goal=state['goal'])
    
    agent = NetworkAgentNode(agent_config)
    return agent.execute(state)
