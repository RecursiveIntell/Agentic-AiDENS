"""
Package & Development Agent for package management and project setup.

Handles pip, venv, apt searching, and project scaffolding.
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class PackageAgentNode(BaseAgent):
    """Specialized agent for package management and development setup.
    
    Handles virtual environments, pip packages, and project scaffolding
    with safety constraints.
    """
    
    AGENT_NAME = "package"
    MAX_STEPS_PER_INVOCATION = 10
    
    SYSTEM_PROMPT = """You are a PACKAGE/DEVELOPMENT agent. You manage packages and set up projects.

Available actions:
- venv_create: { "path": "./venv", "python": "python3" }
- venv_info: { "path": "./venv" }  # Check if venv exists and what's installed
- pip_install: { "packages": ["requests", "flask"], "venv": "./venv" }  # MUST specify venv
- pip_list: { "venv": "./venv" }
- pip_search: { "query": "requests" }  # Search PyPI (limited)
- apt_search: { "query": "python3" }  # Search apt packages (read-only)
- git_clone: { "url": "https://github.com/...", "dest": "./" }
- requirements_parse: { "path": "./requirements.txt" }  # Parse requirements file
- done: { "summary": "what was set up" }

SAFETY RULES:
1. pip install ONLY in virtual environments - never system-wide
2. apt install requires sudo (show command, don't execute)
3. git clone only from https:// URLs
4. Never install from untrusted sources
5. Check venv exists before installing packages

Respond with JSON:
{
  "action": "venv_create|pip_install|...|done",
  "args": { ... },
  "rationale": "brief reason"
}"""

    def __init__(self, config):
        """Initialize package agent."""
        super().__init__(config)
        self._home_dir = Path.home()
    
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
        """Execute package management task."""
        task_context = f"""
PACKAGE TASK: {state['goal']}

Working directory: {Path.cwd()}

Data collected:
{json.dumps(state['extracted_data'], indent=2)[:1000]}

REMINDER: pip install only works in virtual environments!
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.safe_invoke(messages)
            action_data = self._parse_action(response.content)
            
            action = action_data.get("action", "")
            args = action_data.get("args", {})
            
            if action == "done":
                summary = args.get("summary", "Package task completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"package_result": summary},
                )
            
            # Execute the package action
            result = self._execute_action(action, args)
            
            tool_msg = HumanMessage(content=f"Tool '{action}' output:\n{result['message'][:2000]}")
            
            extracted = None
            if result['success'] and result.get('data'):
                extracted = {f"package_{action}": str(result['data'])[:1500]}
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                extracted_data=extracted,
                error=result['message'] if not result['success'] else None,
            )
            
        except Exception as e:
            return self._update_state(
                state,
                error=f"Package agent error: {str(e)}",
            )
    
    def _execute_action(self, action: str, args: dict) -> dict:
        """Execute a package action."""
        handlers = {
            "venv_create": self._venv_create,
            "venv_info": self._venv_info,
            "pip_install": self._pip_install,
            "pip_list": self._pip_list,
            "pip_search": self._pip_search,
            "apt_search": self._apt_search,
            "git_clone": self._git_clone,
            "requirements_parse": self._requirements_parse,
        }
        
        handler = handlers.get(action)
        if not handler:
            return {"success": False, "message": f"Unknown action: {action}"}
        
        try:
            return handler(args)
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def _validate_path(self, path: str, must_exist: bool = True) -> Path:
        """Validate and resolve a path safely."""
        p = Path(path).expanduser().resolve()
        
        # Must be under home directory, cwd, or /tmp
        cwd = Path.cwd().resolve()
        allowed = (str(self._home_dir), str(cwd), '/tmp')
        if not any(str(p).startswith(a) for a in allowed):
            raise ValueError(f"Path must be under home, cwd, or /tmp: {path}")
        
        if must_exist and not p.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        return p
    
    def _venv_create(self, args: dict) -> dict:
        """Create a virtual environment."""
        venv_path = self._validate_path(args.get("path", "./venv"), must_exist=False)
        python = args.get("python", "python3")
        
        if venv_path.exists():
            return {"success": False, "message": f"Venv already exists at {venv_path}"}
        
        cmd = [python, "-m", "venv", str(venv_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {"success": False, "message": f"Failed to create venv: {result.stderr}"}
        
        return {
            "success": True,
            "message": f"Created virtual environment at {venv_path}",
            "data": {"path": str(venv_path), "activate": f"source {venv_path}/bin/activate"}
        }
    
    def _venv_info(self, args: dict) -> dict:
        """Get venv info."""
        venv_path = self._validate_path(args.get("path", "./venv"))
        
        pip_path = venv_path / "bin" / "pip"
        if not pip_path.exists():
            return {"success": False, "message": f"Not a valid venv: {venv_path}"}
        
        # Get python version
        python_path = venv_path / "bin" / "python"
        result = subprocess.run([str(python_path), "--version"], capture_output=True, text=True)
        python_version = result.stdout.strip()
        
        # Get installed packages count
        result = subprocess.run([str(pip_path), "list", "--format=json"], capture_output=True, text=True)
        try:
            packages = json.loads(result.stdout)
            pkg_count = len(packages)
        except:
            pkg_count = "unknown"
        
        return {
            "success": True,
            "message": f"Venv at {venv_path}: {python_version}, {pkg_count} packages",
            "data": {"path": str(venv_path), "python": python_version, "packages": pkg_count}
        }
    
    def _pip_install(self, args: dict) -> dict:
        """Install packages in a venv."""
        packages = args.get("packages", [])
        venv = args.get("venv", "")
        
        if not packages:
            return {"success": False, "message": "No packages specified"}
        
        if not venv:
            return {"success": False, "message": "SAFETY: Must specify venv path. System-wide install not allowed."}
        
        venv_path = self._validate_path(venv)
        pip_path = venv_path / "bin" / "pip"
        
        if not pip_path.exists():
            return {"success": False, "message": f"Not a valid venv (no pip): {venv}"}
        
        # Install packages
        cmd = [str(pip_path), "install"] + packages
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return {"success": False, "message": f"pip install failed: {result.stderr[:500]}"}
        
        return {
            "success": True,
            "message": f"Installed {len(packages)} packages: {', '.join(packages)}",
            "data": {"packages": packages, "venv": str(venv_path)}
        }
    
    def _pip_list(self, args: dict) -> dict:
        """List installed packages."""
        venv = args.get("venv", "")
        
        if not venv:
            return {"success": False, "message": "Must specify venv path"}
        
        venv_path = self._validate_path(venv)
        pip_path = venv_path / "bin" / "pip"
        
        if not pip_path.exists():
            return {"success": False, "message": f"Not a valid venv: {venv}"}
        
        result = subprocess.run([str(pip_path), "list"], capture_output=True, text=True, timeout=30)
        
        return {
            "success": True,
            "message": result.stdout[:2000],
            "data": {"venv": str(venv_path)}
        }
    
    def _pip_search(self, args: dict) -> dict:
        """Search PyPI (via pip index)."""
        query = args.get("query", "")
        
        if not query:
            return {"success": False, "message": "Query is required"}
        
        # pip search is disabled, use pip index versions as workaround
        cmd = ["pip", "index", "versions", query]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        if result.returncode != 0:
            return {
                "success": False, 
                "message": f"Package '{query}' not found or pip index failed. Try: pip install {query}"
            }
        
        return {
            "success": True,
            "message": result.stdout[:1000],
            "data": {"query": query}
        }
    
    def _apt_search(self, args: dict) -> dict:
        """Search apt packages (read-only)."""
        query = args.get("query", "")
        
        if not query:
            return {"success": False, "message": "Query is required"}
        
        cmd = ["apt-cache", "search", query]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        
        lines = result.stdout.strip().split('\n')[:20]  # Top 20 results
        
        return {
            "success": True,
            "message": f"Found {len(lines)} packages:\n" + '\n'.join(lines),
            "data": {"query": query, "count": len(lines)}
        }
    
    def _git_clone(self, args: dict) -> dict:
        """Clone a git repository."""
        url = args.get("url", "")
        dest = args.get("dest", "./")
        
        if not url:
            return {"success": False, "message": "URL is required"}
        
        # Safety: only https URLs
        if not url.startswith("https://"):
            return {"success": False, "message": "Only https:// URLs are allowed for security"}
        
        dest_path = self._validate_path(dest, must_exist=False)
        
        cmd = ["git", "clone", "--depth", "1", url, str(dest_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return {"success": False, "message": f"git clone failed: {result.stderr[:500]}"}
        
        return {
            "success": True,
            "message": f"Cloned {url} to {dest_path}",
            "data": {"url": url, "dest": str(dest_path)}
        }
    
    def _requirements_parse(self, args: dict) -> dict:
        """Parse requirements.txt file."""
        req_path = self._validate_path(args.get("path", "./requirements.txt"))
        
        with open(req_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        
        # Parse package names
        packages = []
        for line in lines:
            # Handle ==, >=, <=, etc.
            for sep in ('==', '>=', '<=', '>', '<', '~='):
                if sep in line:
                    packages.append(line.split(sep)[0].strip())
                    break
            else:
                packages.append(line)
        
        return {
            "success": True,
            "message": f"Found {len(packages)} requirements: {', '.join(packages[:10])}{'...' if len(packages) > 10 else ''}",
            "data": {"packages": packages, "count": len(packages)}
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


def package_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for package agent."""
    from ..tool_registry import get_tools
    
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig(goal=state['goal'])
    
    agent = PackageAgentNode(agent_config)
    return agent.execute(state)
