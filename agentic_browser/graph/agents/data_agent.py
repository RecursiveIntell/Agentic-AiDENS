"""
Data & Transform Agent for file format conversion and data processing.

Handles JSON/CSV/YAML conversions, text search, and archive operations.
"""

import json
import csv
import io
import os
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class DataAgentNode(BaseAgent):
    """Specialized agent for data transformation tasks.
    
    Handles format conversions, text processing, and archive operations
    with built-in safety checks.
    """
    
    AGENT_NAME = "data"
    MAX_STEPS_PER_INVOCATION = 10
    
    SYSTEM_PROMPT = """You are a DATA TRANSFORM agent. You process and convert data files.

Available actions:
- json_to_csv: { "input": "data.json", "output": "data.csv" }
- csv_to_json: { "input": "data.csv", "output": "data.json" }
- json_query: { "input": "data.json", "query": "key.subkey" }  # Simple dot notation
- csv_summary: { "input": "data.csv" }  # Get row count, columns, sample
- text_search: { "pattern": "error", "path": "./logs", "recursive": true }
- text_stats: { "input": "document.txt" }  # Word count, lines, etc.
- compress: { "input": "./folder", "output": "archive.tar.gz", "format": "tar.gz|zip" }
- extract: { "input": "archive.zip", "output": "./" }
- done: { "summary": "what was accomplished" }

RULES:
1. Only process files in user's home directory or specified paths
2. Never overwrite files without confirmation
3. For large files, work with samples first
4. Always verify input files exist before processing

Respond with JSON:
{
  "action": "json_to_csv|csv_to_json|...|done",
  "args": { ... },
  "rationale": "brief reason"
}"""

    def __init__(self, config):
        """Initialize data agent.
        
        Args:
            config: Agent configuration
        """
        super().__init__(config)
        self._home_dir = Path.home()
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute data transformation.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with transformation results
        """
        task_context = f"""
DATA TASK: {state['goal']}

Working directory: {os.getcwd()}
Home directory: {self._home_dir}

Files accessed so far:
{chr(10).join(state['files_accessed'][-5:]) or '(none)'}

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
                summary = args.get("summary", "Data processing completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"data_result": summary},
                )
            
            # Execute the data action
            result = self._execute_action(action, args)
            
            tool_msg = HumanMessage(content=f"Tool '{action}' output:\n{result['message'][:2000]}")
            
            extracted = None
            if result['success'] and result.get('data'):
                extracted = {f"data_{action}": str(result['data'])[:1500]}
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                file_accessed=args.get('input') or args.get('path'),
                extracted_data=extracted,
                error=result['message'] if not result['success'] else None,
            )
            
        except Exception as e:
            return self._update_state(
                state,
                error=f"Data agent error: {str(e)}",
            )
    
    def _execute_action(self, action: str, args: dict) -> dict:
        """Execute a data transformation action."""
        handlers = {
            "json_to_csv": self._json_to_csv,
            "csv_to_json": self._csv_to_json,
            "json_query": self._json_query,
            "csv_summary": self._csv_summary,
            "text_search": self._text_search,
            "text_stats": self._text_stats,
            "compress": self._compress,
            "extract": self._extract,
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
        
        # Must be under home directory or /tmp
        if not (str(p).startswith(str(self._home_dir)) or str(p).startswith('/tmp')):
            raise ValueError(f"Path must be under home directory: {path}")
        
        if must_exist and not p.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        return p
    
    def _json_to_csv(self, args: dict) -> dict:
        """Convert JSON to CSV."""
        input_path = self._validate_path(args.get("input", ""))
        output_path = self._validate_path(args.get("output", ""), must_exist=False)
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Handle list of dicts or single dict
        if isinstance(data, dict):
            data = [data]
        
        if not data or not isinstance(data[0], dict):
            return {"success": False, "message": "JSON must be a list of objects or single object"}
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        return {
            "success": True,
            "message": f"Converted {len(data)} rows to {output_path}",
            "data": {"rows": len(data), "output": str(output_path)}
        }
    
    def _csv_to_json(self, args: dict) -> dict:
        """Convert CSV to JSON."""
        input_path = self._validate_path(args.get("input", ""))
        output_path = self._validate_path(args.get("output", ""), must_exist=False)
        
        with open(input_path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return {
            "success": True,
            "message": f"Converted {len(data)} rows to {output_path}",
            "data": {"rows": len(data), "output": str(output_path)}
        }
    
    def _json_query(self, args: dict) -> dict:
        """Query JSON with dot notation."""
        input_path = self._validate_path(args.get("input", ""))
        query = args.get("query", "")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Simple dot notation query
        result = data
        for key in query.split('.'):
            if key.isdigit():
                result = result[int(key)]
            else:
                result = result[key]
        
        return {
            "success": True,
            "message": f"Query result for '{query}'",
            "data": result
        }
    
    def _csv_summary(self, args: dict) -> dict:
        """Get CSV summary statistics."""
        input_path = self._validate_path(args.get("input", ""))
        
        with open(input_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return {"success": True, "message": "Empty CSV file", "data": {}}
        
        summary = {
            "row_count": len(rows),
            "columns": list(rows[0].keys()),
            "sample": rows[:3] if len(rows) >= 3 else rows,
        }
        
        return {
            "success": True,
            "message": f"CSV has {len(rows)} rows and {len(summary['columns'])} columns",
            "data": summary
        }
    
    def _text_search(self, args: dict) -> dict:
        """Search for pattern in files."""
        path = self._validate_path(args.get("path", "."))
        pattern = args.get("pattern", "")
        recursive = args.get("recursive", False)
        
        if not pattern:
            return {"success": False, "message": "Pattern is required"}
        
        # Use grep for efficiency
        cmd = ["grep", "-n", "-I"]  # Line numbers, skip binary
        if recursive:
            cmd.append("-r")
        cmd.extend([pattern, str(path)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            matches = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            return {
                "success": True,
                "message": f"Found {len(matches)} matches for '{pattern}'",
                "data": {"matches": matches[:50], "pattern": pattern}
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Search timed out"}
    
    def _text_stats(self, args: dict) -> dict:
        """Get text file statistics."""
        input_path = self._validate_path(args.get("input", ""))
        
        with open(input_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        words = content.split()
        
        stats = {
            "lines": len(lines),
            "words": len(words),
            "characters": len(content),
            "bytes": input_path.stat().st_size,
        }
        
        return {
            "success": True,
            "message": f"File has {stats['lines']} lines, {stats['words']} words",
            "data": stats
        }
    
    def _compress(self, args: dict) -> dict:
        """Compress files or directories."""
        input_path = self._validate_path(args.get("input", ""))
        output = args.get("output", "")
        fmt = args.get("format", "tar.gz")
        
        output_path = self._validate_path(output, must_exist=False)
        
        if fmt == "zip":
            cmd = ["zip", "-r", str(output_path), str(input_path)]
        else:  # tar.gz
            cmd = ["tar", "-czf", str(output_path), "-C", str(input_path.parent), input_path.name]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return {"success": False, "message": f"Compression failed: {result.stderr}"}
        
        return {
            "success": True,
            "message": f"Created archive: {output_path}",
            "data": {"output": str(output_path), "format": fmt}
        }
    
    def _extract(self, args: dict) -> dict:
        """Extract archive."""
        input_path = self._validate_path(args.get("input", ""))
        output = args.get("output", ".")
        output_path = self._validate_path(output, must_exist=False)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        name = input_path.name.lower()
        if name.endswith('.zip'):
            cmd = ["unzip", "-o", str(input_path), "-d", str(output_path)]
        elif name.endswith('.tar.gz') or name.endswith('.tgz'):
            cmd = ["tar", "-xzf", str(input_path), "-C", str(output_path)]
        elif name.endswith('.tar'):
            cmd = ["tar", "-xf", str(input_path), "-C", str(output_path)]
        else:
            return {"success": False, "message": f"Unsupported archive format: {name}"}
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            return {"success": False, "message": f"Extraction failed: {result.stderr}"}
        
        return {
            "success": True,
            "message": f"Extracted to: {output_path}",
            "data": {"output": str(output_path)}
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


def data_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for data agent."""
    from ..tool_registry import get_tools
    
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig(goal=state['goal'])
    
    agent = DataAgentNode(agent_config)
    return agent.execute(state)
