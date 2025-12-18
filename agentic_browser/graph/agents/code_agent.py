"""
Code Agent for code analysis and execution.

Specializes in understanding codebases and running code.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from .base import BaseAgent
from ..state import AgentState


class CodeAgentNode(BaseAgent):
    """Specialized agent for code analysis tasks.
    
    Analyzes codebases, reads documentation, and can execute
    sandboxed code for testing.
    """
    
    AGENT_NAME = "code"
    MAX_STEPS_PER_INVOCATION = 10
    
    SYSTEM_PROMPT = """You are a CODE agent. Analyze projects at the appropriate depth.

Available actions:
- os_list_dir: { "path": "." } - List current directory
- os_read_file: { "path": "README.md" } - Read a file
- os_exec: { "cmd": "command" } - Run a shell command
- done: { "summary": "your analysis" } - Complete with findings

ADAPTIVE WORKFLOW:
For SIMPLE tasks ("what is this app?"):
  1. List current directory: {"action": "os_list_dir", "args": {"path": "."}}
  2. Read README: {"action": "os_read_file", "args": {"path": "README.md"}}
  3. Done with summary

For COMPLEX tasks ("analyze architecture"):
  1. List structure (.)
  2. Read config files
  3. Read source
  4. Done

CRITICAL RULES:
1. ALWAYS start with: { "action": "os_list_dir", "args": { "path": "." } }
2. PROJECT VALIDATION:
   - Check if the files in "." match the user's request.
   - If User asks for "Cat App" but you see "Agentic Browser" code -> WRONG DIRECTORY.
   - ACTION: Search nearby: 
     { "action": "os_exec", "args": { "cmd": "find .. -name '*cat*' -type d -maxdepth 3" } }
   - Then listing the CORRECT directory.
3. Do not guess paths! Use "os_exec find ..." if you are lost.
4. "os_list_dir" output is in your HISTORY. Do not re-list same dir repeatedly.
5. Store findings in "code_analysis" key of extracted_data.

FUZZY MATCHING:
- "cat app" could match: CatOS, Cat Info App, cat-tracker, etc.
- Check folder names and READMEs to identify the right project

COMPLETION RULES:
- Always provide a USEFUL summary when calling "done"
- Include: what the project does, tech stack, key features
- If you can't find something, say so and complete with what you have

Respond with JSON:
{
  "action": "os_list_dir|os_read_file|os_exec|done",
  "args": { ... },
  "rationale": "brief reason"
}"""

    def __init__(self, config, os_tools=None):
        """Initialize code agent.
        
        Args:
            config: Agent configuration
            os_tools: OS tools for file access
        """
        super().__init__(config)
        self._os_tools = os_tools
    
    @property
    def system_prompt(self) -> str:
        return self.SYSTEM_PROMPT
    
    def set_os_tools(self, os_tools) -> None:
        """Set OS tools after initialization."""
        self._os_tools = os_tools
    
    def execute(self, state: AgentState) -> AgentState:
        """Execute code analysis.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with code analysis results
        """
        if not self._os_tools:
            return self._update_state(
                state,
                error="OS tools not available for code analysis",
            )
        
        # Collect recently accessed paths to discourage duplicates
        recent_files = state['files_accessed'][-15:]
        files_str = chr(10).join(f"  - {f}" for f in recent_files) if recent_files else "(none yet)"
        
        # Check for repeated patterns
        repeat_warning = ""
        if len(recent_files) >= 4:
            last_4 = recent_files[-4:]
            if len(set(last_4)) <= 2:  # Only 1-2 unique paths in last 4
                repeat_warning = """
⚠️ WARNING: You are repeating the same paths! 
DO NOT explore the same directories again.
Either read NEW files or call 'done' with your analysis.
"""
        
        # Build info summary (compact)
        data_keys = list(state['extracted_data'].keys())
        collected_info = f"Collected {len(data_keys)} items: {data_keys[:5]}" if data_keys else "(none yet)"
        
        task_context = f"""
CODE TASK: {state['goal']}

Files examined:
{files_str}

Data collected: {collected_info}
{repeat_warning}
IMPORTANT RULES:
1. DO NOT re-read the same file twice
2. After seeing README.md and a few key files, call "done" with analysis
3. If step count is high ({state['step_count']}/{state['max_steps']}), complete NOW
"""
        
        messages = self._build_messages(state, task_context)
        
        try:
            response = self.safe_invoke(messages)
            
            # DEBUG: Print raw response
            print(f"\n{'='*60}")
            print(f"[DEBUG] CodeAgent - Raw LLM Response:")
            print(f"{'='*60}")
            print(response.content[:1500] if len(response.content) > 1500 else response.content)
            print(f"{'='*60}\n")
            
            action_data = self._parse_action(response.content)
            
            # DEBUG: Print parsed action
            print(f"[DEBUG] CodeAgent - Parsed Action: {action_data}")
            
            # Force completion if repeating same action too many times
            target_path = action_data.get("args", {}).get("path", "")
            if target_path and target_path in recent_files[-3:]:
                print(f"[DEBUG] CodeAgent - Detected repeat of '{target_path}', forcing done")
                summary = f"Analysis based on explored files: {', '.join(recent_files[:5])}"
                return self._update_state(
                    state,
                    messages=[AIMessage(content="Completing analysis (repeat detected)")],
                    extracted_data={"code_analysis": summary},
                )
            
            if action_data.get("action") == "done":
                print(f"[DEBUG] CodeAgent - Action is 'done', returning findings")
                # Store findings but DON'T mark task_complete - let supervisor decide
                summary = action_data.get("args", {}).get("summary", "Analysis completed")
                return self._update_state(
                    state,
                    messages=[AIMessage(content=response.content)],
                    extracted_data={"code_analysis": summary},
                    # Note: NOT setting task_complete - supervisor handles that
                )
            
            # Execute OS action
            result = self._os_tools.execute(
                action_data.get("action", ""),
                action_data.get("args", {}),
            )
            
            file_accessed = None
            extracted = None
            action_name = action_data.get("action", "unknown")
            output_content = str(result.data if result.data else result.message)
            
            if action_data.get("action") in ("os_read_file", "os_list_dir"):
                file_accessed = action_data.get("args", {}).get("path")
                
                # Store intermediate results to extracted_data so supervisor knows we're working
                # This prevents the "no extracted_data" blocking loop
                if result.success:
                    key = f"code_{action_name}_{len(state['files_accessed'])}"
                    extracted = {key: output_content[:1500]}
            
            tool_msg = HumanMessage(content=f"Tool '{action_name}' output:\n{output_content[:2000]}")
            
            return self._update_state(
                state,
                messages=[AIMessage(content=response.content), tool_msg],
                file_accessed=file_accessed,
                extracted_data=extracted,
                error=result.message if not result.success else None,
            )
            
        except Exception as e:
            return self._update_state(
                state,
                error=f"Code agent error: {str(e)}",
            )
    
    def _parse_action(self, response: str) -> dict:
        """Parse LLM response into action dict.
        
        Handles common LLM formatting issues like newlines in JSON strings.
        """
        import re
        
        try:
            content = response.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                
                # Try to fix common JSON issues: newlines inside strings
                # Replace actual newlines (not \n escape sequences) inside quoted strings
                # This regex finds strings and replaces newlines within them
                def fix_newlines(match):
                    return match.group(0).replace('\n', '\\n').replace('\r', '')
                
                # Match JSON string values (simple approach)
                json_str = re.sub(r'"[^"]*"', fix_newlines, json_str, flags=re.DOTALL)
                
                try:
                    data = json.loads(json_str)
                    
                    # Validate action exists
                    if not data.get("action"):
                        return {"action": "os_list_dir", "args": {"path": "."}}
                    
                    return data
                except json.JSONDecodeError:
                    pass
            
            # Fallback: Try to detect "done" action via regex if JSON fails
            if '"action"' in content and '"done"' in content.lower():
                # Try to extract summary from the content
                summary_match = re.search(r'"summary"\s*:\s*"([^"]+(?:\\.[^"]*)*)"', content)
                if summary_match:
                    summary = summary_match.group(1).replace('\\n', '\n')
                    return {"action": "done", "args": {"summary": summary}}
                else:
                    # Just mark as done with generic message
                    return {"action": "done", "args": {"summary": "Analysis completed."}}
            
            # Default to exploration
            return {"action": "os_list_dir", "args": {"path": "."}}
            
        except Exception:
            # On any parse failure, explore instead of quitting
            return {"action": "os_list_dir", "args": {"path": "."}}


def code_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function for code agent."""
    from ..tool_registry import get_tools
    
    # Get tools from registry using session_id
    tools = get_tools(state.get("session_id", ""))
    if tools:
        agent_config = tools.config
        os_tools = tools.os_tools
    else:
        from ...config import AgentConfig
        agent_config = AgentConfig()
        os_tools = None
    
    agent = CodeAgentNode(agent_config, os_tools)
    return agent.execute(state)


