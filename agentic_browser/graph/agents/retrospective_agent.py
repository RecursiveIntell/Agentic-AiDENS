import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .base import BaseAgent
from ..state import AgentState

class RetrospectiveAgent(BaseAgent):
    """Agent that analyzes completed steps for self-improvement."""
    
    AGENT_NAME = "retrospective"
    LOG_FILE = os.path.expanduser("~/.agentic_browser_learning_log.md")
    MAX_LOG_CHARS = 300_000  # Approx 75k tokens
    
    def execute(self, state: AgentState) -> AgentState:
        """Analyze the ENTIRE session execution and update learning log."""
        
        # 1. READ LOG (Truncate if needed)
        learning_log = self._read_and_truncate_log()
        
        # 2. GET CONTEXT
        # We need the full history to analyze the session
        full_history = state.get("messages", [])
        history_str = "\n".join([f"{type(m).__name__}: {m.content[:500]}..." for m in full_history]) # Truncate individual messages to stay within limits
        
        plan = state.get("implementation_plan", {})
        
        # 3. ANALYZE
        prompt = f"""
You are the Retrospective Agent. Your goal is to analyze the "Plan vs. Reality" of the ENTIRE completed session to improve future performance.

### CURRENT LEARNED KNOWLEDGE (Do not repeat these, but use them):
{learning_log}

### IMPLEMENTATION PLAN
{plan}

### ACTUAL SESSION REALITY
{history_str}

### INSTRUCTIONS
1. Analyze if the overall execution matched the plan.
2. Identify any comprehensive patterns of inefficiency, loops, errors, or UX issues.
3. Suggest high-level improvements for the agent's strategy.
4. If the task was successful, confirm it.

Output your analysis in a concise format to be appended to the learning log.
If there is nothing significant to add, just output "No significant new insights."
"""
        messages = [HumanMessage(content=prompt)]
        
        try:
            response = self.safe_invoke(messages)
            analysis = response.content.strip()
            
            # 4. UPDATE LOG
            if "No significant new insights" not in analysis and len(analysis) > 10:
                self._append_to_log(analysis, {"description": "Full Session Analysis", "agent": "retrospective"})
            
            # 5. MARK AS DONE
            # We set retrospective_ran=True so Supervisor knows we are truly done.
            # We preserve the existing final_answer.
            
            return self._update_state(
                state,
                messages=[AIMessage(content=f"Retrospective Analysis Completed.")],
                # pass custom keys directly to state update
            )
            # Note: _update_state helper might not support arbitrary keys easily if not in schema?
            # AgentState is TypedDict, so we can just return the dict update.
            # But we should use the helper for consistency (token usage etc).
            # I will modify _update_state to allow kwargs or manually merge.
            # _update_state takes **kwargs? 
            # Let's check base.py or just return the dict update manually after calling helper.
            
            base_update = self._update_state(
                state,
                messages=[AIMessage(content=f"Retrospective Analysis Completed.")],
                token_usage=self.update_token_usage(state, response)
            )
            base_update["retrospective_ran"] = True
            base_update["task_complete"] = True # Re-confirm completion
            return base_update
            
        except Exception as e:
            print(f"[RETRO] Error: {e}")
            return {
                **state,
                "retrospective_ran": True, # Fail safe to exit
                "task_complete": True,
                "error": f"Retrospective error: {e}"
            }

    def _read_and_truncate_log(self) -> str:
        """Read log file, truncating if too large."""
        if not os.path.exists(self.LOG_FILE):
            return ""
            
        with open(self.LOG_FILE, "r") as f:
            content = f.read()
            
        if len(content) > self.MAX_LOG_CHARS:
            # Truncate oldest (keep header if exists? Assume simple append)
            # Keep the last MAX_LOG_CHARS
            metrics_len = len(content)
            content = content[-self.MAX_LOG_CHARS:]
            
            # Find first newline to align
            first_newline = content.find("\n")
            if first_newline != -1:
                content = content[first_newline+1:]
                
            # Write back truncated
            with open(self.LOG_FILE, "w") as f:
                f.write(content)
            
            print(f"[RETRO] ✂️ Truncated learning log from {metrics_len} to {len(content)} chars")
            
        return content

    def _append_to_log(self, analysis: str, step: dict):
        """Append new specific insights to log."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        entry = f"\n\n## [{timestamp}] Step: {step.get('description', 'Unknown')}\n"
        entry += f"Agent: {step.get('agent', 'unknown')}\n"
        entry += f"{analysis}\n"
        entry += "-" * 40
        
        with open(self.LOG_FILE, "a") as f:
            f.write(entry)

def retrospective_agent_node(state: AgentState) -> AgentState:
    """LangGraph node function."""
    from ...config import AgentConfig
    # Minimal config, no browser tools needed
    agent = RetrospectiveAgent(AgentConfig(), None)
    return agent.execute(state)
