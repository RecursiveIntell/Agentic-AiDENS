import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .base import BaseAgent
from ..state import AgentState
from ..knowledge_base import get_knowledge_base

class RetrospectiveAgent(BaseAgent):
    """Agent that analyzes completed steps for self-improvement."""
    
    AGENT_NAME = "retrospective"
    LOG_FILE = os.path.expanduser("~/.agentic_browser_learning_log.md")
    MAX_LOG_CHARS = 300_000  # Approx 75k tokens
    
    @property
    def system_prompt(self) -> str:
        return "You are the Retrospective Agent. Your goal is to analyze session history and extract strategies and mistakes."

    def execute(self, state: AgentState) -> AgentState:
        """Analyze the ENTIRE session execution and update learning log."""
        
        # 1. READ LOG (Truncate if needed)
        
        # 1. READ LOG (Truncate if needed)
        learning_log = self._read_and_truncate_log()
        
        # 2. GET CONTEXT
        full_history = state.get("messages", [])
        history_str = "\n".join([f"{type(m).__name__}: {m.content[:500]}..." for m in full_history])
        
        plan = state.get("implementation_plan", {})
        goal = state.get("input", "Unknown Goal")
        final_answer = state.get("final_answer", "")
        error = state.get("error", "")
        step_count = state.get("step_count", 0)
        was_aborted = state.get("was_aborted", False)  # Flag from CLI/Stop button
        
        # 3. ANALYZE
        status_context = "The user ABORTED this run manually." if was_aborted else "The task completed naturally."
        
        prompt = f"""
You are the Retrospective Agent. analyze the "Plan vs. Reality" of the ENTIRE session.

### SESSION CONTEXT
Goal: {goal}
Status: {status_context}
Steps Taken: {step_count}
Result: {final_answer}
Error: {error}

### CURRENT LEARNED KNOWLEDGE (Refine or add to this)
{learning_log}

### IMPLEMENTATION PLAN
{plan}

### ACTUAL SESSION REALITY
{history_str}

### INSTRUCTIONS
1. Analyze if the execution matched the plan.
2. Identify WINNING STRATEGIES (if successful) -> "strategies" list.
3. Identify FAILURE PATTERNS (if failed/inefficient) -> "apocalypse" list.
   CRITICAL: Only include failures that are METHODOLOGICAL and PREVENTABLE by the agent (e.g., infinite loops, bad navigation logic, repeated mistakes). 
   Do NOT record system bugs, crashes, or network errors.
4. Write a brief learning summary -> "summary".

### OUTPUT FORMAT (JSON)
{{
  "strategies": [
    {{
      "name": "Short Name (3-5 words)",
      "description": "High-level reusable approach (1-2 sentences)",
      "agent": "planner|research" 
    }}
  ],
  "apocalypse": [
    {{
      "error_pattern": "Short Pattern Name (3-5 words)",
      "description": "How to avoid this behaviorally (1-2 sentences)",
      "agent": "planner|research"
    }}
  ],
  "summary": "Freeform markdown text for the user learning log"
}}

If no new strategies/failures are found, return empty lists.
"""
        messages = [HumanMessage(content=prompt)]
        
        try:
            # We explicitly ask for JSON mode if supported, or rely on prompt instruction
            # System prompt usually handles this, but let's be safe
            response = self.safe_invoke(messages)
            content = response.content.strip()
            
            # Clean JSON markdown if present
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            
            import json
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Fallback to text only if JSON fails
                print("[RETRO] ⚠️ JSON parse failed, falling back to text log")
                data = {"strategies": [], "apocalypse": [], "summary": content}

            # 4. SAVE KNOWLEDGE
            kb = get_knowledge_base()
            
            # Save Strategies
            for strat in data.get("strategies", []):
                try:
                    kb.save_strategy(
                        strat.get("agent", "research"),
                        strat.get("name"),
                        strat.get("description")
                    )
                    print(f"[RETRO] 💎 Learned Strategy: {strat.get('name')}")
                except Exception as e:
                    print(f"[RETRO] Failed to save strategy: {e}")

            # Save Apocalypse (Preventable Failures)
            for apoc in data.get("apocalypse", []):
                try:
                    kb.save_apocalypse(
                        apoc.get("agent", "research"),
                        apoc.get("error_pattern"),
                        apoc.get("description")
                    )
                    print(f"[RETRO] 💀 Learned from Mistake: {apoc.get('error_pattern')}")
                except Exception as e:
                    print(f"[RETRO] Failed to save apocalypse: {e}")

            # 5. UPDATE LOG
            analysis = data.get("summary", "No summary provided.")
            if len(analysis) > 10:
                self._append_to_log(analysis, {
                    "description": f"Full Session Analysis ({'Aborted' if was_aborted else 'Complete'})", 
                    "agent": "retrospective"
                })
            
            # 6. RETURN UPDATE
            base_update = self._update_state(
                state,
                messages=[AIMessage(content="Retrospective Analysis Completed.")],
                token_usage=self.update_token_usage(state, response)
            )
            base_update["retrospective_ran"] = True
            base_update["task_complete"] = True
            return base_update
            
        except Exception as e:
            print(f"[RETRO] Error: {e}")
            return {
                **state,
                "retrospective_ran": True,
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
