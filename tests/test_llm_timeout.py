import asyncio
import time

import pytest

from agentic_browser.config import AgentConfig
from agentic_browser.graph.agents.base import BaseAgent, LLMTimeoutError
from agentic_browser.graph.state import AgentState


class SlowLLM:
    def __init__(self, sleep_s: float = 0.2):
        self.sleep_s = sleep_s

    async def ainvoke(self, messages):
        await asyncio.sleep(self.sleep_s)
        return "ok"

    def invoke(self, messages):
        time.sleep(self.sleep_s)
        return "ok"


class DummyAgent(BaseAgent):
    AGENT_NAME = "dummy"

    def __init__(self, llm):
        self.config = AgentConfig(goal="test")
        self.llm = llm

    @property
    def system_prompt(self) -> str:
        return "dummy"

    def execute(self, state: AgentState) -> AgentState:
        return state


def test_invoke_llm_with_timeout_sync_times_out():
    agent = DummyAgent(SlowLLM(sleep_s=0.2))

    with pytest.raises(LLMTimeoutError):
        agent.invoke_llm_with_timeout([{"role": "user", "content": "hi"}], timeout_s=0.05)


@pytest.mark.anyio
async def test_invoke_llm_with_timeout_async_times_out():
    agent = DummyAgent(SlowLLM(sleep_s=0.2))

    with pytest.raises(LLMTimeoutError):
        await agent.invoke_llm_with_timeout([{"role": "user", "content": "hi"}], timeout_s=0.05)
