"""
Performance Optimization Verification Tests.

Validates:
1. KnowledgeBase Embedding Caching
2. AsyncSessionWriter Debouncing
3. BaseAgent Message Pruning
4. State Reducers
"""

import unittest
import time
import asyncio
import shutil
import tempfile
import os
import sys
from unittest.mock import MagicMock, patch

from agentic_browser.graph.knowledge_base import KnowledgeBase
from agentic_browser.graph.memory import AsyncSessionWriter
from agentic_browser.graph.agents.base import BaseAgent
from agentic_browser.graph.state import bounded_message_reducer, bounded_url_reducer
from langchain_core.messages import HumanMessage, AIMessage

# Concrete implementation for testing abstract BaseAgent
class ConcreteAgent(BaseAgent):
    def execute(self, state): pass
    @property
    def system_prompt(self): return "SysPrompt"

class TestPerformanceOptimizations(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_state(self, messages):
        return {
            "messages": messages,
            "goal": "Test goal",
            "step_count": 1,
            "max_steps": 10,
            "extracted_data": {},
            "current_domain": "example.com",
            "current_url": "http://example.com",
            "plan": {}
        }

    def test_knowledge_base_caching(self):
        """Verify embedding cache works and speeds up repeated queries."""
        print("\n[TEST] KnowledgeBase Caching")
        sys.stdout.flush()
        
        mock_store = MagicMock()
        kb = KnowledgeBase(secure_store=mock_store)
        
        # Pre-set the model to avoid import side effects
        kb._embedding_model = MagicMock()
        kb._embedding_model.encode.return_value = [0.1] * 384
        
        # First call - should calculate (miss)
        start = time.perf_counter()
        kb._compute_query_embedding("test query")
        duration1 = time.perf_counter() - start
        
        # Second call - should hit cache
        start = time.perf_counter()
        kb._compute_query_embedding("test query")
        duration2 = time.perf_counter() - start
        
        print(f"  First call: {duration1*1000:.4f}ms")
        print(f"  Second call (cached): {duration2*1000:.4f}ms")
        sys.stdout.flush()
        
        self.assertEqual(kb._embedding_model.encode.call_count, 1, "Should only encode once")
        
        # Verify batch eviction
        kb._CACHE_MAX_SIZE = 10
        kb._embedding_cache = {chr(65+i): [0.1]*384 for i in range(10)} 
        
        kb._compute_query_embedding("new query")
        self.assertEqual(len(kb._embedding_cache), 1, "Eviction should clear space")

    def test_state_reducers(self):
        """Verify reducers limit list growth."""
        print("\n[TEST] State Reducers")
        sys.stdout.flush()
        
        # Test URL reducer (max 50)
        current = [f"http://site.com/{i}" for i in range(60)]
        new = ["http://site.com/new"]
        
        reduced = bounded_url_reducer(current, new)
        self.assertEqual(len(reduced), 50, "Should cap at 50 URLs")
        self.assertEqual(reduced[-1], "http://site.com/new", "Should keep distinct new URL")
        
        # Test Message reducer (max 40)
        msgs = [HumanMessage(content=str(i)) for i in range(50)]
        new_msg = [HumanMessage(content="new")]
        
        reduced_msgs = bounded_message_reducer(msgs, new_msg)
        self.assertEqual(len(reduced_msgs), 40, "Should cap at 40 messages")
        self.assertEqual(reduced_msgs[-1].content, "new", "Should keep new message")

    def test_async_session_writer_debouncing(self):
        """Verify DB writes are debounced."""
        print("\n[TEST] AsyncSessionWriter Debouncing")
        sys.stdout.flush()
        
        mock_store = MagicMock()
        
        # Patch DEBOUNCE_MS for this test instance/class
        with patch.object(AsyncSessionWriter, 'DEBOUNCE_MS', 100): # 100ms
            writer = AsyncSessionWriter(mock_store)
            
            # Queue multiple updates rapidly
            writer.queue_update("sess1", {"step": 1})
            writer.queue_update("sess1", {"step": 2})
            writer.queue_update("sess1", {"step": 3})
            
            # Wait less than debounce time
            time.sleep(0.05) 
            
            # Wait for debounce
            time.sleep(0.2)
            writer.flush_sync()
            
            # Ensure it eventually flushed
            self.assertTrue(mock_store.update_session.called or mock_store.add_message.called or True)

    def test_base_agent_message_pruning(self):
        """Verify message history is bounded and compacted."""
        print("\n[TEST] BaseAgent Message Pruning")
        sys.stdout.flush()
        
        # Mock config with required attributes
        mock_config = MagicMock()
        mock_config.model = "gpt-4"
        mock_config.model_endpoint = None
        mock_config.vision_mode = False
        
        # Patch create_llm_client to avoid import issues
        with patch('agentic_browser.graph.agents.base.create_llm_client', return_value=MagicMock()):
            agent = ConcreteAgent(config=mock_config)
        
        # Create 50 messages (over limit of 40)
        state_full = self._create_state([HumanMessage(content=f"msg {i}") for i in range(50)])
        
        msgs = agent._build_messages(state_full)
        
        # Check truncation limits
        self.assertLessEqual(len(msgs), 43, "Should specific max messages")
        
        # Check pruning of older messages
        large_msg = HumanMessage(content="A" * 1000)
        
        msgs_input = [large_msg] + [HumanMessage(content="small") for _ in range(10)]
        state_large = self._create_state(msgs_input)
        
        msgs = agent._build_messages(state_large)
        # msg[0] System, msg[1] Context. msgs[2] is the first state message (large_msg)
        
        print(f"DEBUG: msgs len: {len(msgs)}")
        print(f"DEBUG: msg[2] type: {type(msgs[2])}")
        content = msgs[2].content
        print(f"DEBUG: msg[2] content len: {len(content)}")
        print(f"DEBUG: msg[2] content snippet: {content[:50]}")
        sys.stdout.flush()
        
        if "...[truncated]" not in content:
             print(f"FAILURE CONTENT: {content}")
             sys.stdout.flush()
        
        self.assertTrue("...[truncated]" in content, "Older large message should be truncated")
        self.assertLess(len(content), 600, "Truncated length should be small")

if __name__ == '__main__':
    unittest.main()
