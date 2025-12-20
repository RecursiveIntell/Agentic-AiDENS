"""
Tests for WorkflowAgentNode (n8n integration).
"""
import unittest
from unittest.mock import MagicMock, patch
import json
import requests
from agentic_browser.graph.agents.workflow_agent import WorkflowAgentNode
from agentic_browser.config import AgentConfig
from agentic_browser.settings_store import SettingsStore, Settings

class TestWorkflowAgent(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig(goal="test workflow")
        
        # Mock settings
        self.mock_settings = Settings(
            n8n_url="http://localhost:5678",
            n8n_webhooks={"test_hook": "http://localhost:5678/webhook/test"}
        )
        
        # Patch SettingsStore to return our mock settings
        self.settings_patcher = patch('agentic_browser.settings_store.get_settings', return_value=self.mock_settings)
        self.mock_get_settings = self.settings_patcher.start()
        
        self.agent = WorkflowAgentNode(self.config)
        
    def tearDown(self):
        self.settings_patcher.stop()

    def test_initialization(self):
        """Test agent initialization and config loading."""
        self.assertEqual(self.agent.settings.n8n_url, "http://localhost:5678")
        self.assertTrue(self.agent._is_configured())

    @patch('requests.post')
    def test_trigger_webhook_success(self, mock_post):
        """Test successful webhook trigger."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "id": "123"}
        mock_post.return_value = mock_response
        
        args = {"name": "test_hook", "data": {"foo": "bar"}}
        result = self.agent._trigger_webhook(args)
        
        self.assertTrue(result['success'])
        self.assertIn("Webhook triggered", result['message'])
        mock_post.assert_called_once_with(
            "http://localhost:5678/webhook/test",
            json={"foo": "bar"},
            timeout=10,
            headers={"Content-Type": "application/json"}
        )

    @patch('requests.post')
    def test_trigger_webhook_timeout(self, mock_post):
        """Test webhook timeout handling (async success)."""
        mock_post.side_effect = requests.Timeout("Timeout")
        
        args = {"name": "test_hook"}
        result = self.agent._trigger_webhook(args)
        
        # Should return success True for timeouts (assumed async)
        self.assertTrue(result['success'])
        self.assertIn("timed out waiting for response", result['message'])

    @patch('requests.post')
    def test_trigger_webhook_connection_error(self, mock_post):
        """Test connection error handling."""
        mock_post.side_effect = requests.ConnectionError("Connection refused")
        
        args = {"name": "test_hook"}
        result = self.agent._trigger_webhook(args)
        
        self.assertFalse(result['success'])
        self.assertIn("Connection error", result['message'])

    def test_trigger_unknown_webhook(self):
        """Test triggering a non-existent webhook."""
        args = {"name": "unknown_hook"}
        result = self.agent._trigger_webhook(args)
        
        self.assertFalse(result['success'])
        self.assertIn("not found", result['message'])

    @patch('requests.get')
    def test_check_health(self, mock_get):
        """Test health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": "ok"}'
        mock_get.return_value = mock_response
        
        result = self.agent._check_health({})
        
        self.assertTrue(result['success'])
        mock_get.assert_called_with("http://localhost:5678/healthz", timeout=5)

if __name__ == '__main__':
    unittest.main()
