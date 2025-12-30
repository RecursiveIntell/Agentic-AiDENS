"""
Tests for SettingsStore persistence.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agentic_browser.settings_store import SettingsStore


class TestSettingsStore(unittest.TestCase):
    def test_langsmith_settings_roundtrip(self):
        """LangSmith settings should persist to disk and reload."""
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            with patch("agentic_browser.settings_store.get_base_dir", return_value=base_dir):
                SettingsStore._instance = None
                store = SettingsStore()
                store.update(
                    langsmith_enabled=True,
                    langsmith_api_key="test-api-key",
                    langsmith_project="test-project",
                )

                SettingsStore._instance = None
                reloaded = SettingsStore().settings

        self.assertTrue(reloaded.langsmith_enabled)
        self.assertEqual(reloaded.langsmith_api_key, "test-api-key")
        self.assertEqual(reloaded.langsmith_project, "test-project")


if __name__ == "__main__":
    unittest.main()
