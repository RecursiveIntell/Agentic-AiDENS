"""
Tests for Structured Memory Store.

Tests JSON storage, redaction rules, and memory operations.
"""

import json
import tempfile
from pathlib import Path

import pytest

from agentic_browser.memory_store import (
    KnownSite,
    KnownDirectory,
    Recipe,
    Redactor,
    StructuredMemoryStore,
    create_memory_store,
)


class TestRedactor:
    """Tests for the Redactor class."""
    
    def test_redact_password(self):
        """Test password redaction."""
        text = "password=mysecretpass123"
        result = Redactor.redact(text)
        assert "mysecretpass123" not in result
        assert "REDACTED" in result
    
    def test_redact_api_key(self):
        """Test API key redaction."""
        text = "api_key: sk-abc123def456"
        result = Redactor.redact(text)
        assert "sk-abc123def456" not in result
    
    def test_redact_token(self):
        """Test token redaction."""
        text = "token=eyJhbGciOiJIUzI1NiIs..."
        result = Redactor.redact(text)
        assert "eyJhbGciOiJIUzI1NiIs" not in result
    
    def test_redact_openai_key(self):
        """Test OpenAI key format redaction."""
        text = "key is sk-1234567890123456789012345678901234567890"
        result = Redactor.redact(text)
        assert "sk-123456789012345678901234567890" not in result
        assert "REDACTED_OPENAI_KEY" in result
    
    def test_redact_github_token(self):
        """Test GitHub token redaction."""
        text = "GITHUB_TOKEN=ghp_123456789012345678901234567890123456"
        result = Redactor.redact(text)
        assert "ghp_" not in result or "REDACTED" in result
    
    def test_redact_credit_card(self):
        """Test credit card number redaction."""
        text = "Card: 4111-1111-1111-1111"
        result = Redactor.redact(text)
        assert "4111-1111-1111-1111" not in result
        assert "REDACTED_CARD" in result
    
    def test_redact_ssn(self):
        """Test SSN redaction."""
        text = "SSN: 123-45-6789"
        result = Redactor.redact(text)
        assert "123-45-6789" not in result
        assert "REDACTED_SSN" in result
    
    def test_redact_dict(self):
        """Test dict redaction."""
        data = {
            "username": "john",
            "password": "secret123",
            "api_token": "abc123",
        }
        result = Redactor.redact_dict(data)
        assert result["username"] == "john"
        assert result["password"] == "[REDACTED]"
        assert result["api_token"] == "[REDACTED]"
    
    def test_redact_nested_dict(self):
        """Test nested dict redaction."""
        data = {
            "config": {
                "token": "secret",
                "url": "https://example.com",
            }
        }
        result = Redactor.redact_dict(data)
        assert result["config"]["token"] == "[REDACTED]"
        assert result["config"]["url"] == "https://example.com"
    
    def test_safe_text_unchanged(self):
        """Test safe text is not changed."""
        text = "Hello, this is a normal message without secrets."
        result = Redactor.redact(text)
        assert result == text


class TestKnownSite:
    """Tests for KnownSite model."""
    
    def test_create_minimal(self):
        """Test creating a minimal site."""
        site = KnownSite(domain="github.com")
        assert site.domain == "github.com"
        assert site.common_nav_paths == []
    
    def test_create_full(self):
        """Test creating a full site."""
        site = KnownSite(
            domain="github.com",
            preferred_login_flow="OAuth via Google",
            common_nav_paths=["/issues", "/pulls"],
            selectors={"search": "input[name=q]"},
            constraints=["Requires 2FA"],
            notes="Developer account",
        )
        assert site.preferred_login_flow == "OAuth via Google"
        assert len(site.common_nav_paths) == 2


class TestKnownDirectory:
    """Tests for KnownDirectory model."""
    
    def test_create(self):
        """Test creating a directory."""
        dir = KnownDirectory(
            name="Downloads",
            path="/home/user/Downloads",
            notes="Default download location",
        )
        assert dir.name == "Downloads"
        assert dir.path == "/home/user/Downloads"


class TestRecipe:
    """Tests for Recipe model."""
    
    def test_create(self):
        """Test creating a recipe."""
        recipe = Recipe(
            name="Login to GitHub",
            description="Logs into GitHub via username/password",
            steps=[
                {"action": "goto", "url": "https://github.com/login"},
                {"action": "type", "selector": "#login_field", "text": "username"},
            ],
        )
        assert recipe.name == "Login to GitHub"
        assert len(recipe.steps) == 2
        assert recipe.created_at is not None


class TestStructuredMemoryStore:
    """Tests for StructuredMemoryStore."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield StructuredMemoryStore(Path(tmpdir))
    
    # Site tests
    
    def test_save_and_get_site(self, temp_store):
        """Test saving and retrieving a site."""
        site = KnownSite(domain="github.com", notes="Test site")
        temp_store.save_site(site)
        
        retrieved = temp_store.get_site("github.com")
        assert retrieved is not None
        assert retrieved.domain == "github.com"
    
    def test_get_nonexistent_site(self, temp_store):
        """Test getting a site that doesn't exist."""
        result = temp_store.get_site("nonexistent.com")
        assert result is None
    
    def test_list_sites(self, temp_store):
        """Test listing sites."""
        temp_store.save_site(KnownSite(domain="a.com"))
        temp_store.save_site(KnownSite(domain="b.com"))
        
        sites = temp_store.list_sites()
        assert len(sites) == 2
        assert "a.com" in sites
        assert "b.com" in sites
    
    def test_delete_site(self, temp_store):
        """Test deleting a site."""
        temp_store.save_site(KnownSite(domain="toDelete.com"))
        assert temp_store.delete_site("toDelete.com") is True
        assert temp_store.get_site("toDelete.com") is None
    
    def test_site_case_insensitive(self, temp_store):
        """Test site lookup is case-insensitive."""
        temp_store.save_site(KnownSite(domain="GitHub.com"))
        assert temp_store.get_site("github.com") is not None
    
    # Directory tests
    
    def test_save_and_get_directory(self, temp_store):
        """Test saving and retrieving a directory."""
        dir = KnownDirectory(name="Downloads", path="/home/user/Downloads")
        temp_store.save_directory(dir)
        
        retrieved = temp_store.get_directory("Downloads")
        assert retrieved is not None
        assert retrieved.path == "/home/user/Downloads"
    
    def test_list_directories(self, temp_store):
        """Test listing directories."""
        temp_store.save_directory(KnownDirectory(name="A", path="/a"))
        temp_store.save_directory(KnownDirectory(name="B", path="/b"))
        
        dirs = temp_store.list_directories()
        assert len(dirs) == 2
    
    def test_delete_directory(self, temp_store):
        """Test deleting a directory."""
        temp_store.save_directory(KnownDirectory(name="Temp", path="/tmp"))
        assert temp_store.delete_directory("Temp") is True
        assert temp_store.get_directory("Temp") is None
    
    # Recipe tests
    
    def test_save_and_get_recipe(self, temp_store):
        """Test saving and retrieving a recipe."""
        recipe = Recipe(
            name="Test Recipe",
            description="A test recipe",
            steps=[{"action": "goto", "url": "https://example.com"}],
        )
        temp_store.save_recipe(recipe)
        
        retrieved = temp_store.get_recipe("Test Recipe")
        assert retrieved is not None
        assert retrieved.description == "A test recipe"
    
    def test_list_recipes(self, temp_store):
        """Test listing recipes."""
        temp_store.save_recipe(Recipe(name="R1", description=""))
        temp_store.save_recipe(Recipe(name="R2", description=""))
        
        recipes = temp_store.list_recipes()
        assert len(recipes) == 2
    
    def test_delete_recipe(self, temp_store):
        """Test deleting a recipe."""
        temp_store.save_recipe(Recipe(name="ToDelete", description=""))
        assert temp_store.delete_recipe("ToDelete") is True
        assert temp_store.get_recipe("ToDelete") is None
    
    # Redaction tests
    
    def test_site_saves_redacted(self, temp_store):
        """Test that sites are stored redacted."""
        site = KnownSite(
            domain="test.com",
            notes="password=secret123 api_key=abc",
        )
        temp_store.save_site(site)
        
        # Read raw JSON
        with open(temp_store.sites_file) as f:
            data = json.load(f)
        
        notes = data["sites"]["test.com"]["notes"]
        assert "secret123" not in notes
        assert "abc" not in notes
    
    def test_recipe_steps_redacted(self, temp_store):
        """Test that recipe steps are redacted."""
        recipe = Recipe(
            name="Login",
            description="Test",
            steps=[
                {"action": "type", "text": "password=abc123"},
            ],
        )
        temp_store.save_recipe(recipe)
        
        # Read raw JSON
        with open(temp_store.recipes_file) as f:
            data = json.load(f)
        
        step_text = data["recipes"]["login"]["steps"][0]["text"]
        assert "abc123" not in step_text
    
    # Export/Clear tests
    
    def test_export_all(self, temp_store):
        """Test exporting all data."""
        temp_store.save_site(KnownSite(domain="a.com"))
        temp_store.save_directory(KnownDirectory(name="B", path="/b"))
        
        export = temp_store.export_all()
        assert "a.com" in export["sites"]
        assert "b" in export["directories"]
    
    def test_clear_all(self, temp_store):
        """Test clearing all data."""
        temp_store.save_site(KnownSite(domain="a.com"))
        temp_store.clear_all()
        
        assert temp_store.list_sites() == []
        assert temp_store.list_directories() == []
        assert temp_store.list_recipes() == []


class TestBootstrap:
    """Tests for bootstrap functionality."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary memory store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield StructuredMemoryStore(Path(tmpdir))
    
    def test_bootstrap_creates_directories(self, temp_store):
        """Test bootstrap creates known directories."""
        # This will find whatever dirs exist on the system
        result = temp_store.bootstrap_defaults()
        
        # At least should track that it ran
        assert "directories" in result
    
    def test_bootstrap_idempotent(self, temp_store):
        """Test bootstrap is idempotent."""
        temp_store.bootstrap_defaults()
        initial_count = len(temp_store.list_directories())
        
        temp_store.bootstrap_defaults()
        final_count = len(temp_store.list_directories())
        
        assert initial_count == final_count


class TestFactoryFunction:
    """Tests for factory function."""
    
    def test_create_memory_store(self):
        """Test factory creates store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = create_memory_store(Path(tmpdir))
            assert store is not None
            assert store.memory_dir == Path(tmpdir)
