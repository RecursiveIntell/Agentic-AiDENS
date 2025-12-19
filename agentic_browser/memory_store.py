"""
Structured Memory Store for Agentic Browser.

Provides persistent storage for known sites, directories, and reusable recipes.
Uses JSON files for human-readable storage with aggressive redaction.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .config import get_base_dir


# =============================================================================
# Data Models
# =============================================================================

class KnownSite(BaseModel):
    """Information about a known website."""
    
    domain: str = Field(description="Domain name (e.g., github.com)")
    preferred_login_flow: Optional[str] = Field(
        default=None,
        description="Notes on preferred login method"
    )
    common_nav_paths: list[str] = Field(
        default_factory=list,
        description="Common navigation paths used on this site"
    )
    selectors: dict[str, str] = Field(
        default_factory=dict,
        description="Stable CSS selectors for common elements"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Constraints or notes (e.g., 'requires 2FA')"
    )
    notes: str = Field(default="", description="Free-form notes")
    last_successful_steps: list[dict] = Field(
        default_factory=list,
        description="Last successful action sequence"
    )
    last_accessed: Optional[str] = Field(
        default=None,
        description="ISO timestamp of last access"
    )


class KnownDirectory(BaseModel):
    """Information about a known local directory."""
    
    name: str = Field(description="Human-friendly name (e.g., 'Downloads')")
    path: str = Field(description="Absolute path to directory")
    notes: str = Field(default="", description="Free-form notes")
    safe_write_rules: list[str] = Field(
        default_factory=list,
        description="Rules for safe writes (e.g., 'only .txt files')"
    )


class Recipe(BaseModel):
    """A reusable multi-step playbook."""
    
    name: str = Field(description="Recipe name")
    description: str = Field(description="What this recipe does")
    steps: list[dict] = Field(
        default_factory=list,
        description="Sequence of action dicts"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_used: Optional[str] = Field(default=None)


# =============================================================================
# Redaction
# =============================================================================

class Redactor:
    """Aggressively redacts sensitive information from text."""
    
    # Patterns to redact - never store these
    REDACT_PATTERNS = [
        # Credentials
        (r"password\s*[:=]\s*['\"]?([^\s'\"]+)", "[REDACTED_PASSWORD]"),
        (r"passwd\s*[:=]\s*['\"]?([^\s'\"]+)", "[REDACTED_PASSWORD]"),
        (r"pwd\s*[:=]\s*['\"]?([^\s'\"]+)", "[REDACTED_PASSWORD]"),
        
        # API keys and tokens
        (r"api[_-]?key\s*[:=]\s*['\"]?([^\s'\"]+)", "[REDACTED_API_KEY]"),
        (r"token\s*[:=]\s*['\"]?([^\s'\"]+)", "[REDACTED_TOKEN]"),
        (r"secret\s*[:=]\s*['\"]?([^\s'\"]+)", "[REDACTED_SECRET]"),
        (r"bearer\s+([^\s]+)", "[REDACTED_BEARER]"),
        
        # Cookies and sessions
        (r"cookie\s*[:=]\s*['\"]?([^\s'\"]+)", "[REDACTED_COOKIE]"),
        (r"session[_-]?id\s*[:=]\s*['\"]?([^\s'\"]+)", "[REDACTED_SESSION]"),
        
        # Common API key formats
        (r"sk-[a-zA-Z0-9]{32,}", "[REDACTED_OPENAI_KEY]"),
        (r"ghp_[a-zA-Z0-9]{36}", "[REDACTED_GITHUB_TOKEN]"),
        (r"gho_[a-zA-Z0-9]{36}", "[REDACTED_GITHUB_TOKEN]"),
        (r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_KEY]"),
        
        # Email-like patterns in sensitive contexts
        (r"(user|email)\s*[:=]\s*['\"]?([^@\s]+@[^\s'\"]+)", r"\1=[REDACTED_EMAIL]"),
        
        # Credit card-like numbers
        (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[REDACTED_CARD]"),
        
        # SSN-like patterns
        (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
    ]
    
    @classmethod
    def redact(cls, text: str) -> str:
        """Redact sensitive information from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with sensitive data replaced
        """
        if not text:
            return text
        
        result = text
        for pattern, replacement in cls.REDACT_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    @classmethod
    def redact_dict(cls, data: dict) -> dict:
        """Recursively redact sensitive info from a dictionary.
        
        Args:
            data: Input dictionary
            
        Returns:
            Dictionary with sensitive values redacted
        """
        result = {}
        for key, value in data.items():
            # Skip storing these keys entirely
            key_lower = key.lower()
            if any(s in key_lower for s in ["password", "token", "secret", "cookie", "session"]):
                result[key] = "[REDACTED]"
            elif isinstance(value, str):
                result[key] = cls.redact(value)
            elif isinstance(value, dict):
                result[key] = cls.redact_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    cls.redact(v) if isinstance(v, str)
                    else cls.redact_dict(v) if isinstance(v, dict)
                    else v
                    for v in value
                ]
            else:
                result[key] = value
        
        return result


# =============================================================================
# Memory Store
# =============================================================================

class StructuredMemoryStore:
    """JSON-backed persistent memory for sites, directories, and recipes.
    
    Storage location: ~/.agentic_browser/memory/
    
    Files:
    - known_sites.json: Per-domain site information
    - known_directories.json: Named directory shortcuts
    - recipes.json: Reusable action sequences
    """
    
    def __init__(self, memory_dir: Optional[Path] = None):
        """Initialize the memory store.
        
        Args:
            memory_dir: Directory for memory files (default: ~/.agentic_browser/memory)
        """
        if memory_dir is None:
            memory_dir = get_base_dir() / "memory"
        
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.sites_file = self.memory_dir / "known_sites.json"
        self.dirs_file = self.memory_dir / "known_directories.json"
        self.recipes_file = self.memory_dir / "recipes.json"
        
        # Initialize files if they don't exist
        self._init_files()
    
    def _init_files(self) -> None:
        """Initialize memory files with empty structures."""
        if not self.sites_file.exists():
            self._write_json(self.sites_file, {"sites": {}})
        
        if not self.dirs_file.exists():
            self._write_json(self.dirs_file, {"directories": {}})
        
        if not self.recipes_file.exists():
            self._write_json(self.recipes_file, {"recipes": {}})
    
    def _read_json(self, path: Path) -> dict:
        """Read JSON file with error handling."""
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _write_json(self, path: Path, data: dict) -> None:
        """Write JSON file with pretty formatting."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    # =========================================================================
    # Known Sites
    # =========================================================================
    
    def get_site(self, domain: str) -> Optional[KnownSite]:
        """Get information about a known site.
        
        Args:
            domain: Domain to look up (e.g., github.com)
            
        Returns:
            KnownSite or None if not found
        """
        data = self._read_json(self.sites_file)
        site_data = data.get("sites", {}).get(domain.lower())
        
        if site_data:
            return KnownSite(**site_data)
        return None
    
    def save_site(self, site: KnownSite) -> None:
        """Save site information.
        
        Note: Data is redacted before storage.
        
        Args:
            site: Site information to save
        """
        data = self._read_json(self.sites_file)
        
        # Redact sensitive data
        site_dict = Redactor.redact_dict(site.model_dump())
        site_dict["last_accessed"] = datetime.now(timezone.utc).isoformat()
        
        data.setdefault("sites", {})
        data["sites"][site.domain.lower()] = site_dict
        
        self._write_json(self.sites_file, data)
    
    def list_sites(self) -> list[str]:
        """List all known site domains.
        
        Returns:
            List of domain names
        """
        data = self._read_json(self.sites_file)
        return list(data.get("sites", {}).keys())
    
    def delete_site(self, domain: str) -> bool:
        """Delete a known site.
        
        Args:
            domain: Domain to delete
            
        Returns:
            True if deleted, False if not found
        """
        data = self._read_json(self.sites_file)
        if domain.lower() in data.get("sites", {}):
            del data["sites"][domain.lower()]
            self._write_json(self.sites_file, data)
            return True
        return False
    
    # =========================================================================
    # Known Directories
    # =========================================================================
    
    def get_directory(self, name: str) -> Optional[KnownDirectory]:
        """Get information about a known directory.
        
        Args:
            name: Directory name (e.g., "Downloads")
            
        Returns:
            KnownDirectory or None if not found
        """
        data = self._read_json(self.dirs_file)
        dir_data = data.get("directories", {}).get(name.lower())
        
        if dir_data:
            return KnownDirectory(**dir_data)
        return None
    
    def save_directory(self, directory: KnownDirectory) -> None:
        """Save directory information.
        
        Args:
            directory: Directory information to save
        """
        data = self._read_json(self.dirs_file)
        
        data.setdefault("directories", {})
        data["directories"][directory.name.lower()] = directory.model_dump()
        
        self._write_json(self.dirs_file, data)
    
    def list_directories(self) -> list[str]:
        """List all known directory names.
        
        Returns:
            List of directory names
        """
        data = self._read_json(self.dirs_file)
        return list(data.get("directories", {}).keys())
    
    def delete_directory(self, name: str) -> bool:
        """Delete a known directory.
        
        Args:
            name: Directory name to delete
            
        Returns:
            True if deleted, False if not found
        """
        data = self._read_json(self.dirs_file)
        if name.lower() in data.get("directories", {}):
            del data["directories"][name.lower()]
            self._write_json(self.dirs_file, data)
            return True
        return False
    
    # =========================================================================
    # Recipes
    # =========================================================================
    
    def get_recipe(self, name: str) -> Optional[Recipe]:
        """Get a recipe by name.
        
        Args:
            name: Recipe name
            
        Returns:
            Recipe or None if not found
        """
        data = self._read_json(self.recipes_file)
        recipe_data = data.get("recipes", {}).get(name.lower())
        
        if recipe_data:
            return Recipe(**recipe_data)
        return None
    
    def save_recipe(self, recipe: Recipe) -> None:
        """Save a recipe.
        
        Note: Data is redacted before storage.
        
        Args:
            recipe: Recipe to save
        """
        data = self._read_json(self.recipes_file)
        
        # Redact sensitive data in steps
        recipe_dict = recipe.model_dump()
        recipe_dict["steps"] = [
            Redactor.redact_dict(step) for step in recipe_dict.get("steps", [])
        ]
        
        data.setdefault("recipes", {})
        data["recipes"][recipe.name.lower()] = recipe_dict
        
        self._write_json(self.recipes_file, data)
    
    def list_recipes(self) -> list[str]:
        """List all recipe names.
        
        Returns:
            List of recipe names
        """
        data = self._read_json(self.recipes_file)
        return list(data.get("recipes", {}).keys())
    
    def delete_recipe(self, name: str) -> bool:
        """Delete a recipe.
        
        Args:
            name: Recipe name to delete
            
        Returns:
            True if deleted, False if not found
        """
        data = self._read_json(self.recipes_file)
        if name.lower() in data.get("recipes", {}):
            del data["recipes"][name.lower()]
            self._write_json(self.recipes_file, data)
            return True
        return False
    
    # =========================================================================
    # Bootstrap
    # =========================================================================
    
    def bootstrap_defaults(self) -> dict[str, int]:
        """Pre-seed common directories based on user's system.
        
        Creates entries for:
        - Downloads
        - Documents
        - Pictures
        - Videos
        - Projects (if ~/Projects exists)
        - Desktop
        
        Returns:
            Dict with counts of items added
        """
        home = Path.home()
        added = {"directories": 0, "sites": 0}
        
        # Standard XDG directories
        standard_dirs = [
            ("Downloads", home / "Downloads"),
            ("Documents", home / "Documents"),
            ("Pictures", home / "Pictures"),
            ("Videos", home / "Videos"),
            ("Desktop", home / "Desktop"),
            ("Music", home / "Music"),
        ]
        
        # Check for common project directories
        project_dirs = [
            ("Projects", home / "Projects"),
            ("Code", home / "Code"),
            ("Development", home / "Development"),
            ("dev", home / "dev"),
        ]
        
        for name, path in standard_dirs + project_dirs:
            if path.exists() and path.is_dir():
                existing = self.get_directory(name)
                if not existing:
                    self.save_directory(KnownDirectory(
                        name=name,
                        path=str(path),
                        notes=f"Standard user directory",
                    ))
                    added["directories"] += 1
        
        return added
    
    # =========================================================================
    # Export/Import
    # =========================================================================
    
    def export_all(self) -> dict:
        """Export all memory data.
        
        Returns:
            Complete memory dump
        """
        return {
            "sites": self._read_json(self.sites_file).get("sites", {}),
            "directories": self._read_json(self.dirs_file).get("directories", {}),
            "recipes": self._read_json(self.recipes_file).get("recipes", {}),
        }
    
    def clear_all(self) -> None:
        """Clear all stored memory. Use with caution!"""
        self._write_json(self.sites_file, {"sites": {}})
        self._write_json(self.dirs_file, {"directories": {}})
        self._write_json(self.recipes_file, {"recipes": {}})


def create_memory_store(memory_dir: Optional[Path] = None) -> StructuredMemoryStore:
    """Factory function to create a memory store.
    
    Args:
        memory_dir: Optional custom directory
        
    Returns:
        Configured StructuredMemoryStore instance
    """
    return StructuredMemoryStore(memory_dir)
