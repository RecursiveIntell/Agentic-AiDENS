"""
Settings storage for Agentic Browser GUI.

Provides persistent JSON-based settings storage.
"""

import json
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict

from .config import get_base_dir
from .providers import Provider, ProviderConfig


def get_settings_path() -> Path:
    """Get the path to the settings file."""
    return get_base_dir() / "settings.json"


@dataclass
class Settings:
    """Application settings."""
    
    # Provider settings (browser domain)
    provider: str = "lm_studio"
    api_key: Optional[str] = None  # Deprecated: for backwards compat
    model: Optional[str] = None
    custom_endpoint: Optional[str] = None
    
    # Per-provider API keys (new: stores each provider's key separately)
    provider_api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Per-provider fetched model lists (stores refreshed models for each provider)
    provider_models: Dict[str, list] = field(default_factory=dict)
    
    # OS domain provider settings
    routing_mode: str = "auto"  # auto | browser | os | ask
    os_provider: str = "lm_studio"
    os_api_key: Optional[str] = None
    os_model: Optional[str] = None
    os_custom_endpoint: Optional[str] = None
    
    # Browser settings
    profile_name: str = "default"
    headless: bool = False
    
    # Agent settings
    max_steps: int = 30
    auto_approve: bool = False
    vision_mode: bool = False  # Enable vision (screenshots to LLM)
    debug_mode: bool = True    # Verbose debug output (default ON for now)
    
    # Window settings
    window_width: int = 900
    window_height: int = 700
    
    # n8n Integration Settings
    n8n_url: Optional[str] = None
    n8n_api_key: Optional[str] = None
    n8n_webhooks: Dict[str, str] = field(default_factory=dict)
    
    # LangSmith tracing settings
    langsmith_enabled: bool = False
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "agentic-browser"
    
    def get_api_key_for_provider(self, provider_value: str) -> Optional[str]:
        """Get API key for a specific provider."""
        # First check per-provider keys
        if provider_value in self.provider_api_keys:
            return self.provider_api_keys[provider_value]
        # Fallback to legacy api_key if it's the current provider
        if provider_value == self.provider and self.api_key:
            return self.api_key
        return None
    
    def set_api_key_for_provider(self, provider_value: str, api_key: Optional[str]) -> None:
        """Set API key for a specific provider."""
        if api_key:
            self.provider_api_keys[provider_value] = api_key
        elif provider_value in self.provider_api_keys:
            del self.provider_api_keys[provider_value]
    
    def get_models_for_provider(self, provider_value: str) -> list:
        """Get saved model list for a provider."""
        return self.provider_models.get(provider_value, [])
    
    def set_models_for_provider(self, provider_value: str, models: list) -> None:
        """Save model list for a provider."""
        self.provider_models[provider_value] = models
    
    def get_provider_config(self) -> ProviderConfig:
        """Get the provider configuration."""
        try:
            provider = Provider(self.provider)
        except ValueError:
            provider = Provider.LM_STUDIO
        
        # Get API key for the current provider
        api_key = self.get_api_key_for_provider(self.provider)
            
        return ProviderConfig(
            provider=provider,
            api_key=api_key,
            model=self.model,
            custom_endpoint=self.custom_endpoint,
        )
    
    def set_provider_config(self, config: ProviderConfig) -> None:
        """Set the provider configuration."""
        self.provider = config.provider.value
        self.set_api_key_for_provider(config.provider.value, config.api_key)
        self.model = config.model
        self.custom_endpoint = config.custom_endpoint


class SettingsStore:
    """Thread-safe settings storage."""
    
    _instance: Optional["SettingsStore"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "SettingsStore":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._settings: Settings = Settings()
        self._file_lock = threading.Lock()
        self._load()
    
    @property
    def settings(self) -> Settings:
        """Get current settings."""
        return self._settings
    
    def _load(self) -> None:
        """Load settings from file."""
        path = get_settings_path()
        if not path.exists():
            return
            
        try:
            with self._file_lock:
                data = json.loads(path.read_text())
                self._settings = Settings(
                    # Browser domain settings
                    provider=data.get("provider", "lm_studio"),
                    api_key=data.get("api_key"),  # Legacy
                    model=data.get("model"),
                    custom_endpoint=data.get("custom_endpoint"),
                    # Per-provider API keys
                    provider_api_keys=data.get("provider_api_keys", {}),
                    # Per-provider model lists
                    provider_models=data.get("provider_models", {}),
                    # OS domain settings
                    routing_mode=data.get("routing_mode", "auto"),
                    os_provider=data.get("os_provider", "lm_studio"),
                    os_api_key=data.get("os_api_key"),
                    os_model=data.get("os_model"),
                    os_custom_endpoint=data.get("os_custom_endpoint"),
                    # n8n Integration Settings
                    n8n_url=data.get("n8n_url"),
                    n8n_api_key=data.get("n8n_api_key"),
                    n8n_webhooks=data.get("n8n_webhooks", {}),
                    # Other settings
                    profile_name=data.get("profile_name", "default"),
                    headless=data.get("headless", False),
                    max_steps=data.get("max_steps", 30),
                    auto_approve=data.get("auto_approve", False),
                    vision_mode=data.get("vision_mode", False),
                    debug_mode=data.get("debug_mode", True),
                    window_width=data.get("window_width", 900),
                    window_height=data.get("window_height", 700),
                )
        except (json.JSONDecodeError, KeyError):
            # Use defaults on error
            self._settings = Settings()
    
    def save(self) -> None:
        """Save settings to file."""
        path = get_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            # Browser domain settings
            "provider": self._settings.provider,
            "api_key": self._settings.api_key,  # Legacy
            "model": self._settings.model,
            "custom_endpoint": self._settings.custom_endpoint,
            # Per-provider API keys
            "provider_api_keys": self._settings.provider_api_keys,
            # Per-provider model lists
            "provider_models": self._settings.provider_models,
            # OS domain settings
            "routing_mode": self._settings.routing_mode,
            "os_provider": self._settings.os_provider,
            "os_api_key": self._settings.os_api_key,
            "os_model": self._settings.os_model,
            "os_custom_endpoint": self._settings.os_custom_endpoint,
            # n8n Integration Settings
            "n8n_url": self._settings.n8n_url,
            "n8n_api_key": self._settings.n8n_api_key,
            "n8n_webhooks": self._settings.n8n_webhooks,
            # Other settings
            "profile_name": self._settings.profile_name,
            "headless": self._settings.headless,
            "max_steps": self._settings.max_steps,
            "auto_approve": self._settings.auto_approve,
            "vision_mode": self._settings.vision_mode,
            "debug_mode": self._settings.debug_mode,
            "window_width": self._settings.window_width,
            "window_height": self._settings.window_height,
        }
        
        with self._file_lock:
            path.write_text(json.dumps(data, indent=2))
    
    def update(self, **kwargs) -> None:
        """Update settings and save.
        
        Args:
            **kwargs: Settings fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
        self.save()
    
    def reset(self) -> None:
        """Reset to default settings."""
        self._settings = Settings()
        self.save()


def get_settings() -> Settings:
    """Get the current settings."""
    return SettingsStore().settings


def save_settings() -> None:
    """Save the current settings."""
    SettingsStore().save()


def update_settings(**kwargs) -> None:
    """Update and save settings."""
    SettingsStore().update(**kwargs)
