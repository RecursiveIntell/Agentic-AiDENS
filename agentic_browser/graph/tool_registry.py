"""
Tool registry for LangGraph serialization compatibility.

Stores tool instances outside of state to enable checkpointing.
"""

import threading
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .browser_manager import LazyBrowserManager


class ToolSet:
    """Container for tools associated with a session.
    
    Supports lazy browser initialization via browser_manager.
    """
    
    def __init__(
        self,
        config: Any,
        browser_manager: Optional["LazyBrowserManager"] = None,
        os_tools: Any = None,
    ):
        """Initialize tool set.
        
        Args:
            config: AgentConfig instance
            browser_manager: Optional LazyBrowserManager for on-demand browser
            os_tools: Optional OSTools instance
        """
        self.config = config
        self._browser_manager = browser_manager
        self.os_tools = os_tools
    
    @property
    def browser_tools(self) -> Any:
        """Get browser tools, lazily initializing browser if needed.
        
        Returns:
            BrowserTools instance or None if no browser_manager
        """
        if self._browser_manager is None:
            return None
        return self._browser_manager.get_browser_tools()
    
    @property
    def browser_manager(self) -> Optional["LazyBrowserManager"]:
        """Get the browser manager (without triggering initialization)."""
        return self._browser_manager


class ToolRegistry:
    """Singleton registry for tool instances.
    
    Tools are registered by session_id and looked up by nodes.
    This allows state to store only the session_id (a string)
    which is serializable for checkpointing.
    
    Usage:
        # Register tools when starting a session
        registry = ToolRegistry.get_instance()
        registry.register("session-123", config, os_tools=my_os_tools)
        
        # Look up tools in node functions
        tools = registry.get("session-123")
        os_tools = tools.os_tools
    """
    
    _instance: Optional["ToolRegistry"] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._tools: dict[str, ToolSet] = {}
        self._tools_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def register(
        self,
        session_id: str,
        config: Any,
        browser_manager: Optional["LazyBrowserManager"] = None,
        os_tools: Any = None,
    ) -> None:
        """Register tools for a session.
        
        Args:
            session_id: Unique session identifier
            config: AgentConfig instance
            browser_manager: Optional LazyBrowserManager for on-demand browser
            os_tools: Optional OSTools instance
        """
        with self._tools_lock:
            self._tools[session_id] = ToolSet(
                config=config,
                browser_manager=browser_manager,
                os_tools=os_tools,
            )
    
    def get(self, session_id: str) -> Optional[ToolSet]:
        """Get tools for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ToolSet or None if not found
        """
        with self._tools_lock:
            return self._tools.get(session_id)
    
    def update_browser_manager(self, session_id: str, browser_manager: "LazyBrowserManager") -> None:
        """Update browser manager for a session."""
        with self._tools_lock:
            if session_id in self._tools:
                self._tools[session_id]._browser_manager = browser_manager
    
    def update_os_tools(self, session_id: str, os_tools: Any) -> None:
        """Update OS tools for a session."""
        with self._tools_lock:
            if session_id in self._tools:
                self._tools[session_id].os_tools = os_tools
    
    def unregister(self, session_id: str) -> None:
        """Remove tools for a session.
        
        Args:
            session_id: Session identifier
        """
        with self._tools_lock:
            self._tools.pop(session_id, None)
    
    def clear(self) -> None:
        """Clear all registered tools."""
        with self._tools_lock:
            self._tools.clear()


# Convenience function
def get_tools(session_id: str) -> Optional[ToolSet]:
    """Get tools for a session from the global registry."""
    return ToolRegistry.get_instance().get(session_id)
