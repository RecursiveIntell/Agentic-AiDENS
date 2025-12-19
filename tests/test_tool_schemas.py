"""
Tests for tool schema validation.

Tests Pydantic models for OS and browser tools, validation rules, and schema registry.
"""

import pytest
from pathlib import Path

from agentic_browser.tool_schemas import (
    # OS Schemas
    ListDirRequest,
    ReadFileRequest,
    WriteFileRequest,
    RunCommandRequest,
    MoveFileRequest,
    CopyFileRequest,
    DeleteFileRequest,
    # Browser Schemas  
    GotoRequest,
    ClickRequest,
    TypeRequest,
    PressRequest,
    ScrollRequest,
    WaitForRequest,
    ExtractRequest,
    ExtractVisibleTextRequest,
    ScreenshotRequest,
    DownloadFileRequest,
    # Action Wrapper
    ActionRequest,
    ActionDomain,
    ActionResult,
    RiskLevel,
    # Registry functions
    get_schema_for_action,
    validate_action_args,
    create_typed_request,
    ACTION_SCHEMAS,
)


class TestListDirRequest:
    """Tests for ListDirRequest schema."""
    
    def test_valid_path(self):
        """Test valid directory path."""
        req = ListDirRequest(path="/home/user/docs")
        assert req.path == "/home/user/docs"
    
    def test_tilde_expansion(self):
        """Test tilde expansion in path."""
        req = ListDirRequest(path="~/Downloads")
        assert "~" not in req.path
        assert "Downloads" in req.path
    
    def test_empty_path_rejected(self):
        """Test empty path is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ListDirRequest(path="")


class TestReadFileRequest:
    """Tests for ReadFileRequest schema."""
    
    def test_valid_request(self):
        """Test valid read file request."""
        req = ReadFileRequest(path="/home/user/file.txt")
        assert req.path == "/home/user/file.txt"
        assert req.max_bytes == 1048576  # default
    
    def test_custom_max_bytes(self):
        """Test custom max_bytes."""
        req = ReadFileRequest(path="/file.txt", max_bytes=5000)
        assert req.max_bytes == 5000
    
    def test_max_bytes_limits(self):
        """Test max_bytes limits are enforced."""
        # Too small
        with pytest.raises(ValueError):
            ReadFileRequest(path="/file.txt", max_bytes=0)
        
        # Too large
        with pytest.raises(ValueError):
            ReadFileRequest(path="/file.txt", max_bytes=100000000)


class TestWriteFileRequest:
    """Tests for WriteFileRequest schema."""
    
    def test_valid_request(self):
        """Test valid write file request."""
        req = WriteFileRequest(path="/home/user/file.txt", content="hello")
        assert req.path == "/home/user/file.txt"
        assert req.content == "hello"
        assert req.mode == "overwrite"
    
    def test_append_mode(self):
        """Test append mode."""
        req = WriteFileRequest(path="/file.txt", content="data", mode="append")
        assert req.mode == "append"
    
    def test_invalid_mode(self):
        """Test invalid mode is rejected."""
        with pytest.raises(ValueError):
            WriteFileRequest(path="/file.txt", content="data", mode="invalid")


class TestRunCommandRequest:
    """Tests for RunCommandRequest schema."""
    
    def test_valid_argv(self):
        """Test valid argv list."""
        req = RunCommandRequest(argv=["ls", "-la"])
        assert req.argv == ["ls", "-la"]
        assert req.timeout_s == 30  # default
    
    def test_full_request(self):
        """Test full request with all options."""
        req = RunCommandRequest(
            argv=["grep", "-r", "pattern", "."],
            cwd="/home/user/project",
            timeout_s=60
        )
        assert req.argv == ["grep", "-r", "pattern", "."]
        assert "/home/user/project" in req.cwd
        assert req.timeout_s == 60
    
    def test_empty_argv_rejected(self):
        """Test empty argv is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            RunCommandRequest(argv=[])
    
    def test_timeout_limits(self):
        """Test timeout limits are enforced."""
        # Too small
        with pytest.raises(ValueError):
            RunCommandRequest(argv=["ls"], timeout_s=0)
        
        # Too large
        with pytest.raises(ValueError):
            RunCommandRequest(argv=["ls"], timeout_s=500)


class TestGotoRequest:
    """Tests for GotoRequest schema."""
    
    def test_full_url(self):
        """Test full URL."""
        req = GotoRequest(url="https://example.com")
        assert req.url == "https://example.com"
    
    def test_http_url(self):
        """Test HTTP URL."""
        req = GotoRequest(url="http://localhost:8080")
        assert req.url == "http://localhost:8080"
    
    def test_bare_domain_gets_https(self):
        """Test bare domain gets https:// prefix."""
        req = GotoRequest(url="example.com")
        assert req.url == "https://example.com"
    
    def test_empty_url_rejected(self):
        """Test empty URL is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GotoRequest(url="")


class TestClickRequest:
    """Tests for ClickRequest schema."""
    
    def test_valid_selector(self):
        """Test valid selector."""
        req = ClickRequest(selector="button.submit")
        assert req.selector == "button.submit"
        assert req.timeout_ms == 10000  # default
    
    def test_text_selector(self):
        """Test text selector."""
        req = ClickRequest(selector='text="Click me"')
        assert req.selector == 'text="Click me"'
    
    def test_custom_timeout(self):
        """Test custom timeout."""
        req = ClickRequest(selector="#btn", timeout_ms=5000)
        assert req.timeout_ms == 5000
    
    def test_timeout_limits(self):
        """Test timeout limits."""
        # Too small
        with pytest.raises(ValueError):
            ClickRequest(selector="#btn", timeout_ms=50)
        
        # Too large
        with pytest.raises(ValueError):
            ClickRequest(selector="#btn", timeout_ms=100000)


class TestTypeRequest:
    """Tests for TypeRequest schema."""
    
    def test_valid_request(self):
        """Test valid type request."""
        req = TypeRequest(selector="#input", text="hello")
        assert req.selector == "#input"
        assert req.text == "hello"
        assert req.clear_first is True
    
    def test_no_clear(self):
        """Test without clearing first."""
        req = TypeRequest(selector="#input", text="world", clear_first=False)
        assert req.clear_first is False


class TestActionRequest:
    """Tests for ActionRequest wrapper."""
    
    def test_os_action(self):
        """Test OS action wrapper."""
        req = ActionRequest(
            action="os_list_dir",
            domain=ActionDomain.OS,
            request=ListDirRequest(path="/home"),
            rationale="List home directory"
        )
        assert req.action == "os_list_dir"
        assert req.domain == ActionDomain.OS
        assert req.rationale == "List home directory"
    
    def test_browser_action(self):
        """Test browser action wrapper."""
        req = ActionRequest(
            action="goto",
            domain=ActionDomain.BROWSER,
            request=GotoRequest(url="https://example.com"),
        )
        assert req.action == "goto"
        assert req.domain == ActionDomain.BROWSER
    
    def test_to_legacy_args(self):
        """Test legacy args conversion."""
        req = ActionRequest(
            action="os_list_dir",
            domain=ActionDomain.OS,
            request=ListDirRequest(path="/home/user"),
        )
        args = req.to_legacy_args()
        assert args["path"] == "/home/user"


class TestActionResult:
    """Tests for ActionResult."""
    
    def test_success_result(self):
        """Test success result."""
        result = ActionResult(
            success=True,
            message="Command succeeded",
            data={"returncode": 0},
        )
        assert result.success is True
        assert result.risk_level == RiskLevel.LOW
    
    def test_to_dict(self):
        """Test to_dict conversion."""
        result = ActionResult(success=False, message="Failed")
        d = result.to_dict()
        assert d["success"] is False
        assert d["message"] == "Failed"
        assert d["risk_level"] == "low"


class TestSchemaRegistry:
    """Tests for schema registry functions."""
    
    def test_all_actions_have_schemas(self):
        """Test all documented actions have schemas."""
        os_actions = ["os_list_dir", "os_read_file", "os_write_file", "os_exec"]
        browser_actions = ["goto", "click", "type", "press", "scroll"]
        
        for action in os_actions + browser_actions:
            schema = get_schema_for_action(action)
            assert schema is not None, f"Missing schema for {action}"
    
    def test_get_unknown_action(self):
        """Test getting unknown action returns None."""
        schema = get_schema_for_action("unknown_action")
        assert schema is None
    
    def test_validate_valid_args(self):
        """Test validating valid args."""
        is_valid, model, error = validate_action_args(
            "os_list_dir",
            {"path": "/home"}
        )
        assert is_valid is True
        assert model is not None
        assert error is None
    
    def test_validate_invalid_args(self):
        """Test validating invalid args."""
        is_valid, model, error = validate_action_args(
            "os_list_dir",
            {"path": ""}  # Empty path
        )
        assert is_valid is False
        assert model is None
        assert "empty" in error.lower()
    
    def test_validate_unknown_action(self):
        """Test validating unknown action."""
        is_valid, model, error = validate_action_args(
            "fake_action",
            {"arg": "value"}
        )
        assert is_valid is False
        assert "Unknown action" in error
    
    def test_create_typed_request(self):
        """Test creating typed request."""
        req = create_typed_request("goto", {"url": "https://example.com"})
        assert req is not None
        assert isinstance(req, GotoRequest)
    
    def test_create_typed_request_invalid(self):
        """Test creating typed request with invalid args."""
        req = create_typed_request("goto", {"url": ""})
        assert req is None


class TestOSToolsTypedExecution:
    """Tests for OSTools typed execution (integration)."""
    
    def test_execute_typed_list_dir(self):
        """Test typed list_dir execution."""
        from agentic_browser.os_tools import OSTools
        
        tools = OSTools()
        req = ListDirRequest(path="~")
        result = tools.execute_typed(req)
        
        assert result.success is True
        assert result.data is not None
        assert "entries" in result.data
    
    def test_execute_typed_run_command(self):
        """Test typed run_command execution."""
        from agentic_browser.os_tools import OSTools
        
        tools = OSTools()
        req = RunCommandRequest(argv=["echo", "hello"])
        result = tools.execute_typed(req)
        
        assert result.success is True
        assert "hello" in result.data["stdout"]
        assert result.data["argv"] == ["echo", "hello"]
    
    def test_legacy_argv_support(self):
        """Test legacy execute() with argv list."""
        from agentic_browser.os_tools import OSTools
        
        tools = OSTools()
        result = tools.execute("os_exec", {"argv": ["echo", "world"]})
        
        assert result.success is True
        assert "world" in result.data["stdout"]
