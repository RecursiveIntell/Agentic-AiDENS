import tempfile
from pathlib import Path

from agentic_browser.graph.memory import SessionStore
from agentic_browser.graph.run_history import RecallTool


def test_search_runs_includes_non_success_when_disabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(Path(tmpdir) / "sessions.db")
        recall_tool = RecallTool(store)

        store.create_session("success-1", "query alpha", {"messages": []})
        store._do_update_session("success-1", {"task_complete": True, "final_answer": "ok"})

        store.create_session("failure-1", "query alpha", {"messages": []})
        store._do_update_session("failure-1", {"task_complete": True, "error": "boom"})

        result = recall_tool.search_runs("query", limit=5, success_only=False)
        assert result.success is True
        assert any(run["status"] != "success" for run in result.data)
