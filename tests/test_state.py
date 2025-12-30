"""Unit tests for AgentState reducers."""

from agentic_browser.graph.state import bounded_files_reducer


def test_bounded_files_reducer_dedupes_and_caps_length():
    existing = [f"/path/{i}" for i in range(45)]
    new = [
        "/path/10",
        "/path/46",
        "/path/47",
        "/path/48",
        "/path/49",
        "/path/50",
        "/path/51",
        "/path/52",
        "/path/53",
        "/path/54",
        "/path/55",
    ]

    reduced = bounded_files_reducer(existing, new)

    assert len(reduced) == 50
    assert len(set(reduced)) == len(reduced)
    assert reduced[0] == "/path/5"
    assert reduced[-1] == "/path/55"
    assert reduced.count("/path/10") == 1
