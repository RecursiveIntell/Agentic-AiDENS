import pytest

from agentic_browser.cli import create_parser


def test_explain_flag_rejected():
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "do something", "--explain"])
