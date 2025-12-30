from unittest.mock import patch

from agentic_browser import cli
from agentic_browser.config import AgentConfig


def test_run_command_passes_no_memory_flag():
    parser = cli.create_parser()
    args = parser.parse_args(["run", "Summarize the page", "--no-memory"])

    with (
        patch.object(cli.AgentConfig, "from_cli_args", return_value=AgentConfig(goal="test")) as mock_from_cli_args,
        patch.object(cli, "run_graph_command", return_value=0),
    ):
        result = cli.run_command(args)

    assert result == 0
    assert mock_from_cli_args.call_args.kwargs["no_memory"] is True
