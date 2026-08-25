from click.testing import CliRunner

from research.cli import cli


def test_cli_help_and_capabilities():
    runner = CliRunner()
    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0
    for cmd in ("env", "capabilities", "bench", "quant", "kv-cache", "speculative", "batching", "pareto", "suite", "optimize", "predict"):
        assert cmd in help_result.output

    cap = runner.invoke(cli, ["capabilities"])
    assert cap.exit_code == 0
    assert "experiments" in cap.output
    assert "tensorrt_llm" in cap.output
