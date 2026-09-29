import pytest

from sftmill.cli import build_parser, main


def test_parser_subcommands():
    parser = build_parser()
    subs = {action.dest for action in parser._subparsers._actions if hasattr(action, "choices") and action.choices}
    assert subs == {"command"}
    command_parser = next(
        a for a in parser._subparsers._actions if hasattr(a, "choices") and a.choices
    )
    assert set(command_parser.choices) == {"tasks", "generate"}


@pytest.mark.parametrize("argv", [
    ["train", "--config", "x.yaml"],
    ["smoke", "--config", "x.yaml"],
    ["bench", "--config", "x.yaml"],
    ["probe"],
    ["shell", "--config", "x.yaml"],
])
def test_unknown_commands_fail(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code != 0
