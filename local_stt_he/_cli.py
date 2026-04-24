"""Entry point wrappers — keeps CLI scripts importable from the installed package."""
import sys
import pathlib

# Ensure the repo root is on sys.path so the top-level scripts can be imported.
_REPO = pathlib.Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def file():
    from transcribe_file import main
    main()


def live():
    from transcribe_live import main
    main()


def benchmark():
    from run_benchmark import main
    main()
