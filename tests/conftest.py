import sys
from pathlib import Path

import pytest

# make tests/fakes.py importable as `fakes`
sys.path.insert(0, str(Path(__file__).parent))

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
_STATS: list[str] = []


@pytest.fixture(scope="session")
def stats():
    """Tests append lines here; they are printed in a 'dataset stats' section after the run."""
    return _STATS


def pytest_terminal_summary(terminalreporter):
    if _STATS:
        terminalreporter.section("dataset stats")
        for line in _STATS:
            terminalreporter.write_line(line)


@pytest.fixture(scope="session")
def locomo_data():
    path = RAW / "locomo10.json"
    if not path.exists():
        pytest.skip("data/raw/locomo10.json missing (run `make data`)")
    from eval.loaders.locomo import load_locomo

    return load_locomo(path)


@pytest.fixture(scope="session")
def lme_data():
    path = RAW / "longmemeval_s.json"
    if not path.exists():
        pytest.skip("data/raw/longmemeval_s.json missing (run `make data`)")
    from eval.loaders.longmemeval import load_longmemeval

    return load_longmemeval(path)
