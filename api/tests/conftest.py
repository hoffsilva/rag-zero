import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(API_DIR / "eval"))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
