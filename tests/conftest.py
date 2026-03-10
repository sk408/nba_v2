import pytest
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Create an isolated SQLite DB for a test."""
    from src.database import db
    from src.database.migrations import init_db

    db_path = tmp_path / "nba_test.db"
    monkeypatch.setattr(db, "get_db_path", lambda: str(db_path))

    db.close_all()
    init_db()
    yield db
    db.close_all()
