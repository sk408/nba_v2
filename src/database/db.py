"""SQLite database with in-memory read cache.

Architecture:
  - A single shared in-memory SQLite DB serves ALL reads (instant).
  - A thread-local disk connection handles writes with WAL mode.
  - Every write is applied to BOTH disk and memory atomically
    so reads always reflect the latest state.
  - On startup the disk DB is loaded into memory via sqlite3 backup().
  - Readers-writer lock: multiple concurrent readers, exclusive writer.
    This prevents autotune's heavy DB reads from blocking the UI thread.
"""

import sqlite3
import threading
import os
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.config import get_db_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Readers-Writer Lock — multiple concurrent readers, exclusive writers
# ---------------------------------------------------------------------------

class _RWLock:
    """Readers-writer lock with writer-preference to prevent starvation."""

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._pending_writers = 0

    def read_acquire(self):
        with self._cond:
            while self._writer or self._pending_writers > 0:
                self._cond.wait()
            self._readers += 1

    def read_release(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def write_acquire(self):
        with self._cond:
            self._pending_writers += 1
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._pending_writers -= 1
            self._writer = True

    def write_release(self):
        with self._cond:
            self._writer = False
            self._cond.notify_all()


_rwlock = _RWLock()                       # guards all db access
_disk_local = threading.local()           # thread-local disk connections
_disk_conns: List[sqlite3.Connection] = []  # track all disk connections for cleanup
_disk_conns_lock = threading.Lock()
_mem_conn: Optional[sqlite3.Connection] = None   # single shared memory db
_init_lock = threading.Lock()             # one-shot init guard

DB_BUSY_TIMEOUT = 30000  # 30 seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_db_path() -> str:
    path = get_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def _get_disk_conn() -> sqlite3.Connection:
    """Thread-local disk connection (WAL mode, auto-commit off)."""
    if not hasattr(_disk_local, "conn") or _disk_local.conn is None:
        db_path = _get_db_path()
        conn = sqlite3.connect(db_path, timeout=DB_BUSY_TIMEOUT / 1000)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        _disk_local.conn = conn
        with _disk_conns_lock:
            _disk_conns.append(conn)
    return _disk_local.conn


def _get_mem_conn() -> sqlite3.Connection:
    """Ensure the shared in-memory DB is initialised (thread-safe)."""
    global _mem_conn
    if _mem_conn is not None:
        return _mem_conn
    with _init_lock:
        if _mem_conn is not None:  # double-check after acquiring lock
            return _mem_conn
        _mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
        _mem_conn.execute("PRAGMA foreign_keys=ON")
        _mem_conn.row_factory = sqlite3.Row
        # Load disk → memory
        db_path = _get_db_path()
        if os.path.exists(db_path):
            disk = sqlite3.connect(db_path, timeout=DB_BUSY_TIMEOUT / 1000)
            disk.backup(_mem_conn)
            disk.close()
            logger.info("Database loaded into memory from %s", db_path)
        else:
            logger.info("No database file yet — empty in-memory DB created")
    return _mem_conn


# ---------------------------------------------------------------------------
# Public: backward-compatible API (used by every module via `db.*`)
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return the shared in-memory connection.

    WARNING: internal use only (migrations/init_db).  The write lock is NOT
    held after return — callers must use execute()/execute_script() for
    thread-safe access.
    """
    return _get_mem_conn()


def execute(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    """Execute a write on BOTH disk and memory (exclusive lock)."""
    _rwlock.write_acquire()
    try:
        # 1. Write to disk (crash-safe, WAL)
        disk = _get_disk_conn()
        try:
            disk.execute(sql, params)
            disk.commit()
        except Exception:
            disk.rollback()
            raise

        # 2. Replay on in-memory copy
        mem = _get_mem_conn()
        try:
            cursor = mem.execute(sql, params)
            mem.commit()
            return cursor
        except Exception:
            # Memory is out of sync — full reload
            mem.rollback()
            _reload_memory_unlocked()
            return mem.execute("SELECT 0")  # return a valid cursor
    finally:
        _rwlock.write_release()


def execute_many(sql: str, params_list: List[tuple]) -> None:
    """Execute a write with multiple param sets on BOTH disk and memory."""
    _rwlock.write_acquire()
    try:
        disk = _get_disk_conn()
        try:
            disk.executemany(sql, params_list)
            disk.commit()
        except Exception:
            disk.rollback()
            raise

        mem = _get_mem_conn()
        try:
            mem.executemany(sql, params_list)
            mem.commit()
        except Exception:
            mem.rollback()
            _reload_memory_unlocked()
    finally:
        _rwlock.write_release()


def execute_script(sql: str) -> None:
    """Execute a multi-statement SQL script on BOTH disk and memory.

    NOTE: SQLite's executescript() issues an implicit COMMIT before running,
    so partial failures cannot be rolled back.  Only use for idempotent DDL
    (CREATE TABLE IF NOT EXISTS).  For non-idempotent DML use execute().
    """
    _rwlock.write_acquire()
    try:
        disk = _get_disk_conn()
        try:
            disk.executescript(sql)
        except Exception:
            disk.rollback()
            raise

        mem = _get_mem_conn()
        try:
            mem.executescript(sql)
        except Exception:
            mem.rollback()
            _reload_memory_unlocked()
    finally:
        _rwlock.write_release()


def fetch_one(sql: str, params: tuple = ()) -> Optional[dict]:
    """Fetch a single row from the in-memory DB (shared read lock).

    If the calling thread has a thread-local DB (via thread_local_db()),
    uses that instead of the shared connection — no lock needed.
    """
    local = getattr(_thread_local_db, "conn", None)
    if local is not None:
        row = local.execute(sql, params).fetchone()
        return dict(row) if row else None

    _rwlock.read_acquire()
    try:
        mem = _get_mem_conn()
        row = mem.execute(sql, params).fetchone()
        return dict(row) if row else None
    finally:
        _rwlock.read_release()


def fetch_all(sql: str, params: tuple = ()) -> List[dict]:
    """Fetch all rows from the in-memory DB (shared read lock).

    If the calling thread has a thread-local DB (via thread_local_db()),
    uses that instead of the shared connection — no lock needed.
    """
    local = getattr(_thread_local_db, "conn", None)
    if local is not None:
        return [dict(r) for r in local.execute(sql, params).fetchall()]

    _rwlock.read_acquire()
    try:
        mem = _get_mem_conn()
        return [dict(r) for r in mem.execute(sql, params).fetchall()]
    finally:
        _rwlock.read_release()


# ---------------------------------------------------------------------------
# Thread-local DB for parallel workloads
# ---------------------------------------------------------------------------

_thread_local_db = threading.local()


class thread_local_db:
    """Context manager that gives the calling thread its own in-memory DB.

    Usage::

        with thread_local_db():
            # all db.fetch_one / db.fetch_all calls on this thread
            # use a private in-memory copy — zero lock contention.
            result = db.fetch_one("SELECT ...")

    The copy is created via sqlite3.backup() from the shared in-memory DB,
    and is closed + discarded on exit.  Write operations still go through
    the shared path (disk + memory) and are NOT reflected in the copy.
    """

    def __enter__(self):
        mem = _get_mem_conn()
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        _rwlock.read_acquire()
        try:
            mem.backup(conn)
        finally:
            _rwlock.read_release()
        _thread_local_db.conn = conn
        return self

    def __exit__(self, *exc):
        conn = getattr(_thread_local_db, "conn", None)
        _thread_local_db.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        return False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def reload_memory() -> None:
    """Fully reload the in-memory DB from disk (call after migrations)."""
    _rwlock.write_acquire()
    try:
        _reload_memory_unlocked()
    finally:
        _rwlock.write_release()


def _reload_memory_unlocked() -> None:
    """Reload without acquiring write lock (caller must hold it)."""
    global _mem_conn
    old_conn = _mem_conn

    # Build new connection first — if backup fails, keep old connection
    new_conn = sqlite3.connect(":memory:", check_same_thread=False)
    new_conn.execute("PRAGMA foreign_keys=ON")
    new_conn.row_factory = sqlite3.Row

    db_path = _get_db_path()
    if os.path.exists(db_path):
        disk = sqlite3.connect(db_path, timeout=DB_BUSY_TIMEOUT / 1000)
        try:
            disk.backup(new_conn)
        except Exception:
            disk.close()
            new_conn.close()
            logger.error("Failed to backup disk DB to memory — keeping previous state")
            return
        disk.close()

    # Swap atomically then close old
    _mem_conn = new_conn
    if old_conn is not None:
        try:
            old_conn.close()
        except Exception:
            pass
    logger.info("In-memory database reloaded from disk")


def close_connection():
    """Close the thread-local disk connection and remove from tracking list."""
    if hasattr(_disk_local, "conn") and _disk_local.conn is not None:
        conn = _disk_local.conn
        _disk_local.conn = None
        with _disk_conns_lock:
            try:
                _disk_conns.remove(conn)
            except ValueError:
                pass
        conn.close()


def close_all():
    """Close ALL disk + memory connections (for shutdown)."""
    global _mem_conn
    # Close all tracked disk connections from every thread
    with _disk_conns_lock:
        for conn in _disk_conns:
            try:
                conn.close()
            except Exception:
                pass
        _disk_conns.clear()
    _disk_local.conn = None
    _rwlock.write_acquire()
    try:
        if _mem_conn is not None:
            try:
                _mem_conn.close()
            except Exception:
                pass
            _mem_conn = None
    finally:
        _rwlock.write_release()


def get_db_size() -> str:
    """Return human-readable DB file size."""
    db_path = _get_db_path()
    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    return "0 B"


def delete_database():
    """Delete the database file, WAL/SHM files, and reset memory."""
    global _mem_conn

    # Acquire write lock FIRST to block concurrent execute() calls,
    # then close all disk connections safely inside the lock.
    _rwlock.write_acquire()
    try:
        with _disk_conns_lock:
            for conn in _disk_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            _disk_conns.clear()
        _disk_local.conn = None

        if _mem_conn is not None:
            try:
                _mem_conn.close()
            except Exception:
                pass
            _mem_conn = None
    finally:
        _rwlock.write_release()

    db_path = _get_db_path()
    for suffix in ["", "-wal", "-shm"]:
        p = db_path + suffix
        if os.path.exists(p):
            os.remove(p)
