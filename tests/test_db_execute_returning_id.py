def test_execute_returning_id_persists_correct_row(isolated_db):
    row_id = isolated_db.execute_returning_id(
        """
        INSERT INTO notifications (category, severity, title, message, created_at, read, data)
        VALUES (?, ?, ?, ?, datetime('now'), 0, ?)
        """,
        ("system", "info", "id-check", "msg", ""),
    )

    assert row_id > 0
    row = isolated_db.fetch_one("SELECT id, title FROM notifications WHERE id = ?", (row_id,))
    assert row is not None
    assert row["id"] == row_id
    assert row["title"] == "id-check"
