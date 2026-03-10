from concurrent.futures import ThreadPoolExecutor


def test_create_notification_concurrent_ids_match_rows(isolated_db, monkeypatch):
    from src.notifications import service

    monkeypatch.setattr(service, "_push_notification", lambda _n: None)

    def _create(i: int):
        title = f"title-{i}"
        nid = service.create_notification(
            category="injury",
            severity="info",
            title=title,
            message="m",
            data={"i": i},
        )
        return nid, title

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_create, range(40)))

    ids = [nid for nid, _ in results]
    assert len(ids) == len(set(ids))

    for nid, title in results:
        row = isolated_db.fetch_one("SELECT title FROM notifications WHERE id = ?", (nid,))
        assert row is not None
        assert row["title"] == title
