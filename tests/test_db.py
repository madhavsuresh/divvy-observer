import duckdb

from divvy import config, db


def test_refresh_read_replica_with_pending_writes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "live.duckdb")
    monkeypatch.setattr(config, "READ_DB_PATH", tmp_path / "replica.duckdb")
    monkeypatch.setattr(config, "ENABLE_READ_REPLICA", True)

    with db.session(read_only=False) as conn:
        db.init_schema(conn)
        conn.execute(
            "INSERT INTO stations VALUES "
            "('s1', NULL, 'S1', 'One', 41.0, -87.0, 15, 'classic', NULL, NULL)"
        )
        assert db.refresh_read_replica() is True

    replica = duckdb.connect(str(config.READ_DB_PATH), read_only=True)
    try:
        assert replica.execute("SELECT COUNT(*) FROM stations").fetchone()[0] == 1
    finally:
        replica.close()


def test_refresh_read_replica_disabled_returns_false(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "live.duckdb")
    monkeypatch.setattr(config, "READ_DB_PATH", tmp_path / "replica.duckdb")
    monkeypatch.setattr(config, "ENABLE_READ_REPLICA", False)

    assert db.refresh_read_replica() is False
