from MetaMan import state


def _settings(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "SETTINGS_FILE", tmp_path / "settings.json")
    return state.AppSettings()


def test_save_is_atomic_and_roundtrips(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    s.put_server_root_for_project("ProjX", r"\\server\share")
    # no stray temp files left behind by the atomic write
    leftovers = [p for p in (tmp_path).iterdir() if p.name.startswith(".settings_")]
    assert not leftovers
    s2 = state.AppSettings()
    assert s2.get_server_root_for_project("ProjX") == r"\\server\share"


def test_version_migration(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    assert int(s._data["_version"]) == state.SETTINGS_VERSION


def test_backup_history(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    for i in range(3):
        s.add_backup_run({"project": "P", "destination_kind": "server", "status": "success", "i": i})
    s.add_backup_run({"project": "Q", "destination_kind": "hdd", "status": "error"})
    p_hist = s.get_backup_history(project="P")
    assert len(p_hist) == 3 and p_hist[0]["i"] == 2  # newest first
    last = s.get_last_backup_for("P", kind="server")
    assert last["i"] == 2
    assert s.get_last_backup_for("P", kind="hdd") == {}


def test_history_capped(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    cap = state.AppSettings._MAX_BACKUP_RUNS
    for i in range(cap + 25):
        s.add_backup_run({"project": "P", "i": i})
    assert len(s.get_backup_history(project="P")) == cap


def test_schedule_verify_prune_roundtrip(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    s.put_backup_schedule_for_project("P", True, "02:00", verify=True, prune=True)
    sched = s.get_backup_schedule_for_project("P")
    assert sched["verify"] is True and sched["prune"] is True
    # marking a run preserves the flags
    s.mark_backup_schedule_run("P", "2026-06-18")
    sched2 = s.get_backup_schedule_for_project("P")
    assert sched2["verify"] is True and sched2["prune"] is True
    assert sched2["last_run_date"] == "2026-06-18"


def test_explorer_settings(tmp_path, monkeypatch):
    s = _settings(tmp_path, monkeypatch)
    s.put_explorer_settings({"server_root": r"\\nas\PROJECTS"})
    assert s.get_explorer_settings()["server_root"] == r"\\nas\PROJECTS"
