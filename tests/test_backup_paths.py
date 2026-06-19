from MetaMan.services.staging_service import backup_scope_destination, resolve_backup_data_root


def test_server_backup_root_uses_rawdata_folder(tmp_path):
    server_root = tmp_path / "PROJECTS"
    raw_root = server_root / "rawData"
    raw_root.mkdir(parents=True)

    assert resolve_backup_data_root(str(server_root), "server") == str(raw_root)
    assert backup_scope_destination(str(server_root), "server", "mPFC-NAc") == str(raw_root / "mPFC-NAc")


def test_server_backup_experiment_root_uses_rawdata_folder(tmp_path):
    server_root = tmp_path / "PROJECTS"
    raw_root = server_root / "rawData"
    raw_root.mkdir(parents=True)

    dest = backup_scope_destination(str(server_root), "server", "mPFC-NAc", "NPX")

    assert dest == str(raw_root / "mPFC-NAc" / "NPX")


def test_server_backup_root_falls_back_when_rawdata_missing(tmp_path):
    server_root = tmp_path / "PROJECTS"
    server_root.mkdir()

    assert resolve_backup_data_root(str(server_root), "server") == str(server_root)
    assert backup_scope_destination(str(server_root), "server", "mPFC-NAc") == str(server_root / "mPFC-NAc")


def test_hdd_backup_root_is_unchanged(tmp_path):
    hdd_root = tmp_path / "BackupDrive"
    (hdd_root / "rawData").mkdir(parents=True)

    assert resolve_backup_data_root(str(hdd_root), "hdd") == str(hdd_root)
    assert backup_scope_destination(str(hdd_root), "hdd", "mPFC-NAc") == str(hdd_root / "mPFC-NAc")
