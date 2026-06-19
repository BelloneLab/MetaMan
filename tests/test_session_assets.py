import os

from MetaMan.services import session_assets as sa


def test_import_key_value_csv(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text("Key,Value\nRegion,CA1\nRig,Rig-A\n", encoding="utf-8")
    fields, notes = sa.read_metadata_update_file(str(path))
    assert fields == {"Region": "CA1", "Rig": "Rig-A"}
    assert notes == ""


def test_import_table_csv_first_data_row(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text("Region,Rig,SampleRate\nCA3,Rig-B,30000\nDG,Rig-C,20000\n", encoding="utf-8")
    fields, notes = sa.read_metadata_update_file(str(path))
    assert fields["Region"] == "CA3"
    assert fields["Rig"] == "Rig-B"
    assert fields["SampleRate"] == "30000"
    assert notes == ""


def test_import_txt_fields_and_notes(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("Region: CA1\nRig=Rig-A\nstable unit yield\n", encoding="utf-8")
    fields, notes = sa.read_metadata_update_file(str(path))
    assert fields == {"Region": "CA1", "Rig": "Rig-A"}
    assert notes == "stable unit yield"


def test_notes_sidecar_roundtrip(tmp_path):
    session = tmp_path / "session"
    saved = sa.save_notes(str(session), "first line\nsecond line")
    assert os.path.basename(saved) == sa.SESSION_NOTES_FILE
    assert sa.load_notes(str(session), {}) == "first line\nsecond line"
    assert sa.load_notes(str(session), {"Notes": "metadata wins"}) == "metadata wins"


def test_upload_files_to_project_records_relative_paths(tmp_path):
    project = tmp_path / "ProjectA"
    src = tmp_path / "source.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    records = sa.upload_files_to_project(str(project), [str(src)], ["Exp1", "M1", "S1"])
    assert len(records) == 1
    assert records[0]["relative_path"] == "_metaman_uploads/Exp1/M1/S1/source.csv"
    assert (project / records[0]["relative_path"]).read_text(encoding="utf-8") == "a,b\n1,2\n"
