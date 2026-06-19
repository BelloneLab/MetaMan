import os

from MetaMan.services import backup_report as br
from MetaMan.services.server_sync import new_stats


def _record(**over):
    stats = new_stats()
    stats.update({"copied": 3, "bytes_copied": 100, "files_total": 3})
    base = dict(project="P", experiment="", destination_kind="server",
                destination_root="root", destination_path="root/P",
                source_path="src", trigger="manual", stats=stats)
    base.update(over)
    return br.build_record(**base)


def test_status_success_partial_cancelled_error():
    assert _record()["status"] == "success"

    s = new_stats(); s["failed"] = 1
    assert br.build_record(project="P", experiment="", destination_kind="server",
                           destination_root="r", destination_path="r/P", source_path="s",
                           trigger="manual", stats=s)["status"] == "partial"

    s2 = new_stats(); s2["cancelled"] = True
    assert br.build_record(project="P", experiment="", destination_kind="server",
                           destination_root="r", destination_path="r/P", source_path="s",
                           trigger="manual", stats=s2)["status"] == "cancelled"

    assert _record(error="boom")["status"] == "error"


def test_scope_and_format_text():
    rec = _record(experiment="Exp1")
    assert rec["scope"] == "experiment"
    txt = br.format_report_text(rec)
    assert "MetaMan backup report" in txt and "Exp1" in txt


def test_write_report_files(tmp_path):
    dest_root = str(tmp_path / "server")
    rec = _record()
    jp = br.write_report_files(dest_root, rec)
    assert jp and os.path.isfile(jp)
    rd = br.report_dir(dest_root, "P", "")
    assert os.path.isfile(os.path.join(rd, "last_report.json"))
    assert os.path.isfile(os.path.join(rd, "history.csv"))
    # report dir lives OUTSIDE any project tree
    assert br.REPORT_DIRNAME in rd
