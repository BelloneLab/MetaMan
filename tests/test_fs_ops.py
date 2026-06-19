import os

import pytest

from MetaMan.services import fs_ops


def test_safe_name():
    assert fs_ops.safe_name('a/b:c*?"<>|d') == "abcd"
    assert fs_ops.safe_name("  spaced  ") == "spaced"


def test_copy_into_file_and_dir(tmp_path):
    src_dir = tmp_path / "exp"
    (src_dir / "s1").mkdir(parents=True)
    (src_dir / "s1" / "f.bin").write_text("payload")
    dest = tmp_path / "dest"
    dest.mkdir()

    stats = fs_ops.copy_into(str(src_dir), str(dest), lambda *_: None)
    assert (dest / "exp" / "s1" / "f.bin").read_text() == "payload"
    assert stats["copied"] >= 1 and stats["destination_path"].endswith("exp")

    f = tmp_path / "lone.txt"
    f.write_text("hi")
    s2 = fs_ops.copy_into(str(f), str(dest), lambda *_: None)
    assert (dest / "lone.txt").read_text() == "hi" and s2["copied"] == 1


def test_rename_path_clobber_guard(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    with pytest.raises(FileExistsError):
        fs_ops.rename_path(str(a), "b")
    new = fs_ops.rename_path(str(a), "c")
    assert os.path.basename(new) == "c" and os.path.isdir(new)


def test_rename_path_sanitises(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    new = fs_ops.rename_path(str(a), 'we:ird/name')
    assert os.path.basename(new) == "weirdname"


def test_delete_path_permanent(tmp_path):
    d = tmp_path / "gone"; d.mkdir()
    (d / "x.txt").write_text("x")
    how = fs_ops.delete_path(str(d), permanent=True)
    assert how == "permanent" and not d.exists()
