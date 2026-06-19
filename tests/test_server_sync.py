import os

import pytest

from MetaMan.services import server_sync as ss


class _Cancel:
    """Minimal stand-in for utils.CancelToken (avoids importing PySide6)."""

    def __init__(self, after=0):
        self._after = after
        self._n = 0

    def is_cancelled(self):
        self._n += 1
        return self._n > self._after


def _write(path, data="data"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(data)


def _make_tree(root):
    _write(os.path.join(root, "a.txt"), "hello")
    _write(os.path.join(root, "sub", "b.txt"), "world")
    _write(os.path.join(root, "sub", "c.bin"), "x" * 50)


def test_classify_copy_update_skip(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_tree(str(src))
    s1 = ss.mirror_tree(str(src), str(dst), lambda *_: None)
    assert s1["copied"] == 3 and s1["skipped"] == 0 and s1["failed"] == 0
    assert s1["bytes_copied"] > 0

    # re-run: everything unchanged -> skipped
    s2 = ss.mirror_tree(str(src), str(dst), lambda *_: None)
    assert s2["skipped"] == 3 and s2["copied"] == 0 and s2["updated"] == 0

    # change a file -> updated
    import time
    time.sleep(1.1)
    _write(os.path.join(str(src), "a.txt"), "hello-changed-bigger")
    s3 = ss.mirror_tree(str(src), str(dst), lambda *_: None)
    assert s3["updated"] == 1 and s3["skipped"] == 2


def test_atomic_no_partial_left_on_cancel(tmp_path, monkeypatch):
    # tiny chunk so a small file spans several reads and cancel triggers mid-copy
    monkeypatch.setattr(ss, "COPY_CHUNK_BYTES", 4)
    src = tmp_path / "src.txt"
    src.write_text("abcdefghijklmnop")
    dst = tmp_path / "out" / "src.txt"
    with pytest.raises(ss.CopyCancelled):
        ss.copy_with_progress(str(src), str(dst), lambda *_: None, cancel=_Cancel(after=1))
    assert not dst.exists()
    assert not (tmp_path / "out" / ("src.txt" + ss._PARTIAL_SUFFIX)).exists()


def test_verify_ok_and_mismatch(tmp_path, monkeypatch):
    src = tmp_path / "s.txt"
    src.write_text("verify me")
    dst = tmp_path / "d.txt"
    copied, _ = ss.copy_with_progress(str(src), str(dst), lambda *_: None, verify=True)
    assert copied and dst.read_text() == "verify me"

    dst2 = tmp_path / "d2.txt"
    monkeypatch.setattr(ss, "_hash_file", lambda p: "deadbeef")
    with pytest.raises(ss.VerificationError):
        ss.copy_with_progress(str(src), str(dst2), lambda *_: None, verify=True)
    assert not dst2.exists()
    assert not (tmp_path / ("d2.txt" + ss._PARTIAL_SUFFIX)).exists()


def test_prune_removes_extras(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_tree(str(src))
    ss.mirror_tree(str(src), str(dst), lambda *_: None)
    # an orphan at the destination
    _write(os.path.join(str(dst), "sub", "orphan.txt"), "stale")
    s = ss.mirror_tree(str(src), str(dst), lambda *_: None, prune=True)
    assert s["pruned"] == 1
    assert not (dst / "sub" / "orphan.txt").exists()
    # without prune the orphan would survive
    _write(os.path.join(str(dst), "keep.txt"), "stale")
    s2 = ss.mirror_tree(str(src), str(dst), lambda *_: None, prune=False)
    assert s2["pruned"] == 0 and (dst / "keep.txt").exists()


def test_diff_tree(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_tree(str(src))
    ss.mirror_tree(str(src), str(dst), lambda *_: None)
    _write(os.path.join(str(src), "new.txt"), "n")          # only_source
    _write(os.path.join(str(dst), "extra.txt"), "e")        # only_dest
    import time; time.sleep(1.1)
    _write(os.path.join(str(src), "a.txt"), "changed-bigger")  # different
    d = ss.diff_tree(str(src), str(dst))
    assert "new.txt" in [os.path.basename(x) for x in d["only_source"]]
    assert "extra.txt" in [os.path.basename(x) for x in d["only_dest"]]
    assert "a.txt" in [os.path.basename(x) for x in d["different"]]


def test_tree_size_and_free_space(tmp_path):
    _make_tree(str(tmp_path))
    assert ss.tree_size(str(tmp_path)) > 0
    assert ss.free_space(str(tmp_path)) > 0
