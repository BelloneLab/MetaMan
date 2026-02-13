import os
from typing import List, Dict

def scan_file_list(session_dir: str) -> List[Dict]:
    out: List[Dict] = []
    for root, dirs, files in os.walk(session_dir):
        for d in dirs:
            p = os.path.abspath(os.path.join(root, d))
            out.append({"path": p + (os.sep if not p.endswith(os.sep) else ""), "type": "dir"})
        for f in files:
            p = os.path.abspath(os.path.join(root, f))
            try:
                sz = os.path.getsize(p)
            except Exception:
                sz = None
            out.append({"path": p, "type": "file", "size": sz})
    out.sort(key=lambda x: (x["type"], x["path"]))
    return out
