import json
import os
from typing import List, Dict

def search_in_project(project_dir: str, query: str) -> List[Dict]:
    hits: List[Dict] = []
    q = (query or "").lower()
    for root, dirs, files in os.walk(project_dir):
        if "metadata.json" in files:
            p = os.path.join(root, "metadata.json")
            try:
                data = json.loads(open(p, "r", encoding="utf-8").read())
                for k, v in data.items():
                    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                    if q in (k.lower() + " " + s.lower()):
                        hits.append({"path": root, "key": k, "value": (s[:200] + "..." if len(s) > 200 else s)})
            except Exception:
                pass
    return hits
