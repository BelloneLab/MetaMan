from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

@dataclass
class SessionMetadata:
    data: Dict = field(default_factory=dict)

    @staticmethod
    def new(project: str, animal: str, session: str, root_dir: str) -> "SessionMetadata":
        from uuid import uuid4
        base = {
            "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Project": project,
            "Animal": animal,
            "Subject": animal,
            "Experiment": "",
            "Session": str(session),
            "Trial": "",
            "Condition": "",
            "Recording": "",
            "Region": "",
            "Experimenter": "",
            "Room": "",
            "Box": "",
            "Comments": "",
            "RootDir": root_dir,
            "SessionUUID": str(uuid4()),
            "file_list": [],
            "trial_info": {},
            "trial_assets": {},
            "preprocessing": []
        }
        return SessionMetadata(base)

    def upsert_file_list(self, file_list: List[Dict]):
        self.data["file_list"] = file_list

    def set_trial_info(self, trial_info: Dict):
        self.data["trial_info"] = trial_info
