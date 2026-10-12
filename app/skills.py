"""Load application-owned skills; no model-supplied filesystem paths."""
from pathlib import Path

import yaml


def load_skill(name):
    if name != "document-qa":
        raise ValueError("未知的 Skill")
    path = Path(__file__).resolve().parent.parent / "skills" / name / "SKILL.md"
    content = path.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError("Skill 缺少 metadata")
    metadata = yaml.safe_load(parts[1])
    if (not isinstance(metadata, dict) or metadata.get("name") != name
            or not isinstance(metadata.get("description"), str)
            or not metadata["description"].strip() or not parts[2].strip()):
        raise ValueError("Skill 格式不正確")
    return {"name": name, "description": metadata["description"], "instructions": parts[2].strip()}
