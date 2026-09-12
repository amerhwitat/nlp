"""Registry loader for language, stage and script metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LanguageRecord:
    id: str
    name: str
    family: str
    stages: tuple[str, ...]


@dataclass(frozen=True)
class LanguageStage:
    id: str
    language_id: str


@dataclass(frozen=True)
class ScriptRecord:
    id: str
    iso15924: str
    unicode_script: str
    direction: str
    kind: str


@dataclass(frozen=True)
class RegistryIssue:
    severity: str
    message: str


class AncientLanguageRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.languages_payload = json.loads((root / "languages.json").read_text(encoding="utf-8"))
        self.scripts_payload = json.loads((root / "scripts.json").read_text(encoding="utf-8"))
        self._languages = {
            item["id"]: LanguageRecord(item["id"], item["name"], item["family"], tuple(item["stages"]))
            for item in self.languages_payload["languages"]
        }
        self._stages = {
            stage: LanguageStage(stage, language.id)
            for language in self._languages.values()
            for stage in language.stages
        }
        self._scripts = {item["id"]: ScriptRecord(**item) for item in self.scripts_payload["scripts"]}

    def get_language(self, language_id: str) -> LanguageRecord:
        return self._languages[language_id]

    def get_stage(self, stage_id: str) -> LanguageStage:
        return self._stages[stage_id]

    def get_script(self, script_id: str) -> ScriptRecord:
        return self._scripts[script_id]

    def validate(self) -> list[RegistryIssue]:
        issues: list[RegistryIssue] = []
        for language in self._languages.values():
            for stage in language.stages:
                if stage not in self._stages:
                    issues.append(RegistryIssue("error", f"missing stage {stage}"))
        for script in self._scripts.values():
            if script.direction not in {"ltr", "rtl", "bidi"}:
                issues.append(RegistryIssue("error", f"invalid direction for {script.id}"))
        return issues

    def get_orthography(self, orthography_id: str):
        path = self.root / "orthographies.json"
        if not path.exists():
            raise KeyError(orthography_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("orthographies", []):
            if item["id"] == orthography_id:
                return item
        raise KeyError(orthography_id)


def load_registry(path: str) -> AncientLanguageRegistry:
    return AncientLanguageRegistry(Path(path))
