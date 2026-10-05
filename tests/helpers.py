from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "ops-diagnostics"


class TempDir:
    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="abe-test-"))
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def build_example(workspace: Path) -> Path:
    from abe.builder import Blueprint, build_agent

    blueprint = Blueprint.load(EXAMPLE / "blueprint.yaml")
    docs = sorted((EXAMPLE / "docs").iterdir())
    return build_agent(blueprint, docs, workspace)
