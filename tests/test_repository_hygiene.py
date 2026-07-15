from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DEPENDENCIES = {
    "decord",
    "opencv-python",
    "submitit",
    "timm",
    "torchvision",
}
FORBIDDEN_RUNTIME_TERMS = {
    "decord",
    "opencv",
    "submitit",
    "timm",
    "torchvision",
    "vision_transformer",
}


def test_dependency_graph_has_no_video_or_transformer_packages() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = {
        dependency.split("[")[0].split("<")[0].split(">")[0].split("=")[0].lower()
        for dependency in project["project"]["dependencies"]
    }

    assert dependencies.isdisjoint(FORBIDDEN_DEPENDENCIES)


def test_runtime_has_no_video_or_transformer_imports() -> None:
    runtime_text = "\n".join(
        path.read_text().lower() for path in sorted((ROOT / "src" / "jepa").glob("*.py"))
    )

    assert all(term not in runtime_text for term in FORBIDDEN_RUNTIME_TERMS)


def test_legacy_runtime_directories_are_absent() -> None:
    assert not (ROOT / "app").exists()
    assert not (ROOT / "evals").exists()


def test_distribution_contract_is_wired_into_ci() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert 'python-version: ["3.11", "3.12", "3.13"]' in workflow
    assert "uv sync --locked" in workflow
    assert "uv run ruff check src tests scripts" in workflow
    assert "uv run mypy" in workflow
    assert "uv run pytest" in workflow
    assert "uv build" in workflow
    assert "timeout-minutes: 5" in workflow
    assert "--system-kind all" in workflow
    assert "scripts/verify_smoke.py" in workflow


def test_smoke_config_and_verifier_are_packaged() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    included = set(project["tool"]["uv"]["build-backend"]["source-include"])

    assert ".github/**" in included
    assert "configs/**" in included
    assert "scripts/**" in included
    assert (ROOT / "configs" / "smoke.yaml").is_file()
    assert (ROOT / "scripts" / "verify_smoke.py").is_file()
