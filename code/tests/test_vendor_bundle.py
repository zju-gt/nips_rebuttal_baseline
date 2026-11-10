from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


BASELINES = (
    "sop-world-react",
    "sop-world-flexion",
    "sop-world-AWM",
    "sop-world-Reasoning",
    "sop-world-EvoTest",
    "sop-world",
)

DOMAIN_FILES = {
    "airline": ("db.json", "policy.md", "split_tasks.json", "tasks.json"),
    "retail": ("db.json", "policy.md", "split_tasks.json", "tasks.json"),
    "telecom": (
        "db.toml",
        "user_db.toml",
        "main_policy.md",
        "main_policy_solo.md",
        "tech_support_manual.md",
        "tech_support_workflow.md",
        "tech_support_workflow_solo.md",
        "split_tasks.json",
        "tasks.json",
    ),
}


def _write(path: Path, text: str = "fixture") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_source_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    for name in BASELINES:
        repo = workspace / name
        _write(repo / "src" / "tau2_runner.py", f"METHOD = {name!r}\n")
        _write(repo / "src" / "helper.py", "VALUE = 1\n")
        _write(repo / "src" / "weights.bin", "not-runtime-source")
        _write(repo / "src" / "results" / "old.json", "{}")
        _write(repo / "src" / ".git" / "config", "private")
        _write(repo / "src" / "__pycache__" / "helper.pyc", "bytecode")
        _write(repo / "prompts" / "prior" / "domain_prior_retail.md", "prompt")
        _write(repo / "prompts" / "logs" / "run.log", "private")
        _write(repo / "prompts" / ".pytest_cache" / "state", "cache")
        _write(repo / ".env.local", "LLM_API_KEY=must-not-copy")
        _write(repo / "tests" / "test_runner.py", "assert True")
        _write(repo / "results" / "summary.json", "{}")
        _write(repo / "logs" / "run.log", "private")

    tau2 = workspace / "tau2-bench"
    _write(tau2 / "src" / "tau2" / "__init__.py", "VERSION = 'test'\n")
    _write(tau2 / "LICENSE", "MIT")
    _write(tau2 / "README.md", "tau2")
    _write(tau2 / "pyproject.toml", "[project]\nname='tau2'\nversion='0'\n")
    _write(tau2 / "uv.lock", "lock")
    _write(tau2 / "data" / "tau2" / "user_simulator" / "guidelines.md", "guide")
    for domain, filenames in DOMAIN_FILES.items():
        for filename in filenames:
            _write(tau2 / "data" / "tau2" / "domains" / domain / filename)

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    openai_test = code_dir / "openai_test.py"
    _write(
        openai_test,
        "\n".join(
            [
                "from openai import OpenAI",
                "client = OpenAI(api_key='fixture-secret', base_url='https://example.invalid/v1')",
                "client.chat.completions.create(model='gpt-5.2-test', messages=[])",
            ]
        ),
    )
    return workspace, code_dir, openai_test


def test_baseline_allowlist_copies_runtime_files_and_excludes_artifacts(
    tmp_path: Path,
) -> None:
    from vendor_bundle import sync_vendor_bundle

    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)

    manifest_path = sync_vendor_bundle(workspace, code_dir, openai_test)

    assert manifest_path == code_dir / "vendor" / "bundle_manifest.json"
    for name in BASELINES:
        bundled = code_dir / "vendor" / name
        assert (bundled / "src" / "tau2_runner.py").is_file()
        assert (bundled / "src" / "helper.py").is_file()
        assert (bundled / "prompts" / "prior" / "domain_prior_retail.md").is_file()
        assert not (bundled / ".env.local").exists()
        assert not (bundled / "tests").exists()
        assert not (bundled / "results").exists()
        assert not (bundled / "logs").exists()
        assert not (bundled / "src" / "__pycache__").exists()
        assert not (bundled / "src" / "weights.bin").exists()
        assert not (bundled / "src" / "results").exists()
        assert not (bundled / "src" / ".git").exists()
        assert not (bundled / "prompts" / "logs").exists()
        assert not (bundled / "prompts" / ".pytest_cache").exists()


def test_baseline_allowlisted_symlink_fails_without_replacing_existing_bundle(
    tmp_path: Path,
) -> None:
    from vendor_bundle import sync_vendor_bundle

    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)
    vendor = code_dir / "vendor"
    _write(vendor / "sentinel.txt", "keep-existing")
    target = workspace / "sop-world-react" / "src" / "helper.py"
    (workspace / "sop-world-react" / "src" / "linked.py").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        sync_vendor_bundle(workspace, code_dir, openai_test)

    assert (vendor / "sentinel.txt").read_text(encoding="utf-8") == "keep-existing"
    assert not (vendor / "bundle_manifest.json").exists()


@pytest.mark.parametrize("selected_path", ["metadata", "domain", "root"])
def test_selected_tau2_symlink_fails_without_replacing_existing_bundle(
    tmp_path: Path,
    selected_path: str,
) -> None:
    from vendor_bundle import sync_vendor_bundle

    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)
    vendor = code_dir / "vendor"
    _write(vendor / "sentinel.txt", "keep-existing")
    external = tmp_path / "outside.txt"
    _write(external, "outside-content")
    tau2 = workspace / "tau2-bench"
    if selected_path == "metadata":
        target = tau2 / "README.md"
        target.unlink()
        target.symlink_to(external)
    elif selected_path == "domain":
        target = tau2 / "data" / "tau2" / "domains" / "retail" / "tasks.json"
        target.unlink()
        target.symlink_to(external)
    else:
        target = tau2 / "data" / "tau2" / "user_simulator"
        original = target.rename(tau2 / "data" / "tau2" / "user_simulator-real")
        target.symlink_to(original, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        sync_vendor_bundle(workspace, code_dir, openai_test)

    assert (vendor / "sentinel.txt").read_text(encoding="utf-8") == "keep-existing"
    assert not (vendor / "bundle_manifest.json").exists()


@pytest.mark.parametrize("failure", [OSError("rename failed"), KeyboardInterrupt()])
def test_install_rename_failure_restores_existing_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    from vendor_bundle import sync_vendor_bundle, verify_vendor_bundle

    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)
    sync_vendor_bundle(workspace, code_dir, openai_test)
    vendor = code_dir / "vendor"
    original_helper = vendor / "sop-world-react" / "src" / "helper.py"
    assert original_helper.read_text(encoding="utf-8") == "VALUE = 1\n"
    _write(workspace / "sop-world-react" / "src" / "helper.py", "VALUE = 2\n")

    original_rename = Path.rename

    def fail_install_rename(path: Path, target: Path) -> Path:
        if path.name.startswith(".vendor-build-") and Path(target) == vendor:
            raise failure
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_install_rename)
    with pytest.raises(type(failure), match=str(failure) or None):
        sync_vendor_bundle(workspace, code_dir, openai_test)

    verify_vendor_bundle(vendor)
    assert original_helper.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not list(code_dir.glob(".vendor-backup-*"))


def test_interrupt_after_backup_rename_restores_existing_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vendor_bundle import sync_vendor_bundle, verify_vendor_bundle

    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)
    sync_vendor_bundle(workspace, code_dir, openai_test)
    vendor = code_dir / "vendor"
    original_helper = vendor / "sop-world-react" / "src" / "helper.py"
    _write(workspace / "sop-world-react" / "src" / "helper.py", "VALUE = 2\n")

    original_rename = Path.rename

    def interrupt_after_backup(path: Path, target: Path) -> Path:
        result = original_rename(path, target)
        if path == vendor and Path(target).name.startswith(".vendor-backup-"):
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(Path, "rename", interrupt_after_backup)
    with pytest.raises(KeyboardInterrupt):
        sync_vendor_bundle(workspace, code_dir, openai_test)

    verify_vendor_bundle(vendor)
    assert original_helper.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not list(code_dir.glob(".vendor-backup-*"))


def test_tau2_bundle_contains_only_required_text_runtime_data(tmp_path: Path) -> None:
    from vendor_bundle import sync_vendor_bundle

    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)
    tau2 = workspace / "tau2-bench"
    _write(tau2 / "data" / "tau2" / "domains" / "telecom" / "tasks_voice.json")
    _write(tau2 / "data" / "tau2" / "domains" / "telecom" / "audio_difficulty.json")
    _write(tau2 / "data" / "tau2" / "domains" / "telecom" / "tasks_full.json")
    _write(tau2 / "data" / "tau2" / "domains" / "telecom" / "tasks_small.json")
    _write(tau2 / "data" / "tau2" / "domains" / "banking_knowledge" / "tasks.json")
    _write(tau2 / "data" / "tau2" / "results" / "published.json")
    _write(tau2 / "data" / "voice" / "sample.wav")
    _write(tau2 / "tests" / "test_tau2.py")
    _write(tau2 / "web" / "index.html")

    sync_vendor_bundle(workspace, code_dir, openai_test)

    bundled = code_dir / "vendor" / "tau2-bench"
    assert (bundled / "src" / "tau2" / "__init__.py").is_file()
    assert (bundled / "data" / "tau2" / "user_simulator" / "guidelines.md").is_file()
    for domain, filenames in DOMAIN_FILES.items():
        for filename in filenames:
            assert (bundled / "data" / "tau2" / "domains" / domain / filename).is_file()
    assert not (bundled / "data" / "tau2" / "domains" / "telecom" / "tasks_voice.json").exists()
    assert not (bundled / "data" / "tau2" / "domains" / "telecom" / "audio_difficulty.json").exists()
    assert not (bundled / "data" / "tau2" / "domains" / "telecom" / "tasks_full.json").exists()
    assert not (bundled / "data" / "tau2" / "domains" / "telecom" / "tasks_small.json").exists()
    assert not (bundled / "data" / "tau2" / "domains" / "banking_knowledge").exists()
    assert not (bundled / "data" / "tau2" / "results").exists()
    assert not (bundled / "data" / "voice").exists()
    assert not (bundled / "tests").exists()
    assert not (bundled / "web").exists()


def test_manifest_hashes_every_copied_file_and_records_provenance(
    tmp_path: Path,
) -> None:
    from vendor_bundle import sync_vendor_bundle, verify_vendor_bundle

    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)
    manifest_path = sync_vendor_bundle(workspace, code_dir, openai_test)
    manifest = verify_vendor_bundle(code_dir / "vendor")
    actual_files = sorted(
        path.relative_to(code_dir / "vendor").as_posix()
        for path in (code_dir / "vendor").rglob("*")
        if path.is_file() and path != manifest_path
    )

    assert [item["path"] for item in manifest["files"]] == actual_files
    assert manifest["total_files"] == len(actual_files)
    assert manifest["total_bytes"] == sum(item["size"] for item in manifest["files"])
    assert set(manifest["components"]) == {*BASELINES, "tau2-bench"}
    for item in manifest["files"]:
        path = code_dir / "vendor" / item["path"]
        assert item["size"] == path.stat().st_size
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert item["component"] == Path(item["path"]).parts[0]
    for name, provenance in manifest["components"].items():
        assert provenance["source_path"] == str((workspace / name).resolve())
        assert provenance["git_head"] is None
        assert provenance["dirty"] is None
        assert len(provenance["content_sha256"]) == 64


def test_verifier_rejects_mutated_missing_and_added_files(tmp_path: Path) -> None:
    from vendor_bundle import sync_vendor_bundle, verify_vendor_bundle

    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)
    sync_vendor_bundle(workspace, code_dir, openai_test)
    vendor = code_dir / "vendor"

    target = vendor / "sop-world-react" / "src" / "helper.py"
    original = target.read_text(encoding="utf-8")
    target.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        verify_vendor_bundle(vendor)
    target.write_text(original, encoding="utf-8")

    target.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_vendor_bundle(vendor)
    _write(target, original)

    _write(vendor / "unexpected.txt", "extra")
    with pytest.raises(ValueError, match="unlisted"):
        verify_vendor_bundle(vendor)


def test_verifier_rejects_unsafe_manifest_paths(tmp_path: Path) -> None:
    from vendor_bundle import sync_vendor_bundle, verify_vendor_bundle

    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)
    manifest_path = sync_vendor_bundle(workspace, code_dir, openai_test)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for unsafe in ("../escape.py", "/absolute.py"):
        changed = json.loads(json.dumps(manifest))
        changed["files"][0]["path"] = unsafe
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="unsafe"):
            verify_vendor_bundle(code_dir / "vendor")


def test_secret_in_allowlisted_source_fails_without_replacing_bundle(
    tmp_path: Path,
) -> None:
    from vendor_bundle import sync_vendor_bundle

    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)
    _write(code_dir / "vendor" / "sentinel.txt", "keep-existing")
    _write(workspace / "sop-world-react" / "src" / "leak.py", "VALUE = 'fixture-secret'\n")

    with pytest.raises(ValueError) as exc_info:
        sync_vendor_bundle(workspace, code_dir, openai_test)

    assert "fixture-secret" not in str(exc_info.value)
    assert (code_dir / "vendor" / "sentinel.txt").read_text(encoding="utf-8") == "keep-existing"


def test_sync_cli_prints_only_sanitized_bundle_summary(tmp_path: Path) -> None:
    workspace, code_dir, openai_test = _make_source_workspace(tmp_path)
    script = CODE_DIR / "sync_vendor_bundle.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--source-workspace",
            str(workspace),
            "--code-dir",
            str(code_dir),
            "--openai-test",
            str(openai_test),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["manifest"] == str(code_dir / "vendor" / "bundle_manifest.json")
    assert payload["file_count"] > 0
    assert payload["total_bytes"] > 0
    assert "fixture-secret" not in result.stdout
    assert "fixture-secret" not in result.stderr


def test_real_bundle_is_complete_clean_and_below_transfer_size() -> None:
    from vendor_bundle import verify_vendor_bundle

    vendor = CODE_DIR / "vendor"
    manifest = verify_vendor_bundle(vendor)

    assert set(manifest["components"]) == {*BASELINES, "tau2-bench"}
    for name in BASELINES:
        assert (vendor / name / "src" / "tau2_runner.py").is_file()
        assert any(path.is_file() for path in (vendor / name / "prompts").rglob("*"))
    assert (vendor / "tau2-bench" / "src" / "tau2" / "__init__.py").is_file()
    assert any(
        path.is_file()
        for path in (vendor / "tau2-bench" / "data" / "tau2" / "user_simulator").rglob("*")
    )
    for domain, filenames in DOMAIN_FILES.items():
        for filename in filenames:
            assert (vendor / "tau2-bench" / "data" / "tau2" / "domains" / domain / filename).is_file()
    forbidden_parts = {".git", "__pycache__", "tests", "results", "logs", "multiwoz"}
    for path in vendor.rglob("*"):
        relative = path.relative_to(vendor)
        assert not (set(relative.parts) & forbidden_parts)
        assert path.name != ".env"
        assert not path.name.startswith(".env.")
        assert path.suffix != ".pyc"
    assert manifest["total_bytes"] < 100 * 1024 * 1024
