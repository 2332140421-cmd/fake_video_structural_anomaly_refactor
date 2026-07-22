"""Deterministic Git content classification and release report generation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from semantic3d.dataset_builder.writer import atomic_write_bytes, atomic_write_json

from .model_registry import sha256_file

_TEXT_SUFFIXES = {
    ".cfg", ".csv", ".ini", ".json", ".jsonl", ".md", ".py", ".sh",
    ".toml", ".txt", ".yaml", ".yml",
}
_SENSITIVE_NAME = re.compile(
    r"(^|[._-])(credentials?|secrets?|tokens?|id_rsa|id_ed25519)([._-]|$)", re.IGNORECASE
)
_SENSITIVE_CONTENT = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "assigned_secret",
        re.compile(r"(?i)\b(?:password|passwd|api_key|secret_key|access_token)\s*[:=]\s*['\"][^'\"]{8,}"),
    ),
)
_ABSOLUTE_PATTERNS = (
    ("home_chenyh", re.compile(r"/home/chenyh")),
    ("mnt_e", re.compile(r"/mnt/e")),
    ("windows_c", re.compile(r"\bC:\\", re.IGNORECASE)),
    ("windows_e", re.compile(r"\bE:\\", re.IGNORECASE)),
    ("hostname_cyh", re.compile(r"\bCYH\b", re.IGNORECASE)),
    ("username_chenyh", re.compile(r"\bchenyh\b", re.IGNORECASE)),
)
_FROZEN_LOCATION_CONFIGS = {
    "configs/p4c0_experiment_protocol_v1.yaml",
    "configs/p4c2_formal_data_readiness_v1.yaml",
}
_SERVER_REQUIRED_PREFIXES = (
    "src/", "scripts/", "configs/", "tests/", "docs/", "data/experiment_protocol_v1/",
    "outputs/p4c1_experiment_manifest/", "outputs/p4c2_formal_data_readiness/",
    "outputs/p4c3a_batch_plan/",
)
_SERVER_REQUIRED_FILES = {
    ".gitignore", "README.md", "CHANGELOG.md", "pyproject.toml",
    "requirements-lock.txt", "requirements-inference-lock.txt",
}
_GENERATED_PREFIXES = ("outputs/", "logs/", "tmp/", "cache/")
_MEDIA_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
_MODEL_SUFFIXES = {".pt", ".pth", ".onnx", ".engine"}


def _git(project: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=project, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _git_paths(project: Path, *args: str) -> tuple[str, ...]:
    return tuple(sorted(path for path in _git(project, *args).split("\0") if path))


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _is_required(path: str) -> bool:
    return path in _SERVER_REQUIRED_FILES or path.startswith(_SERVER_REQUIRED_PREFIXES)


def _category(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in _MODEL_SUFFIXES or path == "checkpoints" or path.startswith("checkpoints/"):
        return "model_weight"
    if suffix in _MEDIA_SUFFIXES or path == "data/tests_videos" or path.startswith("data/tests_videos/"):
        return "video_asset"
    if path.startswith("outputs/"):
        return "generated_output"
    if path.startswith("src/"):
        return "source_code"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("scripts/"):
        return "script"
    if path.startswith("configs/"):
        return "configuration"
    if path.startswith("docs/") or path.endswith(".md"):
        return "documentation"
    if path.startswith("data/"):
        return "small_data_or_manifest"
    if path.startswith(".venv/") or path.startswith("venv/"):
        return "local_environment"
    return "project_metadata"


def _machine_specific(path: str) -> bool:
    return path == "configs/runtime/local_wsl.yaml" or path in {
        "docs/DISK_USAGE_AUDIT.md",
        "docs/DISK_CLEANUP_REPORT.md",
        "docs/WSL_MIGRATION_PRECHECK.md",
    }


def _scan_sensitive(path: Path, relative: str) -> tuple[bool, tuple[str, ...]]:
    findings: list[str] = []
    if _SENSITIVE_NAME.search(path.name) or path.name in {".env", ".env.local"}:
        findings.append("sensitive_filename")
    if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES and path.stat().st_size <= 2_000_000:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in _SENSITIVE_CONTENT:
            if pattern.search(text):
                findings.append(name)
    return bool(findings), tuple(sorted(set(findings)))


def _recommend(
    path: str,
    *,
    tracked: bool,
    ignored: bool,
    required: bool,
    sensitive: bool,
    size: int,
) -> str:
    category = _category(path)
    if sensitive:
        return "manual_review"
    if category == "model_weight":
        return "download_on_server"
    if category == "video_asset":
        return "store_outside_git"
    if ignored:
        return "ignore"
    if category in {"local_environment", "generated_output"} and not required:
        return "ignore"
    if path == "configs/runtime/local_wsl.yaml":
        return "commit"
    if _machine_specific(path):
        return "manual_review"
    if size > 50 * 1024 * 1024:
        return "store_with_git_lfs" if required else "store_outside_git"
    if required or tracked:
        return "commit"
    return "ignore" if ignored else "manual_review"


def _file_record(project: Path, relative: str, *, tracked: bool, ignored: bool) -> dict[str, Any]:
    path = project / relative
    is_file = path.is_file()
    size = path.stat().st_size if is_file else _tree_size(path)
    sensitive, findings = _scan_sensitive(path, relative) if is_file else (False, ())
    required = _is_required(relative)
    generated = relative.startswith(_GENERATED_PREFIXES) or "__pycache__" in relative
    digest = sha256_file(path) if is_file else ""
    action = _recommend(
        relative,
        tracked=tracked,
        ignored=ignored,
        required=required,
        sensitive=sensitive,
        size=size,
    )
    distribution = {
        "commit": "git_candidate",
        "ignore": "excluded_generated_or_local",
        "download_on_server": "registry_only_not_binary",
        "store_with_git_lfs": "requires_license_and_user_lfs_approval",
        "store_outside_git": "external_storage_only",
        "manual_review": "blocked_pending_review",
    }[action]
    return {
        "path": relative,
        "category": _category(relative),
        "size": size,
        "sha256": digest,
        "tracked": tracked,
        "required_on_server": required,
        "generated": generated,
        "machine_specific": _machine_specific(relative),
        "contains_sensitive_information": sensitive,
        "distribution_status": distribution,
        "recommended_action": action,
        "ignored": ignored,
        "sensitive_findings": list(findings),
    }


def scan_absolute_paths(project: Path, paths: Iterable[str]) -> tuple[dict[str, Any], ...]:
    """Find location-specific strings and distinguish blockers from frozen history."""

    findings: list[dict[str, Any]] = []
    for relative in sorted(set(paths)):
        path = project / relative
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES or path.stat().st_size > 3_000_000:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for name, pattern in _ABSOLUTE_PATTERNS:
                if not pattern.search(line):
                    continue
                if relative == "configs/runtime/local_wsl.yaml":
                    status = "allowed_local_runtime_profile"
                elif relative in _FROZEN_LOCATION_CONFIGS:
                    status = "frozen_location_specific_snapshot"
                elif relative.startswith("data/experiment_protocol_v1/") or relative.startswith(
                    ("outputs/p4c1_experiment_manifest/", "outputs/p4c2_formal_data_readiness/")
                ):
                    status = "frozen_location_specific_artifact"
                elif relative.startswith("docs/"):
                    status = "documentation_or_historical_metadata"
                elif relative.startswith("tests/"):
                    status = "test_fixture_literal"
                elif relative == "src/semantic3d/git_release/audit.py":
                    status = "scanner_rule_literal"
                else:
                    status = "unresolved_core_or_config_path"
                findings.append(
                    {
                        "path": relative,
                        "line": line_number,
                        "pattern": name,
                        "status": status,
                        "blocking": status == "unresolved_core_or_config_path",
                    }
                )
    return tuple(sorted(findings, key=lambda row: (row["path"], row["line"], row["pattern"])))


def _oversized(project: Path, *, threshold: int = 10 * 1024 * 1024) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    pruned = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
    for root, directories, files in os.walk(project):
        root_path = Path(root)
        relative_root = root_path.relative_to(project).as_posix()
        directories[:] = [name for name in directories if name not in pruned]
        if relative_root.startswith("outputs/p4c3a_git_release"):
            directories[:] = []
            continue
        for name in files:
            path = root_path / name
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= threshold:
                continue
            relative = path.relative_to(project).as_posix()
            category = _category(relative)
            digest = sha256_file(path) if category in {"model_weight", "video_asset"} else ""
            rows.append(
                {
                    "path": relative,
                    "category": category,
                    "size": path.stat().st_size,
                    "sha256": digest,
                    "recommended_action": _recommend(
                        relative,
                        tracked=False,
                        ignored=True,
                        required=_is_required(relative),
                        sensitive=False,
                        size=path.stat().st_size,
                    ),
                }
            )
    return tuple(sorted(rows, key=lambda row: (-row["size"], row["path"])))


def build_git_release_audit(project_root: str | Path) -> dict[str, Any]:
    """Build a deterministic audit without staging, committing, tagging, or pushing."""

    project = Path(project_root).resolve()
    tracked_paths = _git_paths(project, "ls-files", "-z")
    untracked_paths = _git_paths(project, "ls-files", "--others", "--exclude-standard", "-z")
    ignored_paths = _git_paths(
        project, "ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z"
    )
    tracked = tuple(_file_record(project, path, tracked=True, ignored=False) for path in tracked_paths)
    untracked = tuple(_file_record(project, path, tracked=False, ignored=False) for path in untracked_paths)
    ignored = tuple(
        _file_record(project, path.rstrip("/"), tracked=False, ignored=True)
        for path in ignored_paths
        if path.rstrip("/")
    )
    candidates = (*tracked, *untracked)
    sensitive = tuple(
        {
            "path": row["path"],
            "findings": row["sensitive_findings"],
            "tracked": row["tracked"],
            "recommended_action": row["recommended_action"],
        }
        for row in candidates
        if row["contains_sensitive_information"]
    )
    absolute_paths = scan_absolute_paths(project, (row["path"] for row in candidates))
    status_short = _git(project, "status", "--short")
    branch = _git(project, "branch", "--show-current").strip()
    commit = _git(project, "rev-parse", "HEAD").strip()
    remotes = tuple(line for line in _git(project, "remote", "-v").splitlines() if line)
    logical_rows = [
        {key: row[key] for key in ("path", "category", "size", "sha256", "tracked", "required_on_server", "recommended_action")}
        for row in sorted(candidates, key=lambda item: item["path"])
    ]
    logical_hash = hashlib.sha256(
        json.dumps(logical_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "p4c3a_git_release_manifest_v1",
        "branch": branch,
        "commit": commit,
        "remotes": remotes,
        "worktree_clean": not bool(status_short.strip()),
        "git_status_short": status_short.splitlines(),
        "tracked": tracked,
        "untracked": untracked,
        "ignored": ignored,
        "oversized": _oversized(project),
        "sensitive": sensitive,
        "absolute_path_findings": absolute_paths,
        "release_content_sha256": logical_hash,
        "formal_data_downloaded": False,
        "formal_batch_started": False,
        "model_training_performed": False,
    }


def _csv_bytes(rows: Iterable[Mapping[str, Any]], columns: tuple[str, ...]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized = dict(row)
        for key, value in normalized.items():
            if isinstance(value, (list, tuple, dict)):
                normalized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        writer.writerow(normalized)
    return handle.getvalue().encode("utf-8")


def write_git_release_artifacts(output_root: str | Path, audit: Mapping[str, Any]) -> None:
    """Write deterministic CSV/JSON/Markdown release audit artifacts."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    columns = (
        "path", "category", "size", "sha256", "tracked", "required_on_server",
        "generated", "machine_specific", "contains_sensitive_information",
        "distribution_status", "recommended_action",
    )
    atomic_write_bytes(root / "tracked_files.csv", _csv_bytes(audit["tracked"], columns))
    atomic_write_bytes(root / "untracked_files.csv", _csv_bytes(audit["untracked"], columns))
    atomic_write_bytes(root / "ignored_files.csv", _csv_bytes(audit["ignored"], columns))
    atomic_write_bytes(
        root / "oversized_files.csv",
        _csv_bytes(audit["oversized"], ("path", "category", "size", "sha256", "recommended_action")),
    )
    atomic_write_json(
        root / "sensitive_file_audit.json",
        {"finding_count": len(audit["sensitive"]), "findings": audit["sensitive"]},
    )
    manifest = {key: value for key, value in audit.items() if key not in {"tracked", "untracked", "ignored", "oversized", "sensitive"}}
    manifest.update(
        {
            "tracked_file_count": len(audit["tracked"]),
            "untracked_file_count": len(audit["untracked"]),
            "ignored_entry_count": len(audit["ignored"]),
            "oversized_file_count": len(audit["oversized"]),
            "sensitive_finding_count": len(audit["sensitive"]),
        }
    )
    atomic_write_json(root / "git_release_manifest.json", manifest)
    lfs_candidates = [row for row in (*audit["tracked"], *audit["untracked"]) if row["recommended_action"] == "store_with_git_lfs"]
    lfs_text = "# Review only; do not run git lfs track without license and user approval.\n"
    lfs_text += "".join(f"{row['path']} filter=lfs diff=lfs merge=lfs -text\n" for row in lfs_candidates)
    atomic_write_bytes(root / "git_lfs_candidates.gitattributes", lfs_text.encode("utf-8"))
    report = [
        "# P4-C3A-G Git Release Report",
        "",
        f"- branch: `{audit['branch']}`",
        f"- commit: `{audit['commit']}`",
        f"- worktree clean: `{str(audit['worktree_clean']).lower()}`",
        f"- tracked files: {len(audit['tracked'])}",
        f"- untracked files: {len(audit['untracked'])}",
        f"- ignored entries: {len(audit['ignored'])}",
        f"- oversized files: {len(audit['oversized'])}",
        f"- sensitive findings: {len(audit['sensitive'])}",
        f"- release content SHA-256: `{audit['release_content_sha256']}`",
        "",
        "Model weights and six-video media remain outside Git. P4-C1/P4-C2 frozen",
        "artifacts and the empty P4-C3A batch plan are allow-listed as small checkout",
        "validation inputs. No commit, tag, push, download, training, or formal batch was run.",
    ]
    atomic_write_bytes(root / "P4C3A_GIT_RELEASE_REPORT.md", ("\n".join(report) + "\n").encode("utf-8"))
