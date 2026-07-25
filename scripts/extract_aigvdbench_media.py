#!/usr/bin/env python3
"""Extract exact AIGVDBench ZIP members with resumable anonymous HTTP ranges."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any, BinaryIO, Iterable


HF_REPOSITORY = "AIGVDBench/AIGVDBench"
ARCHIVE_REVISION = "bf48acaa4920990af3dd2a511ea88e63354df305"
REAL_ARCHIVE = "AIGVDBench/Real/Real.zip"
OPEN_SORA_ARCHIVE = "AIGVDBench/OpenSource/T2V/Open-Sora.zip"
ARCHIVE_DETAILS = {
    REAL_ARCHIVE: "real_zip_details.json",
    OPEN_SORA_ARCHIVE: "open_sora_zip_details.json",
}
ARCHIVE_OUTPUT_NAMES = {
    REAL_ARCHIVE: "real.mp4",
    OPEN_SORA_ARCHIVE: "open_sora.mp4",
}
INVENTORY_FIELDS = (
    "sample_id",
    "group_id",
    "split",
    "label_for_posthoc_reference",
    "archive_path",
    "archive_revision",
    "archive_member_path",
    "local_video_path",
    "expected_uncompressed_size",
    "actual_file_size",
    "zip_crc",
    "sha256",
    "codec",
    "width",
    "height",
    "fps",
    "duration_seconds",
    "frame_count",
    "range_bytes_transferred",
    "extraction_seconds",
    "ffprobe_ok",
    "status",
    "failure_reason",
    "candidate_manifest_sha256",
)
LOCAL_HEADER = struct.Struct("<4s5H3L2H")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_identity(value: str) -> str:
    """Encode UTF-8 bijectively so distinct group IDs cannot collide."""

    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
    return "gid_" + encoded.rstrip("=")


def archive_url(archive_path: str, revision: str = ARCHIVE_REVISION) -> str:
    quoted = urllib.parse.quote(archive_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{HF_REPOSITORY}/resolve/"
        f"{revision}/{quoted}?download=true"
    )


def _open_range(url: str, start: int, size: int) -> tuple[BinaryIO, dict[str, str]]:
    if size < 1:
        raise ValueError("Range size must be positive.")
    end = start + size - 1
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "paper-core-aigvdbench-range-extractor/1.0",
        },
    )
    response = urllib.request.urlopen(request, timeout=180)
    headers = {key.lower(): value for key, value in response.headers.items()}
    status = getattr(response, "status", None)
    expected_range = f"bytes {start}-{end}/"
    if status != 206:
        response.close()
        raise RuntimeError(f"Range request returned HTTP {status}, expected 206.")
    if not headers.get("content-range", "").startswith(expected_range):
        response.close()
        raise RuntimeError(f"Unexpected Content-Range: {headers.get('content-range')!r}.")
    if int(headers.get("content-length", "-1")) != size:
        response.close()
        raise RuntimeError(f"Unexpected Content-Length: {headers.get('content-length')!r}.")
    return response, headers


def _read_range(url: str, start: int, size: int) -> bytes:
    response, _ = _open_range(url, start, size)
    try:
        data = response.read(size + 1)
    finally:
        response.close()
    if len(data) != size:
        raise RuntimeError(f"Range returned {len(data)} bytes, expected {size}.")
    return data


def _extract_member(
    *,
    url: str,
    member: dict[str, Any],
    expected_member_path: str,
    part_path: Path,
) -> dict[str, Any]:
    local_offset = int(member["local_header_offset"])
    compressed_size = int(member["compressed_size"])
    uncompressed_size = int(member["uncompressed_size"])
    expected_crc = int(str(member["crc"]), 16)
    method = int(member["compression"])
    fixed = _read_range(url, local_offset, LOCAL_HEADER.size)
    (
        signature,
        _needed,
        flags,
        local_method,
        _mtime,
        _mdate,
        _local_crc,
        _local_compressed,
        _local_uncompressed,
        name_len,
        extra_len,
    ) = LOCAL_HEADER.unpack(fixed)
    if signature != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local-header signature.")
    if local_method != method:
        raise ValueError("Central/local ZIP compression methods differ.")
    variable = _read_range(
        url, local_offset + LOCAL_HEADER.size, name_len + extra_len
    )
    encoding = "utf-8" if flags & 0x800 else "cp437"
    local_name = variable[:name_len].decode(encoding).replace("\\", "/")
    if local_name != expected_member_path:
        raise ValueError(
            f"Exact member mismatch: local={local_name!r}, expected={expected_member_path!r}."
        )
    payload_offset = local_offset + LOCAL_HEADER.size + name_len + extra_len
    response, _ = _open_range(url, payload_offset, compressed_size)
    digest = hashlib.sha256()
    crc = 0
    actual = 0
    transferred = LOCAL_HEADER.size + name_len + extra_len
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS) if method == 8 else None
    if method not in {0, 8}:
        response.close()
        raise ValueError(f"Unsupported ZIP compression method {method}.")
    try:
        with part_path.open("wb") as output:
            remaining = compressed_size
            while remaining:
                block = response.read(min(1024 * 1024, remaining))
                if not block:
                    raise RuntimeError("Remote range ended before compressed member payload.")
                remaining -= len(block)
                transferred += len(block)
                decoded = block if method == 0 else decompressor.decompress(block)
                if decoded:
                    output.write(decoded)
                    digest.update(decoded)
                    crc = zlib.crc32(decoded, crc)
                    actual += len(decoded)
            tail = b"" if decompressor is None else decompressor.flush()
            if tail:
                output.write(tail)
                digest.update(tail)
                crc = zlib.crc32(tail, crc)
                actual += len(tail)
            output.flush()
            os.fsync(output.fileno())
    finally:
        response.close()
    if transferred != LOCAL_HEADER.size + name_len + extra_len + compressed_size:
        raise RuntimeError("Transferred-byte count does not match exact ZIP member ranges.")
    if actual != uncompressed_size:
        raise ValueError(f"Uncompressed size mismatch: {actual} != {uncompressed_size}.")
    if crc & 0xFFFFFFFF != expected_crc:
        raise ValueError(f"ZIP CRC mismatch: {crc & 0xFFFFFFFF:08x} != {expected_crc:08x}.")
    return {
        "actual_file_size": actual,
        "sha256": digest.hexdigest(),
        "zip_crc": f"{expected_crc:08x}",
        "range_bytes_transferred": transferred,
    }


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_type,codec_name,width,height,avg_frame_rate,"
            "nb_frames,nb_read_frames,duration:format=duration,size"
        ),
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    videos = [
        stream
        for stream in payload.get("streams", ())
        if stream.get("codec_type") == "video"
    ]
    if len(videos) != 1:
        raise ValueError(f"ffprobe found {len(videos)} video streams.")
    video = videos[0]
    numerator, denominator = (
        int(value) for value in video.get("avg_frame_rate", "0/1").split("/")
    )
    frame_count = video.get("nb_read_frames") or video.get("nb_frames")
    if denominator == 0 or numerator <= 0 or not frame_count:
        raise ValueError("ffprobe returned invalid FPS or frame count.")
    return {
        "codec": str(video["codec_name"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": numerator / denominator,
        "duration_seconds": float(
            video.get("duration") or payload["format"]["duration"]
        ),
        "frame_count": int(frame_count),
        "ffprobe_ok": True,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(
    path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _inventory_paths(value: Path) -> tuple[Path, Path]:
    if value.suffix.lower() == ".csv":
        return value, value.with_name("local_media_manifest.csv")
    return (
        value / "aigvdbench_pilot_v2_media_inventory.csv",
        value / "local_media_manifest.csv",
    )


def _resume_sources(inventory_path: Path) -> list[Path]:
    candidates = [inventory_path]
    parent = inventory_path.parent.parent
    if parent.is_dir():
        candidates.extend(
            path
            for path in parent.glob("*/aigvdbench_pilot_v2_media_inventory.csv")
            if path != inventory_path
        )
    return candidates


def _load_resume_records(inventory_path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for path in _resume_sources(inventory_path):
        if not path.is_file():
            continue
        for row in _read_csv(path):
            if row.get("status") == "COMPLETE":
                records.setdefault(row["sample_id"], row)
    return records


def _manifest_identity(rows: list[dict[str, str]], path: Path) -> str:
    declared = {
        row.get("candidate_manifest_sha256", "").strip()
        for row in rows
        if row.get("candidate_manifest_sha256", "").strip()
    }
    if declared:
        if len(declared) != 1:
            raise ValueError("Manifest has multiple candidate_manifest_sha256 values.")
        return next(iter(declared))
    return sha256_file(path)


def _resume_valid(
    row: dict[str, str],
    prior: dict[str, str] | None,
    local_path: Path,
    manifest_sha256: str,
) -> bool:
    if prior is None or prior.get("status") != "COMPLETE" or not local_path.is_file():
        return False
    expected = {
        "candidate_manifest_sha256": manifest_sha256,
        "archive_revision": ARCHIVE_REVISION,
        "archive_member_path": row["archive_member_path"],
    }
    if any(prior.get(key) != value for key, value in expected.items()):
        return False
    if local_path.stat().st_size != int(prior["actual_file_size"]):
        return False
    if sha256_file(local_path) != prior["sha256"]:
        return False
    try:
        ffprobe(local_path)
    except Exception:
        return False
    return True


def _quarantine(path: Path, output_root: Path, reason: str) -> Path:
    quarantine = output_root / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(
        f"{path}:{path.stat().st_size}:{time.time_ns()}".encode()
    ).hexdigest()[:16]
    target = quarantine / f"{path.name}.{reason}.{suffix}"
    shutil.move(str(path), target)
    return target


def _load_details(manifest: Path) -> dict[str, dict[str, dict[str, Any]]]:
    audit_root = manifest.parent.parent
    access_probe = audit_root / "access_probe"
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for archive, filename in ARCHIVE_DETAILS.items():
        path = access_probe / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required audited ZIP index is missing: {path}.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = {
            str(row["member_path"]): row
            for row in payload.get("entries", ())
            if bool(row.get("is_video"))
        }
        if len(rows) != len(
            [row for row in payload.get("entries", ()) if bool(row.get("is_video"))]
        ):
            raise ValueError(f"Duplicate member path in {path}.")
        output[archive] = rows
    return output


def _selected_rows(
    rows: list[dict[str, str]], split: str | None, sample_limit: int | None
) -> list[dict[str, str]]:
    selected = [row for row in rows if split is None or row["split"] == split]
    if sample_limit is not None:
        if sample_limit < 1:
            raise ValueError("--sample-limit must be positive.")
        selected = selected[:sample_limit]
    return selected


def run(arguments: argparse.Namespace) -> int:
    manifest = Path(arguments.manifest).resolve()
    output_root = Path(arguments.output).resolve()
    inventory_path, local_manifest_path = _inventory_paths(
        Path(arguments.inventory_output).resolve()
    )
    rows = _read_csv(manifest)
    manifest_sha256 = _manifest_identity(rows, manifest)
    expected_pilot = "688c8d7a1995bef1ca23ba42d1072811a7a1be7cae5690cb49773fb598b5dfe9"
    if manifest_sha256 != expected_pilot:
        raise ValueError("Candidate manifest SHA-256 does not match frozen Pilot-v2.")
    selected = _selected_rows(rows, arguments.split, arguments.sample_limit)
    details = _load_details(manifest)
    prior = _load_resume_records(inventory_path) if arguments.resume else {}
    inventory_by_sample = {
        row["sample_id"]: row for row in _read_csv(inventory_path)
    } if inventory_path.is_file() else {}
    local_rows: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    def persist() -> None:
        ordered_inventory = [
            inventory_by_sample[row["sample_id"]]
            for row in selected
            if row["sample_id"] in inventory_by_sample
        ]
        _write_csv_atomic(inventory_path, ordered_inventory, INVENTORY_FIELDS)
        _write_csv_atomic(
            local_manifest_path,
            local_rows,
            list(rows[0]) + [
                "local_video_path",
                "source_video_sha256",
                "media_inventory_status",
                "candidate_manifest_sha256",
            ],
        )

    def write_group_status() -> None:
        groups: dict[str, list[dict[str, str]]] = {}
        for candidate in selected:
            groups.setdefault(candidate["group_id"], []).append(candidate)
        status_rows = []
        for group_id, members in groups.items():
            real = next(
                (member for member in members if member["archive_path"] == REAL_ARCHIVE),
                None,
            )
            fake = next(
                (
                    member
                    for member in members
                    if member["archive_path"] == OPEN_SORA_ARCHIVE
                ),
                None,
            )
            real_status = (
                inventory_by_sample.get(real["sample_id"], {}).get("status", "PENDING")
                if real is not None
                else "MISSING"
            )
            fake_status = (
                inventory_by_sample.get(fake["sample_id"], {}).get("status", "PENDING")
                if fake is not None
                else "MISSING"
            )
            complete = real_status == fake_status == "COMPLETE"
            reasons = [
                inventory_by_sample.get(member["sample_id"], {}).get(
                    "failure_reason", ""
                )
                for member in members
            ]
            status_rows.append(
                {
                    "group_id": group_id,
                    "split": members[0]["split"],
                    "real_status": real_status,
                    "fake_status": fake_status,
                    "group_status": (
                        "GROUP_MEDIA_COMPLETE" if complete else "GROUP_MEDIA_PARTIAL"
                    ),
                    "failure_reason": "|".join(reason for reason in reasons if reason),
                }
            )
        targets = [inventory_path.parent / "group_media_status.csv"]
        if len(selected) == 2000:
            targets.append(inventory_path.parent.parent / "group_media_status.csv")
        for target in targets:
            _write_csv_atomic(
                target,
                status_rows,
                (
                    "group_id",
                    "split",
                    "real_status",
                    "fake_status",
                    "group_status",
                    "failure_reason",
                ),
            )

    for index, row in enumerate(selected, 1):
        started = time.perf_counter()
        sample_id = row["sample_id"]
        archive = row["archive_path"]
        if archive not in ARCHIVE_OUTPUT_NAMES:
            raise ValueError(f"Unsupported official archive path: {archive!r}.")
        group_dir = output_root / row["split"] / safe_identity(row["group_id"])
        local_path = group_dir / ARCHIVE_OUTPUT_NAMES[archive]
        part_path = local_path.with_name(local_path.name + ".part")
        base = {
            "sample_id": sample_id,
            "group_id": row["group_id"],
            "split": row["split"],
            "label_for_posthoc_reference": row["label"],
            "archive_path": archive,
            "archive_revision": ARCHIVE_REVISION,
            "archive_member_path": row["archive_member_path"],
            "local_video_path": str(local_path),
            "expected_uncompressed_size": row["file_size_uncompressed"],
            "actual_file_size": 0,
            "zip_crc": "",
            "sha256": "",
            "codec": "",
            "width": 0,
            "height": 0,
            "fps": 0,
            "duration_seconds": 0,
            "frame_count": 0,
            "range_bytes_transferred": 0,
            "extraction_seconds": 0,
            "ffprobe_ok": False,
            "status": "PENDING",
            "failure_reason": "",
            "candidate_manifest_sha256": manifest_sha256,
        }
        previous = prior.get(sample_id)
        if arguments.resume and _resume_valid(
            row, previous, local_path, manifest_sha256
        ):
            complete = {**base, **previous, "status": "COMPLETE", "failure_reason": ""}
            inventory_by_sample[sample_id] = complete
            local_rows.append(
                {
                    **row,
                    "local_video_path": str(local_path),
                    "source_video_sha256": complete["sha256"],
                    "media_inventory_status": "COMPLETE",
                    "candidate_manifest_sha256": manifest_sha256,
                }
            )
            print(
                f"[MEDIA] sample={sample_id} group={row['group_id']} split={row['split']} "
                f"index={index}/{len(selected)} member={row['archive_member_path']} "
                f"downloaded_bytes=0 file_size={complete['actual_file_size']} "
                f"elapsed={time.perf_counter()-started:.3f} status=COMPLETE_RESUME "
                "failure_reason=",
                flush=True,
            )
            persist()
            continue
        group_dir.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            _quarantine(local_path, output_root, "resume_mismatch")
        if part_path.exists():
            _quarantine(part_path, output_root, "incomplete_part")
        inventory_by_sample[sample_id] = {**base, "status": "EXTRACTING"}
        persist()
        try:
            member = details[archive].get(row["archive_member_path"])
            if member is None:
                raise ValueError("Exact archive member is absent from audited index.")
            if int(member["compressed_size"]) != int(row["file_size_compressed"]):
                raise ValueError("Manifest/index compressed size mismatch.")
            if int(member["uncompressed_size"]) != int(row["file_size_uncompressed"]):
                raise ValueError("Manifest/index uncompressed size mismatch.")
            extracted = _extract_member(
                url=archive_url(archive),
                member=member,
                expected_member_path=row["archive_member_path"],
                part_path=part_path,
            )
            probed = ffprobe(part_path)
            os.replace(part_path, local_path)
            complete = {
                **base,
                **extracted,
                **probed,
                "extraction_seconds": time.perf_counter() - started,
                "status": "COMPLETE",
                "failure_reason": "",
            }
            inventory_by_sample[sample_id] = complete
            local_rows.append(
                {
                    **row,
                    "local_video_path": str(local_path),
                    "source_video_sha256": complete["sha256"],
                    "media_inventory_status": "COMPLETE",
                    "candidate_manifest_sha256": manifest_sha256,
                }
            )
            status = "COMPLETE"
            failure = ""
        except KeyboardInterrupt:
            inventory_by_sample[sample_id] = {
                **base,
                "status": "EXTRACTING",
                "failure_reason": "INTERRUPTED",
                "extraction_seconds": time.perf_counter() - started,
            }
            persist()
            raise
        except Exception as error:
            part_path.unlink(missing_ok=True)
            inventory_by_sample[sample_id] = {
                **base,
                "status": "FAILED",
                "failure_reason": f"{type(error).__name__}:{error}",
                "extraction_seconds": time.perf_counter() - started,
            }
            status = "FAILED"
            failure = inventory_by_sample[sample_id]["failure_reason"]
        persist()
        current = inventory_by_sample[sample_id]
        print(
            f"[MEDIA] sample={sample_id} group={row['group_id']} split={row['split']} "
            f"index={index}/{len(selected)} member={row['archive_member_path']} "
            f"downloaded_bytes={current['range_bytes_transferred']} "
            f"file_size={current['actual_file_size']} "
            f"elapsed={time.perf_counter()-started:.3f} status={status} "
            f"failure_reason={failure}",
            flush=True,
        )
        if status != "COMPLETE":
            write_group_status()
            return 1
    write_group_status()
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--inventory-output", required=True)
    result.add_argument("--split", choices=("train", "validation", "test"))
    result.add_argument("--sample-limit", type=int)
    result.add_argument("--resume", action="store_true")
    return result


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
