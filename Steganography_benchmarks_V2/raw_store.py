"""Incremental JSONL storage for raw model responses."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import MANIFEST_FILE, RAW_RESPONSES_FILE, record_id


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class RawResponseStore:
    """Append-only JSONL + manifest updated after every task."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.raw_path = run_dir / RAW_RESPONSES_FILE
        self.manifest_path = run_dir / MANIFEST_FILE
        self._manifest: dict[str, Any] = {}

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    def init_manifest(self, config: dict[str, Any], total_tasks: int) -> None:
        self._manifest = {
            "phase": "generation",
            "status": "in_progress",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "raw_file": RAW_RESPONSES_FILE,
            "total_tasks": total_tasks,
            "completed_count": 0,
            "completed_task_ids": [],
            **config,
        }
        self._write_manifest()

    def load_existing(self) -> set[str]:
        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self._manifest = {}

        completed: set[str] = set()
        if self.raw_path.exists():
            with self.raw_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    completed.add(record_id(row))
        if completed:
            self._manifest["completed_task_ids"] = sorted(completed)
            self._manifest["completed_count"] = len(completed)
        return completed

    def append_raw(self, record: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with self.raw_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

        rid = record_id(record)
        completed = set(self._manifest.get("completed_task_ids", []))
        completed.add(rid)
        self._manifest["completed_task_ids"] = sorted(completed)
        self._manifest["completed_count"] = len(completed)
        self._manifest["updated_at"] = utc_now_iso()
        self._manifest["last_record_id"] = rid
        self._write_manifest()

    def mark_completed(self, extra: dict[str, Any] | None = None) -> None:
        self._manifest["status"] = "completed"
        self._manifest["completed_at"] = utc_now_iso()
        self._manifest["updated_at"] = utc_now_iso()
        if extra:
            self._manifest.update(extra)
        self._write_manifest()

    def mark_failed(self, error: str) -> None:
        self._manifest["status"] = "failed"
        self._manifest["error"] = error
        self._manifest["updated_at"] = utc_now_iso()
        self._write_manifest()

    def _write_manifest(self) -> None:
        save_json(self.manifest_path, self._manifest)


def find_raw_responses_file(run_dir: Path) -> Path:
    """Standard name or any single *.jsonl in the run folder."""
    standard = run_dir / RAW_RESPONSES_FILE
    if standard.exists():
        return standard
    candidates = sorted(run_dir.glob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No raw responses JSONL in: {run_dir}")
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda p: p.stat().st_size)


def _infer_manifest_from_dir(run_dir: Path, raw_path: Path, records: list[dict]) -> dict[str, Any]:
    name = run_dir.name.lower()
    model_key = "unknown"
    threshold = 0.0
    for key in ("llama", "qwen", "gemma"):
        if key in name:
            model_key = key
            break
    if "0_01" in name or "0.01" in name or "th0_01" in name:
        threshold = 0.01
    elif "0_05" in name or "0.05" in name or "th0_05" in name:
        threshold = 0.05
    elif "0_1" in name or "th0_1" in name:
        if "0_10" not in name and "0.10" not in name:
            threshold = 0.1
    elif "th0_0" in name or name.endswith("_0_0") or "_0_0" in name.split("_")[-2:]:
        threshold = 0.0

    from common import MODELS

    model_id = MODELS.get(model_key, model_key)
    test = "humaneval"
    if records and records[0].get("sample_id"):
        test = str(records[0]["sample_id"]).split("/")[0]

    task_ids = sorted(
        {str(r.get("task_id") or r.get("sample_id")) for r in records if r.get("task_id") or r.get("sample_id")}
    )
    return {
        "phase": "generation",
        "status": "completed",
        "inferred": True,
        "raw_file": raw_path.name,
        "test": test,
        "model_key": model_key,
        "model_id": model_id,
        "threshold": threshold,
        "top_n": 16,
        "max_new_tokens": 512,
        "seed": 1234,
        "humaneval_tasks": None,
        "platform": "kaggle",
        "total_tasks": len(records),
        "completed_count": len(records),
        "completed_task_ids": task_ids,
    }


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / MANIFEST_FILE
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    raw_path = find_raw_responses_file(run_dir)
    records = load_raw_records(run_dir, raw_path=raw_path)
    manifest = _infer_manifest_from_dir(run_dir, raw_path, records)
    save_json(manifest_path, manifest)
    print(f"Inferred manifest -> {manifest_path}", flush=True)
    return manifest


def load_raw_records(run_dir: Path, raw_path: Path | None = None) -> list[dict[str, Any]]:
    raw_path = raw_path or find_raw_responses_file(run_dir)

    records: list[dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {raw_path}") from exc
    return records

