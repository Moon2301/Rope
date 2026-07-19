"""Persistent state machine for the single-character Auto Job workflow.

The Qt layer owns user interaction and GPU workers.  This module deliberately
contains no Qt imports so manifests and transition rules can be tested without
loading PySide6 or CUDA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Iterable


AUTOMATION_JOB_VERSION = 1


class JobState(str, Enum):
    PREFLIGHT = "PREFLIGHT"
    SCANNING = "SCANNING"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    RENDERING = "RENDERING"
    MERGING = "MERGING"
    MUXING = "MUXING"
    QC = "QC"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    FAILED = "FAILED"


ACTIVE_STATES = {
    JobState.PREFLIGHT, JobState.SCANNING, JobState.AWAITING_REVIEW,
    JobState.RENDERING, JobState.MERGING, JobState.MUXING, JobState.QC,
    JobState.PAUSED, JobState.FAILED,
}


@dataclass
class AutomationJob:
    job_id: str
    video_path: str
    output_dir: str
    state: str = JobState.PREFLIGHT.value
    version: int = AUTOMATION_JOB_VERSION
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    video_fingerprint: dict[str, Any] = field(default_factory=dict)
    video_meta: dict[str, Any] = field(default_factory=dict)
    target_embedding: list[float] = field(default_factory=list)
    source_embedding: list[float] = field(default_factory=list)
    source_labels: list[str] = field(default_factory=list)
    scan_config: dict[str, Any] = field(default_factory=dict)
    render_params: dict[str, Any] = field(default_factory=dict)
    segments: list[dict[str, Any]] = field(default_factory=list)
    scan_checkpoint: dict[str, Any] = field(default_factory=dict)
    render_checkpoint: dict[str, Any] = field(default_factory=dict)
    user_confirmed: bool = False
    paused_from: str | None = None
    retry_from: str | None = None
    final_output: str | None = None
    qc: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutomationJob":
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in data.items() if key in fields}
        return cls(**values)


def fingerprint_video(path: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class AutomationJobStore:
    """Atomic JSON-backed job store with one directory per job."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.root / str(job_id)

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def save(self, job: AutomationJob) -> AutomationJob:
        job.updated_at = time.time()
        directory = self.job_dir(job.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = self.manifest_path(job.job_id)
        temp = directory / f"job.{os.getpid()}.tmp"
        temp.write_text(
            json.dumps(job.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp, target)
        return job

    def load(self, job_id: str) -> AutomationJob | None:
        try:
            raw = json.loads(self.manifest_path(job_id).read_text(encoding="utf-8"))
            if int(raw.get("version", 0)) != AUTOMATION_JOB_VERSION:
                return None
            return AutomationJob.from_dict(raw)
        except (OSError, ValueError, TypeError):
            return None

    def list_jobs(self) -> list[AutomationJob]:
        jobs: list[AutomationJob] = []
        try:
            manifests = self.root.glob("*/job.json")
        except OSError:
            return jobs
        for manifest in manifests:
            job = self.load(manifest.parent.name)
            if job is not None:
                jobs.append(job)
        return sorted(jobs, key=lambda item: item.updated_at, reverse=True)

    def latest_resumable(self, video_path: str | None = None) -> AutomationJob | None:
        resolved = str(Path(video_path).resolve()) if video_path else None
        for job in self.list_jobs():
            try:
                state = JobState(job.state)
            except ValueError:
                continue
            if state not in ACTIVE_STATES:
                continue
            if resolved and str(Path(job.video_path).resolve()) != resolved:
                continue
            return job
        return None


class AutomationController:
    """Decision gate and persistence API used by the Qt orchestration layer."""

    _ALLOWED: dict[JobState, set[JobState]] = {
        JobState.PREFLIGHT: {JobState.SCANNING, JobState.FAILED, JobState.PAUSED},
        JobState.SCANNING: {JobState.AWAITING_REVIEW, JobState.FAILED, JobState.PAUSED},
        JobState.AWAITING_REVIEW: {JobState.RENDERING, JobState.FAILED, JobState.PAUSED},
        JobState.RENDERING: {JobState.MERGING, JobState.FAILED, JobState.PAUSED},
        JobState.MERGING: {JobState.MUXING, JobState.FAILED, JobState.PAUSED},
        JobState.MUXING: {JobState.QC, JobState.FAILED, JobState.PAUSED},
        JobState.QC: {JobState.COMPLETED, JobState.FAILED, JobState.PAUSED},
        JobState.PAUSED: set(JobState),
        JobState.FAILED: set(JobState),
        JobState.COMPLETED: set(),
    }

    def __init__(self, root: str | os.PathLike[str]):
        self.store = AutomationJobStore(root)

    def start_job(self, request: dict[str, Any]) -> AutomationJob:
        job = AutomationJob(
            job_id=request.get("job_id") or uuid.uuid4().hex,
            video_path=str(request["video_path"]),
            output_dir=str(request["output_dir"]),
            video_fingerprint=dict(request.get("video_fingerprint") or
                                   fingerprint_video(request["video_path"])),
            video_meta=dict(request.get("video_meta", {})),
            target_embedding=[float(v) for v in request.get("target_embedding", [])],
            source_embedding=[float(v) for v in request.get("source_embedding", [])],
            source_labels=[str(v) for v in request.get("source_labels", [])],
            scan_config=dict(request.get("scan_config", {})),
            render_params=dict(request.get("render_params", {})),
        )
        return self.store.save(job)

    def transition(self, job: AutomationJob, state: JobState | str, **updates) -> AutomationJob:
        current = JobState(job.state)
        target = JobState(state)
        if target != current and target not in self._ALLOWED[current]:
            raise ValueError(f"Invalid Auto Job transition: {current.value} -> {target.value}")
        for key, value in updates.items():
            if key in AutomationJob.__dataclass_fields__:
                setattr(job, key, value)
        job.state = target.value
        if target != JobState.FAILED:
            job.error = updates.get("error")
        return self.store.save(job)

    def begin_scan(self, job: AutomationJob) -> AutomationJob:
        return self.transition(job, JobState.SCANNING, error=None)

    def set_scan_results(self, job: AutomationJob, segments: Iterable[dict]) -> AutomationJob:
        return self.transition(
            job, JobState.AWAITING_REVIEW,
            segments=[dict(item) for item in segments],
            user_confirmed=False, error=None,
        )

    def confirm_segments(self, job: AutomationJob, segments: Iterable[dict],
                         render_params: dict[str, Any]) -> AutomationJob:
        if JobState(job.state) != JobState.AWAITING_REVIEW:
            raise ValueError("Auto Job can only be confirmed from AWAITING_REVIEW")
        selected = [dict(item) for item in segments]
        if not any(bool(item.get("approved")) for item in selected):
            raise ValueError("At least one approved segment is required")
        return self.transition(
            job, JobState.RENDERING, segments=selected,
            render_params=dict(render_params), user_confirmed=True, error=None,
        )

    def pause_job(self, job: AutomationJob) -> AutomationJob:
        current = JobState(job.state)
        if current in (JobState.COMPLETED, JobState.PAUSED):
            return job
        return self.transition(job, JobState.PAUSED, paused_from=current.value)

    def resume_job(self, job_id: str) -> AutomationJob:
        job = self.store.load(job_id)
        if job is None:
            raise ValueError("Auto Job not found")
        state = JobState(job.state)
        if state == JobState.PAUSED:
            target = JobState(job.paused_from or JobState.PREFLIGHT.value)
        elif state == JobState.FAILED:
            target = JobState(job.retry_from or JobState.PREFLIGHT.value)
        else:
            return job
        return self.transition(job, target, error=None)

    def retry_job(self, job_id: str) -> AutomationJob:
        return self.resume_job(job_id)

    def fail_job(self, job: AutomationJob, message: str,
                 retry_from: JobState | str | None = None) -> AutomationJob:
        current = JobState(job.state)
        retry = JobState(retry_from).value if retry_from else current.value
        return self.transition(
            job, JobState.FAILED, error=str(message), retry_from=retry,
        )

    def complete_job(self, job: AutomationJob, final_output: str,
                     qc: dict[str, Any]) -> AutomationJob:
        return self.transition(
            job, JobState.COMPLETED, final_output=str(final_output),
            qc=dict(qc), error=None,
        )
