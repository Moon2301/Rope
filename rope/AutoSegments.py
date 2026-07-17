"""Data model and grouping helpers for character-aware batch rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


@dataclass
class AutoSegment:
    start_frame: int
    end_frame: int
    confidence: float
    approved: bool = True

    def __post_init__(self) -> None:
        self.start_frame = max(0, int(self.start_frame))
        self.end_frame = max(self.start_frame, int(self.end_frame))
        self.confidence = max(0.0, min(100.0, float(self.confidence)))

    def to_dict(self) -> dict:
        return asdict(self)


def group_matches(
    matches: Iterable[tuple[int, float]], *, stride: int, gap_frames: int,
    padding_frames: int, total_frames: int,
) -> list[AutoSegment]:
    """Turn sampled ``(frame, similarity)`` hits into editable ranges.

    Hits separated by at most ``stride + gap_frames`` belong to the same
    appearance. Padding compensates for frames skipped by sampling.
    """
    ordered = sorted((max(0, int(f)), float(c)) for f, c in matches)
    if not ordered or total_frames <= 0:
        return []
    groups: list[list[tuple[int, float]]] = [[ordered[0]]]
    max_delta = max(1, int(stride)) + max(0, int(gap_frames))
    for hit in ordered[1:]:
        if hit[0] - groups[-1][-1][0] <= max_delta:
            groups[-1].append(hit)
        else:
            groups.append([hit])

    result = []
    pad = max(0, int(padding_frames))
    last_frame = max(0, int(total_frames) - 1)
    for group in groups:
        result.append(AutoSegment(
            max(0, group[0][0] - pad),
            min(last_frame, group[-1][0] + max(1, int(stride)) - 1 + pad),
            sum(c for _, c in group) / len(group),
        ))
    return result


def approved_ranges(segments: Iterable[AutoSegment | dict]) -> list[tuple[int, int]]:
    ranges = []
    for segment in segments:
        data = segment if isinstance(segment, dict) else segment.to_dict()
        if data.get("approved", False):
            ranges.append((int(data["start_frame"]), int(data["end_frame"])))
    return sorted(ranges)


def frame_in_ranges(frame: int, ranges: Iterable[tuple[int, int]]) -> bool:
    value = int(frame)
    return any(int(start) <= value <= int(end) for start, end in ranges)


def scan_cache_key(video_path: str, target_embedding, config: dict) -> str:
    """Stable cache key that invalidates when video, face, or scan config changes."""
    path = Path(video_path).resolve()
    stat = path.stat()
    emb = getattr(target_embedding, "tobytes", lambda: bytes(target_embedding))()
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8", "surrogatepass"))
    digest.update(f"|{stat.st_size}|{stat.st_mtime_ns}|".encode())
    digest.update(emb)
    digest.update(payload)
    return digest.hexdigest()


def load_scan_cache(cache_dir: str, key: str) -> list[AutoSegment] | None:
    path = Path(cache_dir) / f"{key}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [AutoSegment(**item) for item in data.get("segments", [])]
    except (OSError, ValueError, TypeError):
        return None


def save_scan_cache(cache_dir: str, key: str, segments: Iterable[AutoSegment]) -> None:
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{key}.json"
    temp = directory / f"{key}.{os.getpid()}.tmp"
    payload = {"version": 1, "segments": [item.to_dict() for item in segments]}
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, target)
