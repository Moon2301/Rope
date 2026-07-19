"""Data model and grouping helpers for character-aware batch rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


SCAN_CACHE_VERSION = 3


class SequentialScanDecoder:
    """Decode requested frames in one forward pass using a private container."""

    def __init__(self, video_path: str, fps: float, total_frames: int):
        self.video_path = str(video_path)
        self.fps = max(0.001, float(fps))
        self.total_frames = max(0, int(total_frames))

    def iter_frames(self, frame_numbers: Iterable[int], cancel_event=None):
        import av

        targets = sorted({
            max(0, min(self.total_frames - 1, int(frame)))
            for frame in frame_numbers if self.total_frames > 0
        })
        if not targets:
            return
        container = av.open(self.video_path)
        try:
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            # Seek slightly before the first target, then only decode forward.
            seek_seconds = max(0.0, targets[0] / self.fps - 2.0)
            try:
                container.seek(
                    int(seek_seconds / time_base), stream=stream,
                    any_frame=False, backward=True,
                )
            except Exception:
                container.seek(0)
            target_index = 0
            for packet in container.demux(stream):
                for decoded in packet.decode():
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    if decoded.pts is None:
                        continue
                    frame_no = int(round(float(decoded.pts) * time_base * self.fps))
                    if frame_no < targets[target_index]:
                        continue
                    # The first decoded frame at/after a requested timestamp
                    # is its sample. Normally this loop runs once; it also
                    # handles sparse/VFR input without reopening the file.
                    rgb = None
                    while target_index < len(targets) and frame_no >= targets[target_index]:
                        if rgb is None:
                            rgb = decoded.to_ndarray(format='rgb24')
                        yield targets[target_index], rgb
                        target_index += 1
                        if target_index >= len(targets):
                            return
        finally:
            container.close()


@dataclass
class AutoSegment:
    start_frame: int
    end_frame: int
    confidence: float
    approved: bool = True
    hit_count: int = 0
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.start_frame = max(0, int(self.start_frame))
        self.end_frame = max(self.start_frame, int(self.end_frame))
        self.confidence = max(0.0, min(100.0, float(self.confidence)))
        self.hit_count = max(0, int(self.hit_count))
        self.review_required = bool(self.review_required)
        self.review_reasons = [str(item) for item in self.review_reasons]

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
            hit_count=len(group),
        ))
    return result


def mark_segments_for_review(segments: Iterable[AutoSegment], *, threshold: float,
                             fps: float) -> list[AutoSegment]:
    """Annotate borderline/short/single-hit ranges without rejecting them.

    The user confirmation gate remains mandatory for every job; these flags
    merely direct attention to the rows most likely to need an edit.
    """
    result = []
    safe_fps = max(0.001, float(fps))
    for segment in segments:
        reasons = []
        if segment.confidence < float(threshold) + 5.0:
            reasons.append("confidence sát threshold")
        duration = (segment.end_frame - segment.start_frame + 1) / safe_fps
        if duration < 0.7:
            reasons.append("segment ngắn dưới 0,7 giây")
        if segment.hit_count <= 1:
            reasons.append("chỉ có một hit")
        segment.review_reasons = reasons
        segment.review_required = bool(reasons)
        result.append(segment)
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


def load_scan_manifest(cache_dir: str, key: str) -> dict | None:
    path = Path(cache_dir) / f"{key}.manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("version", 0)) != SCAN_CACHE_VERSION:
            return None
        return data
    except (OSError, ValueError, TypeError):
        return None


def save_scan_manifest(cache_dir: str, key: str, manifest: dict) -> None:
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{key}.manifest.json"
    temp = directory / f"{key}.{os.getpid()}.manifest.tmp"
    payload = dict(manifest)
    payload["version"] = SCAN_CACHE_VERSION
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, target)
