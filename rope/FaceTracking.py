"""Ordered five-point face tracking and temporal alignment stabilization.

The detector and recognizer remain frame-local and can run on several worker
threads.  :class:`TemporalFaceTracker` only serializes the small CPU-side
association/filter step so the state is always updated in timeline order.
"""

from __future__ import annotations

import base64
import hashlib
import math
import threading
from dataclasses import dataclass
import numpy as np


TRACKING_VERSION = 1

# InsightFace's canonical five-point geometry.  The tracker estimates a
# similarity transform from this stable shape to the detector landmarks,
# filters the transform, then reconstructs five points for the existing Rope
# alignment pipeline.
CANONICAL_KPS = np.asarray([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)
_CANONICAL_CENTER = CANONICAL_KPS.mean(axis=0)
_CANONICAL_CENTERED = CANONICAL_KPS - _CANONICAL_CENTER


def make_slot_key(target_embedding, source_embedding) -> str:
    """Return a stable identity for one target/source assignment."""
    digest = hashlib.sha256()
    for embedding in (target_embedding, source_embedding):
        value = np.ascontiguousarray(embedding, dtype=np.float32).reshape(-1)
        digest.update(value.tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def _rotation_row(angle: float) -> np.ndarray:
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    # Row-vector rotation: [x, y] @ R.
    return np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)


def _unwrap_near(value: float, reference: float) -> float:
    return float(reference + ((value - reference + math.pi) % (2.0 * math.pi) - math.pi))


def _landmarks_to_transform(kps, frame_size) -> tuple[np.ndarray, float]:
    """Return pixel center/log-scale/angle plus a pixel face scale."""
    points = np.asarray(kps, dtype=np.float64).reshape(5, 2)
    _ = frame_size  # sequence resets separately when the video size changes
    center = points.mean(axis=0)
    target = points - center

    covariance = _CANONICAL_CENTERED.T @ target
    left, singular, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    denominator = float(np.square(_CANONICAL_CENTERED).sum())
    scale = max(1e-6, float(singular.sum()) / max(1e-12, denominator))
    angle = math.atan2(float(rotation[0, 1]), float(rotation[0, 0]))

    pairwise = points[:, None, :] - points[None, :, :]
    face_scale = max(1.0, float(np.sqrt(np.square(pairwise).sum(axis=2)).max()))
    transform = np.asarray([
        center[0],
        center[1],
        math.log(scale),
        angle,
    ], dtype=np.float64)
    return transform, face_scale


def _transform_to_landmarks(transform, frame_size) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64).reshape(4)
    _ = frame_size
    center = np.asarray([value[0], value[1]], dtype=np.float64)
    scale = math.exp(float(value[2]))
    points = _CANONICAL_CENTERED @ _rotation_row(float(value[3]))
    return (points * scale + center).astype(np.float32)


class _OneEuroFilter:
    def __init__(self, min_cutoff=1.5, beta=0.08, derivative_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.raw = None
        self.filtered = None
        self.derivative = None

    @staticmethod
    def _alpha(cutoff, dt):
        cutoff = np.maximum(1e-6, np.asarray(cutoff, dtype=np.float64))
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(1e-6, float(dt)))

    def apply(self, value, dt):
        current = np.asarray(value, dtype=np.float64).reshape(4)
        if self.filtered is None:
            self.raw = current.copy()
            self.filtered = current.copy()
            self.derivative = np.zeros_like(current)
            return current.copy()

        derivative = (current - self.raw) / max(1e-6, float(dt))
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        derivative_hat = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self.derivative
        )
        cutoff = self.min_cutoff + self.beta * np.abs(derivative_hat)
        value_alpha = self._alpha(cutoff, dt)
        filtered = value_alpha * current + (1.0 - value_alpha) * self.filtered
        self.raw = current.copy()
        self.filtered = filtered.copy()
        self.derivative = derivative_hat.copy()
        return filtered

    def to_dict(self):
        return {
            "raw": None if self.raw is None else self.raw.tolist(),
            "filtered": None if self.filtered is None else self.filtered.tolist(),
            "derivative": None if self.derivative is None else self.derivative.tolist(),
        }

    def restore(self, data):
        for name in ("raw", "filtered", "derivative"):
            value = (data or {}).get(name)
            setattr(self, name, None if value is None else np.asarray(value, dtype=np.float64))


@dataclass
class _TrackState:
    slot_key: str
    filter: _OneEuroFilter
    raw_transform: np.ndarray
    output_transform: np.ndarray
    velocity: np.ndarray
    frame_size: tuple[int, int]
    face_scale: float
    last_frame: int
    last_confirmed_frame: int
    hit_streak: int
    miss_count: int
    similarity: float

    def to_dict(self):
        return {
            "slot_key": self.slot_key,
            "filter": self.filter.to_dict(),
            "raw_transform": self.raw_transform.tolist(),
            "output_transform": self.output_transform.tolist(),
            "velocity": self.velocity.tolist(),
            "frame_size": list(self.frame_size),
            "face_scale": float(self.face_scale),
            "last_frame": int(self.last_frame),
            "last_confirmed_frame": int(self.last_confirmed_frame),
            "hit_streak": int(self.hit_streak),
            "miss_count": int(self.miss_count),
            "similarity": float(self.similarity),
        }

    @classmethod
    def from_dict(cls, data):
        euro = _OneEuroFilter()
        euro.restore(data.get("filter"))
        return cls(
            slot_key=str(data["slot_key"]),
            filter=euro,
            raw_transform=np.asarray(data["raw_transform"], dtype=np.float64),
            output_transform=np.asarray(data["output_transform"], dtype=np.float64),
            velocity=np.asarray(data.get("velocity", [0.0] * 4), dtype=np.float64),
            frame_size=tuple(int(v) for v in data["frame_size"]),
            face_scale=float(data["face_scale"]),
            last_frame=int(data["last_frame"]),
            last_confirmed_frame=int(data["last_confirmed_frame"]),
            hit_streak=int(data.get("hit_streak", 0)),
            miss_count=int(data.get("miss_count", 0)),
            similarity=float(data.get("similarity", 0.0)),
        )


class TemporalFaceTracker:
    """Timeline-ordered association and adaptive five-point stabilization."""

    scene_cut_threshold = 0.22
    tracked_threshold_margin = 3.0
    max_center_distance = 0.75
    min_scale_ratio = 0.55
    max_scale_ratio = 1.8

    def __init__(self):
        self._condition = threading.Condition()
        self._generation = 0
        self._aborted = True
        self._expected_frame = 0
        self._fps = 30.0
        self._states: dict[str, _TrackState] = {}
        self._slot_signature: tuple[str, ...] = ()
        self._last_processed_frame = None
        self._last_frame_signature = None
        self._last_orientation = None
        self._last_frame_size = None

    @property
    def generation(self):
        with self._condition:
            return self._generation

    def start_sequence(self, start_frame, fps, checkpoint=None):
        with self._condition:
            self._generation += 1
            self._aborted = False
            self._expected_frame = int(start_frame)
            self._fps = max(0.001, float(fps))
            self._states = {}
            self._slot_signature = ()
            self._last_processed_frame = None
            self._last_frame_signature = None
            self._last_orientation = None
            self._last_frame_size = None
            if (
                isinstance(checkpoint, dict)
                and int(checkpoint.get("version", 0)) == TRACKING_VERSION
                and int(checkpoint.get("next_frame", -1)) == self._expected_frame
            ):
                self._restore_checkpoint(checkpoint)
            self._condition.notify_all()
            return self._generation

    def abort_sequence(self, generation=None):
        with self._condition:
            if generation is None or int(generation) == self._generation:
                self._aborted = True
                self._condition.notify_all()

    def process(
        self, frame_number, observations, frame_signature, eligible=True, *,
        slots=(), threshold=55.0, frame_size=(1, 1), orientation=0.0,
        generation=None,
    ):
        frame_number = int(frame_number)
        with self._condition:
            active_generation = self._generation if generation is None else int(generation)
            while (
                active_generation == self._generation
                and not self._aborted
                and frame_number > self._expected_frame
            ):
                self._condition.wait(timeout=0.25)
            if (
                active_generation != self._generation
                or self._aborted
                or frame_number < self._expected_frame
            ):
                return []
            try:
                return self._process_ordered(
                    frame_number, list(observations or []), frame_signature,
                    bool(eligible), list(slots or []), float(threshold),
                    tuple(frame_size), float(orientation),
                )
            finally:
                self._expected_frame = frame_number + 1
                self._condition.notify_all()

    def skip(self, frame_number, frame_signature=None, reset=True, *, generation=None):
        return self.process(
            frame_number, [], frame_signature, eligible=not bool(reset),
            slots=(), threshold=100.0, frame_size=(1, 1), orientation=0.0,
            generation=generation,
        )

    def export_checkpoint(self):
        with self._condition:
            signature = None
            if self._last_frame_signature is not None:
                value = np.ascontiguousarray(self._last_frame_signature, dtype=np.uint8)
                signature = {
                    "shape": list(value.shape),
                    "data": base64.b64encode(value.tobytes()).decode("ascii"),
                }
            return {
                "version": TRACKING_VERSION,
                "next_frame": int(self._expected_frame),
                "fps": float(self._fps),
                "slot_signature": list(self._slot_signature),
                "last_processed_frame": self._last_processed_frame,
                "last_orientation": self._last_orientation,
                "last_frame_size": None if self._last_frame_size is None else list(self._last_frame_size),
                "frame_signature": signature,
                "states": {key: state.to_dict() for key, state in self._states.items()},
            }

    def _restore_checkpoint(self, checkpoint):
        try:
            self._states = {
                str(key): _TrackState.from_dict(value)
                for key, value in (checkpoint.get("states") or {}).items()
            }
            self._slot_signature = tuple(str(v) for v in checkpoint.get("slot_signature", []))
            self._last_processed_frame = checkpoint.get("last_processed_frame")
            self._last_orientation = checkpoint.get("last_orientation")
            size = checkpoint.get("last_frame_size")
            self._last_frame_size = None if size is None else tuple(int(v) for v in size)
            packed = checkpoint.get("frame_signature")
            if packed:
                raw = base64.b64decode(packed["data"])
                self._last_frame_signature = np.frombuffer(raw, dtype=np.uint8).reshape(packed["shape"]).copy()
        except (KeyError, TypeError, ValueError):
            self._states = {}
            self._slot_signature = ()
            self._last_processed_frame = None
            self._last_frame_signature = None
            self._last_orientation = None
            self._last_frame_size = None

    def _reset_tracks(self):
        self._states.clear()

    def _scene_difference(self, signature):
        if signature is None or self._last_frame_signature is None:
            return 0.0
        current = np.asarray(signature, dtype=np.uint8)
        previous = np.asarray(self._last_frame_signature, dtype=np.uint8)
        if current.shape != previous.shape:
            return 1.0
        return float(np.abs(current.astype(np.float32) - previous.astype(np.float32)).mean() / 255.0)

    def _process_ordered(
        self, frame_number, observations, frame_signature, eligible, slots,
        threshold, frame_size, orientation,
    ):
        frame_size = (max(1, int(frame_size[0])), max(1, int(frame_size[1])))
        slot_keys = tuple(sorted(str(slot["key"]) for slot in slots))
        scene_difference = self._scene_difference(frame_signature)
        discontinuity = (
            self._last_processed_frame is not None
            and frame_number != int(self._last_processed_frame) + 1
        )
        context_changed = (
            slot_keys != self._slot_signature
            or self._last_frame_size not in (None, frame_size)
            or (
                self._last_orientation is not None
                # Orientation values are degrees. Ignore the sub-degree EMA
                # fine tuning; reset only for a material/manual rotation.
                and abs(float(orientation) - float(self._last_orientation)) > 1.0
            )
        )
        if not eligible or discontinuity or context_changed or scene_difference >= self.scene_cut_threshold:
            self._reset_tracks()

        self._slot_signature = slot_keys
        self._last_processed_frame = frame_number
        self._last_orientation = float(orientation)
        self._last_frame_size = frame_size
        if frame_signature is not None:
            self._last_frame_signature = np.asarray(frame_signature, dtype=np.uint8).copy()
        if not eligible:
            return []

        prepared = []
        for index, observation in enumerate(observations):
            try:
                transform, face_scale = _landmarks_to_transform(observation["kps"], frame_size)
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                continue
            prepared.append((index, observation, transform, face_scale))

        candidates = []
        for index, observation, transform, face_scale in prepared:
            similarities = observation.get("similarities") or {}
            for slot in slots:
                key = str(slot["key"])
                if key not in similarities:
                    continue
                similarity = float(similarities[key])
                state = self._states.get(key)
                center_distance = 0.0
                scale_change = 0.0
                required = threshold
                if state is not None:
                    previous = state.output_transform
                    previous_center = previous[:2]
                    current_center = transform[:2]
                    center_distance = float(np.linalg.norm(current_center - previous_center)) / max(1.0, state.face_scale)
                    ratio = math.exp(float(transform[2] - previous[2]))
                    if (
                        center_distance > self.max_center_distance
                        or ratio < self.min_scale_ratio
                        or ratio > self.max_scale_ratio
                    ):
                        continue
                    scale_change = abs(math.log(max(1e-6, ratio)))
                    required -= self.tracked_threshold_margin
                if similarity < required:
                    continue
                association_score = similarity - 12.0 * center_distance - 8.0 * scale_change
                candidates.append((association_score, similarity, key, index, transform, face_scale))

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        used_slots = set()
        used_observations = set()
        matches = []
        for _score, similarity, key, index, transform, face_scale in candidates:
            if key in used_slots or index in used_observations:
                continue
            used_slots.add(key)
            used_observations.add(index)
            matches.append(self._confirm_track(
                key, index, transform, face_scale, similarity,
                frame_number, frame_size,
            ))

        # Prediction is deliberately conservative: only a complete detector
        # miss can coast.  If another face was detected but did not match, we
        # do not paste the previous identity over it.
        if not prepared and scene_difference < self.scene_cut_threshold:
            dt = 1.0 / self._fps
            for key, state in list(self._states.items()):
                if key in used_slots:
                    continue
                can_predict = (
                    state.hit_streak >= 2
                    and state.miss_count == 0
                    and state.last_frame == frame_number - 1
                )
                if not can_predict:
                    self._states.pop(key, None)
                    continue
                predicted = state.output_transform + state.velocity * dt
                state.output_transform = predicted
                state.last_frame = frame_number
                state.miss_count = 1
                matches.append({
                    "slot_key": key,
                    "observation_index": None,
                    "kps": _transform_to_landmarks(predicted, frame_size),
                    "similarity": float(state.similarity),
                    "predicted": True,
                })
        else:
            for key in list(self._states):
                if key not in used_slots:
                    self._states.pop(key, None)

        return matches

    def _confirm_track(
        self, key, observation_index, transform, face_scale, similarity,
        frame_number, frame_size,
    ):
        state = self._states.get(key)
        if state is None:
            euro = _OneEuroFilter()
            filtered = euro.apply(transform, 1.0 / self._fps)
            velocity = np.zeros(4, dtype=np.float64)
            hit_streak = 1
        else:
            transform = np.asarray(transform, dtype=np.float64).copy()
            transform[3] = _unwrap_near(transform[3], state.raw_transform[3])
            dt = max(1.0 / self._fps, (frame_number - state.last_confirmed_frame) / self._fps)
            previous = state.filter.filtered.copy()
            filtered = state.filter.apply(transform, dt)
            velocity = (filtered - previous) / dt
            euro = state.filter
            hit_streak = state.hit_streak + 1

        next_state = _TrackState(
            slot_key=key,
            filter=euro,
            raw_transform=np.asarray(transform, dtype=np.float64).copy(),
            output_transform=np.asarray(filtered, dtype=np.float64).copy(),
            velocity=np.asarray(velocity, dtype=np.float64).copy(),
            frame_size=tuple(frame_size),
            face_scale=float(face_scale),
            last_frame=int(frame_number),
            last_confirmed_frame=int(frame_number),
            hit_streak=int(hit_streak),
            miss_count=0,
            similarity=float(similarity),
        )
        self._states[key] = next_state
        return {
            "slot_key": key,
            "observation_index": int(observation_index),
            "kps": _transform_to_landmarks(filtered, frame_size),
            "similarity": float(similarity),
            "predicted": False,
        }


__all__ = [
    "TRACKING_VERSION",
    "TemporalFaceTracker",
    "make_slot_key",
]
