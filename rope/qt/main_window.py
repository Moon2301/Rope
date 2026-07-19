"""Rope-Pearl Qt main window.

Phase C port of rope/GUI.py. Built in stages:
- Stage 1: Top bar (Start, Output, Embed file, Clear VRAM, Build TRT, VRAM)
- Stage 2: Settings persistence (data.json)
- Stage 3: Right-pane parameters (generated from PARAMETERS list)
- Stage 4: Center pane (preview placeholder, timeline, media buttons)
- Stage 5: Left pane (folder buttons + thumbnail lists)

The previous Phase A skeleton's placeholder _tiered_frame() panels are
replaced with real content as each stage lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import os
import shutil
import time
import threading
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QStandardPaths, Qt, QThreadPool, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rope.qt.bus import bus
from rope.qt.panes.center_pane import CenterPane
from rope.qt.panes.left_pane import SourceFacesPanel, TargetMediaPanel
from rope.qt.panes.parameters_pane import ParametersPane
from rope.qt.parameters import PARAMETER_BY_NAME, ButtonParam, default_values, seed_control_dict
from rope.qt.parameters_migration import load as load_params, save as save_params
from rope.qt.settings import Settings, shorten_path
from rope.qt.widgets.button import IconButton
from rope.qt.widgets.capture_viewfinder import CaptureViewfinder
from rope.qt.window_capture import WindowCapture
from rope.qt.widgets.embedding_merge_dialog import (
    EmbeddingMergeDialog,
    FaceEntry,
)
from rope.qt.widgets.text import Text
from rope.qt.widgets.vram_indicator import VRAMIndicator
from rope.AutoSegments import (
    SequentialScanDecoder, approved_ranges, group_matches,
    load_scan_manifest, mark_segments_for_review, save_scan_manifest,
    scan_cache_key,
)
from rope.Automation import AutomationController, JobState, fingerprint_video
from rope.qt.widgets.auto_segments_dialog import AutoSegmentsDialog


SAVED_PARAMETERS_JSON = "saved_parameters.json"


def _tier_frame(tier: int) -> QFrame:
    f = QFrame()
    f.setProperty("panelTier", str(tier))
    f.setFrameShape(QFrame.NoFrame)
    return f


def _bronze_circle_icon(size: int = 256) -> QIcon:
    """A solid bronze-colored circle, used as the window/taskbar icon.
    Drawn antialiased at a large size and let Qt downscale for crisp
    small renders."""
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#CD7F32"))  # bronze
    m = max(1, size // 12)  # small inset so the circle isn't edge-clipped
    painter.drawEllipse(m, m, size - 2 * m, size - 2 * m)
    painter.end()
    return QIcon(pix)


# _BuildTRTSignals / _BuildTRTWorker — removed 2026-05-21. The four main
# pipeline models (inswapper_128, inswapper_512, retinaface, arcface) all
# route through ORT TensorrtExecutionProvider now, which builds and caches
# its own engines internally on first run.


def _cosine_similarity_pct(v1: np.ndarray, v2: np.ndarray) -> float:
    # Matches VideoManager.findCosineDistance: returns 0..100 where
    # 100 = identical, 50 = orthogonal. Direct comparand for ThresholdSlider.
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    cos_dist = 1.0 - float(np.dot(v1, v2)) / float(denom)
    return 100.0 - cos_dist * 50.0


class _AutoScanSignals(QObject):
    progress = Signal(object)
    checkpoint = Signal(object)
    finished = Signal(object)
    failed = Signal(str)


class _AutoScanWorker(QRunnable):
    def __init__(self, vm, target_embedding, threshold, sample_interval,
                 gap, padding, cache_dir, cancel_event):
        super().__init__()
        self.vm = vm
        self.target = np.asarray(target_embedding, dtype=np.float32)
        self.threshold = float(threshold)
        self.sample_interval = float(sample_interval)
        self.gap = int(gap)
        self.padding = int(padding)
        self.cache_dir = cache_dir
        self.cancel_event = cancel_event
        self.signals = _AutoScanSignals()

    @Slot()
    def run(self):
        try:
            import torch
            player = self.vm.player
            params = dict(self.vm.parameters)
            total_frames = int(self.vm.video_frame_total)
            fps = float(self.vm.fps)
            coarse_stride = max(1, int(round(fps * self.sample_interval)))
            refine_stride = 3
            refine_radius = max(1, int(round(fps * 1.0)))
            chunk_frames = max(1, int(round(fps * 300.0)))
            total_chunks = max(1, int(np.ceil(total_frames / chunk_frames)))
            config = {
                'algorithm_version': 3,
                'threshold': self.threshold,
                'sample_interval_seconds': self.sample_interval,
                'refine_radius_seconds': 1.0,
                'refine_stride_frames': refine_stride,
                'chunk_seconds': 300,
                'near_hit_margin': 8.0,
                'gap': self.gap, 'padding': self.padding,
                'detector': str(params.get('DetectTypeTextSel', 'Retinaface')),
                'detect_score': float(params.get('DetectScoreSlider', 50)),
                'input_size': int(params.get('DetectInputSizeTextSel', 640)),
            }
            key = scan_cache_key(self.vm.target_video, self.target, config)
            manifest = load_scan_manifest(self.cache_dir, key) or {
                'config': config, 'completed_chunks': [], 'chunks': {},
            }
            self.signals.checkpoint.emit({
                'cache_key': key, 'cache_dir': self.cache_dir,
                'completed_chunks': list(manifest.get('completed_chunks', [])),
            })
            completed = {int(value) for value in manifest.get('completed_chunks', [])}
            cached = len(completed) == total_chunks
            decoder = SequentialScanDecoder(
                self.vm.target_video, fps, total_frames,
            )
            scan_started = time.perf_counter()

            def emit_progress(chunk_index, stage, stage_current, stage_total):
                stage_fraction = stage_current / max(1, stage_total)
                stage_offset = 0.0 if stage == 'coarse' else 0.5
                chunk_fraction = stage_offset + stage_fraction * 0.5
                fraction = min(1.0, (chunk_index + chunk_fraction) / total_chunks)
                elapsed = time.perf_counter() - scan_started
                eta = int(elapsed * (1.0 - fraction) / fraction) if fraction > 0 else 0
                self.signals.progress.emit({
                    'chunk': chunk_index + 1, 'chunks': total_chunks,
                    'stage': stage, 'percent': fraction * 100.0,
                    'eta_seconds': eta,
                })

            def score_rgb(rgb):
                image = torch.from_numpy(
                    np.ascontiguousarray(rgb, dtype=np.uint8)
                ).to('cuda').permute(2, 0, 1)
                kpss = self.vm.models.run_detect(
                    image, config['detector'], max_num=20,
                    score=config['detect_score'] / 100.0,
                    input_size=config['input_size'],
                )
                best = 0.0
                for kps in kpss:
                    embedding, _crop = self.vm.models.run_recognize(image, kps)
                    best = max(best, _cosine_similarity_pct(embedding, self.target))
                return best

            checkpoint_calls = 100
            checkpoint_seconds = 10.0

            def save_chunk(chunk_index, chunk_state, *, mark_complete=False):
                manifest.setdefault('chunks', {})[str(chunk_index)] = chunk_state
                if mark_complete:
                    completed.add(chunk_index)
                manifest['completed_chunks'] = sorted(completed)
                save_scan_manifest(self.cache_dir, key, manifest)
                self.signals.checkpoint.emit({
                    'cache_key': key, 'cache_dir': self.cache_dir,
                    'completed_chunks': sorted(completed),
                    'chunk': int(chunk_index),
                    'stage': str(chunk_state.get('stage', 'coarse')),
                    'coarse_cursor': int(chunk_state.get('coarse_cursor', 0)),
                    'refine_cursor': int(chunk_state.get('refine_cursor', 0)),
                })

            for chunk_index in range(total_chunks):
                if chunk_index in completed:
                    emit_progress(chunk_index, 'cache', 1, 1)
                    continue
                start = chunk_index * chunk_frames
                end = min(total_frames, start + chunk_frames)
                coarse_targets = list(range(start, end, coarse_stride))
                chunk_state = dict(manifest.get('chunks', {}).get(str(chunk_index), {}))
                chunk_state.setdefault('start_frame', start)
                chunk_state.setdefault('end_frame', end - 1)
                stage = str(chunk_state.get('stage', 'coarse'))
                coarse_hits = {
                    int(frame): float(score)
                    for frame, score in chunk_state.get('coarse_hits', [])
                }
                near_hits = {
                    int(frame): float(score)
                    for frame, score in chunk_state.get('near_hits', [])
                }
                coarse_cursor = max(0, min(
                    len(coarse_targets), int(chunk_state.get('coarse_cursor', 0))
                ))

                if stage == 'coarse':
                    last_checkpoint = time.perf_counter()
                    calls_since_checkpoint = 0
                    pending = coarse_targets[coarse_cursor:]
                    for offset, (frame_no, rgb) in enumerate(
                        decoder.iter_frames(pending, self.cancel_event), 1
                    ):
                        similarity = score_rgb(rgb)
                        if similarity >= self.threshold:
                            coarse_hits[frame_no] = similarity
                        elif similarity >= self.threshold - 8.0:
                            near_hits[frame_no] = similarity
                        coarse_cursor += 1
                        calls_since_checkpoint += 1
                        chunk_state.update({
                            'stage': 'coarse', 'coarse_cursor': coarse_cursor,
                            'coarse_hits': [[frame, score] for frame, score in sorted(coarse_hits.items())],
                            'near_hits': [[frame, score] for frame, score in sorted(near_hits.items())],
                        })
                        now = time.perf_counter()
                        if (calls_since_checkpoint >= checkpoint_calls
                                or now - last_checkpoint >= checkpoint_seconds):
                            save_chunk(chunk_index, chunk_state)
                            calls_since_checkpoint = 0
                            last_checkpoint = now
                        emit_progress(
                            chunk_index, 'coarse', coarse_cursor, len(coarse_targets)
                        )
                    if self.cancel_event.is_set():
                        save_chunk(chunk_index, chunk_state)
                        self.signals.failed.emit(
                            "Đã dừng scan; tiến độ trong chunk hiện tại đã được lưu."
                        )
                        return
                    chunk_state.update({
                        'stage': 'refine', 'coarse_cursor': len(coarse_targets),
                        'coarse_hits': [[frame, score] for frame, score in sorted(coarse_hits.items())],
                        'near_hits': [[frame, score] for frame, score in sorted(near_hits.items())],
                    })
                    save_chunk(chunk_index, chunk_state)

                candidates = sorted(set(coarse_hits) | set(near_hits))
                refine_targets = sorted({
                    frame
                    for candidate in candidates
                    for frame in range(
                        max(start, candidate - refine_radius),
                        min(end, candidate + refine_radius + 1),
                        refine_stride,
                    )
                })
                refined_hits = {
                    int(frame): float(score)
                    for frame, score in chunk_state.get('hits', [])
                } or dict(coarse_hits)
                refine_cursor = max(0, min(
                    len(refine_targets), int(chunk_state.get('refine_cursor', 0))
                ))
                last_checkpoint = time.perf_counter()
                calls_since_checkpoint = 0
                pending_refine = refine_targets[refine_cursor:]
                for offset, (frame_no, rgb) in enumerate(
                    decoder.iter_frames(pending_refine, self.cancel_event), 1
                ):
                    similarity = score_rgb(rgb)
                    if similarity >= self.threshold:
                        refined_hits[frame_no] = max(
                            similarity, refined_hits.get(frame_no, 0.0)
                        )
                    refine_cursor += 1
                    calls_since_checkpoint += 1
                    chunk_state.update({
                        'stage': 'refine', 'refine_cursor': refine_cursor,
                        'hits': [[frame, score] for frame, score in sorted(refined_hits.items())],
                    })
                    now = time.perf_counter()
                    if (calls_since_checkpoint >= checkpoint_calls
                            or now - last_checkpoint >= checkpoint_seconds):
                        save_chunk(chunk_index, chunk_state)
                        calls_since_checkpoint = 0
                        last_checkpoint = now
                    emit_progress(
                        chunk_index, 'refine', refine_cursor, len(refine_targets)
                    )
                if not refine_targets:
                    emit_progress(chunk_index, 'refine', 1, 1)
                if self.cancel_event.is_set():
                    save_chunk(chunk_index, chunk_state)
                    self.signals.failed.emit(
                        "Đã dừng scan; tiến độ refine hiện tại đã được lưu."
                    )
                    return

                chunk_state.update({
                    'stage': 'complete', 'refine_cursor': len(refine_targets),
                    'hits': [[frame, score] for frame, score in sorted(refined_hits.items())],
                    'near_hits': [[frame, score] for frame, score in sorted(near_hits.items())],
                })
                save_chunk(chunk_index, chunk_state, mark_complete=True)

            matches = []
            for chunk in manifest.get('chunks', {}).values():
                matches.extend((int(frame), float(score)) for frame, score in chunk.get('hits', []))
            segments = group_matches(
                matches, stride=refine_stride, gap_frames=self.gap,
                padding_frames=self.padding, total_frames=total_frames,
            )
            segments = mark_segments_for_review(
                segments, threshold=self.threshold, fps=fps,
            )

            thumbnails = {}
            thumb_dir = Path(self.cache_dir) / f"{key}.thumbs"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            for row, segment in enumerate(segments):
                if self.cancel_event.is_set():
                    self.signals.failed.emit("Đã hủy scan.")
                    return
                points = {
                    'start': segment.start_frame,
                    'middle': (segment.start_frame + segment.end_frame) // 2,
                    'end': segment.end_frame,
                }
                thumbnails[row] = {}
                for name, frame_no in points.items():
                    thumb_path = thumb_dir / (
                        f"{segment.start_frame}_{segment.end_frame}_{name}.jpg"
                    )
                    cached_thumb = cv2.imread(str(thumb_path), cv2.IMREAD_COLOR)
                    if cached_thumb is not None:
                        thumbnails[row][name] = cv2.cvtColor(
                            cached_thumb, cv2.COLOR_BGR2RGB,
                        )
                        continue
                    frame, _pts = player.get_frame_at(frame_no)
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    frame = np.ascontiguousarray(frame)
                    h, w = frame.shape[:2]
                    scale = min(96 / max(1, w), 64 / max(1, h))
                    thumbnails[row][name] = cv2.resize(
                        frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                    cv2.imwrite(
                        str(thumb_path),
                        cv2.cvtColor(thumbnails[row][name], cv2.COLOR_RGB2BGR),
                    )
            self.signals.finished.emit({
                'segments': segments, 'thumbnails': thumbnails, 'cached': cached,
            })
        except Exception as exc:
            self.signals.failed.emit(f"Quét thất bại: {exc}")


class _AutoQCSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class _AutoQCWorker(QRunnable):
    """Lightweight output validation with bounded identity sampling."""

    def __init__(self, vm, job, output_path):
        super().__init__()
        self.vm = vm
        self.job = job
        self.output_path = str(output_path)
        self.signals = _AutoQCSignals()

    @Slot()
    def run(self):
        cap = None
        try:
            import av
            import torch
            expected = self.job.video_meta
            cap = cv2.VideoCapture(self.output_path)
            if not cap.isOpened():
                raise RuntimeError("Output không mở được bằng decoder")
            width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            expected_frames = int(expected.get('total_frames', 0))
            expected_fps = float(expected.get('fps', 0.0))
            checks = {
                'readable': True,
                'resolution': [width, height],
                'fps': fps,
                'frame_count': frames,
            }
            failures = []
            warnings = []
            if [width, height] != [int(expected.get('width', width)), int(expected.get('height', height))]:
                failures.append("resolution output không khớp input")
            if expected_fps > 0 and abs(fps - expected_fps) > 0.05:
                failures.append(f"FPS sai: {fps:.3f} thay vì {expected_fps:.3f}")
            if expected_frames > 0 and abs(frames - expected_frames) > 2:
                failures.append(f"frame count sai: {frames} thay vì {expected_frames}")
            duration = frames / fps if fps > 0 else 0.0
            expected_duration = (
                expected_frames / expected_fps
                if expected_frames > 0 and expected_fps > 0 else duration
            )
            checks['duration_seconds'] = duration
            checks['expected_duration_seconds'] = expected_duration
            if abs(duration - expected_duration) > max(0.1, 2.0 / max(1.0, expected_fps)):
                failures.append(
                    f"duration sai: {duration:.3f}s thay vì {expected_duration:.3f}s"
                )

            with av.open(self.job.video_path) as source_container:
                source_has_audio = bool(source_container.streams.audio)
            with av.open(self.output_path) as output_container:
                output_has_audio = bool(output_container.streams.audio)
            checks['source_has_audio'] = source_has_audio
            checks['output_has_audio'] = output_has_audio
            if source_has_audio and not output_has_audio:
                failures.append("output bị thiếu audio gốc")

            points = []
            for segment in self.job.segments:
                if not segment.get('approved'):
                    continue
                start = int(segment['start_frame'])
                end = int(segment['end_frame'])
                points.extend((start, (start + end) // 2, end))
            points = sorted(set(max(0, min(max(0, frames - 1), value)) for value in points))
            if len(points) > 30:
                indices = np.linspace(0, len(points) - 1, 30).astype(int)
                points = [points[index] for index in indices]

            black = 0
            decoded = 0
            similarities = []
            source_embedding = np.asarray(self.job.source_embedding, dtype=np.float32)
            config = self.job.scan_config
            for frame_no in points:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                decoded += 1
                if float(bgr.mean()) < 1.0 and float(bgr.std()) < 1.0:
                    black += 1
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to('cuda').permute(2, 0, 1)
                kpss = self.vm.models.run_detect(
                    tensor, str(config.get('detector', 'Retinaface')), max_num=20,
                    score=float(config.get('detect_score', 50)) / 100.0,
                    input_size=int(config.get('input_size', 640)),
                )
                best = None
                for kps in kpss:
                    embedding, _crop = self.vm.models.run_recognize(tensor, kps)
                    value = _cosine_similarity_pct(embedding, source_embedding)
                    best = value if best is None else max(best, value)
                if best is not None:
                    similarities.append(float(best))

            checks['sample_points'] = len(points)
            checks['decoded_samples'] = decoded
            checks['black_samples'] = black
            if points and decoded == 0:
                failures.append("không decode được frame QC")
            elif decoded > 0 and black == decoded:
                failures.append("toàn bộ frame QC đều đen")

            identity_threshold = max(45.0, float(config.get('threshold', 55)) - 10.0)
            identity_passes = sum(value >= identity_threshold for value in similarities)
            identity_rate = identity_passes / max(1, len(similarities))
            checks['identity_threshold'] = identity_threshold
            checks['identity_samples'] = len(similarities)
            checks['identity_pass_rate'] = identity_rate
            checks['identity_mean'] = float(np.mean(similarities)) if similarities else None
            if len(similarities) >= 3 and identity_rate < 0.5:
                failures.append("identity sau swap không đủ gần source")
            elif len(similarities) < 3:
                warnings.append("không đủ 3 mẫu mặt đo được để kết luận identity")

            result = {
                'passed': not failures,
                'checks': checks,
                'warnings': warnings,
                'failures': failures,
            }
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.failed.emit(f"QC thất bại: {exc}")
        finally:
            if cap is not None:
                cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rope")
        self.setWindowIcon(_bronze_circle_icon())

        self.settings = Settings.load()
        app_data = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not app_data:
            app_data = os.path.join(os.getcwd(), ".rope_app_data")
        self._automation = AutomationController(
            os.path.join(app_data, "automation_jobs")
        )
        self._active_auto_job = self._automation.store.latest_resumable()
        w, h, x, y = self.settings.dock_win_geom
        self.setGeometry(int(x), int(y), int(w), int(h))

        # Per-widget registries (mirrors self.widget / self.static_widget in Tk).
        self._buttons: dict[str, IconButton] = {}
        self._static_widgets: dict[str, QWidget] = {}

        # VideoManager-facing control snapshot. Use seed_control_dict() so
        # the legacy widget-registry aliases ('SwapFacesButton',
        # 'MaskViewButton', etc.) are present — VideoManager.swap_core does
        # `self.control['MaskViewButton']` and the shared schema name
        # ('MaskView') alone would KeyError on the first swap.
        self._control: dict[str, object] = seed_control_dict()

        # Face-pipeline state.
        # _found_faces:    [{Embedding, SourceFaceAssignments, AssignedEmbedding, Thumbnail,
        #                    HFCorrectionGap, HFCorrectionGapSamples, HFRefinePending}, ...]
        #                  Each entry is a face detected in the current frame; sent to VM via
        #                  bus.target_faces. SourceFaceAssignments holds the source-face paths
        #                  whose embeddings have been assigned to this slot; AssignedEmbedding
        #                  is the embedding the swapper uses for this slot. HFCorrectionGap
        #                  caches the High-Fidelity (s_e - output_emb) vector on first HF use
        #                  so subsequent frames skip the 2-pass; HFCorrectionGapSamples is the
        #                  running-mean sample count for that cache; HFRefinePending forces a
        #                  one-shot re-measurement on the next swap to blend a new sample in.
        #                  All three are cleared when assignment changes.
        # _source_face_embeddings: cache of {abs_path -> 512-d embedding} so a face image
        #                  is only ever recognized once per session.
        # _selected_source_paths: the source faces the user has clicked, in click order.
        #                  A click on a Found Face thumbnail assigns these to it.
        self._found_faces: list[dict] = []
        self._source_face_embeddings: dict[str, np.ndarray] = {}
        self._selected_source_paths: list[str] = []
        # Active merged embedding from the embeddings pane: when set,
        # _on_found_face_clicked uses it instead of combining the
        # source-faces selection. Cleared whenever the source-faces
        # selection changes (the two are mutually exclusive sources).
        self._active_merged_embedding: tuple[str, np.ndarray] | None = None

        # Playback state. Tracks whether the Play button + VM are currently
        # in the "playing" position so the button's icon and the click
        # action stay in sync. Recording is a separate flag because Record
        # arms playback rather than replacing it.
        self._is_playing: bool = False
        # Recording has three states: idle, "armed" (Record clicked, waiting
        # for Play to start it), and actively recording (_is_recording). Play
        # while armed sends the VM the "record" command; Play or Record while
        # actively recording stops and finalizes (VideoManager closes the
        # writer and muxes the original audio back in).
        self._is_recording: bool = False
        self._record_armed: bool = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._build_middle(root)
        self._build_bottom_bar(root)
        self._center_pane.set_auto_job_pending(self._active_auto_job is not None)

        # Pull refs to the Settings-tab "Actions" group buttons into
        # self._buttons so the existing enable/disable code (which
        # toggles the Build TRT button while a build runs) keeps
        # working against a stable handle, even though the buttons
        # themselves now live inside the params pane.
        for btn_name in (
            "ClearVramButton",
            "BenchmarkButton", "BenchmarkHeadlessButton",
        ):
            btn = self._params_pane.widgets.get(btn_name)
            if btn is not None:
                self._buttons[btn_name] = btn

        # Tooltip label used by widgets' add_info_frame() to surface their
        # info_text on hover. Rendered in the bottom bar.
        for btn in self._buttons.values():
            btn.add_info_frame(self._tooltip_label)
        self._center_pane.attach_info_label(self._tooltip_label)

        self._install_shortcuts()

        # Seed VideoManager.control by emitting the initial snapshot. Without
        # this, `vm.control` stays as an empty list and the first frame
        # request crashes (TypeError: list indices must be integers).
        bus.control_changed.emit(dict(self._control))

    def _install_shortcuts(self) -> None:
        """Replicate the Tk GUI's keyboard bindings (preview_control)."""
        bindings: list[tuple[str, callable]] = [
            ("Space", self._on_play_pressed),
            ("Q", lambda: self._center_pane.timeline.set(0)),
            ("A", lambda: self._nudge_timeline(-30)),
            ("D", lambda: self._nudge_timeline(30)),
            ("Left", lambda: self._nudge_timeline(-1)),
            ("Right", lambda: self._nudge_timeline(1)),
            ("Home", lambda: self._seek_to_frame(0)),
            ("End", lambda: self._seek_to_frame(
                self._center_pane.timeline.get_length()
            )),
            ("M", self._on_add_marker),
            ("Shift+M", self._on_del_marker),
            ("Shift+,", self._on_prev_marker),   # < key
            ("Shift+.", self._on_next_marker),   # > key
            ("Ctrl+S", lambda: self._on_params_io("save")),
            ("F3", self._center_pane.preview.toggle_hud),
        ]
        self._shortcuts: list[QShortcut] = []
        for keyseq, handler in bindings:
            sc = QShortcut(QKeySequence(keyseq), self)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)

    # ----- Model preload ----------------------------------------------------------

    def _on_preload_models(self) -> None:
        """Preload the swap-pipeline sessions on demand, sized to the
        user's current ThreadsSlider and session mode. Non-blocking —
        VideoManager.preload_models builds on background / worker threads
        so the UI stays interactive while engines build. The Preload
        Models button (center-pane toggle row) shows progress and turns
        green once the sessions are live (bus.models_preloaded)."""
        coord = getattr(self, "_coordinator", None)
        vm = getattr(coord, "vm", None) if coord is not None else None
        if vm is None or not hasattr(vm, "preload_models"):
            self._tooltip_label.setText("Preload Models: VideoManager unavailable")
            return
        vals = self._params_pane.values
        swapper_type = str(vals.get("SwapperTypeTextSel", "128"))
        detect_mode = str(vals.get("DetectTypeTextSel", "Retinaface"))
        # Require a live Models instance up front. Setting the button to
        # "loading" only makes sense if a build will actually start and
        # later emit bus.models_preloaded to settle it — a models-less VM
        # would early-return silently and leave the button stuck.
        models = self._get_models()
        if models is None:
            self._tooltip_label.setText("Preload Models: Models unavailable")
            return
        if hasattr(models, "swap_pipeline_files_present"):
            try:
                if not models.swap_pipeline_files_present(swapper_type, detect_mode):
                    self._tooltip_label.setText(
                        "Preload Models: model files for the current detector / "
                        "swapper selection not found — check the models folder "
                        "in Settings."
                    )
                    return
            except Exception:
                pass
        # Re-apply the current params so Models sees the latest session
        # mode and VideoManager sees the latest ThreadsSlider before the
        # preload sizes/builds sessions. bus.parameters_changed is a
        # same-thread (direct) connection, so vm.parameters is updated
        # synchronously here, before preload_models reads ThreadsSlider.
        try:
            self._on_params_changed(dict(self._params_pane.values))
        except Exception:
            pass
        try:
            n = int(self._params_pane.values.get("ThreadsSlider", 1))
        except (TypeError, ValueError):
            n = 1
        try:
            vm.preload_models()
        except Exception as exc:
            self._tooltip_label.setText(f"Preload Models failed to start: {exc}")
            return
        # Flip the button into its in-progress state; bus.models_preloaded
        # will settle it to loaded/idle when the background build finishes.
        self._center_pane.set_preload_button_state("loading")
        self._tooltip_label.setText(
            f"Preload Models: building pipeline sessions for {n} thread(s)… "
            "first build on a cold cache can take ~30-60s per session; the "
            "window stays responsive meanwhile."
        )

    @Slot()
    def _on_models_preloaded(self) -> None:
        """VideoManager finished a preload build (bus.models_preloaded).
        Verify what actually loaded for the current selection and settle
        the button: green "Models Loaded" if every needed session is
        live, else back to "Preload Models" with an error toast."""
        vals = self._params_pane.values
        swapper_type = str(vals.get("SwapperTypeTextSel", "128"))
        detect_mode = str(vals.get("DetectTypeTextSel", "Retinaface"))
        models = self._get_models()
        loaded = False
        if models is not None and hasattr(models, "pipeline_sessions_loaded"):
            try:
                loaded = bool(models.pipeline_sessions_loaded(swapper_type, detect_mode))
            except Exception:
                loaded = False
        if loaded:
            self._center_pane.set_preload_button_state("loaded")
            self._tooltip_label.setText("Preload Models: sessions loaded.")
        else:
            self._center_pane.set_preload_button_state("idle")
            self._tooltip_label.setText(
                "Preload Models: build finished but some sessions did not "
                "load — see console for details."
            )

    # ----- Middle splitters -------------------------------------------------------

    def _build_middle(self, root_layout: QVBoxLayout) -> None:
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(True)
        self.main_splitter.setHandleWidth(3)

        # Left pane (vertical splitter: videos + faces)
        self.left_splitter = QSplitter(Qt.Vertical)
        self.left_splitter.setChildrenCollapsible(True)
        self.left_splitter.setHandleWidth(3)
        self._videos_panel = TargetMediaPanel(self.settings.source_videos)
        self._faces_panel = SourceFacesPanel(self.settings.source_faces)
        self._videos_panel.pick_folder.connect(self._on_pick_videos_folder)
        self._videos_panel.activated.connect(self._on_target_media_clicked)
        self._faces_panel.pick_folder.connect(self._on_pick_faces_folder)
        # Selection model fires whenever the user clicks / ctrl-clicks /
        # shift-clicks any tile; we react with embedding-compute on newly
        # added paths and reblend assignments. Replaces the legacy
        # itemClicked-only flow that had to hand-roll ctrl detection.
        self._faces_panel.selection_changed.connect(self._on_source_face_selection_changed)
        self.left_splitter.addWidget(self._videos_panel)
        self.left_splitter.addWidget(self._faces_panel)
        self.left_splitter.setSizes(self.settings.splitter_left_sizes)

        # Center pane
        self._center_pane = CenterPane()
        self._wire_center_signals()

        # Right pane (params)
        self._params_pane = ParametersPane()
        self._params_pane.params_changed.connect(self._on_params_changed)
        self._params_pane.io_action.connect(self._on_params_io)
        self._params_pane.button_clicked.connect(self._on_params_button_clicked)
        self._params_pane.apply_collapsed_state(self.settings.params_collapsed)
        self._params_pane.section_toggled.connect(self._on_param_section_toggled)
        self._params_pane.models_folder_pick_requested.connect(self._on_pick_models_folder)
        self._params_pane.set_models_folder(self.settings.models_folder)
        self._params_pane.output_folder_pick_requested.connect(self._on_pick_output_folder)
        self._params_pane.set_output_folder(self.settings.saved_videos)
        self._params_pane.embeddings_file_pick_requested.connect(self._on_pick_embed_file)
        self._params_pane.set_embeddings_file(self.settings.merged_embeddings_file)
        self._params_pane.model_unload_requested.connect(self._on_model_unload)
        self._params_pane.model_backend_changed.connect(self._on_model_backend_changed)
        # Auto-refresh the Settings tab's loaded-state column whenever
        # VRAM changes (Models.__setattr__ flips vram_dirty on load /
        # unload, coordinator emits vram_updated on the next tick).
        bus.vram_updated.connect(self._on_vram_for_inventory)
        self._load_saved_parameters()

        self.main_splitter.addWidget(self.left_splitter)
        self.main_splitter.addWidget(self._center_pane)
        self.main_splitter.addWidget(self._params_pane)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes(self.settings.splitter_main_sizes)

        # Restore the center pane's vertical split (video/chrome above,
        # embeddings below) if the center pane exposes a splitter.
        center_splitter = getattr(self._center_pane, "center_splitter", None)
        if center_splitter is not None:
            center_splitter.setSizes(self.settings.splitter_center_sizes)

        # Persist pane positions immediately when the user drags a handle,
        # not only in closeEvent. closeEvent doesn't fire if the process is
        # killed (e.g. closing the console window Rope.bat spawned), and a
        # mid-session settings.save() (from a section toggle) would
        # otherwise write the *startup* splitter sizes back over a fresh
        # drag. A short debounce coalesces the stream of splitterMoved
        # emits during a drag into one write when it settles.
        self._splitter_save_timer = QTimer(self)
        self._splitter_save_timer.setSingleShot(True)
        self._splitter_save_timer.setInterval(400)
        self._splitter_save_timer.timeout.connect(self._persist_splitter_sizes)
        self.main_splitter.splitterMoved.connect(
            lambda *_: self._splitter_save_timer.start()
        )
        self.left_splitter.splitterMoved.connect(
            lambda *_: self._splitter_save_timer.start()
        )
        if center_splitter is not None:
            center_splitter.splitterMoved.connect(
                lambda *_: self._splitter_save_timer.start()
            )

        root_layout.addWidget(self.main_splitter, stretch=1)

    def _persist_splitter_sizes(self) -> None:
        """Snapshot the live splitter geometry into settings and save it.
        Driven by the debounce timer on splitterMoved so pane positions
        survive even a non-clean exit. closeEvent still saves as a
        belt-and-braces final capture."""
        self.settings.splitter_main_sizes = list(self.main_splitter.sizes())
        self.settings.splitter_left_sizes = list(self.left_splitter.sizes())
        center_splitter = getattr(self._center_pane, "center_splitter", None)
        if center_splitter is not None:
            self.settings.splitter_center_sizes = list(center_splitter.sizes())
        try:
            self.settings.save()
        except OSError as exc:
            print(f"[main_window] failed to save splitter sizes: {exc}")

    def _make_placeholder(self, text: str, *, tier: int) -> QFrame:
        f = _tier_frame(tier)
        lay = QVBoxLayout(f); lay.setContentsMargins(8, 8, 8, 8)
        lbl = QLabel(text); lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        return f

    # Preserve attribute names used by existing smoke tests
    @property
    def videos_panel(self) -> QFrame: return self._videos_panel
    @property
    def faces_panel(self) -> QFrame: return self._faces_panel
    @property
    def center_pane(self) -> QFrame: return self._center_pane
    @property
    def params_pane(self) -> QFrame: return self._params_pane
    @property
    def preview(self) -> QWidget:
        return self._center_pane.preview if hasattr(self._center_pane, "preview") else QLabel("")
    def _wire_center_signals(self) -> None:
        cp = self._center_pane
        cp.frame_requested.connect(bus.get_requested_video_frame)
        bus.frame_ready.connect(cp.preview.set_frame)
        # Drive the timeline playhead + entry field from VM-delivered
        # frame numbers. timeline.set() is the position-only API: it
        # updates the visual without re-emitting frame_requested, so no
        # feedback loop with the scrub path.
        bus.playback_frame_changed.connect(cp.timeline.set)
        cp.scrub_started.connect(self._on_scrub_started)
        cp.play_pressed.connect(self._on_play_pressed)
        cp.record_pressed.connect(self._on_record_pressed)
        # VM raises stop_play when playback ends naturally (EOF) or stop
        # marker hits. Reset the Play button so the user can click it
        # again without it being in the wrong state.
        bus.stop_play.connect(self._on_vm_stopped)
        cp.toggle_audio.connect(lambda: self._toggle_control("AudioButton"))
        cp.toggle_mask_view.connect(lambda: self._toggle_control("MaskViewButton"))
        cp.toggle_swap_faces.connect(self._on_toggle_swap_faces)
        cp.find_faces_pressed.connect(self._on_find_faces)
        cp.clear_faces_pressed.connect(self._on_clear_faces)
        cp.found_faces_gallery.face_clicked.connect(self._on_found_face_clicked)
        cp.embeddings_pane.selection_activated.connect(self._on_embedding_selection_activated)
        cp.embeddings_pane.save_current_requested.connect(self._on_embedding_save_current)
        cp.embeddings_pane.set_source_path(self.settings.merged_embeddings_file)
        # Mouse wheel over the preview cycles embeddings.
        cp.preview.wheel_scrolled.connect(cp.embeddings_pane.cycle)
        cp.add_marker.connect(self._on_add_marker)
        cp.del_marker.connect(self._on_del_marker)
        cp.prev_marker.connect(self._on_prev_marker)
        cp.next_marker.connect(self._on_next_marker)
        cp.nudge_left.connect(lambda: self._nudge_timeline(-30))
        cp.nudge_right.connect(lambda: self._nudge_timeline(30))
        cp.jump_to_start.connect(lambda: self._seek_to_frame(0))
        cp.preview_mode_changed.connect(self._on_preview_mode_changed)
        cp.save_image.connect(self._on_save_image)
        cp.preload_pressed.connect(self._on_preload_models)
        cp.auto_segments_pressed.connect(self._open_auto_segments)

        # Reflect VM-emitted slider length back into the timeline.
        bus.slider_length_changed.connect(cp.timeline.set_length)
        # Preload build finished → settle the Preload Models button.
        bus.models_preloaded.connect(self._on_models_preloaded)
        bus.auto_render_progress.connect(self._on_auto_render_progress)
        bus.auto_render_stage.connect(self._on_auto_render_stage)
        bus.auto_render_finished.connect(self._on_auto_render_finished)
        bus.auto_render_failed.connect(self._on_auto_render_failed)

    def _open_auto_segments(self) -> None:
        existing_dialog = getattr(self, '_auto_segments_dialog', None)
        if existing_dialog is not None:
            existing_dialog.show()
            existing_dialog.raise_()
            existing_dialog.activateWindow()
            return
        vm = getattr(getattr(self, '_coordinator', None), 'vm', None)
        if vm is None:
            QMessageBox.warning(self, "Auto Job", "Backend video chưa sẵn sàng.")
            return
        current_video = vm.target_video if getattr(vm, 'is_video_loaded', False) else None
        pending = self._automation.store.latest_resumable(current_video)
        if pending is None and not self._found_faces:
            pending = self._automation.store.latest_resumable()
        if pending is not None:
            self._active_auto_job = pending
            self._restore_auto_job_context(pending)
        if not self._found_faces:
            QMessageBox.information(
                self, "Auto Job",
                "Hãy mở frame có nhân vật, bấm Find Faces và gán source face trước.",
            )
            return
        gallery = self._center_pane.found_faces_gallery
        selected_getter = getattr(gallery, 'selected_index', None)
        selected = selected_getter() if callable(selected_getter) else -1
        if not isinstance(selected, int) or not (0 <= selected < len(self._found_faces)):
            selected = 0
        self._auto_target_index = selected
        dialog = AutoSegmentsDialog(self)
        dialog.scan_requested.connect(self._start_auto_scan)
        dialog.render_requested.connect(self._start_auto_render)
        dialog.seek_requested.connect(self._seek_to_frame)
        dialog.preview_segment_requested.connect(self._preview_auto_segment)
        dialog.pause_job_requested.connect(self._pause_auto_job)
        dialog.resume_job_requested.connect(self._resume_auto_job)
        dialog.retry_job_requested.connect(self._retry_auto_job)
        dialog.open_output_requested.connect(self._open_auto_output)
        dialog.segments_changed.connect(self._on_auto_segments_edited)
        self._center_pane.timeline.position_changed.connect(dialog.set_playhead)
        self._auto_segments_dialog = dialog
        if pending is not None:
            self._show_job_in_dialog(pending)
        dialog.show()

    def _on_auto_segments_edited(self, segments) -> None:
        job = self._active_auto_job
        if job is None or JobState(job.state) != JobState.AWAITING_REVIEW:
            return
        data = [item.to_dict() for item in segments]
        try:
            self._active_auto_job = self._automation.transition(
                job, JobState.AWAITING_REVIEW, segments=data,
            )
            self._center_pane.timeline.set_segments(data)
        except ValueError:
            pass

    def _restore_auto_job_context(self, job) -> None:
        vm = getattr(getattr(self, '_coordinator', None), 'vm', None)
        if vm is None:
            return
        if job.output_dir and os.path.isdir(job.output_dir):
            # A job owns its output location even if the app-wide setting was
            # changed between restarts; render checkpoint paths are derived
            # from this directory.
            vm.saved_video_path = job.output_dir
        if (not getattr(vm, 'is_video_loaded', False)
                or os.path.abspath(str(vm.target_video)) != os.path.abspath(job.video_path)):
            bus.load_target_video.emit(job.video_path)
            self._current_media_path = job.video_path
        thumb_path = self._automation.store.job_dir(job.job_id) / "target.jpg"
        thumb = cv2.imread(str(thumb_path), cv2.IMREAD_COLOR)
        if thumb is not None:
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        else:
            thumb = np.zeros((64, 64, 3), dtype=np.uint8)
        slot = {
            'Embedding': np.asarray(job.target_embedding, dtype=np.float32),
            'SourceFaceAssignments': list(job.source_labels),
            'AssignedEmbedding': np.asarray(job.source_embedding, dtype=np.float32),
            'Thumbnail': thumb,
            'HFCorrectionGap': None,
            'HFCorrectionGapSamples': 0,
            'HFRefinePending': False,
        }
        self._found_faces = [slot]
        self._auto_target_index = 0
        models = self._get_models()
        if models is not None:
            try:
                models._mean_emb = np.asarray(job.source_embedding, dtype=np.float32)
            except Exception:
                pass
        self._refresh_found_faces_gallery()
        bus.target_faces.emit(self._found_faces)

    def _show_job_in_dialog(self, job) -> None:
        dialog = getattr(self, '_auto_segments_dialog', None)
        if dialog is None:
            return
        if job.segments:
            try:
                from rope.AutoSegments import AutoSegment
                segments = [AutoSegment(**item) for item in job.segments]
                dialog.set_segments(
                    segments, self._load_job_thumbnails(job), cached=True,
                    fps=float(job.video_meta.get('fps', 1.0)),
                )
                self._center_pane.timeline.set_segments(job.segments)
            except (TypeError, ValueError):
                pass
        dialog.set_job_state(
            job.state,
            job.error or f"Auto Job {job.job_id[:8]} — {job.state}",
            output=job.final_output,
        )
        if job.state in {
            JobState.SCANNING.value, JobState.RENDERING.value,
            JobState.MERGING.value, JobState.MUXING.value,
            JobState.QC.value, JobState.PAUSED.value,
        }:
            dialog.resume_button.setEnabled(True)

    def _load_job_thumbnails(self, job) -> dict:
        result = {}
        directory = self._automation.store.job_dir(job.job_id) / "thumbnails"
        for row, _segment in enumerate(job.segments):
            for label in ('start', 'middle', 'end'):
                path = directory / f"{row}_{label}.jpg"
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is not None:
                    result.setdefault(row, {})[label] = cv2.cvtColor(
                        image, cv2.COLOR_BGR2RGB,
                    )
        return result

    def _save_job_assets(self, job, slot, thumbnails: dict | None = None) -> None:
        directory = self._automation.store.job_dir(job.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        thumb = slot.get('Thumbnail')
        if isinstance(thumb, np.ndarray) and thumb.ndim == 3:
            cv2.imwrite(
                str(directory / "target.jpg"),
                cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR),
            )
        if thumbnails:
            thumbs_dir = directory / "thumbnails"
            thumbs_dir.mkdir(parents=True, exist_ok=True)
            for row, images in thumbnails.items():
                for label, image in images.items():
                    if isinstance(image, np.ndarray) and image.ndim == 3:
                        cv2.imwrite(
                            str(thumbs_dir / f"{row}_{label}.jpg"),
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                        )

    def _auto_preflight_errors(self, vm, slot) -> list[str]:
        errors = []
        output = self.settings.saved_videos
        if not output or not os.path.isdir(output) or not os.access(output, os.W_OK):
            errors.append("Output Folder chưa tồn tại hoặc không ghi được")
        if vm is None or not getattr(vm, 'is_video_loaded', False) \
                or not os.path.isfile(str(getattr(vm, 'target_video', ''))):
            errors.append("Video input chưa được tải hoặc không còn tồn tại")
        if not slot.get('SourceFaceAssignments') or slot.get('AssignedEmbedding') is None:
            errors.append("Target face chưa được gán source face/embedding")
        if vm is None or float(getattr(vm, 'fps', 0.0) or 0.0) <= 0 \
                or int(getattr(vm, 'video_frame_total', 0) or 0) <= 0:
            errors.append("FPS hoặc tổng frame của video không hợp lệ")
        try:
            from rope.VideoManager import _find_ffmpeg
            if _find_ffmpeg() is None:
                errors.append("Không tìm thấy FFmpeg để encode/ghép/mux audio")
        except Exception:
            errors.append("Không kiểm tra được FFmpeg")
        models = self._get_models()
        values = self._params_pane.values
        if models is None:
            errors.append("Models backend chưa sẵn sàng")
        else:
            try:
                if not models.swap_pipeline_files_present(
                    str(values.get('SwapperTypeTextSel', '128')),
                    str(values.get('DetectTypeTextSel', 'Retinaface')),
                ):
                    errors.append("Thiếu detector, recognizer hoặc swapper đang chọn")
            except Exception as exc:
                errors.append(f"Không kiểm tra được model pipeline: {exc}")
            optional = []
            if values.get('OccluderSwitch'):
                optional.append(('occluder.onnx', 'Occluder'))
            if values.get('DFLXSegSwitch'):
                optional.append(('dfl_xseg.onnx', 'DFL XSeg'))
            if values.get('FaceParserSwitch'):
                optional.append(('faceparser_resnet34.onnx', 'Face Parser'))
            if values.get('RestorerSwitch'):
                restorer_files = {
                    'GFPGAN': 'GFPGANv1.4.onnx', 'CF': 'codeformer_fp16.onnx',
                    'GPEN256': 'GPEN-BFR-256.onnx', 'GPEN512': 'GPEN-BFR-512.onnx',
                }
                name = str(values.get('RestorerTypeTextSel', 'GFPGAN'))
                optional.append((restorer_files.get(name, ''), f'Restorer {name}'))
            for filename, label in optional:
                if filename and not os.path.isfile(os.path.join(models.models_folder, filename)):
                    errors.append(f"Đang bật {label} nhưng thiếu {filename}")
        if output and os.path.isdir(output) and vm is not None \
                and os.path.isfile(str(getattr(vm, 'target_video', ''))):
            try:
                source_size = os.path.getsize(vm.target_video)
                required = int(source_size * 2.5 + 256 * 1024 * 1024)
                if shutil.disk_usage(output).free < required:
                    errors.append(
                        f"Không đủ dung lượng trống (cần khoảng {required / 1024**3:.1f} GB)"
                    )
            except OSError:
                errors.append("Không kiểm tra được dung lượng Output Folder")
        return errors

    def _auto_job_inputs_locked(self) -> bool:
        job = self._active_auto_job
        if job is None:
            return False
        try:
            state = JobState(job.state)
            if state == JobState.FAILED and job.retry_from == JobState.PREFLIGHT.value:
                return False
            if state == JobState.PAUSED and job.paused_from == JobState.PREFLIGHT.value:
                return False
            return state in {
                JobState.SCANNING, JobState.AWAITING_REVIEW, JobState.RENDERING,
                JobState.MERGING, JobState.MUXING, JobState.QC,
                JobState.PAUSED, JobState.FAILED,
            }
        except ValueError:
            return False

    def _warn_auto_job_locked(self) -> None:
        QMessageBox.information(
            self, "Auto Job đang khóa input",
            "Video, target face và source face được khóa để checkpoint/resume luôn "
            "dùng đúng identity. Hãy hoàn tất hoặc retry Auto Job hiện tại.",
        )

    def _start_auto_scan(self, sample_interval: float, gap: int, padding: int) -> None:
        vm = getattr(getattr(self, '_coordinator', None), 'vm', None)
        dialog = getattr(self, '_auto_segments_dialog', None)
        if vm is None or dialog is None or not getattr(vm, 'is_video_loaded', False):
            if dialog is not None:
                dialog.set_job_state("NEW", "Hãy tải video trước khi bắt đầu Auto Job.")
            return
        if (not self._found_faces or not hasattr(self, '_auto_target_index')
                or not (0 <= self._auto_target_index < len(self._found_faces))):
            dialog.set_job_state("NEW", "Hãy Find Faces, chọn target và gán source trước.")
            return
        existing = self._automation.store.latest_resumable()
        if existing is not None:
            self._active_auto_job = existing
            self._restore_auto_job_context(existing)
            self._show_job_in_dialog(existing)
            dialog.status.setText(
                "Chỉ một GPU job được phép hoạt động. Hãy Resume/Retry job hiện tại."
            )
            return
        if self._is_playing or self._is_recording:
            bus.play_video.emit('stop_from_gui')
            self._is_playing = self._is_recording = False
        slot = self._found_faces[self._auto_target_index]
        errors = self._auto_preflight_errors(vm, slot)
        if errors:
            message = "Preflight thất bại:\n• " + "\n• ".join(errors)
            dialog.set_job_state("NEW", message)
            return
        threshold = float(self._params_pane.values.get('ThresholdSlider', 55))
        scan_config = {
            'sample_interval_seconds': float(sample_interval),
            'gap_frames': int(gap), 'padding_frames': int(padding),
            'threshold': threshold, 'chunk_seconds': 300,
            'refine_radius_seconds': 1.0, 'refine_stride_frames': 3,
            'near_hit_margin': 8.0,
            'detector': str(self._params_pane.values.get('DetectTypeTextSel', 'Retinaface')),
            'detect_score': float(self._params_pane.values.get('DetectScoreSlider', 50)),
            'input_size': int(self._params_pane.values.get('DetectInputSizeTextSel', 640)),
            'algorithm_version': 3,
        }
        request = {
            'video_path': vm.target_video,
            'output_dir': self.settings.saved_videos or '',
            'video_fingerprint': fingerprint_video(vm.target_video),
            'video_meta': {
                'fps': float(vm.fps), 'total_frames': int(vm.video_frame_total),
                'width': int(vm.player.width), 'height': int(vm.player.height),
            },
            'target_embedding': np.asarray(slot['Embedding']).reshape(-1).tolist(),
            'source_embedding': np.asarray(slot.get('AssignedEmbedding')).reshape(-1).tolist(),
            'source_labels': list(slot.get('SourceFaceAssignments') or []),
            'scan_config': scan_config,
            'render_params': dict(self._params_pane.values),
        }
        job = self._automation.start_job(request)
        self._active_auto_job = job
        self._save_job_assets(job, slot)
        self._center_pane.set_auto_job_pending(True)
        self._active_auto_job = self._automation.begin_scan(job)
        self._params_pane.setEnabled(False)
        dialog.set_job_state("SCANNING", "Preflight thành công. Đang scan…")
        self._launch_auto_scan(self._active_auto_job)

    def _launch_auto_scan(self, job) -> None:
        vm = getattr(getattr(self, '_coordinator', None), 'vm', None)
        dialog = getattr(self, '_auto_segments_dialog', None)
        if vm is None or dialog is None:
            return
        config = job.scan_config
        self._auto_scan_cancel = threading.Event()
        cache_dir = os.path.join(QStandardPaths.writableLocation(QStandardPaths.CacheLocation),
                                 'auto_segments')
        worker = _AutoScanWorker(
            vm, np.asarray(job.target_embedding, dtype=np.float32),
            float(config.get('threshold', 55)),
            float(config.get('sample_interval_seconds', 0.5)),
            int(config.get('gap_frames', 20)),
            int(config.get('padding_frames', 10)),
            cache_dir, self._auto_scan_cancel,
        )
        worker.signals.progress.connect(dialog.set_scan_progress)
        worker.signals.checkpoint.connect(self._on_auto_scan_checkpoint)
        worker.signals.finished.connect(self._on_auto_scan_finished)
        worker.signals.failed.connect(self._on_auto_scan_failed)
        self._auto_scan_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_auto_scan_checkpoint(self, checkpoint: dict) -> None:
        job = self._active_auto_job
        if job is None or JobState(job.state) != JobState.SCANNING:
            return
        try:
            self._active_auto_job = self._automation.transition(
                job, JobState.SCANNING, scan_checkpoint=dict(checkpoint),
            )
        except ValueError:
            pass

    def _cancel_auto_scan(self) -> None:
        self._pause_auto_job()

    def _on_auto_scan_failed(self, message: str) -> None:
        job = self._active_auto_job
        event = getattr(self, '_auto_scan_cancel', None)
        cancelled = bool(event is not None and event.is_set())
        if job is not None:
            try:
                if cancelled:
                    job = self._automation.pause_job(job)
                else:
                    job = self._automation.fail_job(
                        job, message, retry_from=JobState.SCANNING,
                    )
                self._active_auto_job = job
            except ValueError:
                pass
        dialog = getattr(self, '_auto_segments_dialog', None)
        if dialog is not None:
            dialog.set_job_state("PAUSED" if cancelled else "FAILED", message)
        self._params_pane.setEnabled(True)
        self._center_pane.set_auto_job_pending(True)

    def _on_auto_scan_finished(self, result) -> None:
        segments = result['segments']
        job = self._active_auto_job
        if job is not None:
            try:
                job = self._automation.set_scan_results(
                    job, [item.to_dict() for item in segments],
                )
                self._active_auto_job = job
                self._save_job_assets(
                    job, self._found_faces[self._auto_target_index],
                    result.get('thumbnails'),
                )
            except ValueError as exc:
                self._on_auto_scan_failed(str(exc))
                return
        dialog = getattr(self, '_auto_segments_dialog', None)
        if dialog is not None:
            dialog.set_segments(
                segments, result.get('thumbnails'),
                cached=result.get('cached', False), fps=float(getattr(
                    getattr(self._coordinator, 'vm', None), 'fps', 1.0,
                )),
            )
        self._params_pane.setEnabled(True)
        self._center_pane.timeline.set_segments([item.to_dict() for item in segments])

    def _start_auto_render(self, segments) -> None:
        ranges = approved_ranges(segments)
        if not ranges:
            return
        job = self._active_auto_job
        if job is None:
            QMessageBox.warning(self, "Auto Job", "Không tìm thấy job đang chờ duyệt.")
            return
        if not job.output_dir or not os.path.isdir(job.output_dir):
            QMessageBox.warning(self, "Auto Job", "Output Folder của job không còn tồn tại.")
            return
        try:
            if fingerprint_video(job.video_path) != job.video_fingerprint:
                raise ValueError("Video input đã thay đổi sau khi scan; cần quét lại job.")
        except (OSError, ValueError) as exc:
            self._active_auto_job = self._automation.fail_job(
                job, str(exc), retry_from=JobState.PREFLIGHT,
            )
            dialog = getattr(self, '_auto_segments_dialog', None)
            if dialog is not None:
                dialog.set_job_state("FAILED", str(exc))
            return
        try:
            job = self._automation.confirm_segments(
                job, [item.to_dict() for item in segments],
                dict(self._params_pane.values),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Auto Job", str(exc))
            return
        self._active_auto_job = job
        self._params_pane.setEnabled(False)
        bus.parameters_changed.emit(dict(job.render_params))
        self._center_pane.timeline.set_segments([item.to_dict() for item in segments])
        bus.auto_render_segments.emit(self._auto_render_request(job, ranges))
        if (self._active_auto_job is None
                or JobState(self._active_auto_job.state) != JobState.RENDERING):
            return
        self._is_playing = True
        self._is_recording = True
        self._center_pane.set_play_state(True)
        self._center_pane.set_record_state(True)
        dialog = getattr(self, '_auto_segments_dialog', None)
        if dialog is not None:
            dialog.set_job_state("RENDERING", "Đang render part đầu tiên…")

    def _auto_render_request(self, job, ranges=None) -> dict:
        source_label = os.path.splitext(os.path.basename(job.source_labels[0]))[0] \
            if job.source_labels else "source"
        return {
            'ranges': list(ranges if ranges is not None else approved_ranges(job.segments)),
            'job_id': job.job_id, 'source_label': source_label,
            'part_seconds': 60, 'render_params': dict(job.render_params),
        }

    def _cancel_auto_render(self) -> None:
        self._pause_auto_job()

    def _resume_auto_render(self) -> None:
        self._resume_auto_job()

    def _pause_auto_job(self) -> None:
        job = self._active_auto_job
        if job is None:
            return
        state = JobState(job.state)
        if state == JobState.SCANNING:
            event = getattr(self, '_auto_scan_cancel', None)
            if event is not None:
                event.set()
        elif state == JobState.RENDERING:
            vm = getattr(getattr(self, '_coordinator', None), 'vm', None)
            if vm is not None:
                vm.cancel_auto_segment_render()
        else:
            try:
                self._active_auto_job = self._automation.pause_job(job)
            except ValueError:
                return
        dialog = getattr(self, '_auto_segments_dialog', None)
        if dialog is not None:
            dialog.status.setText("Đang dừng an toàn và lưu checkpoint…")
            dialog.pause_button.setEnabled(False)

    def _resume_auto_job(self) -> None:
        job = self._active_auto_job or self._automation.store.latest_resumable()
        if job is None:
            return
        try:
            if JobState(job.state) in (JobState.PAUSED, JobState.FAILED):
                job = self._automation.resume_job(job.job_id)
        except ValueError as exc:
            dialog = getattr(self, '_auto_segments_dialog', None)
            if dialog is not None:
                dialog.set_error(str(exc))
            return
        self._active_auto_job = job
        self._restore_auto_job_context(job)
        state = JobState(job.state)
        if state in (JobState.SCANNING, JobState.AWAITING_REVIEW):
            try:
                changed = fingerprint_video(job.video_path) != job.video_fingerprint
            except OSError:
                changed = True
            if changed:
                message = "Video input đã thay đổi; bấm Retry để preflight và scan lại."
                self._active_auto_job = self._automation.fail_job(
                    job, message, retry_from=JobState.PREFLIGHT,
                )
                self._show_job_in_dialog(self._active_auto_job)
                return
        if state == JobState.PREFLIGHT:
            errors = self._auto_preflight_errors(
                getattr(self._coordinator, 'vm', None), self._found_faces[0],
            )
            if errors:
                self._active_auto_job = self._automation.fail_job(
                    job, "Preflight thất bại:\n• " + "\n• ".join(errors),
                    retry_from=JobState.PREFLIGHT,
                )
                self._show_job_in_dialog(self._active_auto_job)
                return
            job = self._automation.transition(
                job, JobState.PREFLIGHT,
                output_dir=self.settings.saved_videos or job.output_dir,
                render_params=dict(self._params_pane.values),
                video_fingerprint=fingerprint_video(job.video_path),
                video_meta={
                    'fps': float(self._coordinator.vm.fps),
                    'total_frames': int(self._coordinator.vm.video_frame_total),
                    'width': int(self._coordinator.vm.player.width),
                    'height': int(self._coordinator.vm.player.height),
                },
                segments=[], user_confirmed=False, scan_checkpoint={},
            )
            job = self._automation.begin_scan(job)
            self._active_auto_job = job
            self._params_pane.setEnabled(False)
            self._launch_auto_scan(job)
        elif state == JobState.SCANNING:
            self._params_pane.setEnabled(False)
            dialog = getattr(self, '_auto_segments_dialog', None)
            if dialog is not None:
                dialog.set_job_state("SCANNING", "Đang tiếp tục scan từ checkpoint…")
            self._launch_auto_scan(job)
        elif state == JobState.AWAITING_REVIEW:
            self._show_job_in_dialog(job)
        elif state in (JobState.RENDERING, JobState.MERGING, JobState.MUXING):
            vm = getattr(getattr(self, '_coordinator', None), 'vm', None)
            if vm is not None:
                self._params_pane.setEnabled(state != JobState.RENDERING)
                if state == JobState.RENDERING:
                    self._is_playing = self._is_recording = True
                    self._center_pane.set_play_state(True)
                    self._center_pane.set_record_state(True)
                    dialog = getattr(self, '_auto_segments_dialog', None)
                    if dialog is not None:
                        dialog.set_job_state("RENDERING", "Đang tiếp tục render từ checkpoint…")
                manifest_path = vm._auto_manifest_for_video(job.job_id)
                if os.path.isfile(manifest_path):
                    vm.resume_auto_segment_render(job.job_id)
                else:
                    bus.auto_render_segments.emit(self._auto_render_request(job))
        elif state == JobState.QC and job.final_output:
            self._start_auto_qc(job.final_output)

    def _retry_auto_job(self) -> None:
        self._resume_auto_job()

    def _open_auto_output(self) -> None:
        job = self._active_auto_job
        path = job.final_output if job and job.final_output else None
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))

    def _preview_auto_segment(self, start: int, end: int) -> None:
        self._auto_previewing = True
        bus.set_stop.emit(int(end))
        self._seek_to_frame(int(start))
        bus.play_video.emit('play')
        self._is_playing = True
        self._center_pane.set_play_state(True)

    def _on_auto_render_progress(self, data) -> None:
        payload = dict(data)
        dialog = getattr(self, '_auto_segments_dialog', None)
        if dialog is not None:
            dialog.set_render_progress(payload)
        job = self._active_auto_job
        part = int(payload.get('part', 0))
        if (job is not None and JobState(job.state) == JobState.RENDERING
                and part != getattr(self, '_auto_job_checkpoint_part', -1)):
            self._auto_job_checkpoint_part = part
            try:
                self._active_auto_job = self._automation.transition(
                    job, JobState.RENDERING, render_checkpoint=payload,
                )
            except ValueError:
                pass

    def _on_auto_render_stage(self, stage: str) -> None:
        job = self._active_auto_job
        if job is None:
            return
        try:
            target = JobState(stage)
            if target == JobState.PAUSED:
                job = self._automation.pause_job(job)
            else:
                job = self._automation.transition(job, target)
            self._active_auto_job = job
        except (ValueError, KeyError):
            return
        dialog = getattr(self, '_auto_segments_dialog', None)
        if dialog is not None:
            dialog.set_job_state(job.state, f"Auto Job: {job.state}")
        if JobState(job.state) != JobState.RENDERING:
            self._params_pane.setEnabled(True)

    def _on_auto_render_finished(self, final_output: str) -> None:
        self._is_playing = False
        self._is_recording = False
        self._record_armed = False
        self._center_pane.set_play_state(False)
        self._center_pane.set_record_state(False)
        self._params_pane.setEnabled(True)
        job = self._active_auto_job
        if job is None:
            return
        try:
            job = self._automation.transition(
                job, JobState.QC, final_output=str(final_output),
            )
            self._active_auto_job = job
        except ValueError as exc:
            self._on_auto_render_failed(str(exc))
            return
        self._start_auto_qc(final_output)

    def _on_auto_render_failed(self, message: str) -> None:
        self._is_playing = False
        self._is_recording = False
        self._record_armed = False
        self._center_pane.set_play_state(False)
        self._center_pane.set_record_state(False)
        self._params_pane.setEnabled(True)
        job = self._active_auto_job
        if job is not None:
            try:
                self._active_auto_job = self._automation.fail_job(
                    job, message, retry_from=JobState.RENDERING,
                )
            except ValueError:
                pass
        dialog = getattr(self, '_auto_segments_dialog', None)
        if dialog is not None:
            dialog.set_job_state("FAILED", message)

    def _start_auto_qc(self, final_output: str) -> None:
        job = self._active_auto_job
        vm = getattr(getattr(self, '_coordinator', None), 'vm', None)
        if job is None or vm is None:
            return
        dialog = getattr(self, '_auto_segments_dialog', None)
        if dialog is not None:
            dialog.set_job_state("QC", "Đang kiểm tra duration, audio, frame và identity…")
        worker = _AutoQCWorker(vm, job, final_output)
        worker.signals.finished.connect(self._on_auto_qc_finished)
        worker.signals.failed.connect(self._on_auto_qc_failed)
        self._auto_qc_worker = worker
        QThreadPool.globalInstance().start(worker)

    def _on_auto_qc_finished(self, qc: dict) -> None:
        job = self._active_auto_job
        if job is None:
            return
        if qc.get('passed'):
            try:
                job = self._automation.complete_job(
                    job, job.final_output or '', qc,
                )
                self._active_auto_job = job
            except ValueError as exc:
                self._on_auto_qc_failed(str(exc))
                return
            vm = getattr(getattr(self, '_coordinator', None), 'vm', None)
            if vm is not None:
                vm.complete_auto_render_cleanup(job.job_id)
            self._center_pane.set_auto_job_pending(False)
            dialog = getattr(self, '_auto_segments_dialog', None)
            if dialog is not None:
                warning_note = ""
                if qc.get('warnings'):
                    warning_note = " — " + "; ".join(qc['warnings'])
                dialog.set_job_state(
                    "COMPLETED", f"Hoàn tất QC{warning_note}",
                    output=job.final_output,
                )
            QMessageBox.information(
                self, "Auto Job hoàn tất",
                f"Video đã được render và QC:\n{job.final_output}",
            )
        else:
            message = "QC không đạt: " + "; ".join(qc.get('failures') or ['unknown'])
            try:
                job.qc = dict(qc)
                self._active_auto_job = self._automation.fail_job(
                    job, message, retry_from=JobState.QC,
                )
            except ValueError:
                pass
            dialog = getattr(self, '_auto_segments_dialog', None)
            if dialog is not None:
                dialog.set_job_state("FAILED", message, output=job.final_output)

    def _on_auto_qc_failed(self, message: str) -> None:
        job = self._active_auto_job
        if job is not None:
            try:
                self._active_auto_job = self._automation.fail_job(
                    job, message, retry_from=JobState.QC,
                )
            except ValueError:
                pass
        dialog = getattr(self, '_auto_segments_dialog', None)
        if dialog is not None:
            dialog.set_job_state("FAILED", message, output=(
                job.final_output if job else None
            ))

    def _on_play_pressed(self) -> None:
        job = self._active_auto_job
        if job is not None:
            state = JobState(job.state)
            if state == JobState.RENDERING:
                self._pause_auto_job()
                return
            if state in {JobState.SCANNING, JobState.MERGING, JobState.MUXING, JobState.QC}:
                self._tooltip_label.setText(f"Auto Job đang {state.value}; transport tạm khóa.")
                return
        # The preview canvas is wired to this handler too, so clicking the
        # output window during a recording routes here and stops it.
        if self._is_recording:
            # Actively recording → Play stops and finalizes: VideoManager
            # closes the writer and muxes the original audio back in.
            bus.play_video.emit("stop_from_gui")
            self._is_recording = False
            self._is_playing = False
            self._record_armed = False
        elif self._record_armed:
            # Record armed → Play starts the recording. The VM "record"
            # command sets up the writer and drives the decoder from the
            # current frame; the preview stays live while recording.
            bus.play_video.emit("record")
            self._is_recording = True
            self._is_playing = True
            self._record_armed = False
        elif self._is_playing:
            bus.play_video.emit("stop_from_gui")
            self._is_playing = False
        else:
            bus.play_video.emit("play")
            self._is_playing = True
        self._center_pane.set_play_state(self._is_playing)
        self._center_pane.set_record_state(self._is_recording or self._record_armed)

    def _on_record_pressed(self) -> None:
        job = self._active_auto_job
        if job is not None:
            state = JobState(job.state)
            if state == JobState.RENDERING:
                self._pause_auto_job()
                return
            if state in {JobState.SCANNING, JobState.MERGING, JobState.MUXING, JobState.QC}:
                self._tooltip_label.setText(f"Auto Job đang {state.value}; Record tạm khóa.")
                return
        if self._is_recording:
            # Actively recording → stop and finalize (same as Play here).
            bus.play_video.emit("stop_from_gui")
            self._is_recording = False
            self._is_playing = False
            self._record_armed = False
        elif self._record_armed:
            # Armed but not yet started → cancel the arm.
            self._record_armed = False
        else:
            # Idle (or previewing) → arm recording for the next Play. Guard
            # on the output folder so we fail loudly here instead of crashing
            # in VideoManager's os.path.join at record time.
            if not self.settings.saved_videos:
                self._tooltip_label.setText(
                    "Record: pick an output folder first (Settings → Output Folder)."
                )
                return
            if self._is_playing:
                # Stop the running preview so recording starts cleanly.
                bus.play_video.emit("stop_from_gui")
                self._is_playing = False
            self._record_armed = True
        self._center_pane.set_play_state(self._is_playing)
        self._center_pane.set_record_state(self._is_recording or self._record_armed)

    def _on_scrub_started(self) -> None:
        if self._is_playing or self._is_recording:
            bus.play_video.emit("stop_from_gui")
        if self._is_playing or self._is_recording or self._record_armed:
            self._is_playing = False
            self._is_recording = False
            self._record_armed = False
            self._center_pane.set_play_state(False)
            self._center_pane.set_record_state(False)

    def _on_vm_stopped(self) -> None:
        # VideoManager finished playback / hit a stop marker / finished a
        # recording (audio already muxed) / errored — reset the transport
        # buttons so a fresh click starts again.
        self._is_playing = False
        self._is_recording = False
        self._record_armed = False
        self._center_pane.set_play_state(False)
        self._center_pane.set_record_state(False)
        if getattr(self, '_auto_previewing', False):
            self._auto_previewing = False
            bus.set_stop.emit(-1)

    def _seek_to_frame(self, frame: int) -> None:
        """Move the timeline scrubber AND request the frame from VM.
        timeline.set() is position-only; without the bus emit the
        preview never updates. Every transport control that jumps to a
        specific frame should funnel through here."""
        tl = self._center_pane.timeline
        n = max(0, min(tl.get_length(), int(frame)))
        tl.set(n)
        bus.get_requested_video_frame.emit(n)

    def _nudge_timeline(self, delta: int) -> None:
        self._seek_to_frame(self._center_pane.timeline.get() + delta)

    # Marker handling: keep a self._markers list with {'frame': int, 'parameters': dict}
    # entries. add adds at current frame, del removes, prev/next jumps.
    def _on_add_marker(self) -> None:
        frame = self._center_pane.timeline.get()
        if not hasattr(self, "_markers"):
            self._markers = []
        if any(m["frame"] == frame for m in self._markers):
            return
        self._markers.append({"frame": frame, "parameters": dict(self._params_pane.values)})
        self._markers.sort(key=lambda m: m["frame"])
        self._center_pane.timeline.set_markers([m["frame"] for m in self._markers])
        bus.markers_changed.emit(self._markers)

    def _on_del_marker(self) -> None:
        frame = self._center_pane.timeline.get()
        if not hasattr(self, "_markers"):
            self._markers = []
            return
        self._markers = [m for m in self._markers if m["frame"] != frame]
        self._center_pane.timeline.set_markers([m["frame"] for m in self._markers])
        bus.markers_changed.emit(self._markers)

    def _on_prev_marker(self) -> None:
        frames = sorted(m["frame"] for m in getattr(self, "_markers", []))
        current = self._center_pane.timeline.get()
        prev = [f for f in frames if f < current]
        if prev:
            self._center_pane.timeline.set(prev[-1])
            bus.get_requested_video_frame.emit(prev[-1])

    def _on_next_marker(self) -> None:
        frames = sorted(m["frame"] for m in getattr(self, "_markers", []))
        current = self._center_pane.timeline.get()
        nxt = [f for f in frames if f > current]
        if nxt:
            self._center_pane.timeline.set(nxt[0])
            bus.get_requested_video_frame.emit(nxt[0])

    def _on_preview_mode_changed(self, mode: str) -> None:
        self._tooltip_label.setText(f"Mode: {mode}")
        # Mirror into the control dict so VideoManager sees mode changes.
        self._control["PreviewModeTextSel"] = mode
        bus.control_changed.emit(dict(self._control))
        if mode == "Capture":
            self.open_capture_viewfinder()
        else:
            # Leaving Capture: hide the viewfinder and pause the worker
            # (threads stay alive so re-entering Capture doesn't reload
            # models). See _exit_capture_mode for the flash-prevention.
            self._exit_capture_mode()

    def _toggle_control(self, name: str) -> None:
        """Flip a boolean control entry and re-emit the snapshot. Used by
        the center-pane Audio / MaskView buttons (and any future toggle
        wired via cp.toggle_*)."""
        prev = bool(self._control.get(name, False))
        self._control[name] = not prev
        bus.control_changed.emit(dict(self._control))
        self._refresh_current_frame()

    def _on_toggle_swap_faces(self) -> None:
        # Turning SwapFaces on without a complete target->source assignment
        # used to fail silently: VideoManager correctly skipped every
        # target-only slot, but the user only saw an unchanged preview and
        # no inswapper load in the console. Block that invalid state here
        # and explain the required workflow. Turning an already-on toggle
        # off must always remain available.
        turning_on = not bool(self._control.get("SwapFacesButton", False))
        if turning_on:
            assigned = [
                slot for slot in self._found_faces
                if slot.get("SourceFaceAssignments")
                and slot.get("AssignedEmbedding") is not None
            ]
            if not assigned:
                if not self._found_faces:
                    detail = (
                        "Chưa có Found Face.\n\n"
                        "1. Tới frame thấy rõ nhân vật.\n"
                        "2. Bấm Find Faces.\n"
                        "3. Chọn Source Face và gán vào Found Face."
                    )
                else:
                    detail = (
                        "Found Face chưa được gán Source Face hoặc Embedding.\n\n"
                        "1. Chọn ảnh trong Source Faces (hoặc một embedding).\n"
                        "2. Click thumbnail nhân vật trong Found Faces để gán.\n"
                        "3. Bật SwapFaces lại."
                    )
                self._tooltip_label.setText("SwapFaces: chưa gán Source Face")
                QMessageBox.warning(self, "Không thể bật SwapFaces", detail)
                return
        btn = self._center_pane.buttons.get("SwapFacesButton")
        if btn is not None:
            btn.toggle_button()
        # vm.control['SwapFacesButton'] is the canonical VM-side key.
        # _toggle_control itself triggers a frame refresh.
        self._toggle_control("SwapFacesButton")
        state = "on" if self._control.get("SwapFacesButton") else "off"
        self._tooltip_label.setText(f"SwapFaces {state}")

    # ----- Face pipeline ----------------------------------------------------------

    def _get_models(self):
        coord = getattr(self, "_coordinator", None) or getattr(self, "_coordinator_ref", None)
        return getattr(coord, "models", None) if coord is not None else None

    def _detect_and_recognize(self, rgb: np.ndarray, *, max_num: int = 20):
        """Run detect + recognize on an RGB HxWx3 uint8 numpy frame.

        Returns a list of (embedding (512,), thumbnail (HxWx3 uint8 RGB)).
        The thumbnail is the 112x112 aligned face crop the recognizer
        already produced. Empty list if nothing detected or models busy.
        """
        models = self._get_models()
        if models is None:
            return []
        try:
            import torch
            from torchvision.transforms import v2 as _v2  # noqa: F401  (ensures torchvision present)
        except ImportError:
            return []

        detect_mode = str(self._params_pane.values.get("DetectTypeTextSel", "Retinaface"))
        detect_score = float(self._params_pane.values.get("DetectScoreSlider", 50)) / 100.0
        detect_input_size = int(self._params_pane.values.get("DetectInputSizeTextSel", 640))

        # Mirror VideoManager.swap_video: HWC uint8 -> CHW CUDA tensor.
        img_chw = torch.from_numpy(rgb.astype("uint8")).to("cuda").permute(2, 0, 1)
        try:
            kpss = models.run_detect(img_chw, detect_mode, max_num=max_num, score=detect_score, input_size=detect_input_size)
        except Exception as exc:
            self._tooltip_label.setText(f"Find Faces: detect failed ({exc})")
            return []

        out = []
        for kps in kpss:
            try:
                emb, cropped = models.run_recognize(img_chw, kps)
            except Exception as exc:
                print(f"[main_window] recognize failed: {exc}")
                continue
            # cropped is HWC torch tensor at 112x112; emb is numpy (512,).
            try:
                thumb = cropped.cpu().numpy().astype(np.uint8)
            except Exception:
                thumb = np.zeros((112, 112, 3), dtype=np.uint8)
            out.append((np.asarray(emb, dtype=np.float32), thumb))
        return out

    def _on_find_faces(self) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        # Run detect+recognize on the currently displayed preview frame
        # and append any *new* faces — existing entries are preserved
        # (along with their SourceFaceAssignments). A detection is
        # considered "already known" when its cosine similarity to any
        # existing found face is >= ThresholdSlider, the same test the
        # swap pipeline uses for per-frame face matching.
        rgb = self._center_pane.preview.last_frame_rgb()
        if rgb is None:
            self._tooltip_label.setText("Find Faces: no frame in preview yet — scrub to a frame first")
            return
        if self._get_models() is None:
            self._tooltip_label.setText("Find Faces: Models unavailable")
            return

        detections = self._detect_and_recognize(rgb)
        if not detections:
            self._tooltip_label.setText("Find Faces: no faces detected")
            return

        threshold = float(self._params_pane.values.get("ThresholdSlider", 55))
        existing_count = len(self._found_faces)
        # Build the comparison set lazily so newly-added detections in
        # this same call also block their own duplicates (two near-
        # identical faces in one frame won't both get added).
        known = [np.asarray(f["Embedding"], dtype=np.float32) for f in self._found_faces]
        first_new_index = None
        for emb, thumb in detections:
            if any(_cosine_similarity_pct(emb, ex) >= threshold for ex in known):
                continue
            if first_new_index is None:
                first_new_index = len(self._found_faces)
            self._found_faces.append({
                "Embedding": emb,
                "SourceFaceAssignments": [],
                "AssignedEmbedding": None,
                "Thumbnail": thumb,
                "HFCorrectionGap": None,
                "HFCorrectionGapSamples": 0,
                "HFRefinePending": False,
            })
            known.append(emb)

        added = len(self._found_faces) - existing_count
        if added == 0:
            self._tooltip_label.setText(
                f"Find Faces: {len(detections)} detected, all already in Found Faces. "
                f"Total: {existing_count}."
            )
            return

        # Preserve the prior selection where possible; if nothing was
        # selected before, focus the first newly added face so the
        # next source-face / embedding pick targets it.
        gallery = self._center_pane.found_faces_gallery
        prior_selection = gallery.selected_index()
        self._refresh_found_faces_gallery()
        new_selection = prior_selection if prior_selection >= 0 else first_new_index
        if new_selection is not None and 0 <= new_selection < len(self._found_faces):
            gallery.set_selected(new_selection)
            # Sync the source/embedding highlights to whichever slot ends
            # up selected — without this, a previously-selected slot's
            # tint would persist when Find Faces auto-selects a newly
            # added (empty-assignment) slot.
            self._apply_assigned_highlight(
                self._found_faces[new_selection].get("SourceFaceAssignments") or []
            )
        bus.target_faces.emit(self._found_faces)
        self._refresh_current_frame()
        if added == len(detections):
            self._tooltip_label.setText(
                f"Find Faces: {added} new added. Total: {len(self._found_faces)}."
            )
        else:
            skipped = len(detections) - added
            self._tooltip_label.setText(
                f"Find Faces: {added} new added ({skipped} already present). "
                f"Total: {len(self._found_faces)}."
            )

    def _on_clear_faces(self) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        self._found_faces = []
        self._refresh_found_faces_gallery()
        self._center_pane.found_faces_gallery.set_selected(-1)
        # No selected Found Face → drop the assigned-source tint from
        # both panels so leftover gold doesn't survive a clear.
        self._apply_assigned_highlight([])
        bus.target_faces.emit(self._found_faces)
        coord = getattr(self, "_coordinator", None) or getattr(self, "_coordinator_ref", None)
        vm = getattr(coord, "vm", None) if coord is not None else None
        if vm is not None:
            try:
                vm.found_faces = []
            except Exception:
                pass
        self._refresh_current_frame()
        self._tooltip_label.setText("Clear Faces: gallery cleared")

    def _refresh_found_faces_gallery(self) -> None:
        gallery = self._center_pane.found_faces_gallery
        gallery.clear()
        for ff in self._found_faces:
            idx = gallery.add(ff["Thumbnail"], assigned=bool(ff["SourceFaceAssignments"]))
            del idx  # tile index matches list index by construction

    def _on_found_face_clicked(self, index: int) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        """Select a target slot and apply any already-selected source.

        Both natural interaction orders must work:
        ``Found Face -> Source Face`` is handled by the source selection
        callback, while ``Source Face -> Found Face`` is handled here.
        Previously the latter only changed highlights, leaving the slot
        unassigned even though both tiles visibly looked selected.
        """
        if not (0 <= index < len(self._found_faces)):
            return
        self._center_pane.found_faces_gallery.set_selected(index)
        slot = self._found_faces[index]

        # If a source or saved embedding was selected first, assign it now.
        if self._active_merged_embedding is not None:
            name, embedding = self._active_merged_embedding
            self._apply_to_selected_slot(
                embedding, [f"emb:{name}"], label=f"Embedding '{name}'",
            )
            return
        usable = [
            path for path in self._selected_source_paths
            if path in self._source_face_embeddings
        ]
        if usable:
            embeddings = [self._source_face_embeddings[path] for path in usable]
            mode = str(self._params_pane.values.get("MergeTextSel", "Mean"))
            try:
                from rope.EmbeddingMerge import combine
                merged = combine(embeddings, mode)
            except Exception:
                merged = np.mean(np.stack(embeddings, axis=0), axis=0)
            self._apply_to_selected_slot(
                merged, usable,
                label=f"{len(usable)} source face(s) ({mode})",
            )
            return

        assigned = slot.get("SourceFaceAssignments") or []
        # Push the assignment back to the source/embedding panels so
        # they tint the contributing items in gold. Tags use the same
        # split convention as _apply_to_selected_slot: "emb:<name>"
        # for saved embeddings, anything else is a source-face path.
        self._apply_assigned_highlight(assigned)
        if assigned:
            tag = assigned[0] if isinstance(assigned, list) and assigned else ""
            self._tooltip_label.setText(
                f"Found Face #{index} selected (currently: {tag}). "
                "Pick source faces or an embedding to change it."
            )
        else:
            self._tooltip_label.setText(
                f"Found Face #{index} selected. Pick source faces or an embedding to assign."
            )

    def _apply_assigned_highlight(self, assignments) -> None:
        """Tint the source-face thumbnails and embedding tiles whose
        identifiers appear in `assignments` (the SourceFaceAssignments
        list of a Found Face slot). Pass an empty iterable to clear
        both panels."""
        emb_names: set[str] = set()
        face_paths: set[str] = set()
        for tag in assignments or ():
            if isinstance(tag, str) and tag.startswith("emb:"):
                emb_names.add(tag[4:])
            elif isinstance(tag, str):
                face_paths.add(tag)
        self._faces_panel.set_assigned_paths(face_paths)
        self._center_pane.embeddings_pane.set_assigned_names(emb_names)

    def _apply_to_selected_slot(
        self, embedding: np.ndarray, assignment_tag: list, *, label: str
    ) -> None:
        """Write `embedding` into the currently-selected Found Face slot
        and refresh. No-op (with a toast) if nothing is selected."""
        gallery = self._center_pane.found_faces_gallery
        index = gallery.selected_index()
        if not (0 <= index < len(self._found_faces)):
            self._tooltip_label.setText(
                "Select a Found Face first, then pick source(s) or an embedding."
            )
            return
        slot = self._found_faces[index]
        slot["SourceFaceAssignments"] = list(assignment_tag)
        slot["AssignedEmbedding"] = np.asarray(embedding, dtype=np.float32)
        slot["HFCorrectionGap"] = None
        slot["HFCorrectionGapSamples"] = 0
        slot["HFRefinePending"] = False
        gallery.set_assigned(index, True)
        # The selected slot just changed sources — re-tint the panels
        # so the gold highlight reflects the new assignment.
        self._apply_assigned_highlight(slot["SourceFaceAssignments"])
        bus.target_faces.emit(self._found_faces)
        self._refresh_current_frame()
        self._tooltip_label.setText(f"{label} -> Found Face #{index}")

    # ----- Embeddings pane callbacks ---------------------------------------------

    def _on_embedding_selection_activated(self, entries) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        """Apply the picked embedding(s) to the currently-selected
        Found Face. With ExtendedSelection on the embeddings list,
        ctrl/shift+click can build a multi-selection; merging across
        entries uses the same MergeTextSel mode the source-faces
        path uses, so the two assignment routes stay consistent."""
        if not entries:
            return
        vecs = [np.asarray(v, dtype=np.float32) for _, v in entries]
        names = [str(n) for n, _ in entries]
        if len(vecs) == 1:
            merged = vecs[0]
            active_label = names[0]
            log_label = f"Embedding '{names[0]}'"
        else:
            mode = str(self._params_pane.values.get("MergeTextSel", "Mean"))
            try:
                from rope.EmbeddingMerge import combine
                merged = combine(vecs, mode)
            except Exception:
                merged = np.mean(np.stack(vecs, axis=0), axis=0)
            preview = " + ".join(names[:3])
            if len(names) > 3:
                preview += f" + {len(names) - 3} more"
            active_label = preview
            log_label = f"{len(entries)} embeddings ({mode}): {preview}"
        self._active_merged_embedding = (active_label, np.asarray(merged, dtype=np.float32))
        # Drop source-face selection: this embedding (or merge thereof)
        # is now the exclusive source.
        if self._selected_source_paths:
            self._faces_panel.list.clearSelection()
            self._selected_source_paths = []
        self._apply_to_selected_slot(
            merged, [f"emb:{n}" for n in names], label=log_label,
        )

    def _on_embedding_save_current(self) -> None:
        """Compute the active source embedding from the source-faces
        selection (or echo the active merged embedding) and save it
        under a user-chosen name."""
        embedding = None
        default_name = "merged"
        if self._active_merged_embedding is not None:
            # Save a copy of the active merged entry under a new name.
            old_name, embedding = self._active_merged_embedding
            default_name = f"{old_name} copy"
        elif self._selected_source_paths:
            embs = [self._source_face_embeddings[p]
                    for p in self._selected_source_paths
                    if p in self._source_face_embeddings]
            if not embs:
                self._tooltip_label.setText("Save Embedding: no usable source embeddings to combine")
                return
            mode = str(self._params_pane.values.get("MergeTextSel", "Mean"))
            try:
                from rope.EmbeddingMerge import combine
                embedding = combine(embs, mode)
            except Exception:
                embedding = np.mean(np.stack(embs, axis=0), axis=0)
            default_name = "merged"
        else:
            self._tooltip_label.setText(
                "Save Embedding: select source faces (or an existing embedding) first"
            )
            return

        if not self.settings.merged_embeddings_file:
            self._tooltip_label.setText(
                "Save Embedding: pick an embeddings file (Settings tab > Embeddings File) first"
            )
            return

        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "Save Embedding", "Name:", text=default_name,
        )
        if not ok or not name:
            return
        self._center_pane.embeddings_pane.add_entry(name, embedding)
        self._tooltip_label.setText(f"Saved embedding '{name}'")

    # ----- Secondary-window launchers ---------------------------------------------

    def open_capture_viewfinder(self) -> "CaptureViewfinder":
        if getattr(self, "_capture_vf", None) is None:
            self._capture_vf = CaptureViewfinder(border_px=4, always_on_top=True)
            self._capture_vf.closed.connect(self._on_capture_viewfinder_closed)
            # Start the WindowCapture worker — capture thread grabs the
            # transparent inner area via dxcam/mss, swap thread runs
            # vm.swap_video and pushes frames through vm._publish_frame
            # so the preview sees them via bus.frame_ready.
            coord = getattr(self, "_coordinator", None)
            vm = getattr(coord, "vm", None) if coord is not None else None
            if vm is not None:
                worker = WindowCapture(
                    self._capture_vf, vm, lambda: dict(self._params_pane.values)
                )
                worker.start()
                self._capture_worker = worker
            else:
                self._tooltip_label.setText(
                    "Capture: VideoManager unavailable — viewfinder shown without swap"
                )
        else:
            # Re-entering Capture: resume the existing worker instead of
            # rebuilding its threads, so per-thread model sessions stay
            # warm (no reload).
            worker = getattr(self, "_capture_worker", None)
            if worker is not None:
                worker.resume()
        self._capture_vf.show()
        self._capture_vf.raise_()
        return self._capture_vf

    def _exit_capture_mode(self) -> None:
        """Leave Capture mode: pause the worker (keep threads/sessions
        alive), hide the viewfinder box, and repaint the current video
        frame so the last captured frame doesn't linger or flash in the
        preview after the switch."""
        worker = getattr(self, "_capture_worker", None)
        if worker is not None:
            worker.pause()
        vf = getattr(self, "_capture_vf", None)
        if vf is not None:
            vf.hide()
        # Replace the stale capture frame with the real video frame.
        self._refresh_current_frame()

    def _on_capture_viewfinder_closed(self) -> None:
        # Explicit close (X): fully tear the worker down. This is the only
        # path that stops the threads; mode switches only pause/resume.
        worker = getattr(self, "_capture_worker", None)
        if worker is not None:
            worker.stop()
            self._capture_worker = None
        self._capture_vf = None

    def open_embedding_merge(self, faces: list | None = None) -> "EmbeddingMergeDialog":
        # If no faces are supplied (Phase E is pre-Models wiring), seed
        # with the source-faces panel's current list as name-only entries so
        # the dialog at least shows something. The merge call will be deferred
        # until embeddings are loaded.
        if faces is None:
            entries = []
            for i in range(self._faces_panel.list.count()):
                item = self._faces_panel.list.item(i)
                path = item.data(Qt.UserRole) if item is not None else None
                if path:
                    entries.append(FaceEntry(name=item.text(), thumbnail_path=path))
            faces = entries
        dlg = EmbeddingMergeDialog(faces, parent=self)
        if dlg.exec() == dlg.Accepted:
            payload = dlg.result_payload()
            if payload is not None:
                self._tooltip_label.setText(
                    f"Merge {payload.mode}: '{payload.name}' "
                    f"from {len(payload.selected_indices)} face(s)"
                )
        return dlg

    # ----- Left pane callbacks ----------------------------------------------------

    def _on_pick_videos_folder(self) -> None:
        start = self.settings.source_videos or ""
        path = QFileDialog.getExistingDirectory(self, "Select Target Videos Folder", start)
        if not path:
            return
        self.settings.source_videos = path
        self.settings.save()
        self._videos_panel.set_folder(path)
        self._tooltip_label.setText(f"Loaded {self._videos_panel.list.count()} media files")

    def _on_pick_faces_folder(self) -> None:
        start = self.settings.source_faces or ""
        path = QFileDialog.getExistingDirectory(self, "Select Source Faces Folder", start)
        if not path:
            return
        self.settings.source_faces = path
        self.settings.save()
        self._faces_panel.set_folder(path)
        self._tooltip_label.setText(f"Loaded {self._faces_panel.list.count()} face images")

    def _on_target_media_clicked(self, path: str) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        if suffix in {"jpg", "jpeg", "png", "bmp", "webp"}:
            bus.load_target_image.emit(path)
        else:
            bus.load_target_video.emit(path)
        # Track for Save Image's default filename derivation.
        self._current_media_path = path
        # Reset stale state: new media means the old playhead position
        # and the old face detections no longer make sense. Found faces
        # also propagate the empty list to VM via _on_clear_faces.
        self._center_pane.timeline.set(0)
        self._on_clear_faces()
        self._tooltip_label.setText(f"Loading: {path}")

    def _on_source_face_selection_changed(self, paths: list) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        """Driven by the source-faces list's native selection model.
        Computes embeddings for any newly-selected paths, then applies
        the combined embedding to the currently-selected Found Face.
        """
        # Selecting source faces takes over as the active source; clear
        # any merged-embedding pane selection so the two sources stay
        # mutually exclusive.
        if paths and self._active_merged_embedding is not None:
            self._active_merged_embedding = None
        # Compute embeddings for newly-selected, uncached paths.
        skipped: list[str] = []
        cache_grew = False
        for path in paths:
            if path in self._source_face_embeddings:
                continue
            emb = self._compute_source_embedding(path)
            if emb is None:
                skipped.append(os.path.basename(path))
                continue
            self._source_face_embeddings[path] = emb
            cache_grew = True
        # The mean of all known source-face embeddings drives the
        # Distinctiveness slider's extrapolation. Recompute + push to
        # Models whenever the cache changes, and clear the latent
        # cache so existing entries don't keep returning stale
        # extrapolations against the old mean.
        if cache_grew:
            self._update_session_mean_embedding()

        # Drop paths whose embedding compute failed (e.g. no face detected
        # in the image). Keeps the active selection in sync with what we
        # can actually use.
        usable = [p for p in paths if p in self._source_face_embeddings]
        self._selected_source_paths = usable

        if skipped:
            self._tooltip_label.setText(
                f"Source: {len(usable)} selected; "
                f"skipped (no face detected): {', '.join(skipped[:3])}"
                + ("…" if len(skipped) > 3 else "")
            )

        # Apply to the currently-selected Found Face only. Each found
        # face holds its own embedding, so the user clicks a target slot
        # first and then picks sources/embeddings to fill it.
        if not usable:
            return
        embs = [self._source_face_embeddings[p] for p in usable]
        mode = str(self._params_pane.values.get("MergeTextSel", "Mean"))
        try:
            from rope.EmbeddingMerge import combine
            merged = combine(embs, mode)
        except Exception:
            merged = np.mean(np.stack(embs, axis=0), axis=0)
        self._apply_to_selected_slot(
            merged, list(usable),
            label=f"{len(usable)} source face(s) ({mode})",
        )

    def _compute_source_embedding(self, path: str):
        """Run detect+recognize on a source-face image file. Returns 512-d
        np.ndarray on success, None on failure."""
        models = self._get_models()
        if models is None:
            return None
        try:
            bgr = cv2.imread(path)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
        results = self._detect_and_recognize(rgb, max_num=1)
        if not results:
            return None
        emb, _thumb = results[0]
        return emb

    def _update_session_mean_embedding(self) -> None:
        """Recompute the mean of all known source-face embeddings and
        push it to Models for the Distinctiveness slider. Also clears
        the swapper-latent cache because cached extrapolations were
        computed against the old mean."""
        if not self._source_face_embeddings:
            return
        embs = list(self._source_face_embeddings.values())
        mean = np.mean(np.stack(embs, axis=0), axis=0)
        models = self._get_models()
        if models is None:
            return
        try:
            models.set_session_mean_embedding(mean)
        except Exception as exc:
            print(f"[main_window] set_session_mean_embedding failed: {exc}")
            return
        # Clear the latent cache so the next swap recomputes against
        # the fresh mean.
        coord = getattr(self, "_coordinator", None) or getattr(self, "_coordinator_ref", None)
        vm = getattr(coord, "vm", None) if coord is not None else None
        if vm is not None and hasattr(vm, "clear_latent_cache"):
            try:
                vm.clear_latent_cache()
            except Exception:
                pass

# ----- Bottom bar (status + tooltip text) -------------------------------------

    def _build_bottom_bar(self, root_layout: QVBoxLayout) -> None:
        bar = _tier_frame(1)
        bar.setFixedHeight(28)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Bottom-left: PayPal donation hotlink, replacing the old hover-help
        # / status text. Rich-text QLabel; setOpenExternalLinks opens the
        # URL in the default browser when the link is clicked.
        donate = QLabel(
            '<a href="https://www.paypal.com/donate/'
            '?hosted_button_id=Y5SB9LSXFGRF2" '
            'style="color:#E5B854;text-decoration:none;">'
            '♥ Support Rope — Donate via PayPal</a>'
        )
        donate.setOpenExternalLinks(True)
        donate.setTextInteractionFlags(Qt.TextBrowserInteraction)
        donate.setCursor(Qt.PointingHandCursor)
        donate.setStyleSheet("font-size: 9pt;")
        donate.setToolTip("Opens the PayPal donation page in your browser")
        layout.addWidget(donate)

        layout.addStretch(1)

        # The hover-info / status label is retained as a hidden sink: the
        # many widgets that call add_info_frame(self._tooltip_label) plus
        # the ~40 setText(...) status updates keep working, but nothing is
        # rendered — the help text is no longer shown (the donation link
        # takes its place in the bottom bar).
        self._tooltip_label = Text(text="", tier=1)
        self._tooltip_label.setParent(bar)
        self._tooltip_label.setVisible(False)

        # VRAM indicator anchored bottom-right. Updates flow in via
        # bus.vram_updated, which the coordinator polls on its idle tick.
        self._static_widgets["vram_indicator"] = VRAMIndicator()
        self._static_widgets["vram_indicator"].setFixedSize(200, 20)
        bus.vram_updated.connect(self._static_widgets["vram_indicator"].set)
        layout.addWidget(self._static_widgets["vram_indicator"])
        root_layout.addWidget(bar)

    # ----- Settings persistence ---------------------------------------------------

    def closeEvent(self, event) -> None:
        geom = self.geometry()
        self.settings.dock_win_geom = [geom.width(), geom.height(), geom.x(), geom.y()]
        self.settings.splitter_main_sizes = list(self.main_splitter.sizes())
        self.settings.splitter_left_sizes = list(self.left_splitter.sizes())
        center_splitter = getattr(self._center_pane, "center_splitter", None)
        if center_splitter is not None:
            self.settings.splitter_center_sizes = list(center_splitter.sizes())
        try:
            self.settings.save()
        except OSError as exc:
            print(f"[main_window] failed to save settings: {exc}")
        super().closeEvent(event)

    # ----- Top-bar callbacks ------------------------------------------------------

    def _on_save_image(self) -> None:
        """Write the current preview frame to the saved-videos folder.

        Mirrors the Tk GUI's save_image: derive a filename from the
        currently-loaded media (or 'rope' if none) plus a unix timestamp,
        cv2.imwrite as PNG. Surfaces failure via the bottom-bar tooltip
        rather than a modal so it doesn't interrupt a recording flow.
        """
        rgb = self._center_pane.preview.last_frame_rgb()
        if rgb is None:
            self._tooltip_label.setText("Save Image: no frame in preview yet")
            return
        folder = self.settings.saved_videos
        if not folder or not os.path.isdir(folder):
            self._tooltip_label.setText("Save Image: pick a valid Output Folder first")
            return
        media = getattr(self, "_current_media_path", None)
        base = os.path.splitext(os.path.basename(media))[0] if media else "rope"
        stamp = str(int(time.time()))
        filename = os.path.join(folder, f"{base}_{stamp}.png")
        try:
            cv2.imwrite(filename, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except cv2.error as exc:
            self._tooltip_label.setText(f"Save Image failed: {exc}")
            return
        self._tooltip_label.setText(f"Saved {filename}")

    def _on_pick_output_folder(self, *_args) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        start = self.settings.saved_videos or ""
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", start)
        if not path:
            return
        self.settings.saved_videos = path
        self.settings.save()
        self._params_pane.set_output_folder(path)
        bus.saved_video_path.emit(path)

    def _on_pick_models_folder(self) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        start = self.settings.models_folder or ""
        path = QFileDialog.getExistingDirectory(self, "Select Models Folder", start)
        if not path:
            return
        self.settings.models_folder = path
        self.settings.save()
        self._params_pane.set_models_folder(path)
        # Push into the running Models instance so the next inference
        # call loads from the new folder. set_models_folder unloads any
        # currently-loaded models so the swap of paths takes effect
        # without a restart.
        models = self._get_models()
        if models is not None and hasattr(models, "set_models_folder"):
            try:
                models.set_models_folder(path)
            except Exception as exc:
                self._tooltip_label.setText(
                    f"Models folder set to {path}, but live-apply failed: {exc}"
                )
                return
        self._tooltip_label.setText(f"Models folder set to {path}")

    def _on_pick_embed_file(self, *_args) -> None:
        current = self.settings.merged_embeddings_file or "merged_embeddings.txt"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Embedding File",
            current,
            "Embedding files (*.txt);;All files (*)",
        )
        if not path:
            return
        self.settings.merged_embeddings_file = path
        self.settings.save()
        self._params_pane.set_embeddings_file(path)
        # set_source_path triggers EmbeddingsPane.reload_from_disk so the
        # tile grid populates from the selected file. Subsequent "Save
        # Current..." actions write back to this same path.
        self._center_pane.embeddings_pane.set_source_path(path)

    def _on_clear_vram(self, *_args) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        coord = getattr(self, "_coordinator", None) or getattr(self, "_coordinator_ref", None)
        models = getattr(coord, "models", None) if coord is not None else None
        if models is None or not hasattr(models, "delete_models"):
            self._tooltip_label.setText("Clear VRAM: Models instance unavailable")
            return
        try:
            models.delete_models()
        except Exception as exc:
            self._tooltip_label.setText(f"Clear VRAM failed: {exc}")
            return
        self._tooltip_label.setText("Cleared all ONNX models from VRAM")

    @Slot(str)
    def _on_model_unload(self, attr_name: str) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        coord = getattr(self, "_coordinator", None)
        models = getattr(coord, "models", None) if coord is not None else None
        if models is None or not hasattr(models, "unload_model"):
            self._tooltip_label.setText("Unload: Models instance unavailable")
            return
        try:
            models.unload_model(attr_name)
        except Exception as exc:
            self._tooltip_label.setText(f"Unload {attr_name} failed: {exc}")
            return
        self._tooltip_label.setText(f"Unloaded {attr_name}")

    @Slot(str, str)
    def _on_model_backend_changed(self, attr_name: str, backend: str) -> None:
        if self._auto_job_inputs_locked():
            self._warn_auto_job_locked()
            return
        coord = getattr(self, "_coordinator", None)
        models = getattr(coord, "models", None) if coord is not None else None
        if models is None or not hasattr(models, "set_backend_preference"):
            self._tooltip_label.setText("Backend: Models instance unavailable")
            return
        try:
            # unload=True drops the model if loaded so the next inference
            # call reloads it via the new backend. The vram_updated emit
            # from the assignment refreshes the inventory table.
            models.set_backend_preference(attr_name, backend, unload=True)
        except Exception as exc:
            self._tooltip_label.setText(f"Backend {attr_name}: {exc}")
            return
        # Persist the choice. Settings.model_backends is empty by default
        # and grows entry-by-entry as the user toggles. Removing back to
        # 'auto' isn't exposed in the UI today, but the Models method
        # accepts None for that case.
        self.settings.model_backends[attr_name] = backend
        self.settings.save()
        self._tooltip_label.setText(f"{attr_name}: using {backend.upper()} at next load")
        # Force a refresh in case the model wasn't loaded (no
        # vram_updated will fire) — the button label still needs to flip.
        self._params_pane.refresh_models_inventory(
            self.settings.models_folder or "./models"
        )

    @Slot(float, float)
    def _on_vram_for_inventory(self, _used_gb: float, _total_gb: float) -> None:
        # vram_updated fires whenever any tracked Models attr is
        # (re)assigned, so it's the right signal to drive the Loaded
        # column refresh. We don't need the actual GB numbers here.
        if hasattr(self, "_params_pane"):
            self._params_pane.refresh_models_inventory(
                self.settings.models_folder or "./models"
            )

    def _on_benchmark(self, *_args) -> None:
        # Toggle: if a play/record is already active, treat the click as
        # a stop (which will print the bench report if bench mode is
        # active). Otherwise start a benchmark run, which behaves like
        # Play but with audio/wall-clock sync disabled in VideoManager.
        if self._is_playing or self._is_recording:
            bus.play_video.emit("stop_from_gui")
            self._is_playing = False
            self._is_recording = False
            self._tooltip_label.setText("Benchmark: stopping (report prints to console)")
        else:
            bus.play_video.emit("benchmark")
            self._is_playing = True
            self._tooltip_label.setText("Benchmark: running (no sync — stop manually or wait for EOF)")
        self._record_armed = False
        self._center_pane.set_play_state(self._is_playing)
        self._center_pane.set_record_state(self._is_recording or self._record_armed)

    def _on_benchmark_headless(self, *_args) -> None:
        # Same toggle semantics as Benchmark, but routes through the
        # benchmark_headless command which tells VideoManager to skip
        # _publish_frame. Frame outputs are counted into the bench
        # report and dropped, so the wall-clock excludes preview /
        # GL upload / Qt paint. Useful for telling whether display
        # rendering is in the critical path vs. swap throughput.
        if self._is_playing or self._is_recording:
            bus.play_video.emit("stop_from_gui")
            self._is_playing = False
            self._is_recording = False
            self._tooltip_label.setText(
                "Benchmark (Headless): stopping (report prints to console)"
            )
        else:
            bus.play_video.emit("benchmark_headless")
            self._is_playing = True
            self._tooltip_label.setText(
                "Benchmark (Headless): running, preview disabled — "
                "stop manually or wait for EOF"
            )
        self._record_armed = False
        self._center_pane.set_play_state(self._is_playing)
        self._center_pane.set_record_state(self._is_recording or self._record_armed)

    # ----- Parameter round-trip ---------------------------------------------------

    def _load_saved_parameters(self) -> None:
        values = load_params(SAVED_PARAMETERS_JSON)
        if values:
            self._params_pane.apply_values(values, emit=True)

    def _on_params_changed(self, snapshot: dict) -> None:
        bus.parameters_changed.emit(snapshot)
        # Settings-level toggles that affect Models internals — apply
        # them eagerly so the next swap call sees the new state.
        # set_model_session_mode is idempotent (no-op when mode
        # already matches), so calling on every params_changed is
        # cheap. We also re-prime the [N/M] load-progress expectation
        # because both mode and ThreadsSlider changes invalidate it.
        models = self._get_models()
        if models is not None:
            mode = str(snapshot.get("ModelSessionsTextSel", "Shared"))
            setter = getattr(models, "set_model_session_mode", None)
            if callable(setter):
                setter(mode)
            update = getattr(models, "update_load_expectation", None)
            if callable(update):
                threads = int(snapshot.get("ThreadsSlider", 1))
                update(threads)
        # Re-render the current frame so the new parameter takes effect
        # immediately. Skipped during playback — the worker pipeline is
        # already producing frames with the latest parameters baked in.
        self._refresh_current_frame()

    def _on_param_section_toggled(self, title: str, collapsed: bool) -> None:
        # Persist per-section collapsed state to data.json on every toggle.
        # Cheap (single file write) and crash-safe — no risk of losing the
        # state if the app exits abnormally.
        self.settings.params_collapsed[title] = collapsed
        self.settings.save()

    def _refresh_current_frame(self) -> None:
        """Re-request the timeline's current frame from VideoManager so
        the preview reflects any state change (parameters, source-face
        assignments, control toggles).

        Rapid bursts (slider drags emitting on every micro-tick) are
        absorbed by the coordinator's leading-edge debouncer — the first
        call fires immediately, subsequent calls within ~16ms are
        coalesced to the latest. So we can call this freely from any
        change handler without queuing up decode work.
        """
        if self._is_playing or self._is_recording:
            return
        frame = self._center_pane.timeline.get()
        bus.get_requested_video_frame.emit(frame)

    def show_splash(self) -> None:
        """Stage the startup splash image into the preview.

        Shown at launch and left in place until the first real frame
        (media scrub / playback) replaces it via bus.frame_ready. The
        preview is a QOpenGLWidget, so the staged frame simply waits in
        the pending buffer until the first paintGL after the window is
        shown. Best-effort: a missing or unreadable splash.png is
        silently skipped so it can never block launch.
        """
        try:
            splash_path = Path(__file__).resolve().parents[1] / "media" / "splash.png"
            if not splash_path.is_file():
                return
            bgr = cv2.imread(str(splash_path), cv2.IMREAD_COLOR)
            if bgr is None:
                return
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            self._center_pane.preview.set_frame(rgb)
        except Exception as exc:  # never let a decorative splash break startup
            print(f"[main_window] splash display skipped: {exc}")

    def _on_params_io(self, action: str) -> None:
        if action == "save":
            try:
                save_params(self._params_pane.values, SAVED_PARAMETERS_JSON)
                self._tooltip_label.setText(f"Saved {len(self._params_pane.values)} parameters")
            except OSError as exc:
                QMessageBox.warning(self, "Save failed", str(exc))
        elif action == "load":
            values = load_params(SAVED_PARAMETERS_JSON)
            if not values:
                self._tooltip_label.setText("No saved_parameters.json found")
                return
            self._params_pane.apply_values(values, emit=True)
            # apply_values(emit=True) routes through _on_params_changed
            # which already calls _refresh_current_frame.
            self._tooltip_label.setText(f"Loaded {len(values)} parameters")
        elif action == "default":
            self._params_pane.load_defaults(emit=True)
            self._tooltip_label.setText("Parameters reset to defaults")

    def _on_params_button_clicked(self, name: str) -> None:
        """Route parameter-pane button clicks. Covers in-section
        buttons (HF refine / clear) and the Settings-tab "Actions"
        group (Build TRT / Clear VRAM / Benchmark). Add elif cases as
        more pane buttons are introduced."""
        if name == "ClearVramButton":
            self._on_clear_vram()
            return
        if name == "BenchmarkButton":
            self._on_benchmark()
            return
        if name == "BenchmarkHeadlessButton":
            self._on_benchmark_headless()
            return
        if name == "HFRefineButton":
            touched = 0
            for slot in self._found_faces:
                if slot.get("AssignedEmbedding") is None:
                    continue
                slot["HFRefinePending"] = True
                touched += 1
            if touched == 0:
                self._tooltip_label.setText(
                    "HF Refine: no assigned target faces to refine."
                )
                return
            self._refresh_current_frame()
            self._tooltip_label.setText(
                f"HF Refine: queued re-measurement on {touched} face(s) "
                "for the next swap."
            )
        elif name == "HFClearCacheButton":
            cleared = 0
            for slot in self._found_faces:
                if slot.get("HFCorrectionGap") is not None or slot.get("HFCorrectionGapSamples"):
                    cleared += 1
                slot["HFCorrectionGap"] = None
                slot["HFCorrectionGapSamples"] = 0
                slot["HFRefinePending"] = False
            self._refresh_current_frame()
            self._tooltip_label.setText(
                f"HF Clear: dropped cached corrections on {cleared} face(s)."
            )
