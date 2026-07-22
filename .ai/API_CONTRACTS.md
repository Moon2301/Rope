# Internal API Contracts

These are internal contracts for the current branch, not a compatibility promise for third-party packages. Change them deliberately, update all producers/consumers in one commit, and version durable data separately.

## Shared data conventions

| Concept | Contract |
|---|---|
| Decoded/preview frame | RGB `uint8`, H×W×3; NumPy on CPU fallback or torch tensor on CPU/CUDA |
| Model frame | Torch tensor on CUDA, RGB `uint8`, C×H×W unless the model helper documents normalization |
| Face landmarks | NumPy-compatible float array shaped 5×2 in current frame coordinates |
| Identity embedding | 512 values, normally NumPy `float32`; cosine similarity is reported as a percentage |
| FFmpeg raw frame | Contiguous NumPy `uint8`, RGB, H×W×3 |
| Frame range | Inclusive `(start_frame, end_frame)` using absolute zero-based frame numbers |
| Time conversion | `seconds = frame / fps`; do not use an integer stride as a substitute for real FPS |

Color conversions must happen at explicit boundaries. OpenCV files are BGR; preview, decoder, swap pipeline, and rawvideo output are RGB.

## Qt Bus contracts

`rope.qt.bus.bus` is a singleton. Qt `AutoConnection` gives direct delivery on the same thread and queued delivery across threads.

### GUI → backend

| Signal | Payload | Consumer/effect |
|---|---|---|
| `load_target_video` | `str path` | `VideoManager.load_target_video()` replaces the media session |
| `load_target_image` | `str path` | Loads a static target image |
| `get_requested_image` | none | Declared legacy signal; no current Coordinator connection |
| `play_video` | command string such as `play`, `stop`, `record`, `benchmark`, `benchmark_headless` | `VideoManager.play_video()` |
| `get_requested_video_frame` | `int frame` | Asynchronous scrub with marker parameters |
| `get_requested_video_frame_without_markers` | `int frame` | Asynchronous scrub using current parameters |
| `target_faces` | Found Face list | Replaces `VideoManager.found_faces` |
| `parameters_changed` | complete parameter `dict` snapshot | Replaces `VideoManager.parameters` |
| `markers_changed` | list of `{frame, parameters}` snapshots | Replaces marker state |
| `control_changed` | complete control `dict` snapshot | Replaces VM control state |
| `ui_vars_changed` | `dict` | Replaces auxiliary VM UI data |
| `saved_video_path` | `str directory` | Sets record/Auto Render output root |
| `vid_qual` | `int` | Sets legacy quality field |
| `set_stop` | `int frame` | Sets inclusive playback/record stop marker |
| `perf_test` | `bool` | Enables legacy per-frame diagnostic output |
| `auto_swap` | none | Declared legacy signal; no current Coordinator connection |
| `auto_render_segments` | Auto Render request object | Starts persistent part rendering |

### Backend → GUI

| Signal | Payload | Meaning |
|---|---|---|
| `frame_ready` | `(image, requested: bool)` | Pushes an RGB frame; `requested=True` denotes one-shot load/scrub output |
| `playback_frame_changed` | `int frame` | Moves the timeline to the published frame |
| `stop_play` | none | Synchronizes UI transport state after VM stops |
| `slider_length_changed` | `int max_frame` | Sets timeline maximum, normally `total_frames - 1` |
| `vram_updated` | two numeric values | Currently receives MiB from `Models.get_gpu_memory()` even though argument/UI names say GB; percentage is valid, absolute unit label is not |
| `queue_depths` | `(frame_q_length, requested_q_length)` | Telemetry emitted at most about 10 Hz |
| `models_preloaded` | none | Preload attempt finished; GUI must re-check `pipeline_sessions_loaded()` |
| `auto_render_progress` | progress dictionary | Frame/part progress and optional completed checkpoint |
| `auto_render_stage` | state string | `MERGING`, `MUXING`, or `PAUSED` from VM |
| `auto_render_finished` | `str final_output` | Mux completed and output is ready for QC |
| `auto_render_failed` | `str message` | Render pipeline stopped with recoverable artifacts preserved when possible |

Auto Render progress fields are:

```text
frame: int
frames: int
part: int
parts: int
percent: float
eta_seconds: int
checkpoint_complete?: bool
tracking_version?: int
tracking_checkpoint?: dict
```

## Found Face contract

```text
{
  Embedding: float32[512],
  SourceFaceAssignments: list[str],
  AssignedEmbedding: float32[512] | None,
  Thumbnail: uint8[H,W,3],
  HFCorrectionGap: optional data,
  HFCorrectionGapSamples: int,
  HFRefinePending: bool
}
```

- `Embedding` is the target identity used for per-frame matching.
- `AssignedEmbedding` is the source identity passed through latent projection.
- `SourceFaceAssignments` is also the explicit assignment-presence gate; an empty list causes the slot to be skipped.
- A slot key for temporal tracking hashes target and source embeddings, so changing either invalidates active track state.

## Models facade

### `run_detect(img, detect_mode, max_num=1, score=0.5, input_size=640)`

- Input: CUDA RGB `uint8` tensor C×H×W.
- `detect_mode`: currently `Retinaface` or `SCRDF`.
- `score`: fractional detector threshold in `[0, 1]`.
- `input_size`: square edge, normally 320/416/480/640.
- Output: iterable of five-point landmark arrays in input-frame coordinates.

### `run_recognize(img, kps, dim=1)`

- Input image: same CUDA RGB C×H×W frame.
- Input landmarks: raw detector 5×2 points.
- Output: `(embedding, cropped_image)` where embedding is 512-D NumPy and crop is the aligned RGB torch image.
- Identity recognition must continue using raw detector landmarks even when temporal stabilization is enabled.

### `run_swapper(image, embedding, output)`

- Logical input: NCHW aligned target batch and projected 512-D source latent.
- Callers provide an output torch buffer; the helper handles model FP16/FP32 boundary conversion.
- Batch support depends on the loaded graph/provider. `run_swapper_batched()` falls back to per-call execution after the first rejected batch.

Restorer and mask helpers use caller-owned output buffers except `run_dfl_xseg()`, which returns a CUDA mask and handles missing models as a no-op.

## MediaPlayer contract

| Method | Contract |
|---|---|
| `MediaPlayer(path, prefer_gpu_decode=True)` | Opens metadata/audio with PyAV and probes GPU decoder backends |
| `get_frame_at(frame_number)` | Random-access fetch returning `(RGB frame, pts)`; GPU fetch failure falls back to PyAV for that call |
| `get_first_frame()` | Convenience first-frame fetch |
| `seek(frame_number)` | Invalidates the active decode generation and moves the next playback position |
| `start_playback(audio_enabled=True)` | Starts background video decoding and optional audio output |
| `get_next_frame(timeout=0.0)` | Returns the next queued decoded item without blocking when timeout is zero; may return `None` |
| `get_audio_position()` | Returns DAC-clock position when audio is active, otherwise `None` |
| `stop_playback()` | Stops decode/audio workers for the current generation |
| `close()` | Releases containers, decoder, audio, and worker resources; do not reuse afterward |

The preferred decoder order is PyNvVideoCodec, torchcodec, then PyAV CPU fallback.

## VideoManager frame contracts

### `submit_scrub(frame, marker=True)`

Asynchronously appends an integer request to the bounded scrub deque. Current semantics are FIFO with `maxlen=5`; overflow drops the oldest queued entry, but every remaining request executes.

### `swap_video(target_image, frame_number, use_markers, temporal_generation=None)`

- Accepts a NumPy H×W×3 RGB frame or torch H×W×3 RGB tensor.
- Moves/casts to CUDA `uint8`, converts to C×H×W, and may upscale dimensions below 512 for processing.
- Copies parameter/control snapshots before processing.
- Uses the current worker CUDA stream, then inserts a default-stream dependency before returning.
- Returns an RGB frame suitable for preview or recording, normally a CUDA H×W×3 tensor from the swap path.
- When temporal generation is provided, every dispatched frame must call tracker `process()` or `skip()` exactly once.

### `swap_core(img, kps, source_embedding, parameters, control, precomputed_latent=None, slot=None)`

- Input image: CUDA RGB C×H×W.
- Landmarks: raw or temporally stabilized 5×2 coordinates in the same image space.
- Source embedding: assigned 512-D identity embedding.
- Parameters/control: immutable per-call snapshots.
- Output: CUDA RGB C×H×W composited frame.
- The function owns alignment, swapper selection/polyphase passes, masks, color, optional restoration, and inverse composite.

Callers must not share mutable scratch tensors between worker threads.

## Auto Segment contract

`AutoSegment` serializes as:

```text
{
  start_frame: int,
  end_frame: int,
  confidence: float in [0, 100],
  approved: bool,
  hit_count: int,
  review_required: bool,
  review_reasons: list[str]
}
```

Both frame boundaries are inclusive. `approved_ranges()` returns sorted inclusive tuples. Grouping uses absolute frames and clamps ranges to the source timeline.

## AutomationController contract

Primary methods:

- `start_job(request)`: creates and atomically saves a `PREFLIGHT` job.
- `begin_scan(job)`: transitions `PREFLIGHT → SCANNING`.
- `set_scan_results(job, segments)`: transitions `SCANNING → AWAITING_REVIEW` and clears confirmation.
- `confirm_segments(job, segments, render_params)`: accepts only `AWAITING_REVIEW`, requires at least one approved segment, snapshots render parameters, and transitions to `RENDERING`.
- `pause_job(job)`: records `paused_from` and enters `PAUSED`.
- `resume_job(job_id)` / `retry_job(job_id)`: restore `paused_from` or `retry_from`.
- `fail_job(job, message, retry_from=None)`: enters `FAILED` while preserving recovery state.
- `complete_job(job, final_output, qc)`: enters `COMPLETED` after QC success.

There is no separate `cancel_job()` method in the current controller. UI cancellation is implemented as a scan cancel event or graceful render cancellation followed by persisted `PAUSED` state.

## Auto Render request contract

`VideoManager.start_auto_segment_render(request)` accepts:

```text
{
  ranges: list[[inclusive_start, inclusive_end]],
  job_id: str,
  source_label: str,
  part_seconds: int,          # currently 60 from MainWindow
  render_params: dict
}
```

VM validates ranges, loaded video, and FFmpeg; it creates a versioned render manifest and emits lifecycle signals. Input video is never physically split.

## TemporalFaceTracker contract

### Sequence lifecycle

- `start_sequence(start_frame, fps, checkpoint=None) -> generation` resets state and optionally restores a version-compatible checkpoint whose `next_frame` equals `start_frame`.
- `abort_sequence(generation=None)` wakes waiters and invalidates that generation.
- `export_checkpoint() -> dict` serializes state for the next frame.

### `process(...)`

Core inputs:

```text
frame_number: absolute int
observations: [{kps: 5x2, similarities: {slot_key: percent}}]
frame_signature: uint8[18,32] luma or None
eligible: bool
slots: [{key: str}]
threshold: float percentage
frame_size: (width, height)
orientation: float degrees
generation: int
```

Output matches:

```text
{
  slot_key: str,
  observation_index: int | None,
  kps: stabilized 5x2,
  similarity: float,
  predicted: bool
}
```

Calls block at a `Condition` until `frame_number` equals the expected frame. Every successful, failed, skipped, and ineligible dispatched frame must advance or abort the sequence to prevent deadlock.

## Durable schema versions

| Schema | Current version | Owner |
|---|---:|---|
| Auto Job | `AUTOMATION_JOB_VERSION = 1` | `Automation.py` |
| Scan manifest | `SCAN_CACHE_VERSION = 3` | `AutoSegments.py` |
| Render manifest | `version = 3` | `VideoManager.py` |
| Temporal checkpoint | `TRACKING_VERSION = 1` | `FaceTracking.py` |

Never change durable field meaning without incrementing the owning version and defining migration or fallback behavior.
