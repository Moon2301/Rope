# Verified Performance Backlog

The changes below are **not considered implemented**. Priorities reflect expected return, correctness impact, and implementation risk. Do not combine every item into one commit; that would make regressions difficult to isolate.

## Recommended order

| Priority | Item | Expected return | Risk |
|---|---|---:|---:|
| P0 | Fix atomic `MediaCache` writes and add face caching | High; fixes a real bug | Low |
| P1 | Direct ArcFace ROI warp and face batching | Very high for large frames or multiple faces | Medium–high |
| P1 | Scrub debounce/latest-only and single signal emission | High UX impact | Medium |
| P1 | Use rawvideo for manual FFmpeg recording | High during recording | Medium |
| P1 | Move heavy GUI-triggered work to a serialized worker | High responsiveness impact | Medium–high |
| P2 | Reuse RetinaFace output buffers | Medium | Medium |
| P2 | Keep SCRFD post-processing on GPU and cache anchors | High when SCRFD is used | High |
| P2 | Timer, allocator, and process-slot quick wins | Low–medium | Low |

## P0 — Make MediaCache reliable and persist source faces

### Verified current behavior

- `MediaCache._atomic_savez()` builds `tmp = target + '.tmp'` and calls `np.savez(tmp, ...)`.
- NumPy creates `*.tmp.npz`, while `os.replace()` looks for `*.tmp`; the destination is never created.
- `load_face()` exists, but `store_face()` does not.
- `ThumbnailLoader` reads the face cache but, after a cache miss, only decodes a thumbnail and never persists an embedding or crop.
- `MainWindow._source_face_embeddings` caches embeddings only in memory for the active process.

### Target design

- Write `np.savez()` to an open binary file handle, or use a temporary path that already ends with `.npz` and replace the exact resulting file.
- Add `store_face(path, fingerprint, thumbnail, embedding, metadata)`.
- The minimum key/fingerprint includes absolute path, size, `mtime_ns`, detector, detection score, input size, and recognizer/model version.
- Separate thumbnail-only entries from embedding-ready entries so the gallery does not need to load models.
- Persist a source embedding after successful detection/recognition; recompute corrupt entries automatically.

### Acceptance criteria

- A round-trip creates the exact destination and leaves no orphan `*.tmp.npz` files.
- After restarting the app, selecting the same source does not rerun the detector or ArcFace.
- Editing the image or recognition configuration produces a cache miss.
- Two processes writing the same key cannot expose a partial file.

## P1 — Direct ArcFace ROI warp and batching

### Verified current behavior

`Models.recognize()` estimates a similarity transform, calls `v2.functional.affine()` on the **entire frame**, and only then crops the `dim*112 × dim*112` region. Every face runs the session separately for each `dim × dim` tile, and the `(1, 512)` output is allocated per call.

At 1920×1080 versus 112×112, full-frame warping touches approximately 165 times as many pixels as the ROI for each face. This ratio describes pixel count, not a guaranteed FPS improvement.

### Target design

- Derive an affine grid for a direct 112×112 output from the detector-to-ArcFace transform.
- Use `torch.nn.functional.grid_sample()` or an equivalent kernel on the source frame without materializing a full warped frame.
- Batch all faces from one frame into `N×3×112×112` and run ArcFace once when the session/profile supports dynamic batching.
- Reuse input/output buffers by batch capacity, with a batch-1 fallback for static TensorRT engines.
- Preserve the old path's interpolation, coordinate convention, BGR order, and normalization exactly.

### Safe implementation sequence

1. Add a direct-crop batch helper while still running inference once per face.
2. Golden-test crops and embeddings against the old path for centered, edge, scaled, and rotated faces.
3. Enable batched ArcFace only after alignment equivalence is established.
4. Benchmark detector, recognizer, and end-to-end performance separately.

### Acceptance criteria

- Crop differences remain within an explicit tolerance, and new/old embedding cosine similarity remains sufficiently high across a pose set.
- Faces near image boundaries are not truncated, and padding matches the old path.
- Batch 1 and batch N preserve detection ordering.
- VRAM does not grow without bound, and throughput improvements are measured at 720p, 1080p, and 4K.

## P1 — Scrub debounce and latest-only execution

### Verified current behavior

- VM uses `deque(maxlen=5)`, and the scrub thread consumes it with FIFO `popleft()`; every remaining request is decoded and swapped.
- Coordinator does not coalesce events.
- `ParameterSlider._on_entry_committed()` can emit once from `setValue()` and then emit again because its post-set condition is always true.
- `ParameterSlider.set(request_frame=True)` can receive `valueChanged` and also emit explicitly.

### Target design

- Use a 16–33 ms single-shot GUI debounce during dragging.
- VM retains only the latest pending request; in-flight work receives a generation ID and must not publish when stale.
- Mouse release and entry commit always publish the final requested frame.
- Fix slider emission by comparing `old_pos` before `setValue()`, or block the slider signal and emit from exactly one location.

### Acceptance criteria

- A burst of 100 drag events produces at most one in-flight and one pending request.
- The final preview always displays the frame where the user released the control.
- One entry commit or `set(..., request_frame=True)` call emits exactly once.
- Seek and stop cannot deadlock the temporal coordinator.

## P1 — Rawvideo for manual FFmpeg recording

### Verified current behavior

- Auto Render starts FFmpeg with `-f rawvideo -pixel_format rgb24 -video_size ...` and writes contiguous bytes.
- Manual FFMPEG recording uses an input pipe without declaring rawvideo; it creates a `PIL.Image` and BMP-encodes every frame into stdin.

### Target design

- Extract a shared FFmpeg rawvideo encoder helper for manual recording and Auto Render.
- Write `np.ascontiguousarray(image, dtype=np.uint8).tobytes()` from the manual path.
- Preserve original-audio muxing and start/stop marker semantics.
- Keep manual recording on `libx264` initially; NVENC policy can remain a separate change.

### Acceptance criteria

- No `Image.fromarray(...).save(..., 'BMP')` remains in the recording hot path.
- Output has correct RGB color, resolution, FPS, frame count, and duration.
- A broken pipe closes recording cleanly without freezing the GUI.
- Audio muxing and playhead-based recording boundaries remain synchronized.

## P1 — Move heavy work off the GUI thread

### Verified current behavior

- `_on_find_faces()` runs detection and recognition synchronously.
- `_on_source_face_selection_changed()` computes every uncached image embedding sequentially.
- Selecting a target emits a same-thread signal into `VideoManager.load_target_video()`, which creates `MediaPlayer` and reads the first frame.
- Auto scan and QC already provide a suitable pattern using `QRunnable` and signals.

### Target design

- Add one serialized GPU task queue for preload, Find Faces, source embedding, and media-open/first-frame work so sessions and VRAM are not contended.
- Attach a generation to every request; changing media or selection discards stale results.
- Keep repaint and cancellation responsive while exposing busy/progress state.
- Split media open into prepare and commit phases: a worker reads metadata/first frame, then GUI/VM commits only if the generation is still current.

### Acceptance criteria

- The window remains movable and resizable during cold model loading and multi-image selection.
- Selecting video A and then B quickly can never publish A's first frame after B.
- No two GPU inference tasks from this queue run simultaneously.
- Cancellation and exceptions always release busy state.

## P2 — Reuse RetinaFace output buffers

### Verified current behavior

RetinaFace already has a thread-local input letterbox buffer and an anchor cache. However, every `detect_retinaface()` call still allocates nine CUDA output tensors: score, bbox, and keypoints for strides 8, 16, and 32.

### Target design

- Add a thread-local output cache keyed by input `(H, W)`, dtype, and device.
- Bind ORT outputs to cached buffers; consumers must finish before the same worker reuses them.
- Bound the number of retained input profiles, or clear them when unloading or changing backend.

### Acceptance criteria

- Steady-state execution no longer creates nine `torch.empty` tensors per frame.
- Output and NMS are bitwise-equivalent or within an explicit numerical tolerance.
- Per-thread sessions with 1–5 workers never overwrite another worker's buffers.

## P2 — Keep SCRFD post-processing on GPU

### Verified current behavior

- SCRFD calls `io_binding.copy_outputs_to_cpu()`.
- `center_cache = {}` is recreated for every detection.
- After CPU decode and sorting, boxes and scores are uploaded to CUDA again for `torchvision.ops.nms()`.
- Preprocessing also allocates a new letterbox tensor and performs several permute/materialize steps.

### Target design

- Bind outputs to CUDA tensors as RetinaFace already does.
- Persist anchor centers by `(input_h, input_w, stride, device, dtype)`.
- Decode bbox/keypoints, threshold, sort, and run NMS on GPU; copy only the small final keypoint result when truly necessary.
- Reuse preprocessing and output buffers per thread.

### Acceptance criteria

- No full-output GPU→CPU→GPU round trip remains.
- Detection and keypoints match the old path on a regression set.
- VRAM remains stable across frames and repeated input-size changes.

## P2 — Resource quick wins

### Coordinator timer

`Coordinator._tick()` now handles only queue and VRAM telemetry, but its `QTimer` interval remains 0 ms. A 50–100 ms interval is sufficient for the HUD and reduces idle CPU wakeups because frame delivery is already push-based.

### Stop and the CUDA allocator

Both `play_video('stop')` and `stop_from_gui` call `torch.cuda.empty_cache()` on every stop. Normal Stop should abort work, stop queues, and release references. Empty the cache only for explicit unload, backend rebuild, or a justified recovery path.

### ProcessedFrame lifetime

After preview publication, a process slot clears status, thread, frame number, timing, and PTS but does not reset `ProcessedFrame`. The recording path behaves similarly. Release the reference immediately after the final consumer to allow earlier tensor/CPU-frame reclamation.

### Shared acceptance criteria

- Idle CPU usage decreases measurably while VRAM telemetry still updates within 100 ms.
- Repeated play/stop cycles do not cause memory growth.
- Peak allocated/reserved memory does not regress, and Stop latency improves.
- Process slots retain no tensor from a frame that has already been published or written.

## Suggested commit sequence

1. `fix(cache): make npz writes atomic and persist face embeddings`
2. `perf(scrub): debounce and keep only the latest request`
3. `perf(record): share rawvideo ffmpeg writer`
4. `perf(ui): serialize heavy media and face tasks off the GUI thread`
5. `perf(arcface): crop aligned ROIs directly`
6. `perf(arcface): batch recognition where backend supports it`
7. `perf(detector): reuse retinaface outputs`
8. `perf(scrfd): keep postprocess on CUDA`
9. `perf(runtime): reduce idle polling and release frame references`

Each commit requires its own tests and benchmarks so quality or stability regressions can be bisected reliably.
