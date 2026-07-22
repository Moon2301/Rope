# Code Modification Rules

These rules are derived from the current architecture. Their purpose is to preserve GUI-thread safety, frame ordering, GPU memory behavior, and resumability.

## 1. Module boundaries

- UI and dialogs belong in `rope/qt`; model kernels belong in `Models`; media lifecycle belongs in `MediaPlayer` and `VideoManager`.
- Keep the pure-Python state machine in `Automation.py`; it must not import Qt, torch, or CUDA.
- Put reusable scan/group/cache logic in `AutoSegments.py`; UI workers should only orchestrate work and emit progress.
- Do not add model inference to widget event handlers. Handlers validate input, create immutable snapshots, and dispatch workers.
- `Bus` contains signals with explicit contracts; never use it as a state store.

## 2. GUI thread

- Only the GUI thread may mutate widgets.
- Detection, ArcFace, model loading, video open/decode, long FFmpeg probes, and bulk I/O must run in workers.
- Workers return results through Qt signals; each result must carry a generation or job ID so stale results can be rejected.
- Never wait on `Future.result()`, `Thread.join()`, or a long subprocess from the GUI thread.
- An idle timer must not spin at 0 ms when it only reports telemetry; choose an interval appropriate for display updates.

## 3. Frame ordering and concurrency

- Every frame dispatched into a temporal sequence must advance the ordered coordinator, including pass-through frames, no-detection frames, and exception paths.
- A worker must mark its process slot completed or failed in `finally` or an equivalent error path; never leave `Status='started'` indefinitely.
- Seek, stop, video changes, assignment changes, and orientation changes must increment the generation or abort the sequence so old waiters exit.
- Presentation and recording must consume completed frames by `FrameNumber`, not worker completion order.
- Do not share mutable scratch buffers or I/O bindings across workers; use thread-local storage or narrowly scoped locks.
- Release frame tensors from process slots immediately after the final consumer publishes or writes them.

## 4. Tensor, color, and shape contracts

- The preferred decode/preview contract is RGB `uint8` with shape H×W×3.
- Model paths usually consume CUDA tensors with shape C×H×W; convert once at the boundary.
- FFmpeg `rawvideo rgb24` input must be contiguous `uint8` H×W×3.
- OpenCV reads and writes BGR by default; document color conversion at every boundary.
- Avoid GPU→CPU→GPU transfers in hot paths. Call `.cpu().numpy()` only for output, file I/O, or a UI fallback that actually requires it.
- Cache fixed-shape buffers by `(device, dtype, shape, worker)`.
- Do not use torch/CUDA scalars for metadata arithmetic such as FPS, ratios, or dimensions.

## 5. Model and CUDA lifecycle

- Load models lazily through existing helpers; never construct `InferenceSession` directly from UI code.
- Respect backend preferences and TensorRT input profiles.
- I/O-bound output buffers must remain alive until inference and all consumers finish.
- Buffer caches must be bounded and invalidated when device, model, or input profile changes.
- Do not call `torch.cuda.empty_cache()` during frequent operations such as Stop. Reserve it for explicit unload or recovery after all references have been released.
- Every GPU optimization must measure warmup, throughput, peak VRAM, and correctness on identical input.

## 6. Recognition, matching, and tracking

- ArcFace uses raw detector landmarks unless a separately tested migration changes that contract.
- Landmark smoothing affects alignment/swap only, never identity embeddings.
- Multi-face association must remain one-to-one; one detection cannot be assigned to two target slots.
- New tracks must satisfy the identity threshold. Do not widen active-track hysteresis beyond current rules without tests.
- Miss prediction remains conservative at one frame; a scene cut or second miss resets the track.
- Temporal Stabilization must not affect scan-cache validity.

## 7. Parameters and compatibility

- Declare new parameters in `parameters.py` with explicit defaults and scope.
- Render snapshots must contain every parameter that affects output, including the tracking switch and version.
- Loaders must accept older snapshots and apply deterministic fallback behavior.
- Increment the relevant version whenever a cache, job, or render schema changes.
- Never silently reinterpret an existing field with a new meaning.

## 8. Caches and files

- Atomic write sequence: temporary file in the same directory → flush/close → `os.replace()`.
- Test atomic writers by verifying that the destination exists and can be read back.
- Cache keys include the input fingerprint, model/configuration, and algorithm version.
- Cache misses and corruption are normal paths: discard the entry and recompute instead of crashing the app.
- Never commit developer-specific absolute paths into source, documentation, or defaults.
- Delete parts and checkpoints only after QC passes; failed and paused jobs retain them for resume.

## 9. Error handling and UX

- Logs should include stage, model/backend, and frame/part/chunk whenever available.
- Worker exceptions must emit a failure result and release all waiters and process slots.
- Do not catch `Exception` silently when it can invalidate output; at minimum log it or move the job to `FAILED`.
- User-facing errors should state the next action: assign a source, select an output folder, resume/retry, or change backend.
- Preflight blocks incomplete input early; rendering must never begin before `confirm_segments()`.

## 10. Performance workflow

Before merging a hot-path optimization:

1. Record a baseline: video, resolution, FPS, worker count, backend, warmup, and parameter snapshot.
2. Profile individual stages (`detect`, `recognize`, `swap`, preview/write), not only end-to-end FPS.
3. Verify numerical output: embedding cosine, landmark/alignment results, frame count, duration, and audio synchronization.
4. Measure 1–5 workers and peak VRAM on an RTX 3060 12 GB.
5. Provide a fallback when a dependency, codec, or backend is unavailable.

An optimization is not acceptable merely because it reduces allocations if output changes unexpectedly or workers can deadlock.

## 11. Tests and pre-commit checks

Current unit tests live in `rope/qt/tests`:

```powershell
python -m unittest discover -s rope/qt/tests -p "test_*.py"
```

Minimum checks by change type:

- State/cache: transition, versioning, invalidation, and atomic round-trip tests.
- Tracking: out-of-order completion, scene cuts, one-frame misses, angle wrapping, and checkpoint resume.
- UI signals: one user action emits once, and a stale generation never updates UI.
- Record/render: readable output with correct frame count, duration, audio, NVENC behavior, and x264 fallback.
- GPU kernels: compare output against the old path across poses and sizes, and measure peak memory.

Then run the appropriate UI smoke tests in `rope/qt/tests/`. Real model/GPU checks must be clearly marked as smoke or integration tests rather than mixed into pure unit tests.

## 12. Quick review checklist

- [ ] Widgets are touched only on the GUI thread.
- [ ] No unbounded queues or waiters without an abort path.
- [ ] Frame ordering, generations, and error paths all complete correctly.
- [ ] Tensor color, shape, and device contracts are explicit.
- [ ] Cache/schema versions and invalidation are correct.
- [ ] Pause/failure preserves checkpoints.
- [ ] Regression tests and benchmarks match the change risk.
- [ ] `.ai` is updated when a flow or contract changes.
