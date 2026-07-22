# Development Guide

This guide covers the current Windows-first development workflow. Architecture and contracts are documented separately in [ARCHITECTURE.md](ARCHITECTURE.md) and [API_CONTRACTS.md](API_CONTRACTS.md).

## Known-good environment

The reproducible environment in `requirements.lock.txt` was captured from:

- Windows
- Python 3.12.10
- PyTorch 2.11.0 with CUDA 12.8 wheels
- ONNX Runtime GPU 1.26.0
- TensorRT 10.16 packages
- PySide6 6.11
- PyNvVideoCodec 2.1.0

`requirements.txt` is the curated dependency list. `requirements.lock.txt` is the exact known-good snapshot and should be preferred when reproducing bugs or benchmarks.

## Initial setup

From the repository root:

```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
```

If PowerShell activation is blocked, use Command Prompt:

```cmd
venv\Scripts\activate.bat
```

Place model files in `./models` or select another folder from Settings. See [MODEL_CATALOG.md](MODEL_CATALOG.md) before diagnosing a missing model. Model files, TensorRT caches, virtual environments, runtime caches, and user settings are intentionally ignored by Git.

## Running the application

```powershell
venv\Scripts\python.exe Rope.py
```

`Rope.bat` is also available for the normal Windows launch flow. Always launch from the repository root because current settings and media cache paths are working-directory relative.

For a UI-only smoke instance without loading models or `VideoManager`:

```powershell
venv\Scripts\python.exe -c "from rope.qt.app import run; raise SystemExit(run(skip_backend=True))"
```

## Repository map

- `Rope.py`: entrypoint.
- `rope/qt`: PySide6 UI, signals, panes, widgets, and UI tests.
- `rope/VideoManager.py`: playback/scrub/swap/record/Auto Render scheduler.
- `rope/MediaPlayer.py`: video and audio decoding.
- `rope/Models.py`: model sessions and inference paths.
- `rope/FaceTracking.py`: temporal association and stabilization.
- `rope/Automation.py`: durable Auto Job state machine.
- `rope/AutoSegments.py`: scan decoder, segment grouping, and cache helpers.
- `rope/scripts`: diagnostics such as the inswapper FP16 probe.
- `.ai`: architecture, contracts, operations, and performance documentation.

## Normal development loop

1. Reproduce with a fixed input and save the active parameter snapshot.
2. Identify the owning module and data/thread boundary before editing.
3. Add or update a focused unit test.
4. Make the smallest implementation change.
5. Run unit tests and the relevant UI/GPU smoke test.
6. For performance work, capture before/after measurements with identical settings.
7. Update `.ai` when behavior, data formats, models, or contracts change.
8. Commit generated documentation only; never commit weights, engines, caches, output videos, or personal embeddings.

## Unit tests

Run the complete pure test set:

```powershell
venv\Scripts\python.exe -m unittest discover -s rope/qt/tests -p "test_*.py"
```

Run one module while iterating:

```powershell
venv\Scripts\python.exe -m unittest rope.qt.tests.test_face_tracking
venv\Scripts\python.exe -m unittest rope.qt.tests.test_auto_segments
venv\Scripts\python.exe -m unittest rope.qt.tests.test_automation
venv\Scripts\python.exe -m unittest rope.qt.tests.test_parameters
```

Current test ownership:

- `test_automation.py`: state transitions, persistence, resume, and confirmation gate.
- `test_auto_segments.py`: grouping, review flags, ranges, and cache behavior.
- `test_face_tracking.py`: smoothing, matching, prediction, reset, ordering, and checkpoints.
- `test_parameters.py`: typed parameter schema and defaults.

## UI smoke tests

These tests open Qt windows and can write screenshots under `rope/qt/tests`:

```powershell
venv\Scripts\python.exe -m rope.qt.tests.smoke
venv\Scripts\python.exe -m rope.qt.tests.phase_c_e2e
venv\Scripts\python.exe -m rope.qt.tests.phase_d_preview
venv\Scripts\python.exe -m rope.qt.tests.phase_e_dialogs
```

Run them interactively on a machine with a display. Generated PNG files are ignored by Git.

## Real GPU smoke matrix

At minimum, validate one short video with:

| Dimension | Cases |
|---|---|
| Detector | RetinaFace; SCRFD when its model is available |
| Backend | CUDA EP; TensorRT EP for supported models |
| Session mode | Shared; Per-Thread |
| Workers | 1 and the intended production value; 1–5 for throughput changes |
| Swapper output | 128, 256, 512 |
| Tracking | On and off |
| Decode | GPU path and PyAV fallback when the change touches media code |
| Encode | NVENC and libx264 fallback when the change touches recording |

Verify the face visually and check frame count, duration, resolution, FPS, audio, VRAM peak, and console errors.

## Reproducible benchmarking

Use the built-in **Benchmark** and **Benchmark (Headless)** controls:

- Benchmark removes audio and wall-clock pacing but still publishes preview frames.
- Headless benchmark also drops preview publication, isolating decode and swap throughput more closely.
- Stop manually, reach EOF, or set a stop marker; the console prints frame count, elapsed time, mean generation time, and effective FPS.

Record this metadata with every result:

```text
commit:
GPU / driver / VRAM:
Python / torch / ORT / TensorRT:
video codec / resolution / FPS / frame range:
detector / input size / detect score:
swapper resolution / backend:
session mode / workers:
tracking / restorer / masks:
warm cache or cold build:
benchmark vs headless:
mean generation ms / effective FPS / peak VRAM:
```

Discard cold model-build frames or report them separately. Compare the same frame range and warmup policy before and after a change.

## Nsight Systems and NVTX

Hot paths already use `nvtx_range()` labels for detector, recognizer, swapper, and restoration stages. When Nsight Systems is available, profile the application process and filter by these ranges instead of inferring cost only from Python wall time.

Recommended profiling discipline:

1. Preload models first when measuring steady state.
2. Use a fixed stop marker and headless benchmark.
3. Capture CUDA kernels, CUDA API, OS runtime, and NVTX.
4. Compare one worker against the production worker count.
5. Check stream overlap and host synchronization, not only kernel duration.
6. Store trace files outside the repository.

## Inswapper FP16 diagnostic probe

`rope/scripts/probe_inswapper_fp16.py` performs an expensive TensorRT precision bisection. It requires Polygraphy and can take 10–60 minutes:

```powershell
venv\Scripts\python.exe -m rope.scripts.probe_inswapper_fp16
```

Read the script header and use an explicit temporary work directory when preserving results. This is a diagnostic tool, not part of normal app startup.

## Common change recipes

### Add a parameter

1. Add its legacy-compatible default fields to `rope/qt/_default_data.py`.
2. Add a typed entry to `rope/qt/parameters.py` with the correct scope.
3. Route it through the generated parameter pane; avoid bespoke widget state.
4. Snapshot it in Auto Job render parameters if it affects output.
5. Add migration/default tests and update API contracts when the payload changes.

### Add a model

Follow the checklist in [MODEL_CATALOG.md](MODEL_CATALOG.md). Model inference belongs in `Models`; UI code only selects and reports it.

### Add a Bus signal

1. Define a typed `Signal` in `rope/qt/bus.py`.
2. Document direction and payload in [API_CONTRACTS.md](API_CONTRACTS.md).
3. Connect it in the owner that translates UI intent to backend state.
4. Verify cross-thread delivery and stale-generation behavior.

### Change a cache or manifest

1. Increment the relevant schema version.
2. Keep writes atomic in the same directory.
3. Define old-version fallback or migration.
4. Test missing, corrupt, interrupted, and stale-key cases.
5. Update [STATE_AND_DATA.md](STATE_AND_DATA.md).

## Definition of done

- Scope-specific tests pass.
- The relevant UI/GPU smoke path passes.
- No new GUI-thread blocking or unbounded queue is introduced.
- Tensor color/shape/device and frame ordering remain explicit.
- Performance claims include reproducible measurements.
- New errors have actionable messages and preserve recovery artifacts.
- Documentation, schema versions, and model inventory are synchronized with code.
