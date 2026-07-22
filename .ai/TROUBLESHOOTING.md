# Troubleshooting Runbook

Start with the symptom, collect the listed evidence, and make the smallest reversible change first. Do not delete model files, job parts, or caches while the application is running.

## Quick environment checks

Run these from the repository root with the project virtual environment active:

```powershell
venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
ffmpeg -version
ffmpeg -hide_banner -encoders | findstr /I nvenc
```

Expected baseline:

- `torch.cuda.is_available()` is `True`.
- ONNX Runtime lists `CUDAExecutionProvider`; TensorRT users also see `TensorrtExecutionProvider`.
- FFmpeg is on `PATH` or discoverable through the bundled fallback.
- The Settings model table shows the exact baseline filenames as present.

## Face is detected but never swapped

### Evidence

- Found Faces contains a target thumbnail.
- The console never prints `[swap_diag] matched`, or `assignments=[]`.
- Clicking the target slot says to choose a source face or embedding.

### Cause and action

The target slot has no usable source assignment. Select the Found Face slot, then select a source image or saved embedding. Confirm that:

- `SourceFaceAssignments` is non-empty.
- `AssignedEmbedding` is not `None`.
- **Swap Faces** is enabled for preview/manual recording.
- The similarity threshold is not excluding the target identity.

Auto Job preflight rejects this condition; ordinary preview silently skips target-only slots.

## Swap output is garbled, noisy, ghosted, or badly colored

### Evidence

Look for the backend line:

```text
[Models] inswapper: TensorrtExecutionProvider ...
```

and the one-shot diagnostics:

```text
[swap_diag] matched ...
[swap_diag] model_output min=... max=... mean_abs_delta=... changed_values=...
```

### Recovery

1. In Settings, switch the inswapper backend to `ONNX` and unload that model so the next frame reloads it.
2. Prefer `inswapper_128.onnx` when it is available; the loader chooses it before the FP16 file.
3. Keep the current numeric `128`, `256`, or `512` modes; native 256/512 paths have different precision and memory behavior.
4. Restart the process after a CUDA fault because the CUDA context may remain poisoned.

Inswapper intentionally defaults to CUDA EP for reliability on RTX 3060-class systems. If an old TensorRT cache is suspected, close Rope and **move** the affected `models/ort_trt_cache` subdirectory aside instead of permanently deleting it; this preserves rollback and forces a clean rebuild.

## CUDA error 700 followed by NVDEC/PyNvVideoCodec failure

### Typical sequence

- ORT reports `CUDA failure 700: an illegal memory access was encountered`.
- A later `PyNvVideoCodec` call fails to decode a frame or reports `Error Code: 700`.

### Interpretation

The first invalid CUDA operation can poison the process-wide CUDA context. The later decoder error may be a consequence rather than the original cause.

### Recovery

1. Stop Rope completely and start a new process.
2. Select the ONNX/CUDA backend for inswapper.
3. Use `Shared` sessions and one worker to establish a stable baseline.
4. Re-enable Per-Thread mode and additional workers one at a time.
5. Capture the first Python/ORT stack trace, not only the final NVDEC error.

Do not repeatedly retry in the same process after an illegal-memory-access error.

## Find Faces or source selection appears to freeze the UI

Detection and ArcFace for **Find Faces** and uncached source images currently run synchronously from GUI handlers. A cold TensorRT build or a large source selection can therefore block repainting.

- Watch the console for detector/ArcFace load messages.
- Click **Preload Models** before Find Faces when practical.
- Select one source image first to distinguish cold-load time from multi-image work.
- Use CUDA EP temporarily if TensorRT engine construction is the delay.

Moving these operations to a serialized background GPU worker is an open P1 backlog item; the current pause is not always a deadlock.

## Find Faces returns no faces

- Scrub to a sharp frame where the face is large enough and unobstructed.
- Lower `DetectScoreSlider` conservatively.
- Increase `DetectInputSizeTextSel` from 320/416 to 480/640 for smaller faces.
- Try the other detector (`Retinaface` or `SCRDF`) only if its exact model file is present.
- Check for `Find Faces: detect failed (...)` in the bottom status and console.
- Verify the input is RGB H×W×3 and the selected detector session actually loaded.

## Model is present but Settings reports Missing

The inventory uses exact filenames. Common mismatches:

- `faceparser_fp16.onnx` does not satisfy `faceparser_resnet34.onnx`.
- `inswapper_128.onnx` is preferred by the runtime, but the current required inventory row is `inswapper_128.fp16.onnx`.
- Native 256/512 files are not part of the current visible inventory.

Use [MODEL_CATALOG.md](MODEL_CATALOG.md) and compare names character-for-character. Press **Refresh** after changing the folder.

## TensorRT was selected but the log shows CUDAExecutionProvider

ORT can fall back when TensorRT EP cannot initialize.

- Confirm `TensorrtExecutionProvider` is listed by ONNX Runtime.
- On Windows, look for `[rope] native DLL search path: added tensorrt_libs`.
- Verify compatible NVIDIA driver, CUDA-enabled PyTorch, ONNX Runtime GPU, and TensorRT packages in the same virtual environment.
- Inspect the earlier `TRT-EP init failed (...)` message.
- Ensure the selected model folder is writable so `ort_trt_cache` can be created.

A fallback is not automatically a correctness failure. The loaded-state column and startup log report the provider actually selected.

## Face Parser, Occluder, XSeg, or Restorer fails

- **Face Parser** requires `faceparser_resnet34.onnx`.
- **Occluder** requires `occluder.onnx`.
- **DFL XSeg** requires a user-supplied `dfl_xseg.onnx`; when absent it warns once and uses a no-op mask.
- The selected restorer requires its exact model from the catalog.
- `RestorerDetTypeTextSel=Reference` additionally requires `res50.onnx`.

Auto Job checks Occluder, XSeg, Face Parser, and the selected restorer during preflight. It does not currently preflight `res50.onnx` for Reference alignment. Manual preview can also fail later during lazy loading, so first disable the feature switch to isolate the baseline swap.

## Video does not open or decode

`MediaPlayer` probes PyAV for metadata/audio, then prefers PyNvVideoCodec, then torchcodec, and finally PyAV CPU decode.

- Look for `PyNvVideoCodec init failed` or `torchcodec init failed`; these messages identify the active fallback.
- A system FFmpeg DLL can conflict with PyNvVideoCodec's bundled FFmpeg. The bootstrap pins package DLLs first, but the exact exception still matters.
- Test a conventional H.264 MP4 to separate codec/container issues from the pipeline.
- After CUDA error 700, restart before testing decode again.
- `gpu_decode_active=False` is slower but can still be valid.

## Scrubbing lags behind the cursor

The current scrub implementation retains up to five FIFO requests and renders every request that remains queued. Rapid dragging can therefore display obsolete intermediate frames.

- Release the slider and wait for the final request.
- Temporarily disable swapping to distinguish seek cost from model cost.
- Do not increase the queue length.

Debounce and latest-only execution are tracked in the P1 backlog.

## Record or Auto Job cannot find the output folder

- Open Settings and select a writable **Output Folder**.
- Confirm `data.json` contains a valid `saved videos` path after the picker closes.
- For an existing job, the snapshotted output directory must still exist.
- Check free disk space; Auto Job keeps video-only parts until QC succeeds.

Manual recording and **Save Image** also require this folder.

## FFmpeg, NVENC, merge, or audio mux fails

- Confirm `ffmpeg -version` succeeds from the same environment.
- Auto Render probes `h264_nvenc` with a real one-frame encode. If it fails, it selects `libx264`.
- A live NVENC failure retries the current part with x264 and records `encoder_fallback_reason`.
- Missing audio is a QC failure only when the input contained audio.
- Preserve the render manifest and parts on failure; use Retry/Resume instead of manually concatenating files.

## Auto Job will not resume

Read the latest job and render error before modifying files:

- `Render checkpoint belongs to another video`: the loaded video path does not match.
- `Input video changed after the job was created`: size or modification time changed; create a new job/scan.
- `No auto-render checkpoint to resume`: render never created its manifest or it was removed.
- A scan-cache miss after changing video, target embedding, detector, threshold, or scan settings is expected.

Do not rename the video, render manifest, part directory, or part files between pause and resume.

## Thumbnail or source embedding cache does not persist

This is a verified current bug, not a user configuration issue:

- `_atomic_savez()` can create `*.tmp.npz` while attempting to replace `*.tmp`.
- Face cache has `load_face()` but no `store_face()`.

The P0 item in [PERFORMANCE_BACKLOG.md](PERFORMANCE_BACKLOG.md) defines the fix and acceptance tests.

## VRAM display appears 1024× too large

`Models.get_gpu_memory()` currently returns MiB, while `vram_updated` arguments and `VRAMIndicator` formatting call the values GB. Percentage remains correct because both values use the same unit, but the displayed absolute unit is mislabeled. Treat the numeric label as MiB until this contract is corrected.

## Evidence to include in a bug report

- Commit hash and branch.
- GPU model, VRAM, driver version, and Windows version.
- Python, torch, ONNX Runtime, TensorRT, and PyNvVideoCodec versions.
- Active detector, swapper resolution, backend, session mode, and worker count.
- Input resolution/FPS/codec and whether GPU decode is active.
- Full log from the first model-load line through the first exception.
- Minimal steps and whether the issue reproduces with Temporal Stabilization, restoration, and masking disabled.
- Auto Job state plus manifest stage, without sharing private video, embeddings, or face images.
