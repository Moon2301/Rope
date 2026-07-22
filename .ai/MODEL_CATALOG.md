# Model Catalog

This catalog describes model files referenced by the current code. The source of truth for the Settings inventory is `MODEL_INVENTORY` in [`rope/Models.py`](../rope/Models.py); runtime selection additionally contains native swapper paths that are not exposed by the current UI selector.

## Model folder and loading policy

- The default folder is `./models`; users can select another folder in Settings.
- Model files and generated engines are ignored by Git and must never be committed.
- Sessions load lazily. **Preload Models** warms only the selected detector, ArcFace, and selected swapper.
- Feature-gated restorers and mask models remain shared singletons and load on first use.
- The current UI supports `Shared` and `Per-Thread` session modes. Per-thread sessions increase parallelism and VRAM use.
- The Settings `Backend` button selects ORT `TensorrtExecutionProvider` (`TRT`) or `CUDAExecutionProvider` (`ONNX`) for supported models.
- TRT EP builds and caches engines internally under `<models>/ort_trt_cache/`; a standalone `.engine` file is not required by the runtime loader.

## Baseline swap pipeline

| File | Required when | Logical contract | Backend policy | Missing behavior |
|---|---|---|---|---|
| `det_10g.onnx` | RetinaFace is selected; required by the default pipeline | RGB frame → face detections with five landmarks; square input 320/416/480/640 | TRT EP by default with a dynamic 320–640 profile; CUDA EP fallback | Detection, Find Faces, scan, and swap fail to initialize |
| `scrfd_2.5g_bnkps.onnx` | SCRFD is selected | RGB frame → face detections with five landmarks; dynamic square input | CUDA EP shared singleton; no backend toggle in the inventory | The SCRFD path fails; RetinaFace remains available if present |
| `w600k_r50.onnx` | Every identity-aware operation | Normalized 112×112 BGR face chip → 512-D ArcFace embedding | TRT EP by default; CUDA EP fallback | Source embedding, Find Faces identity, scanning, tracking association, and swapping cannot complete |
| `inswapper_128.onnx` | Preferred baseline swapper when present | 128×128 aligned target plus 512-D source latent → 128×128 swapped face | CUDA EP by default for reliability; TRT only when explicitly selected | Loader falls back to `inswapper_128.fp16.onnx` |
| `inswapper_128.fp16.onnx` | Baseline fallback when `inswapper_128.onnx` is absent | Same logical 128 swap contract; model I/O can be FP16 | CUDA EP by default; TRT only when explicitly selected | No baseline swapper is available if the FP32 file is also absent |

Important inventory detail: `MODEL_INVENTORY` currently marks `inswapper_128.fp16.onnx` as the required row, while `_swapper_128_filename()` prefers `inswapper_128.onnx`. A folder containing only the preferred FP32 file can therefore work even if the Settings inventory marks the FP16 row missing.

## Swapper resolution behavior

The current selector exposes `128`, `256`, and `512`:

- `128`: one 128×128 inswapper pass.
- `256`: polyphase tiling through the 128 model; it does not require a native 256 model.
- `512`: polyphase tiling through the 128 model; it does not require a native 512 model.

Additional code paths exist but are not exposed in `SwapperTypeTextSelModes`:

| File | Internal selection | Contract | Notes |
|---|---|---|---|
| `inswapper_256_phase1.onnx` | `256-Native` | FP32, batch 1, 256×256 target + 512-D latent → 256×256 output | TRT uses a fixed 256 profile with FP16 disabled because FP16 variance computation produced noisy output |
| `inswapper_512_level2.onnx` | `512-Native` | Usually static batch 1, 512×512 target + 512-D latent → 512×512 output | TRT EP or CUDA EP; runtime detects whether batching is supported |

Do not add these files to release requirements unless the corresponding UI modes are intentionally exposed and tested.

## Restorers

All restorers are optional and loaded only when `RestorerSwitch` is enabled with the matching `RestorerTypeTextSel` value.

| File | UI value | Input/output | Backend |
|---|---|---|---|
| `GFPGANv1.4.onnx` | `GFPGAN` | NCHW float32, 512×512 → 512×512 | CUDA EP with CPU fallback declared |
| `codeformer_fp16.onnx` | `CF` | NCHW float32 image plus scalar weight → 512×512 | Configured provider list, normally CUDA EP |
| `GPEN-BFR-256.onnx` | `GPEN256` | NCHW float32, 256×256 → 256×256 | Configured provider list, normally CUDA EP |
| `GPEN-BFR-512.onnx` | `GPEN512` | NCHW float32, 512×512 → 512×512 | Configured provider list, normally CUDA EP |
| `res50.onnx` | `RestorerDetTypeTextSel=Reference` | 512×512 aligned face → five reference landmarks | Configured provider list; used for restorer alignment, not the primary video detector |

If a selected restorer is missing, its lazy loader raises during the frame pipeline. Auto Job preflight checks the selected restorer file, but it does not currently check `res50.onnx` when `RestorerDetTypeTextSel=Reference`; that is a known preflight coverage gap.

## Mask and parsing models

| File | Feature switch | Logical contract | Missing behavior |
|---|---|---|---|
| `occluder.onnx` | `OccluderSwitch` | NCHW 256×256 face → one-channel 256×256 occlusion mask | Lazy load raises; Auto Job preflight reports the exact missing file |
| `faceparser_resnet34.onnx` | `FaceParserSwitch` | NCHW 512×512 face → 19-class 512×512 parsing logits | Lazy load raises; Auto Job preflight reports the exact missing file |
| `dfl_xseg.onnx` | `DFLXSegSwitch` | User-supplied NHWC or NCHW face → one-channel mask; names, layout, dtype, and resolution are introspected | Warns once and returns an all-ones no-op mask |

The current face parser loader requires the exact filename `faceparser_resnet34.onnx`. A file named `faceparser_fp16.onnx` is not discovered automatically.

## Files visible in some model bundles but unused by current code

The current Python runtime does not reference files such as:

- `7999j_iter.pth`
- `rfd64-uni-refined.pth`
- `e4e-*.ckpt`
- `yoloface_8n.onnx`
- `faceparser_fp16.onnx`

They may belong to older Rope variants, removed features, or experimental pipelines. Their presence does not make a current feature available unless a loader explicitly references them.

## TensorRT cache and backend semantics

| Model family | Default request | Important behavior |
|---|---|---|
| RetinaFace | TRT EP | One dynamic engine profile covers 320–640; CUDA EP fallback uses heuristic cuDNN selection |
| ArcFace | TRT EP | Fixed 112×112 input; engine/timing cache is shared |
| Inswapper 128 | CUDA EP | TRT is opt-in because some RTX 3060/ORT/TRT combinations returned garbled faces despite successful execution |
| Native 256/512 | TRT EP unless forced to ONNX | Native 256 keeps TRT computation in FP32; native 512 is normally static batch 1 |
| SCRFD, restorers, masks | CUDA EP/provider list | They do not use the per-model TRT toggle in the current inventory |

Always trust the startup log and `session.get_providers()` result over the requested setting. ORT can silently fall back from TensorRT EP to CUDA EP.

## Adding or replacing a model

1. Add the exact filename and role to `MODEL_INVENTORY` when it should appear in Settings.
2. Add `MODEL_INVENTORY_DETAILS` metadata when the model has tracked loaded state.
3. Implement lazy load/unload through `Models`; do not construct sessions in UI code.
4. Document input names, shapes, layout, color order, range, dtype, and output meaning.
5. Add preflight checks for any feature whose missing model would otherwise fail during rendering.
6. Decide whether the model is shared or per-thread and include it in VRAM tracking if applicable.
7. Add backend fallback and model-version/cache invalidation rules.
8. Add a smoke test with the real model and a pure unit test for selection/fallback logic where possible.
