# State, Data, and Caches

## Durable data sources

| Data | Location | Format/owner |
|---|---|---|
| Folders, window geometry, and backend preferences | `data.json` in the working directory | `qt.settings.Settings` |
| User-saved parameter snapshot | `saved_parameters.json` | `parameters_migration.py` / `MainWindow` |
| Auto Jobs | `QStandardPaths.AppDataLocation/automation_jobs/<job_id>/job.json` | `AutomationJobStore` |
| Scan manifest | `QStandardPaths.CacheLocation/auto_segments/<key>.manifest.json` | `_AutoScanWorker` + `AutoSegments` |
| Scan thumbnails | `<key>.thumbs/` under the same cache directory | `_AutoScanWorker` |
| Media thumbnail cache | `./cache/*.npz` | `MediaCache` / `ThumbnailLoader` |
| Auto Render manifest | `<output>/.<video>_<job_id>_auto_render.json` | `VideoManager` |
| Auto Render parts | `<output>/.rope_<video>_<job_id>.parts/` | `VideoManager` |

If `AppDataLocation` is empty, the job store falls back to `./.rope_app_data/automation_jobs`.

## Settings

`Settings` accepts missing keys and applies defaults to remain compatible with older `data.json` files. Primary keys are:

- `source videos`, `source faces`, `saved videos`, `merged_embeddings_file`, and `models_folder`.
- `dock_win_geom`, `splitter_main_sizes`, `splitter_left_sizes`, and `splitter_center_sizes`.
- `params_collapsed`.
- `model_backends`: model attribute → `trt` or `onnx`; a missing key means automatic selection.

Every new setting requires a safe default, and its loader must tolerate old or damaged files.

## Found Face slot

The logical shape passed between the UI and VM is:

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

`Embedding` identifies the target. `AssignedEmbedding` identifies the source to swap in. Never infer assignment state only from a UI highlight.

## Auto Job manifest

`AutomationJob.version` is currently `1`. The manifest contains:

- Identity: `job_id`, timestamps, `video_path`, and `video_fingerprint`.
- Input snapshot: metadata, target/source embeddings, and source labels.
- Configuration snapshot: `scan_config` and `render_params`.
- Review state: segments and `user_confirmed`.
- Recovery state: `scan_checkpoint`, `render_checkpoint`, `paused_from`, and `retry_from`.
- Result state: `final_output`, `qc`, and `error`.

Job JSON is written atomically through a temporary file in the same directory followed by `os.replace()`. `AutomationController` is the only valid place for state transitions; `MainWindow` orchestrates workers and updates the controller.

## Scan cache

`SCAN_CACHE_VERSION = 3`. The cache key hashes:

- Resolved video path, file size, and `mtime_ns`.
- Target embedding.
- The complete scan configuration: algorithm version, detector, detection score, input size, identity threshold, sampling interval, refine settings, chunk size, near-hit margin, grouping gap, and padding.

The manifest stores `completed_chunks` and, for each chunk:

- `start_frame`, `end_frame`, and `stage`.
- `coarse_cursor` and `refine_cursor`.
- `coarse_hits`, `near_hits`, and `hits` in absolute frame coordinates.

Changing tracking, restoration, or masking does not invalidate the scan cache because those settings affect rendering only. Thumbnails are generated after segment merging and cached separately from inference results.

## Render manifest

The render manifest is currently version `3` and contains:

- Input fingerprint, approved ranges, and render parameters.
- `part_frames`, completed part paths, `next_frame`, working directory, and final output path.
- Encoder (`h264_nvenc` or `libx264`) and current stage.
- `tracking_version` and the `tracking_checkpoint` at a part boundary.

When resuming an older job without `TemporalTrackingSwitch`, tracking remains disabled so later parts do not differ from already-rendered parts. A crash inside a part restores the checkpoint from the previous completed part and re-renders only the incomplete part.

## Temporal tracking checkpoint

`TRACKING_VERSION = 1`. A checkpoint is restored only when its version is valid. It stores the final sequence state: slot key, transform/filter derivatives, last frame, hit/miss streaks, and frame signature. The slot key hashes the target and source embeddings, preventing reuse after an assignment change.

## MediaCache: known current issue

`MediaCache._atomic_savez()` currently builds a temporary path as `target.tmp` and passes it to `np.savez()`. NumPy appends `.npz`, producing `target.tmp.npz`, while `os.replace()` looks for `target.tmp`. The destination cache file may therefore never be created.

In addition, `load_face()` exists but `store_face()` does not, and `ThumbnailLoader` does not persist face embeddings/crops after a cache miss. Treat face disk caching as nonfunctional until the P0 backlog item has a passing regression test.

## Atomic write and invalidation rules

1. Durable files must be written to a temporary file in the **same directory** and committed with `os.replace()`.
2. For APIs that append extensions, such as `np.savez`, use an open file handle or a temporary name with the correct extension; tests must verify the destination exists.
3. Cache keys must include every input that affects output; never use the current timestamp as cache identity.
4. Parsers must tolerate missing, corrupt, or older-version files and fall back safely.
5. Do not delete checkpoints or parts while paused or failed; clean them only after QC passes or after an explicit user action.
6. Persisted embeddings must include a source-file fingerprint and enough detector/recognizer metadata for correct invalidation.
