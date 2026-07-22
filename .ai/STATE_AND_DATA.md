# State, dữ liệu và cache

## Nguồn dữ liệu bền vững

| Dữ liệu | Vị trí | Format/owner |
|---|---|---|
| Folder, geometry, backend preference | `data.json` ở working directory | `qt.settings.Settings` |
| Parameter snapshot do user lưu | `saved_parameters.json` | `parameters_migration.py` / `MainWindow` |
| Auto Job | `QStandardPaths.AppDataLocation/automation_jobs/<job_id>/job.json` | `AutomationJobStore` |
| Scan manifest | `QStandardPaths.CacheLocation/auto_segments/<key>.manifest.json` | `_AutoScanWorker` + `AutoSegments` |
| Scan thumbnails | Cùng cache, thư mục `<key>.thumbs/` | `_AutoScanWorker` |
| Media thumbnail cache | `./cache/*.npz` | `MediaCache` / `ThumbnailLoader` |
| Auto Render manifest | `<output>/.<video>_<job_id>_auto_render.json` | `VideoManager` |
| Auto Render parts | `<output>/.rope_<video>_<job_id>.parts/` | `VideoManager` |

Nếu `AppDataLocation` rỗng, job store fallback về `./.rope_app_data/automation_jobs`.

## Settings

`Settings` chấp nhận key thiếu và dùng default để giữ tương thích với `data.json` cũ. Key chính:

- `source videos`, `source faces`, `saved videos`, `merged_embeddings_file`, `models_folder`.
- `dock_win_geom`, `splitter_main_sizes`, `splitter_left_sizes`, `splitter_center_sizes`.
- `params_collapsed`.
- `model_backends`: map model attribute → `trt` hoặc `onnx`; thiếu key là auto.

Thêm setting mới phải có default an toàn và loader phải chịu được file cũ/hỏng.

## Found Face slot

Shape logic đang được truyền giữa UI và VM:

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

`Embedding` là target identity. `AssignedEmbedding` là source identity sẽ được swap vào. Không suy ra assignment chỉ từ highlight UI.

## Auto Job manifest

`AutomationJob.version` hiện là `1`. Manifest chứa:

- Identity: `job_id`, timestamp, `video_path`, `video_fingerprint`.
- Input snapshot: metadata, target/source embedding, source labels.
- Config snapshot: `scan_config`, `render_params`.
- Review: segment và `user_confirmed`.
- Recovery: `scan_checkpoint`, `render_checkpoint`, `paused_from`, `retry_from`.
- Result: `final_output`, `qc`, `error`.

Job JSON ghi nguyên tử bằng file tạm cùng thư mục rồi `os.replace()`. `AutomationController` là nơi duy nhất hợp lệ để transition state; `MainWindow` điều phối worker và cập nhật controller.

## Scan cache

`SCAN_CACHE_VERSION = 3`. Cache key hash từ:

- Video path đã resolve, size và `mtime_ns`.
- Target embedding.
- Toàn bộ scan config: algorithm version, detector, detect score, input size, identity threshold, interval, refine, chunk, near-hit, gap và padding.

Manifest lưu `completed_chunks` và từng chunk:

- `start_frame`, `end_frame`, `stage`.
- `coarse_cursor`, `refine_cursor`.
- `coarse_hits`, `near_hits`, `hits` theo absolute frame.

Thay tracking/restorer/mask không invalid scan cache vì chúng chỉ ảnh hưởng render. Thumbnail sinh sau khi segment đã merge và cache tách khỏi inference result.

## Render manifest

Render manifest hiện version `3`, gồm:

- Fingerprint input, approved ranges, render params.
- `part_frames`, danh sách part đã xong, `next_frame`, work dir và final output.
- Encoder (`h264_nvenc` hoặc `libx264`) và stage.
- `tracking_version` cùng `tracking_checkpoint` tại ranh giới part.

Job cũ không có `TemporalTrackingSwitch` được resume với tracking tắt để tránh nửa đầu/nửa sau khác hành vi. Crash giữa part dùng checkpoint cuối part trước và render lại part chưa hoàn tất.

## Temporal tracking checkpoint

`TRACKING_VERSION = 1`. Checkpoint chỉ restore nếu version hợp lệ và chứa state cuối sequence: slot key, transform/filter derivative, frame cuối, hit/miss streak và frame signature. Slot key là hash target + source embedding nên assignment đổi sẽ không dùng nhầm track cũ.

## MediaCache: hiện trạng cần lưu ý

`MediaCache._atomic_savez()` hiện tạo temp path dạng `target.tmp` rồi gọi `np.savez(temp, ...)`. NumPy tự thêm `.npz`, thành `target.tmp.npz`, trong khi `os.replace()` tìm `target.tmp`. Vì vậy cache media có thể không tạo được file đích.

Ngoài ra `load_face()` có nhưng chưa có `store_face()`, và `ThumbnailLoader` không lưu face embedding/crop sau cache miss. Hai vấn đề này là P0 trong performance backlog; không được coi face disk cache là hoạt động cho tới khi có test chứng minh.

## Quy tắc atomic và invalidation

1. File bền vững phải ghi temp trong **cùng thư mục** rồi `os.replace()`.
2. Với API tự thêm extension như `np.savez`, mở file handle hoặc dùng temp name có extension đúng; test phải xác nhận file đích tồn tại.
3. Cache key phải chứa mọi input ảnh hưởng output; không dùng timestamp hiện tại làm cache identity.
4. Parser phải chịu được file thiếu/hỏng/version cũ và fallback an toàn.
5. Không xóa checkpoint/part khi pause hoặc fail; chỉ cleanup sau QC pass hoặc thao tác user rõ ràng.
6. Embedding lưu ra đĩa phải có fingerprint source file và thông tin detector/recognizer cần thiết để invalid đúng.
