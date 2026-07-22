# Performance backlog đã đối chiếu code

Đây là các thay đổi **chưa được xem là hoàn tất**. Ưu tiên dựa trên ROI, độ đúng và phạm vi rủi ro. Không gộp tất cả vào một commit vì rất khó xác định regression.

## Thứ tự đề xuất

| Ưu tiên | Hạng mục | ROI dự kiến | Rủi ro |
|---|---|---:|---:|
| P0 | Sửa `MediaCache` atomic save + thêm face cache | Cao, sửa bug thật | Thấp |
| P1 | ArcFace direct ROI warp + batch faces | Rất cao khi frame lớn/nhiều face | Trung bình-cao |
| P1 | Scrub debounce/latest-only + single emit | Cao cho UX | Trung bình |
| P1 | Manual FFmpeg dùng rawvideo | Cao khi record | Trung bình |
| P1 | Worker tuần tự cho task nặng từ GUI | Cao cho responsiveness | Trung bình-cao |
| P2 | Reuse RetinaFace output buffers | Trung bình | Trung bình |
| P2 | SCRFD GPU postprocess/cache anchors | Cao nếu dùng SCRFD | Cao |
| P2 | Quick wins timer/cache/slot cleanup | Nhỏ-trung bình | Thấp |

## P0 — MediaCache hoạt động đúng và cache source face

### Hiện trạng đã xác nhận

- `MediaCache._atomic_savez()` tạo `tmp = target + '.tmp'` rồi gọi `np.savez(tmp, ...)`.
- NumPy tạo file `*.tmp.npz`, còn `os.replace()` tìm `*.tmp`; destination không xuất hiện.
- Có `load_face()` nhưng không có `store_face()`.
- `ThumbnailLoader` đọc face cache nhưng sau cache miss chỉ decode thumbnail, không lưu embedding/crop.
- `MainWindow._source_face_embeddings` chỉ cache embedding trong RAM của phiên chạy.

### Thiết kế đích

- Ghi `np.savez()` vào binary file handle đã mở, hoặc dùng temp path kết thúc `.npz` và replace đúng tên.
- Thêm `store_face(path, fingerprint, thumbnail, embedding, metadata)`.
- Key/fingerprint tối thiểu: absolute path, size, `mtime_ns`, detector, detect score, input size, recognizer/model version.
- Tách thumbnail-only khỏi embedding-ready để gallery không bắt buộc load model.
- Source embedding worker ghi cache sau detect/recognize thành công; corrupt entry tự tính lại.

### Nghiệm thu

- Round-trip tạo đúng destination và không còn orphan `*.tmp.npz`.
- Restart app rồi chọn cùng source không gọi detector/ArcFace lại.
- Sửa ảnh hoặc config recognition làm cache miss.
- Hai process ghi cùng key không tạo file nửa vời.

## P1 — ArcFace direct ROI warp và batch

### Hiện trạng đã xác nhận

`Models.recognize()` tính similarity transform rồi gọi `v2.functional.affine()` trên **toàn bộ frame**, sau đó mới crop vùng `dim*112 × dim*112`. Mỗi face chạy session riêng theo tile trong vòng lặp `dim × dim`; output `(1, 512)` cũng cấp phát mỗi call.

Ở 1920×1080 so với 112×112, warp toàn frame xử lý xấp xỉ 165 lần số pixel của ROI cho mỗi face. Tỷ lệ này chỉ mô tả pixel count, không phải cam kết FPS.

### Thiết kế đích

- Từ transform detector→ArcFace, dựng affine grid trực tiếp cho output 112×112.
- Dùng `torch.nn.functional.grid_sample()` hoặc kernel tương đương trên source frame, không materialize warped full frame.
- Gom mọi face của cùng frame thành batch `N×3×112×112` và chạy ArcFace một lần nếu session/profile hỗ trợ dynamic batch.
- Reuse input/output buffer theo batch capacity; có fallback batch=1 cho TRT engine static.
- Giữ đúng interpolation, coordinate convention, BGR và normalization của đường cũ.

### Thứ tự triển khai an toàn

1. Thêm helper direct crop batch nhưng vẫn inference từng face.
2. Golden-test crop/embedding với đường cũ trên center, edge, scale và rotated face.
3. Sau khi alignment tương đương mới bật batch ArcFace.
4. Benchmark detector riêng, recognize riêng và end-to-end.

### Nghiệm thu

- Crop sai khác trong tolerance đã định; cosine embedding đường mới/đường cũ đủ cao trên tập pose.
- Không cắt mất mặt sát mép; padding giống đường cũ.
- Batch 1 và batch N trả đúng thứ tự detection.
- VRAM không tăng không giới hạn; throughput tăng có số đo ở 720p/1080p/4K.

## P1 — Scrub debounce và latest-only

### Hiện trạng đã xác nhận

- VM dùng `deque(maxlen=5)` và scrub thread `popleft()` FIFO; mọi request còn lại đều decode/swap.
- Coordinator không coalesce event.
- `ParameterSlider._on_entry_committed()` có thể emit từ `setValue()` rồi emit lại vì điều kiện sau set luôn đúng.
- `ParameterSlider.set(request_frame=True)` cũng có thể vừa nhận `valueChanged`, vừa emit explicit.

### Thiết kế đích

- GUI debounce single-shot 16–33 ms cho drag.
- VM giữ duy nhất request pending mới nhất; frame đang xử lý có generation id và không publish nếu stale.
- Mouse release/entry commit bắt buộc publish final frame.
- Sửa slider: so `old_pos` trước `setValue()`; chỉ explicit emit khi position không đổi hoặc block signal và emit đúng một nơi.

### Nghiệm thu

- 100 event drag nhanh dẫn tới tối đa một in-flight + một pending.
- Preview cuối cùng luôn là frame user thả.
- Một entry commit/`set(..., request_frame=True)` emit đúng một lần.
- Seek/stop không deadlock temporal coordinator.

## P1 — Manual FFmpeg rawvideo

### Hiện trạng đã xác nhận

- Auto Render đã mở FFmpeg với `-f rawvideo -pixel_format rgb24 -video_size ...` và ghi bytes contiguous.
- Manual FFMPEG dùng input pipe không khai báo rawvideo; mỗi frame tạo `PIL.Image` và encode BMP vào stdin.

### Thiết kế đích

- Tách helper tạo FFmpeg rawvideo encoder dùng chung cho manual và Auto Render.
- Manual ghi `np.ascontiguousarray(image, dtype=np.uint8).tobytes()`.
- Giữ mux audio gốc sau encode và semantics start/stop marker.
- Codec policy có thể giữ `libx264` cho manual trước; NVENC là thay đổi riêng.

### Nghiệm thu

- Không còn `Image.fromarray(...).save(..., 'BMP')` trong record hot path.
- Output RGB đúng màu, resolution/FPS/frame count/duration đúng.
- Broken pipe đóng record sạch, không treo GUI.
- Audio mux và đoạn record theo playhead không lệch.

## P1 — Đưa task nặng khỏi GUI thread

### Hiện trạng đã xác nhận

- `_on_find_faces()` gọi detect/recognize đồng bộ.
- `_on_source_face_selection_changed()` tuần tự tính embedding ảnh chưa cache.
- Chọn target emit signal cùng thread vào `VideoManager.load_target_video()`, nơi tạo `MediaPlayer` và lấy first frame.
- Auto scan và QC đã là mẫu tốt: `QRunnable` + signal.

### Thiết kế đích

- Một serialized GPU task queue cho preload/find/source embedding/open-first-frame để không tranh session/VRAM.
- Mỗi request có generation; đổi media/selection làm result cũ bị discard.
- UI hiển thị busy/progress và vẫn xử lý repaint/cancel.
- Tách open metadata/decode first frame khỏi commit state: worker chuẩn bị, GUI/VM commit nếu generation còn đúng.

### Nghiệm thu

- Window kéo/resize được trong cold model load và chọn nhiều ảnh.
- Chọn video A rồi B nhanh không publish first frame của A sau B.
- Không có hai GPU inference job loại này chạy đồng thời.
- Cancel/exception luôn release busy state.

## P2 — Reuse RetinaFace output buffers

### Hiện trạng đã xác nhận

RetinaFace đã có thread-local input letterbox buffer và cache anchor. Tuy nhiên mỗi `detect_retinaface()` vẫn tạo 9 CUDA tensor output: score/bbox/kps cho stride 8/16/32.

### Thiết kế đích

- Thread-local output cache theo input `(H,W)`, dtype và device.
- Bind ORT vào buffer cache; consumer hoàn tất trước khi buffer cùng worker bị reuse.
- Giới hạn số profile size được giữ hoặc clear khi unload/backend đổi.

### Nghiệm thu

- Steady-state không còn 9 `torch.empty` mỗi frame.
- Output/NMS bitwise hoặc tolerance-equivalent.
- Per-thread session 1–5 worker không ghi đè buffer nhau.

## P2 — SCRFD giữ postprocess trên GPU

### Hiện trạng đã xác nhận

- SCRFD gọi `io_binding.copy_outputs_to_cpu()`.
- `center_cache = {}` được tạo lại trong mỗi detect.
- Sau CPU decode/sort, boxes/scores lại được upload CUDA để `torchvision.ops.nms()`.
- Preprocess cũng cấp phát letterbox tensor mới và permute/materialize nhiều bước.

### Thiết kế đích

- Bind output vào CUDA tensor như RetinaFace.
- Cache anchor center lâu dài theo `(input_h,input_w,stride,device,dtype)`.
- Decode bbox/kps, threshold, sort và NMS trên GPU; chỉ copy final small kps về nơi thật sự cần.
- Reuse preprocess/output buffer thread-local.

### Nghiệm thu

- Không có full output GPU→CPU→GPU round-trip.
- Detection/kps tương đương đường cũ trên bộ regression.
- Không tăng VRAM theo số frame hoặc đổi input size lặp lại.

## P2 — Quick wins tài nguyên

### Coordinator timer

`Coordinator._tick()` hiện chỉ telemetry queue/VRAM nhưng `QTimer` interval vẫn là 0 ms. Đổi 50–100 ms là đủ cho HUD và giảm wake-up CPU; frame delivery đã push-based.

### Stop và CUDA allocator

`play_video('stop')` và `stop_from_gui` gọi `torch.cuda.empty_cache()` mỗi lần. Stop thường chỉ nên abort/dừng queue và bỏ reference. Chỉ empty cache khi explicit unload, backend rebuild hoặc recovery có lý do.

### ProcessedFrame lifetime

Sau publish preview, slot reset status/thread/frame/time/PTS nhưng chưa đặt `ProcessedFrame` về `None`/`[]`. Record path cũng vậy. Xóa reference ngay sau consumer cuối cùng để tensor/CPU frame được thu hồi sớm.

### Nghiệm thu chung

- Idle CPU giảm có số đo; VRAM indicator vẫn cập nhật trong ≤100 ms.
- Play/stop lặp lại không tạo memory growth.
- Peak allocated/reserved memory không xấu hơn và stop nhanh hơn.
- Slot không giữ tensor của frame đã publish/write.

## Cách chia commit

1. `fix(cache): make npz writes atomic and persist face embeddings`
2. `perf(scrub): debounce and keep only the latest request`
3. `perf(record): share rawvideo ffmpeg writer`
4. `perf(ui): serialize heavy media and face tasks off the GUI thread`
5. `perf(arcface): crop aligned ROIs directly`
6. `perf(arcface): batch recognition where backend supports it`
7. `perf(detector): reuse retinaface outputs`
8. `perf(scrfd): keep postprocess on CUDA`
9. `perf(runtime): reduce idle polling and release frame references`

Mỗi commit cần test/benchmark riêng để dễ bisect khi chất lượng swap hoặc stability thay đổi.
