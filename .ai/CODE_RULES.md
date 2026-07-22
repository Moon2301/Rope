# Quy tắc sửa code

Các quy tắc này rút từ kiến trúc hiện tại. Mục tiêu là giữ đúng UI thread, thứ tự frame, GPU memory và khả năng resume.

## 1. Ranh giới module

- UI/dialog ở `rope/qt`; model kernel ở `Models`; media lifecycle ở `MediaPlayer/VideoManager`.
- Logic state machine thuần Python ở `Automation.py`, không import Qt, torch hoặc CUDA.
- Logic scan/group/cache thuần dùng lại được đặt ở `AutoSegments.py`; UI worker chỉ orchestrate và emit progress.
- Không thêm model inference vào widget event handler. Handler chỉ validate, snapshot và dispatch worker.
- `Bus` chỉ chứa signal có contract rõ; không dùng signal như nơi lưu state.

## 2. GUI thread

- Chỉ GUI thread được mutate widget.
- Detect, ArcFace, model load, mở/decode video, FFmpeg probe dài và I/O hàng loạt phải chạy worker.
- Worker trả kết quả qua Qt signal; result phải mang generation/job id để UI bỏ result cũ.
- Không chờ `Future.result()`, `Thread.join()` hoặc subprocess dài trên GUI thread.
- Timer idle không được spin ở 0 ms nếu chỉ làm telemetry; chọn interval theo nhu cầu hiển thị.

## 3. Frame order và concurrency

- Mọi frame dispatch vào temporal sequence phải advance coordinator, kể cả pass-through, không detection và exception.
- Worker phải hoàn tất/fail process slot trong `finally` hoặc error path; không để `Status='started'` vô hạn.
- Seek, stop, đổi video/assignment/orientation phải tăng generation hoặc abort sequence để waiter cũ thoát.
- Presentation/record phải lấy frame hoàn tất theo `FrameNumber`, không theo thứ tự thread xong.
- Shared mutable scratch/I/O binding không dùng giữa worker; dùng thread-local hoặc khóa có phạm vi nhỏ.
- Không giữ tensor frame trong slot sau khi frame đã publish/write xong.

## 4. Tensor, màu và shape

- Decode/preview contract ưu tiên RGB `uint8`, shape H×W×3.
- Model path thường dùng CUDA tensor C×H×W; convert đúng một lần ở biên.
- Ghi FFmpeg `rawvideo rgb24` phải dùng contiguous `uint8` H×W×3.
- OpenCV đọc/ghi mặc định BGR; conversion phải được ghi rõ ngay tại biên.
- Tránh GPU→CPU→GPU trong hot path. Chỉ `.cpu().numpy()` ở output/file/UI fallback thật sự cần.
- Buffer shape cố định theo model/input profile nên cache theo `(device, dtype, shape, worker)`.
- Không dùng scalar torch/CUDA cho toán học metadata như FPS, ratio hoặc size.

## 5. Model và CUDA lifecycle

- Lazy-load qua helper hiện có; không tạo `InferenceSession` trực tiếp trong UI.
- Tôn trọng backend preference và TRT profile.
- I/O binding output có lifetime ít nhất đến khi inference và consumer hoàn tất.
- Cache buffer phải bounded và được invalid khi device/model/input profile đổi.
- Không gọi `torch.cuda.empty_cache()` trong thao tác thường xuyên như Stop; chỉ dùng ở explicit unload/recovery sau khi đã bỏ mọi reference.
- Thêm optimization GPU phải đo warmup, throughput, VRAM peak và correctness trên cùng input.

## 6. Recognition, matching và tracking

- ArcFace luôn dùng landmark detector thô trừ khi có migration được test rõ.
- Smoothing landmark chỉ ảnh hưởng alignment/swap, không thay identity embedding.
- Association nhiều mặt phải one-to-one; một detection không gán cho hai target slot.
- Track mới phải qua threshold; hysteresis của track đang sống không được mở rộng quá rule hiện tại mà không có test.
- Prediction miss giữ bảo thủ một frame; scene cut/second miss reset.
- Scan cache không phụ thuộc Temporal Stabilization.

## 7. Parameter và compatibility

- Tham số mới khai báo trong `parameters.py`, có default và scope rõ.
- Snapshot render phải chứa mọi parameter ảnh hưởng output, gồm tracking switch/version.
- Loader phải chấp nhận snapshot cũ; behavior fallback phải deterministic.
- Đổi cache/job/render schema phải tăng version tương ứng.
- Không silently reinterpret một field cũ với nghĩa mới.

## 8. Cache và file

- Atomic write: temp file cùng directory → flush/close → `os.replace()`.
- Test atomic writer bằng cách kiểm tra destination thật sự tồn tại và đọc lại được.
- Cache key gồm fingerprint input, model/config và algorithm version.
- Cache miss/corruption là đường bình thường: bỏ entry và tính lại, không crash app.
- Không ghi absolute path riêng của developer vào source/docs/default.
- Part/checkpoint chỉ bị xóa sau QC pass; fail/pause giữ để resume.

## 9. Error handling và UX

- Log phải nêu stage, model/backend và frame/part/chunk khi có thể.
- Exception trong worker phải emit failed và release waiter/slot.
- Không bắt `Exception` rồi im lặng nếu lỗi làm sai output; ít nhất log hoặc chuyển job `FAILED`.
- User-facing message nói hành động tiếp theo: gán source, chọn output, resume/retry, hoặc đổi backend.
- Preflight chặn sớm input thiếu; render không được bắt đầu trước `confirm_segments()`.

## 10. Performance workflow

Trước khi merge tối ưu hot path:

1. Lưu baseline: video, resolution, FPS, worker count, backend, warmup và parameter snapshot.
2. Profile theo stage (`detect`, `recognize`, `swap`, preview/write), không chỉ nhìn FPS tổng.
3. Kiểm tra output số: embedding cosine, landmark/alignment, frame count, duration và audio sync.
4. Đo 1–5 worker và VRAM peak trên RTX 3060 12 GB.
5. Có fallback khi dependency/codec/backend không khả dụng.

Không chấp nhận tối ưu chỉ vì allocation ít hơn nếu output lệch hoặc worker deadlock.

## 11. Test và kiểm tra trước commit

Các unit test hiện ở `rope/qt/tests`:

```powershell
python -m unittest discover -s rope/qt/tests -p "test_*.py"
```

Kiểm tra tối thiểu theo loại thay đổi:

- State/cache: unit test transition, version, invalidation và atomic round-trip.
- Tracking: out-of-order, scene cut, one-frame miss, angle wrap, checkpoint resume.
- UI signal: một thao tác emit đúng một lần, stale generation không cập nhật UI.
- Record/render: output đọc được, frame count/duration/audio đúng; NVENC và fallback x264.
- GPU kernel: so output với đường cũ trên nhiều pose/size và đo peak memory.

Sau đó chạy smoke UI phù hợp trong `rope/qt/tests/`. Test thật model/GPU phải được ghi rõ là smoke/integration, không trộn vào unit test thuần.

## 12. Checklist review nhanh

- [ ] Widget chỉ được chạm trên GUI thread.
- [ ] Không có queue không giới hạn hoặc waiter không có abort.
- [ ] Frame order, generation và error path đều hoàn tất.
- [ ] Tensor màu/shape/device được ghi rõ.
- [ ] Cache/schema version và invalidation đúng.
- [ ] Pause/fail không làm mất checkpoint.
- [ ] Test regression và benchmark tương xứng với rủi ro.
- [ ] `.ai` được cập nhật nếu flow/contract thay đổi.
