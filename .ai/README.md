# Tài liệu kỹ thuật Rope

Thư mục này là bản đồ dành cho người và AI agent khi sửa Rope. Nội dung mô tả **code đang tồn tại trên nhánh hiện tại**; các thay đổi chưa làm được ghi riêng trong [PERFORMANCE_BACKLOG.md](PERFORMANCE_BACKLOG.md), không trộn với hành vi runtime.

## Đọc theo nhu cầu

- [ARCHITECTURE.md](ARCHITECTURE.md): entrypoint, module, ownership, thread và GPU pipeline.
- [FLOWS.md](FLOWS.md): flow startup, chọn media, nhận diện/gán mặt, preview, scrub, record và Auto Job.
- [STATE_AND_DATA.md](STATE_AND_DATA.md): state machine, manifest, cache, tham số và quy tắc invalidation.
- [CODE_RULES.md](CODE_RULES.md): quy tắc bắt buộc khi sửa code hiện tại.
- [PERFORMANCE_BACKLOG.md](PERFORMANCE_BACKLOG.md): các hotspot đã đối chiếu với code, thứ tự ưu tiên và tiêu chí nghiệm thu.

## Phạm vi hiện tại

- UI chính là Qt/PySide6; entrypoint là [`Rope.py`](../Rope.py).
- Một `VideoManager` giữ media session, playback, scrub, swap và record.
- Model được lazy-load qua `Models`; backend có thể là ONNX Runtime CUDA hoặc TensorRT EP.
- Auto Job xử lý một video, một target slot và một source embedding, bắt buộc user duyệt segment trước render.
- Temporal stabilization dùng 5 landmark và chỉ áp dụng cho video preview/manual record/Auto Job khi switch bật.
- Scan segment và render là hai pha độc lập: tracking là render parameter, không làm invalid scan cache.

## Quy ước cập nhật tài liệu

Khi thay đổi kiến trúc hoặc format dữ liệu:

1. Sửa tài liệu cùng commit với code.
2. Dẫn chiếu theo tên class/hàm thay vì số dòng vì line number thay đổi nhanh.
3. Gắn rõ một mục là `Hiện trạng`, `Đề xuất` hoặc `Đã hoàn tất`.
4. Nếu đổi manifest/cache schema, tăng version và ghi đường nâng cấp hoặc hành vi fallback.
5. Không ghi secret, đường dẫn máy cá nhân hoặc model binary vào `.ai`.
