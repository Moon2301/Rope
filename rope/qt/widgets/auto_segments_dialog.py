from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from rope.AutoSegments import AutoSegment


class AutoSegmentsDialog(QDialog):
    scan_requested = Signal(int, int, int)  # stride, gap, padding
    render_requested = Signal(object)
    seek_requested = Signal(int)
    cancel_scan_requested = Signal()
    cancel_render_requested = Signal()
    resume_render_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Auto Character Segments")
        self.resize(760, 480)
        self._segments: list[AutoSegment] = []

        root = QVBoxLayout(self)
        info = QLabel(
            "Quét nhân vật đang chọn trong Found Faces. Kiểm tra và sửa các "
            "khoảng bên dưới trước khi render."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        options = QHBoxLayout()
        form = QFormLayout()
        self.stride = QSpinBox(); self.stride.setRange(1, 300); self.stride.setValue(10)
        self.gap = QSpinBox(); self.gap.setRange(0, 900); self.gap.setValue(20)
        self.padding = QSpinBox(); self.padding.setRange(0, 300); self.padding.setValue(10)
        form.addRow("Sample mỗi (frame)", self.stride)
        form.addRow("Nối khoảng hở (frame)", self.gap)
        form.addRow("Padding (frame)", self.padding)
        options.addLayout(form)
        options.addStretch()
        self.scan_button = QPushButton("Quét video")
        self.scan_button.clicked.connect(self._request_scan)
        options.addWidget(self.scan_button)
        root.addLayout(options)

        self.status = QLabel("Chưa quét")
        root.addWidget(self.status)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Duyệt", "Frame bắt đầu", "Frame kết thúc", "Độ tin cậy",
            "Đầu", "Giữa", "Cuối",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.cellDoubleClicked.connect(self._seek_row)
        root.addWidget(self.table, stretch=1)

        edit_row = QHBoxLayout()
        split_btn = QPushButton("Split tại playhead")
        split_btn.clicked.connect(self._split_selected)
        merge_btn = QPushButton("Merge hàng đã chọn")
        merge_btn.clicked.connect(self._merge_selected)
        self.cancel_scan_button = QPushButton("Hủy scan")
        self.cancel_scan_button.setEnabled(False)
        self.cancel_scan_button.clicked.connect(self.cancel_scan_requested)
        edit_row.addWidget(split_btn); edit_row.addWidget(merge_btn)
        edit_row.addStretch(); edit_row.addWidget(self.cancel_scan_button)
        root.addLayout(edit_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.render_button = buttons.addButton("Render các đoạn đã duyệt", QDialogButtonBox.AcceptRole)
        self.cancel_render_button = buttons.addButton("Hủy render", QDialogButtonBox.ActionRole)
        self.resume_button = buttons.addButton("Tiếp tục render", QDialogButtonBox.ActionRole)
        self.cancel_render_button.clicked.connect(self.cancel_render_requested)
        self.resume_button.clicked.connect(self.resume_render_requested)
        self.render_button.setEnabled(False)
        self.render_button.clicked.connect(self._request_render)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _request_scan(self) -> None:
        self.scan_button.setEnabled(False)
        self.cancel_scan_button.setEnabled(True)
        self.render_button.setEnabled(False)
        self.status.setText("Đang quét…")
        self.scan_requested.emit(self.stride.value(), self.gap.value(), self.padding.value())

    def set_progress(self, current: int, total: int) -> None:
        self.status.setText(f"Đang quét: {current}/{total} frame mẫu")

    def set_error(self, message: str) -> None:
        self.scan_button.setEnabled(True)
        self.cancel_scan_button.setEnabled(False)
        self.status.setText(message)

    def set_segments(self, segments: list[AutoSegment], thumbnails: dict | None = None,
                     *, cached: bool = False) -> None:
        self._segments = list(segments)
        self.table.setRowCount(len(segments))
        for row, segment in enumerate(segments):
            check = QCheckBox(); check.setChecked(segment.approved)
            check.setStyleSheet("margin-left: 16px")
            self.table.setCellWidget(row, 0, check)
            self.table.setItem(row, 1, QTableWidgetItem(str(segment.start_frame)))
            self.table.setItem(row, 2, QTableWidgetItem(str(segment.end_frame)))
            confidence = QTableWidgetItem(f"{segment.confidence:.1f}%")
            confidence.setFlags(confidence.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, confidence)
            for column, label in zip((4, 5, 6), ("start", "middle", "end")):
                image = (thumbnails or {}).get(row, {}).get(label)
                thumb = QLabel(); thumb.setAlignment(Qt.AlignCenter)
                if image is not None and getattr(image, "ndim", 0) == 3:
                    h, w, _ = image.shape
                    qimage = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                    thumb.setPixmap(QPixmap.fromImage(qimage))
                self.table.setCellWidget(row, column, thumb)
            self.table.setRowHeight(row, 72)
        self.table.resizeColumnsToContents()
        self.scan_button.setEnabled(True)
        self.cancel_scan_button.setEnabled(False)
        self.render_button.setEnabled(bool(segments))
        source = "cache" if cached else "video"
        self.status.setText(f"Tìm thấy {len(segments)} đoạn từ {source}. Double-click để xem.")

    def _read_segments(self) -> list[AutoSegment]:
        result = []
        for row in range(self.table.rowCount()):
            try:
                start = int(self.table.item(row, 1).text())
                end = int(self.table.item(row, 2).text())
                confidence = float(self.table.item(row, 3).text().rstrip("%"))
            except (AttributeError, ValueError):
                continue
            check = self.table.cellWidget(row, 0)
            result.append(AutoSegment(start, end, confidence, bool(check.isChecked())))
        return result

    def _seek_row(self, row: int, _column: int) -> None:
        segments = self._read_segments()
        if 0 <= row < len(segments):
            self.seek_requested.emit(segments[row].start_frame)

    def _request_render(self) -> None:
        segments = self._read_segments()
        if any(item.approved for item in segments):
            self.render_requested.emit(segments)

    def _selected_rows(self) -> list[int]:
        return sorted({index.row() for index in self.table.selectionModel().selectedRows()})

    def _split_selected(self) -> None:
        rows = self._selected_rows()
        if len(rows) != 1:
            self.status.setText("Chọn đúng một segment để split.")
            return
        row = rows[0]
        segment = self._read_segments()[row]
        # seek_requested keeps the dialog decoupled; parent exposes playhead
        # through set_playhead() before this action.
        frame = int(getattr(self, "_playhead", segment.start_frame))
        if not (segment.start_frame < frame <= segment.end_frame):
            self.status.setText("Playhead phải nằm bên trong segment.")
            return
        items = self._read_segments()
        items[row:row + 1] = [
            AutoSegment(segment.start_frame, frame - 1, segment.confidence, segment.approved),
            AutoSegment(frame, segment.end_frame, segment.confidence, segment.approved),
        ]
        self.set_segments(items)

    def set_playhead(self, frame: int) -> None:
        self._playhead = int(frame)

    def _merge_selected(self) -> None:
        rows = self._selected_rows()
        if len(rows) < 2:
            self.status.setText("Chọn ít nhất hai segment để merge.")
            return
        items = self._read_segments()
        chosen = [items[row] for row in rows]
        merged = AutoSegment(
            min(item.start_frame for item in chosen),
            max(item.end_frame for item in chosen),
            sum(item.confidence for item in chosen) / len(chosen),
            any(item.approved for item in chosen),
        )
        first = rows[0]
        items = [item for index, item in enumerate(items) if index not in rows]
        items.insert(first, merged)
        self.set_segments(items)
