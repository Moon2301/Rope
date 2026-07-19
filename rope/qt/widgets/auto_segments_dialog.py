from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
    QDoubleSpinBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from rope.AutoSegments import AutoSegment


class AutoSegmentsDialog(QDialog):
    """Review gate and progress surface for a persistent Auto Job."""

    scan_requested = Signal(float, int, int)  # interval seconds, gap, padding
    render_requested = Signal(object)
    seek_requested = Signal(int)
    preview_segment_requested = Signal(int, int)
    pause_job_requested = Signal()
    resume_job_requested = Signal()
    retry_job_requested = Signal()
    open_output_requested = Signal()
    segments_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Auto Job — Single Character")
        self.resize(940, 600)
        self._segments: list[AutoSegment] = []
        self._fps = 1.0
        self._job_state = "NEW"
        self._loading_segments = False

        root = QVBoxLayout(self)
        info = QLabel(
            "Auto Job sẽ preflight và scan trước, sau đó bắt buộc dừng ở đây "
            "để bạn duyệt segment trước khi thay mặt."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        options = QHBoxLayout()
        form = QFormLayout()
        self.sample_interval = QDoubleSpinBox()
        self.sample_interval.setRange(0.1, 5.0)
        self.sample_interval.setSingleStep(0.1)
        self.sample_interval.setDecimals(1)
        self.sample_interval.setValue(0.5)
        self.sample_interval.setSuffix(" s")
        self.gap = QSpinBox(); self.gap.setRange(0, 900); self.gap.setValue(20)
        self.padding = QSpinBox(); self.padding.setRange(0, 300); self.padding.setValue(10)
        form.addRow("Coarse sample", self.sample_interval)
        form.addRow("Nối khoảng hở (frame)", self.gap)
        form.addRow("Padding (frame)", self.padding)
        options.addLayout(form)
        options.addStretch()
        self.scan_button = QPushButton("Start Auto Job")
        self.scan_button.clicked.connect(self._request_scan)
        options.addWidget(self.scan_button)
        root.addLayout(options)

        self.state_label = QLabel("READY")
        self.state_label.setStyleSheet("font-weight: bold; color: #d89b62;")
        self.status = QLabel("Chưa bắt đầu")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        root.addWidget(self.state_label)
        root.addWidget(self.status)
        root.addWidget(self.progress)

        self.summary = QLabel("0 segment")
        root.addWidget(self.summary)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Duyệt", "Frame bắt đầu", "Frame kết thúc", "Độ tin cậy",
            "Cần xem kỹ", "Đầu", "Giữa", "Cuối",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.table.cellDoubleClicked.connect(self._seek_row)
        self.table.cellChanged.connect(self._on_table_changed)
        root.addWidget(self.table, stretch=1)

        edit_row = QHBoxLayout()
        split_btn = QPushButton("Split tại playhead")
        split_btn.clicked.connect(self._split_selected)
        merge_btn = QPushButton("Merge hàng đã chọn")
        merge_btn.clicked.connect(self._merge_selected)
        preview_btn = QPushButton("Phát segment")
        preview_btn.clicked.connect(self._preview_selected)
        select_all_btn = QPushButton("Chọn tất cả")
        select_all_btn.clicked.connect(lambda: self._set_all_approved(True))
        select_none_btn = QPushButton("Bỏ tất cả")
        select_none_btn.clicked.connect(lambda: self._set_all_approved(False))
        edit_row.addWidget(split_btn)
        edit_row.addWidget(merge_btn)
        edit_row.addWidget(preview_btn)
        edit_row.addStretch()
        edit_row.addWidget(select_all_btn)
        edit_row.addWidget(select_none_btn)
        root.addLayout(edit_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.render_button = buttons.addButton(
            "Confirm & Render", QDialogButtonBox.AcceptRole
        )
        self.pause_button = buttons.addButton(
            "Pause / Cancel", QDialogButtonBox.ActionRole
        )
        self.resume_button = buttons.addButton(
            "Resume Previous Job", QDialogButtonBox.ActionRole
        )
        self.retry_button = buttons.addButton("Retry", QDialogButtonBox.ActionRole)
        self.open_output_button = buttons.addButton(
            "Open Output", QDialogButtonBox.ActionRole
        )
        self.pause_button.clicked.connect(self.pause_job_requested)
        self.resume_button.clicked.connect(self.resume_job_requested)
        self.retry_button.clicked.connect(self.retry_job_requested)
        self.open_output_button.clicked.connect(self.open_output_requested)
        self.render_button.setEnabled(False)
        self.retry_button.setEnabled(False)
        self.open_output_button.setEnabled(False)
        self.render_button.clicked.connect(self._request_render)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _request_scan(self) -> None:
        self.scan_button.setEnabled(False)
        self.render_button.setEnabled(False)
        self.set_job_state("PREFLIGHT", "Đang kiểm tra job…")
        self.scan_requested.emit(
            self.sample_interval.value(), self.gap.value(), self.padding.value()
        )

    def set_scan_progress(self, data: dict) -> None:
        percent = float(data.get("percent", 0.0))
        chunk = int(data.get("chunk", 0))
        chunks = int(data.get("chunks", 0))
        stage = str(data.get("stage", "coarse"))
        eta = max(0, int(data.get("eta_seconds", 0)))
        self._job_state = "SCANNING"
        self.state_label.setText("SCANNING")
        self.scan_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.retry_button.setEnabled(False)
        self.progress.setValue(max(0, min(1000, int(percent * 10))))
        cache_note = " — cache hit" if stage == "cache" else ""
        self.status.setText(
            f"Scan {stage}: chunk {chunk}/{chunks} — {percent:.1f}% — "
            f"ETA {eta // 60:02d}:{eta % 60:02d}{cache_note}"
        )

    def set_render_progress(self, data: dict) -> None:
        percent = float(data.get("percent", 0.0))
        part = int(data.get("part", 0))
        parts = int(data.get("parts", 0))
        eta = max(0, int(data.get("eta_seconds", 0)))
        self._job_state = "RENDERING"
        self.state_label.setText("RENDERING")
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.retry_button.setEnabled(False)
        self.progress.setValue(max(0, min(1000, int(percent * 10))))
        self.status.setText(
            f"Render part {part}/{parts} — {percent:.1f}% — "
            f"ETA {eta // 60:02d}:{eta % 60:02d}"
        )

    def set_job_state(self, state: str, message: str = "", output: str | None = None) -> None:
        self._job_state = str(state)
        self.state_label.setText(self._job_state)
        if message:
            self.status.setText(message)
        self.scan_button.setEnabled(self._job_state in {"NEW", "COMPLETED"})
        self.pause_button.setEnabled(self._job_state in {"SCANNING", "RENDERING"})
        self.resume_button.setEnabled(self._job_state == "PAUSED")
        self.retry_button.setEnabled(self._job_state == "FAILED")
        self.render_button.setEnabled(
            self._job_state == "AWAITING_REVIEW" and bool(self._segments)
        )
        self.open_output_button.setEnabled(bool(output))
        if self._job_state == "COMPLETED":
            self.progress.setValue(1000)

    def set_error(self, message: str) -> None:
        self.set_job_state("FAILED", message)

    def set_segments(self, segments: list[AutoSegment], thumbnails: dict | None = None,
                     *, cached: bool = False, fps: float = 1.0) -> None:
        self._segments = list(segments)
        self._fps = max(0.001, float(fps))
        self._loading_segments = True
        self.table.setRowCount(len(segments))
        for row, segment in enumerate(segments):
            check = QCheckBox(); check.setChecked(segment.approved)
            check.setStyleSheet("margin-left: 16px")
            check.stateChanged.connect(self._on_review_changed)
            self.table.setCellWidget(row, 0, check)
            self.table.setItem(row, 1, QTableWidgetItem(str(segment.start_frame)))
            self.table.setItem(row, 2, QTableWidgetItem(str(segment.end_frame)))
            confidence = QTableWidgetItem(f"{segment.confidence:.1f}%")
            confidence.setFlags(confidence.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, confidence)
            review = QTableWidgetItem("; ".join(segment.review_reasons) if segment.review_required else "—")
            review.setFlags(review.flags() & ~Qt.ItemIsEditable)
            if segment.review_required:
                review.setForeground(QBrush(QColor("#f0c75e")))
            self.table.setItem(row, 4, review)
            for column, label in zip((5, 6, 7), ("start", "middle", "end")):
                image = (thumbnails or {}).get(row, {}).get(label)
                thumb = QLabel(); thumb.setAlignment(Qt.AlignCenter)
                if image is not None and getattr(image, "ndim", 0) == 3:
                    h, w, _ = image.shape
                    qimage = QImage(image.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                    thumb.setPixmap(QPixmap.fromImage(qimage))
                self.table.setCellWidget(row, column, thumb)
            self.table.setRowHeight(row, 72)
        self.table.resizeColumnsToContents()
        self._loading_segments = False
        source = "cache" if cached else "video"
        self.set_job_state(
            "AWAITING_REVIEW",
            f"Tìm thấy {len(segments)} segment từ {source}. Hãy duyệt trước khi render.",
        )
        self._update_summary()

    def _on_review_changed(self, *_args) -> None:
        self._update_summary()
        self._emit_segments_changed()

    def _on_table_changed(self, _row: int, column: int) -> None:
        if column in (1, 2):
            self._update_summary()
            self._emit_segments_changed()

    def _emit_segments_changed(self) -> None:
        if not self._loading_segments and self._job_state == "AWAITING_REVIEW":
            self.segments_changed.emit(self._read_segments())

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
            base = self._segments[row] if row < len(self._segments) else None
            result.append(AutoSegment(
                start, end, confidence, bool(check.isChecked()),
                hit_count=base.hit_count if base else 0,
                review_required=base.review_required if base else False,
                review_reasons=list(base.review_reasons) if base else [],
            ))
        return result

    def _update_summary(self, *_args) -> None:
        segments = self._read_segments()
        approved = [item for item in segments if item.approved]
        frames = sum(item.end_frame - item.start_frame + 1 for item in approved)
        seconds = frames / self._fps
        flagged = sum(item.review_required for item in approved)
        self.summary.setText(
            f"Đã chọn {len(approved)}/{len(segments)} segment — "
            f"{seconds / 60.0:.2f} phút sẽ swap — {flagged} segment cần xem kỹ"
        )
        self.render_button.setEnabled(
            self._job_state == "AWAITING_REVIEW" and bool(approved)
        )

    def _set_all_approved(self, value: bool) -> None:
        self._loading_segments = True
        for row in range(self.table.rowCount()):
            check = self.table.cellWidget(row, 0)
            if isinstance(check, QCheckBox):
                check.setChecked(bool(value))
        self._loading_segments = False
        self._update_summary()
        self._emit_segments_changed()

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

    def _preview_selected(self) -> None:
        rows = self._selected_rows()
        if len(rows) != 1:
            self.status.setText("Chọn đúng một segment để phát.")
            return
        segment = self._read_segments()[rows[0]]
        self.preview_segment_requested.emit(segment.start_frame, segment.end_frame)

    def _split_selected(self) -> None:
        rows = self._selected_rows()
        if len(rows) != 1:
            self.status.setText("Chọn đúng một segment để split.")
            return
        row = rows[0]
        segment = self._read_segments()[row]
        frame = int(getattr(self, "_playhead", segment.start_frame))
        if not (segment.start_frame < frame <= segment.end_frame):
            self.status.setText("Playhead phải nằm bên trong segment.")
            return
        items = self._read_segments()
        items[row:row + 1] = [
            AutoSegment(
                segment.start_frame, frame - 1, segment.confidence,
                segment.approved, review_required=True,
                review_reasons=["segment đã được split thủ công"],
            ),
            AutoSegment(
                frame, segment.end_frame, segment.confidence,
                segment.approved, review_required=True,
                review_reasons=["segment đã được split thủ công"],
            ),
        ]
        self.set_segments(items, fps=self._fps)
        self._emit_segments_changed()

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
            hit_count=sum(item.hit_count for item in chosen),
            review_required=True,
            review_reasons=["segment đã được merge thủ công"],
        )
        first = rows[0]
        items = [item for index, item in enumerate(items) if index not in rows]
        items.insert(first, merged)
        self.set_segments(items, fps=self._fps)
        self._emit_segments_changed()
