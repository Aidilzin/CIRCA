import csv
import logging
import os
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSlot, QVariantAnimation, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush, QPen
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QFrame,
    QFileDialog,
    QCheckBox,
    QScrollArea,
)

from core.models import DetectionResult
from ui.theme import (
    COLOR_ACCENT_CYAN,
    COLOR_STATUS_OK,
    COLOR_STATUS_WARN,
    COLOR_STATUS_ERROR,
    FONT_UI,
    FONT_MONO,
    DEFECT_CLASS_COLORS,
    DEFECT_CLASS_COLOR_FALLBACK,
    ThemeManager,
)

logger = logging.getLogger(__name__)

class PulsingCard(QFrame):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("AnalyticsCard")
        palette = ThemeManager().get_palette()
        self._bg_color = QColor(palette["BG_SURFACE"])

    def set_bg_color(self, color: QColor) -> None:
        self._bg_color = color
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        palette = ThemeManager().get_palette()
        border_color = QColor(palette["BORDER"])
        
        # Draw background and border
        painter.setBrush(QBrush(self._bg_color))
        painter.setPen(QPen(border_color, 1))
        
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)
        painter.end()


class ChecklistItem(QWidget):
    hovered = pyqtSignal(int, bool)

    def __init__(self, index: int, box, box_index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._box_index = box_index
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(8)

        # Checkbox
        self.checkbox = QCheckBox(f"{index}.")
        self.checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.checkbox)

        # Colored pill label for accessibility (color + text label)
        self.tag_label = QLabel()
        self.tag_label.setFont(QFont(FONT_MONO, 8, QFont.Weight.Bold))
        
        hex_color = DEFECT_CLASS_COLORS.get(box.class_name, DEFECT_CLASS_COLOR_FALLBACK)
        # Abbreviate label (e.g. cold_solder_joint -> COLD, missing_hole -> MISSING)
        short_code = box.class_name.split("_")[0].upper()
        
        self.tag_label.setText(f" {short_code} ")
        self.tag_label.setStyleSheet(
            f"background-color: {hex_color}; "
            f"color: #09090B; "
            f"border-radius: 4px; "
            f"padding: 2px 6px;"
        )
        layout.addWidget(self.tag_label)

        # Full description text
        desc = box.class_name.replace("_", " ").upper()
        pct = int(box.confidence * 100)
        self.desc_label = QLabel(f"{desc} ({pct}%)")
        self.desc_label.setFont(QFont(FONT_UI, 10, QFont.Weight.Medium))
        layout.addWidget(self.desc_label, stretch=1)

    def enterEvent(self, event) -> None:  # noqa: N802
        self.hovered.emit(self._box_index, True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.hovered.emit(self._box_index, False)
        super().leaveEvent(event)


class AnalyticsDashboard(QWidget):
    defect_hovered = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("AnalyticsDashboard")
        
        # Session Counters
        self._diagnosed_count = 0
        self._faulty_count = 0
        self._rework_logged_count = 0
        
        # Active Board State
        self._active_defects = 0
        self._active_avg_conf = 1.0
        self._active_sharpness = 0.0
        self._active_boxes = []

        # Advisory pulsing animation setup
        self._pulse_anim = QVariantAnimation(self)
        self._pulse_anim.valueChanged.connect(self._on_pulse_value_changed)
        self._pulse_anim.setLoopCount(-1)
        
        self._build_ui()
        self._update_stats_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(16)

        # 1. Header
        header = QLabel("PCB DIAGNOSTICS & REWORK STATION")
        header.setObjectName("AnalyticsHeader")
        header.setFont(QFont(FONT_UI, 10, QFont.Weight.Bold))
        layout.addWidget(header)

        # 2. System Advisory / Repair Advice
        self.status_card = PulsingCard()
        self.status_card.setObjectName("AnalyticsCard")
        status_layout = QVBoxLayout(self.status_card)
        status_layout.setContentsMargins(16, 16, 16, 16)
        status_layout.setSpacing(6)
        
        status_title = QLabel("REPAIR ADVISORY")
        status_title.setFont(QFont(FONT_UI, 9, QFont.Weight.Bold))
        status_title.setProperty("secondary", "true")
        status_layout.addWidget(status_title)
        
        self.status_val = QLabel("STATION IDLE")
        self.status_val.setObjectName("AdvisoryStatusVal")
        self.status_val.setFont(QFont(FONT_UI, 16, QFont.Weight.Bold))
        status_layout.addWidget(self.status_val)
        
        self.status_desc = QLabel("Position a PCB under the camera to begin diagnostics.")
        self.status_desc.setObjectName("AdvisoryStatusDesc")
        self.status_desc.setFont(QFont(FONT_UI, 10))
        self.status_desc.setWordWrap(True)
        status_layout.addWidget(self.status_desc)
        
        layout.addWidget(self.status_card)

        # 3. Active Board Metrics Grid
        metrics_grid_widget = QWidget()
        grid = QGridLayout(metrics_grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        
        # Active Faults Card
        self.faults_card = QFrame()
        self.faults_card.setObjectName("AnalyticsCard")
        fa_lay = QVBoxLayout(self.faults_card)
        fa_lay.addWidget(self._stat_label("ACTIVE FAULTS"))
        self.faults_val = QLabel("0")
        self.faults_val.setFont(QFont(FONT_MONO, 20, QFont.Weight.Bold))
        self.faults_val.setStyleSheet(f"color: {COLOR_STATUS_OK};")
        fa_lay.addWidget(self.faults_val)
        grid.addWidget(self.faults_card, 0, 0)
        
        # Avg Confidence Card
        self.conf_card = QFrame()
        self.conf_card.setObjectName("AnalyticsCard")
        co_lay = QVBoxLayout(self.conf_card)
        co_lay.addWidget(self._stat_label("AVG CONFIDENCE"))
        self.conf_val = QLabel("— %")
        self.conf_val.setFont(QFont(FONT_MONO, 18, QFont.Weight.Bold))
        co_lay.addWidget(self.conf_val)
        grid.addWidget(self.conf_card, 0, 1)
        
        # Sharpness/Blur Card
        self.sharpness_card = QFrame()
        self.sharpness_card.setObjectName("AnalyticsCard")
        sh_lay = QVBoxLayout(self.sharpness_card)
        sh_lay.addWidget(self._stat_label("SHARPNESS"))
        self.sharpness_val = QLabel("—")
        self.sharpness_val.setFont(QFont(FONT_MONO, 18, QFont.Weight.Bold))
        sh_lay.addWidget(self.sharpness_val)
        grid.addWidget(self.sharpness_card, 0, 2)
        
        layout.addWidget(metrics_grid_widget)

        # 4. Rework Checklist (Interactive Task Checklist)
        layout.addWidget(self._section_header("REWORK CHECKLIST"))
        
        self.scroll = QScrollArea()
        self.scroll.setObjectName("AnalyticsScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedHeight(180)
        
        self.checklist_container = QWidget()
        self.checklist_container.setObjectName("ChecklistContainer")
        self.checklist_layout = QVBoxLayout(self.checklist_container)
        self.checklist_layout.setContentsMargins(0, 0, 0, 0)
        self.checklist_layout.setSpacing(10)
        self.checklist_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.scroll.setWidget(self.checklist_container)
        layout.addWidget(self.scroll)

        # 5. Session Overview Card
        layout.addWidget(self._section_header("SESSION DIAGNOSTIC LOG"))
        
        self.session_card = QFrame()
        self.session_card.setObjectName("AnalyticsCard")
        session_layout = QGridLayout(self.session_card)
        session_layout.setContentsMargins(12, 12, 12, 12)
        session_layout.setSpacing(10)
        
        session_layout.addWidget(self._stat_label("BOARDS DIAGNOSED"), 0, 0)
        self.session_diag_val = QLabel("0")
        self.session_diag_val.setFont(QFont(FONT_MONO, 11, QFont.Weight.Bold))
        session_layout.addWidget(self.session_diag_val, 0, 1, Qt.AlignmentFlag.AlignRight)
        
        session_layout.addWidget(self._stat_label("DEFECTIVE ENCOUNTERED"), 1, 0)
        self.session_fault_val = QLabel("0")
        self.session_fault_val.setFont(QFont(FONT_MONO, 11, QFont.Weight.Bold))
        session_layout.addWidget(self.session_fault_val, 1, 1, Qt.AlignmentFlag.AlignRight)
        
        session_layout.addWidget(self._stat_label("INSPECTION STREAM SPEED"), 2, 0)
        self.fps_val = QLabel("— fps")
        self.fps_val.setFont(QFont(FONT_MONO, 11, QFont.Weight.Bold))
        session_layout.addWidget(self.fps_val, 2, 1, Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(self.session_card)

        # 6. Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        
        self.commit_btn = QPushButton("Log Rework Done")
        self.commit_btn.setObjectName("CommitReworkButton")
        self.commit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.commit_btn.clicked.connect(self._on_commit_clicked)
        btn_row.addWidget(self.commit_btn, stretch=2)
        
        self.export_btn = QPushButton("Export Ticket")
        self.export_btn.setObjectName("ExportTicketButton")
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.clicked.connect(self._on_export_clicked)
        btn_row.addWidget(self.export_btn, stretch=1)
        
        self.reset_btn = QPushButton("Next Board")
        self.reset_btn.setObjectName("NextBoardButton")
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        btn_row.addWidget(self.reset_btn, stretch=1)
        
        layout.addLayout(btn_row)
        layout.addStretch(1)

    def _stat_label(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont(FONT_UI, 8, QFont.Weight.Bold))
        lbl.setProperty("secondary", "true")
        return lbl

    def _section_header(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("AnalyticsSectionHeader")
        lbl.setFont(QFont(FONT_UI, 9, QFont.Weight.Bold))
        lbl.setProperty("secondary", "true")
        return lbl

    def _on_pulse_value_changed(self, color: QColor) -> None:
        self.status_card.set_bg_color(color)

    @pyqtSlot(object)
    def handle_detections(self, result: DetectionResult):
        # Update live defect counts
        self._active_defects = len(result.boxes)
        self._active_avg_conf = result.average_confidence
        self._active_boxes = result.boxes
        
        # Display active metrics
        self.faults_val.setText(str(self._active_defects))
        if self._active_defects > 0:
            self.faults_val.setStyleSheet(f"color: {COLOR_STATUS_ERROR};")
            self.conf_val.setText(f"{self._active_avg_conf * 100:.0f}%")
        else:
            self.faults_val.setStyleSheet(f"color: {COLOR_STATUS_OK};")
            self.conf_val.setText("— %")

        # Sharpness metric
        sharpness = getattr(result, "sharpness_variance", -1.0)
        if sharpness >= 0:
            self._active_sharpness = sharpness
            self.sharpness_val.setText(f"{sharpness:.1f}")
        else:
            self.sharpness_val.setText("—")

        # Repopulate Rework Checklist
        self._repopulate_checklist()

        # Update Advisory Panel & Animations
        self._update_advisory_ui()

    def _update_advisory_ui(self):
        palette = ThemeManager().get_palette()
        self._pulse_anim.stop()

        if self._active_defects > 0:
            # Calibrate pulse color based on confidence (Constraint #5)
            conf = self._active_avg_conf
            
            # Base warning color selection
            if conf < 0.65:
                base_color = QColor(255, 193, 7) # Amber
                self.status_val.setText("CHECK COLD SOLDER")
                self.status_val.setStyleSheet("color: #FFC107;")
                self.status_desc.setText(
                    "Warning: low confidence detections. Inspect highlighted joints "
                    "manually to verify potential cold solder or micro-cracks."
                )
            else:
                base_color = QColor(244, 67, 54) # Red
                self.status_val.setText("REWORK REQUIRED")
                self.status_val.setStyleSheet("color: #F44336;")
                
                # List dominant defect
                counts = {}
                for box in self._active_boxes:
                    counts[box.class_name] = counts.get(box.class_name, 0) + 1
                max_class = max(counts, key=counts.get).replace("_", " ").upper()
                
                self.status_desc.setText(
                    f"Found {self._active_defects} fault(s). Dominant type: {max_class}. "
                    "Locate bounding boxes on feed and repair."
                )

            # Scale alpha based on confidence: range 30 to 110 (Constraint #5)
            peak_alpha = int(30 + conf * 80)
            peak_color = QColor(base_color.red(), base_color.green(), base_color.blue(), peak_alpha)
            start_color = QColor(palette["BG_SURFACE"])

            # Setup pulsing animation values
            self._pulse_anim.setDuration(1200)
            self._pulse_anim.setStartValue(start_color)
            self._pulse_anim.setEndValue(start_color)
            self._pulse_anim.setKeyValueAt(0.5, peak_color)
            self._pulse_anim.start()
        else:
            self.status_val.setText("BOARD CLEAN")
            self.status_val.setStyleSheet("color: #4CAF50;") # Green
            self.status_desc.setText("Visual inspection complete. No surface defects detected in current view.")
            self.status_card.set_bg_color(QColor(palette["BG_SURFACE"]))

    @pyqtSlot(float)
    def handle_fps(self, fps: float):
        if fps >= 0:
            self.fps_val.setText(f"{fps:.1f} fps")
        else:
            self.fps_val.setText("— fps")

    def _repopulate_checklist(self):
        # Clear checklist layout
        while self.checklist_layout.count():
            child = self.checklist_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        if self._active_defects == 0:
            lbl = QLabel("No active faults. Board is clear.")
            lbl.setFont(QFont(FONT_UI, 10))
            lbl.setStyleSheet("color: #71717A; italic; padding: 10px;")
            self.checklist_layout.addWidget(lbl)
            return

        for i, box in enumerate(self._active_boxes, 1):
            item = ChecklistItem(i, box, box_index=i-1)
            item.checkbox.stateChanged.connect(self._on_checklist_state_changed)
            item.hovered.connect(self._on_item_hovered)
            self.checklist_layout.addWidget(item)

    def _on_item_hovered(self, idx: int, is_hovered: bool) -> None:
        if is_hovered:
            self.defect_hovered.emit(idx)
        else:
            self.defect_hovered.emit(-1)

    def _on_checklist_state_changed(self, state):
        # Optional tracker for interactive operator actions
        pass

    def _flash_success(self) -> None:
        self._pulse_anim.stop()
        
        # Set background to success green
        success_color = QColor(76, 175, 80, 80) # Soft success green
        self.status_card.set_bg_color(success_color)
        self.status_val.setText("REWORK LOGGED")
        self.status_val.setStyleSheet("color: #4CAF50;")
        self.status_desc.setText("Rework recorded successfully in system diagnostic database.")

        # Revert back after 1.0 seconds
        QTimer.singleShot(1000, self._restore_advisory_state)

    def _restore_advisory_state(self):
        self._update_advisory_ui()

    def _on_commit_clicked(self):
        # Record diagnosis log
        self._diagnosed_count += 1
        if self._active_defects > 0:
            self._faulty_count += 1
            self._rework_logged_count += 1
            logger.info("Rework logged successfully for board with %d defects.", self._active_defects)
            # Flash the success green confirmation
            self._flash_success()
        else:
            logger.info("Board logged clean.")
            self._flash_success()
            
        self._update_stats_ui()

    def _on_reset_clicked(self):
        # Advance to next board, keep session totals but clear active view metrics
        self._pulse_anim.stop()
        self._active_defects = 0
        self._active_avg_conf = 1.0
        self._active_boxes = []
        
        self.faults_val.setText("0")
        self.faults_val.setStyleSheet(f"color: {COLOR_STATUS_OK};")
        self.conf_val.setText("— %")
        self.sharpness_val.setText("—")
        self.status_val.setText("STATION IDLE")
        self.status_val.setStyleSheet("color: #71717A;")
        self.status_desc.setText("Ready for next board scan. Place PCB under lens.")
        
        palette = ThemeManager().get_palette()
        self.status_card.set_bg_color(QColor(palette["BG_SURFACE"]))
        
        self._repopulate_checklist()
        logger.info("Cleared active board view diagnostics.")

    def _on_export_clicked(self):
        # Save inspection ticket
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Diagnostics Ticket",
            os.path.expanduser("~/circa_diagnostics_ticket.txt"),
            "Text Files (*.txt);;CSV Files (*.csv)",
        )
        if not path:
            return
            
        try:
            if path.endswith(".csv"):
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["CIRCA PCB Diagnostics Ticket"])
                    writer.writerow([])
                    writer.writerow(["Active Faults Count", self._active_defects])
                    writer.writerow(["Average Confidence", f"{self._active_avg_conf * 100:.1f}%"])
                    writer.writerow(["Board Sharpness score", f"{self._active_sharpness:.1f}"])
                    writer.writerow([])
                    writer.writerow(["Item", "Fault Type", "Confidence"])
                    for i, box in enumerate(self._active_boxes, 1):
                         writer.writerow([i, box.class_name.upper(), f"{box.confidence*100:.0f}%"])
            else:
                with open(path, "w") as f:
                    f.write("=========================================\n")
                    f.write("       CIRCA PCB DIAGNOSTICS TICKET      \n")
                    f.write("=========================================\n\n")
                    f.write(f"Active Faults Count : {self._active_defects}\n")
                    f.write(f"Average Confidence  : {self._active_avg_conf * 100:.1f}%\n")
                    f.write(f"Board Sharpness     : {self._active_sharpness:.1f}\n\n")
                    f.write("Detected Fault Rework Checklist:\n")
                    f.write("-----------------------------------------\n")
                    if self._active_boxes:
                        for i, box in enumerate(self._active_boxes, 1):
                            name = box.class_name.replace("_", " ").upper()
                            pct = int(box.confidence * 100)
                            f.write(f"[ ] {i}. {name} (Conf: {pct}%)\n")
                    else:
                        f.write("No surface defects detected.\n")
                    f.write("\n=========================================\n")
            logger.info("Diagnostics ticket exported successfully to %s", path)
        except Exception as exc:
            logger.error("Failed to export diagnostics ticket: %s", exc)

    def _update_stats_ui(self):
        self.session_diag_val.setText(str(self._diagnosed_count))
        self.session_fault_val.setText(str(self._faulty_count))
