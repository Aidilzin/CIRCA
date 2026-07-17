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

REWORK_DB = {
    "missing_hole": {
        "std": "IPC-A-600 §3.4 Drill breakthrough failure",
        "action": "Inspect drill alignment and validate drill bit clearance."
    },
    "mouse_bite": {
        "std": "IPC-A-600 §3.3 Board edge notching",
        "action": "File edge smooth; inspect traces for physical break."
    },
    "open_circuit": {
        "std": "IPC-A-600 §3.2 Broken conductor trace",
        "action": "Bridge trace gap using micro-wire jumper and solder."
    },
    "short": {
        "std": "IPC-A-600 §3.2 Unintended conductor bridge",
        "action": "Remove solder bridge using copper braid or solder sucker."
    },
    "excess_solder": {
        "std": "IPC-A-610H §5 Solder volume above spec",
        "action": "Remove excess solder alloy using copper desoldering braid."
    },
    "insufficient_solder": {
        "std": "IPC-A-610H §5 Solder volume below spec",
        "action": "Apply fresh solder alloy and flux to form a proper fillet."
    },
    "cold_solder_joint": {
        "std": "IPC-A-610H §5 Dull/granular joint",
        "action": "Reheat joint with fresh flux to reflow into a clean fillet."
    }
}

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
        
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 6, 6)
        painter.end()


class ChecklistItem(QWidget):
    hovered = pyqtSignal(int, bool)
    clicked = pyqtSignal(int)
    fp_marked = pyqtSignal(int)

    def __init__(self, index: int, box, box_index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._box_index = box_index

        # Main Layout (Vertical)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        # Header Row Widget
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 4, 0, 4)
        header_layout.setSpacing(8)

        # Checkbox
        self.checkbox = QCheckBox(f"{index}.")
        self.checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
        header_layout.addWidget(self.checkbox)

        # Colored pill label for accessibility (color + text label)
        self.tag_label = QLabel()
        self.tag_label.setFont(QFont(FONT_MONO, 8, QFont.Weight.Bold))
        
        hex_color = DEFECT_CLASS_COLORS.get(box.class_name, DEFECT_CLASS_COLOR_FALLBACK)
        short_code = box.class_name.split("_")[0].upper()
        
        self.tag_label.setText(f" {short_code} ")
        self.tag_label.setStyleSheet(
            f"background-color: {hex_color}; "
            f"color: #09090B; "
            f"border-radius: 4px; "
            f"padding: 2px 6px;"
        )
        header_layout.addWidget(self.tag_label)

        # Full description text
        desc = box.class_name.replace("_", " ").upper()
        pct = int(box.confidence * 100)
        self.desc_label = QLabel(f"{desc} ({pct}%)")
        self.desc_label.setFont(QFont(FONT_UI, 10, QFont.Weight.Medium))
        header_layout.addWidget(self.desc_label, stretch=1)

        # FP Button (Mark False Positive)
        self.fp_btn = QPushButton("Mark FP")
        self.fp_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fp_btn.setToolTip("Mark as False Positive (exclude from report and save for model training)")
        self.fp_btn.setFont(QFont(FONT_UI, 8, QFont.Weight.Medium))
        self.fp_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #27272A;"
            "  color: #EF4444;"
            "  border: 1px solid #3F3F46;"
            "  border-radius: 4px;"
            "  padding: 2px 6px;"
            "}"
            "QPushButton:hover {"
            "  background-color: #7F1D1D;"
            "  color: #FCA5A5;"
            "}"
        )
        self.fp_btn.clicked.connect(self._on_fp_clicked)
        header_layout.addWidget(self.fp_btn)

        main_layout.addWidget(header_widget)

        # Expandable defect description label (Plain English Glossary info)
        GLOSSARY_DESCS = {
            "missing_hole": "A required drill hole is absent. Typically a drilling toolpath or process fault.",
            "mouse_bite": "An edge erosion on a copper trace. Decreases trace cross-section and changes impedance.",
            "open_circuit": "A physical break/cut in a copper trace that completely cuts off electrical signal flow.",
            "short": "An accidental solder/copper bridge between traces, causing current leakage or short.",
            "excess_solder": "Too much solder deposited, risking solder bridges and masking structural faults.",
            "insufficient_solder": "Too little solder deposited, causing mechanically weak joints prone to cracking.",
            "cold_solder_joint": "Granular, dull joint due to inadequate melting or quick cooling; poor electrical connection."
        }
        
        desc_text = GLOSSARY_DESCS.get(box.class_name, "No explanation available.")
        self.exp_label = QLabel(f"ℹ️ {desc_text}")
        self.exp_label.setWordWrap(True)
        self.exp_label.setFont(QFont(FONT_UI, 9))
        self.exp_label.setStyleSheet(
            "QLabel {"
            "  color: #94A3B8;"
            "  background-color: #1E293B;"
            "  border-left: 2px solid #00D2FF;"
            "  border-radius: 4px;"
            "  padding: 6px 10px;"
            "  margin-left: 24px;"
            "}"
        )
        self.exp_label.setVisible(False)
        main_layout.addWidget(self.exp_label)

    def toggle_expansion(self) -> None:
        self.exp_label.setVisible(not self.exp_label.isVisible())

    def _on_fp_clicked(self) -> None:
        self.fp_marked.emit(self._box_index)

    def enterEvent(self, event) -> None:  # noqa: N802
        self.hovered.emit(self._box_index, True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.hovered.emit(self._box_index, False)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle_expansion()
            self.clicked.emit(self._box_index)
        super().mousePressEvent(event)


class AnalyticsDashboard(QWidget):
    defect_hovered = pyqtSignal(int)
    defect_clicked = pyqtSignal(int)
    fp_marked = pyqtSignal(int)
    next_board_requested = pyqtSignal()

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
        self._image_name = "—"
        self._model_name = "—"

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
        header = QLabel("PCB DIAGNOSTICS & REPAIR STATION")
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
        layout.addWidget(self._section_header("DEFECT CHECKLIST"))
        
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

        # 5. Action Buttons (immediately below checklist)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        
        self.commit_btn = QPushButton("Log Repairs Done")
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

        # 6. Session Overview Card (moved to bottom)
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
        
        session_layout.addWidget(self._stat_label("INSPECTION LATENCY"), 2, 0)
        self.latency_val = QLabel("— ms")
        self.latency_val.setFont(QFont(FONT_MONO, 11, QFont.Weight.Bold))
        session_layout.addWidget(self.latency_val, 2, 1, Qt.AlignmentFlag.AlignRight)
        
        layout.addWidget(self.session_card)
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
        try:
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
        except (AttributeError, TypeError, ValueError) as exc:
            logger.error("Failed to update active metrics: %s", exc, exc_info=True)
            self.faults_val.setText("Error")
            self.conf_val.setText("Error")

        # Sharpness metric
        try:
            sharpness = getattr(result, "sharpness_variance", -1.0)
            if sharpness >= 0:
                self._active_sharpness = sharpness
                self.sharpness_val.setText(f"{sharpness:.1f}")
            else:
                self.sharpness_val.setText("—")
        except (AttributeError, TypeError, ValueError) as exc:
            logger.error("Failed to update sharpness metric: %s", exc, exc_info=True)
            self.sharpness_val.setText("Error")

        # Repopulate Rework Checklist
        try:
            self._repopulate_checklist()
        except Exception as exc:
            logger.error("Failed to repopulate rework checklist: %s", exc, exc_info=True)
            while self.checklist_layout.count():
                child = self.checklist_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            err_lbl = QLabel("Checklist unavailable (rendering error)")
            err_lbl.setFont(QFont(FONT_UI, 10))
            err_lbl.setStyleSheet("color: #EF4444; font-style: italic; padding: 10px;")
            self.checklist_layout.addWidget(err_lbl)

        # Update Advisory Panel & Animations
        try:
            self._update_advisory_ui()
        except Exception as exc:
            logger.error("Failed to update advisory UI: %s", exc, exc_info=True)
            self.status_val.setText("ADVISORY ERROR")
            self.status_val.setStyleSheet("color: #EF4444;")
            self.status_desc.setText("Repair advice is currently unavailable.")

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
                self.status_val.setText("REPAIR REQUIRED")
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
    def handle_inference_time(self, latency_ms: float):
        if latency_ms >= 0:
            self.latency_val.setText(f"{latency_ms:.0f} ms")
        else:
            self.latency_val.setText("— ms")

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
            item.clicked.connect(self.defect_clicked.emit)
            item.fp_marked.connect(self.fp_marked.emit)
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
        self.status_val.setText("REPAIR LOGGED")
        self.status_val.setStyleSheet("color: #4CAF50;")
        self.status_desc.setText("Repairs recorded successfully in system diagnostic database.")

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
            logger.info("Repairs logged successfully for board with %d defects.", self._active_defects)
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
        self._image_name = "—"
        
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
        self.next_board_requested.emit()
        logger.info("Cleared active board view diagnostics.")

    def set_image_name(self, name: str) -> None:
        self._image_name = name

    def set_model_name(self, name: str) -> None:
        self._model_name = name

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
            
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        device = os.environ.get("CIRCA_DEVICE", "AUTO")
        latency = self.latency_val.text()

        # Count defects by standard
        bare_board_count = 0
        solder_assembly_count = 0
        for box in self._active_boxes:
            if box.class_name in ["missing_hole", "mouse_bite", "open_circuit", "short"]:
                bare_board_count += 1
            else:
                solder_assembly_count += 1

        try:
            if path.endswith(".csv"):
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["CIRCA PCB Diagnostics Ticket"])
                    writer.writerow([])
                    writer.writerow(["Timestamp", timestamp])
                    writer.writerow(["Inspected Board", self._image_name])
                    writer.writerow(["Diagnostic Model", self._model_name])
                    writer.writerow(["Execution Device", device])
                    writer.writerow(["Inference Latency", latency])
                    writer.writerow(["Active Faults Count", self._active_defects])
                    writer.writerow(["Average Confidence", f"{self._active_avg_conf * 100:.1f}%"])
                    writer.writerow(["Board Status", "REPAIR REQUIRED" if self._active_defects > 0 else "PASSED"])
                    writer.writerow([])
                    writer.writerow(["Item", "Fault Class", "IPC Standard Section", "Confidence", "Repair Action"])
                    for i, box in enumerate(self._active_boxes, 1):
                         db = REWORK_DB.get(box.class_name, {"std": "Custom Standard", "action": "General inspection required."})
                         writer.writerow([i, box.class_name.upper(), db["std"], f"{box.confidence*100:.0f}%", db["action"]])
            else:
                with open(path, "w") as f:
                    f.write("================================================================================\n")
                    f.write("                               CIRCA PCB DIAGNOSTICS TICKET\n")
                    f.write("================================================================================\n")
                    f.write(f"Timestamp           : {timestamp}\n")
                    f.write(f"Inspected Board     : {self._image_name}\n")
                    f.write(f"Diagnostic Model    : {self._model_name}\n")
                    f.write(f"Execution Device    : {device}\n")
                    f.write(f"Inference Latency   : {latency}\n")
                    if self._active_defects > 0:
                        f.write(f"Board Status        : ❌ REPAIR REQUIRED ({self._active_defects} defect(s) detected)\n")
                    else:
                        f.write("Board Status        : ✅ PASSED (No surface defects detected)\n")
                    f.write("\n[ DEFECT BREAKDOWN ]\n")
                    f.write(f"IPC-A-600 (Bare-Board) Defects      : {bare_board_count}\n")
                    f.write(f"IPC-A-610H (Solder Assembly) Defects: {solder_assembly_count}\n")
                    f.write("\n[ DETECTED REPAIR CHECKLIST ]\n")
                    f.write("--------------------------------------------------------------------------------\n")
                    if self._active_boxes:
                        for i, box in enumerate(self._active_boxes, 1):
                            name = box.class_name.upper()
                            pct = int(box.confidence * 100)
                            db = REWORK_DB.get(box.class_name, {"std": "General Quality Concern", "action": "Inspect visually and repair manually."})
                            f.write(f"[ ] {i}. {name} (Confidence: {pct}%)\n")
                            f.write(f"    - Reference Standard: {db['std']}\n")
                            f.write(f"    - Action Required   : {db['action']}\n\n")
                    else:
                        f.write("No surface defects detected. Fillets and conductor lines conform to standard specifications.\n")
                    f.write("================================================================================\n")
                    f.write("                     Universiti Teknologi MARA — CIRCA FYP                      \n")
                    f.write("================================================================================\n")
            logger.info("Diagnostics ticket exported successfully to %s", path)
        except OSError as exc:
            logger.error("Failed to export diagnostics ticket: %s", exc, exc_info=True)

    def _update_stats_ui(self):
        self.session_diag_val.setText(str(self._diagnosed_count))
        self.session_fault_val.setText(str(self._faulty_count))
