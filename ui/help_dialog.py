"""
ui/help_dialog.py
-----------------
HelpDialog — Interactive onboarding guide and help system.
Designed using HCI best practices for Learnability and Documentation.
"""

from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QWidget,
    QFrame,
    QScrollArea,
)
from ui.theme import FONT_UI, FONT_MONO, COLOR_ACCENT_CYAN, ThemeManager

class HelpDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CIRCA Onboarding Guide")
        self.setFixedSize(520, 420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._setup_ui()
        self._update_navigation()

    def _setup_ui(self):
        palette = ThemeManager().get_palette()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {palette['BG_BASE']};
                color: {palette['TEXT_PRIMARY']};
            }}
            QLabel {{
                color: {palette['TEXT_PRIMARY']};
            }}
            QLabel[secondary="true"] {{
                color: {palette['TEXT_SECONDARY']};
            }}
            QPushButton {{
                background-color: {palette['BG_ELEVATED']};
                border: 1px solid {palette['BORDER']};
                border-radius: 6px;
                padding: 6px 16px;
                color: {palette['TEXT_PRIMARY']};
                font-family: {FONT_UI};
                font-weight: 500;
            }}
            QPushButton:hover {{
                border-color: {COLOR_ACCENT_CYAN};
            }}
            QPushButton#PrimaryAction {{
                background-color: {COLOR_ACCENT_CYAN};
                color: {palette['BG_BASE']};
                border: none;
            }}
            QPushButton#PrimaryAction:hover {{
                background-color: {COLOR_ACCENT_CYAN};
            }}
            #StepIndicator {{
                color: {palette['TEXT_SECONDARY']};
                font-family: {FONT_MONO};
                font-size: 11px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Header Title
        self.title_lbl = QLabel("CIRCA Diagnostic Console")
        self.title_lbl.setFont(QFont(FONT_UI, 14, QFont.Weight.Bold))
        layout.addWidget(self.title_lbl)

        # Stacked Pages
        self.stack = QStackedWidget()
               # Page 1: Overview
        p1 = QWidget()
        p1_lay = QVBoxLayout(p1)
        p1_lay.setContentsMargins(0, 0, 0, 0)
        p1_lay.setSpacing(12)
        desc1 = QLabel(
            "Welcome to CIRCA, your intelligent decision-support assistant for printed circuit board (PCB) diagnostics and repairs.\n\n"
            "The workspace uses a split-screen design:\n"
            "• Left Pane: High-resolution static image inspection workspace.\n"
            "• Right Pane: Dynamic advisory card, statistics, and repair checklist.\n\n"
            "This layout keeps critical inspection results and log diagnostics visible simultaneously."
        )
        desc1.setWordWrap(True)
        desc1.setFont(QFont(FONT_UI, 11))
        desc1.setProperty("secondary", "true")
        p1_lay.addWidget(desc1)
        p1_lay.addStretch(1)
        self.stack.addWidget(p1)
 
        # Page 2: HUD & Viewport
        p2 = QWidget()
        p2_lay = QVBoxLayout(p2)
        p2_lay.setContentsMargins(0, 0, 0, 0)
        p2_lay.setSpacing(12)
        desc2 = QLabel(
            "The inspection panel features a digital HUD overlay:\n"
            "• Image Resolution & Analysis Time displayed in the top-right.\n"
            "• PCB Scene Guard in the bottom-left validates that the loaded image is a PCB.\n"
            "• Glowing Bounding Boxes surround board defects with class tags.\n"
            "• High-Contrast outlines ensure bounding boxes remain legible over any PCB color mask."
        )
        desc2.setWordWrap(True)
        desc2.setFont(QFont(FONT_UI, 11))
        desc2.setProperty("secondary", "true")
        p2_lay.addWidget(desc2)
        p2_lay.addStretch(1)
        self.stack.addWidget(p2)
 
        # Page 3: Repair Checklist
        p3 = QWidget()
        p3_lay = QVBoxLayout(p3)
        p3_lay.setContentsMargins(0, 0, 0, 0)
        p3_lay.setSpacing(12)
        desc3 = QLabel(
            "Managing & Repairing defects is simple:\n"
            "• The Repair Checklist logs active board faults sequentially.\n"
            "• Hovering over checklist items highlights the defect's exact location on the image with a thick glowing border.\n"
            "• Each fault includes an accessibility pill (e.g. [COLD] or [SHORT]) pairing text with color to support color-blind operators.\n"
            "• Log repairs by checking them off or clicking 'Log Repairs Done'."
        )
        desc3.setWordWrap(True)
        desc3.setFont(QFont(FONT_UI, 11))
        desc3.setProperty("secondary", "true")
        p3_lay.addWidget(desc3)
        p3_lay.addStretch(1)
        self.stack.addWidget(p3)

        # Page 4: Settings & Optimization
        p4 = QWidget()
        p4_lay = QVBoxLayout(p4)
        p4_lay.setContentsMargins(0, 0, 0, 0)
        p4_lay.setSpacing(12)
        desc4 = QLabel(
            "Auto-Tuning and Configuration:\n"
            "• Auto-Optimise: Instantly adjusts contrast (CLAHE) and brightness (Gamma) for your desk lighting.\n"
            "• Manual Overrides: Adjusting any slider disables auto-optimise automatically to hand over manual control.\n"
            "• Auto-Detect Best Model: Benchmarks your workstation and auto-selects the optimal model variant (Nano, Small, Medium) matching your GPU/CPU capability."
        )
        desc4.setWordWrap(True)
        desc4.setFont(QFont(FONT_UI, 11))
        desc4.setProperty("secondary", "true")
        p4_lay.addWidget(desc4)
        p4_lay.addStretch(1)
        self.stack.addWidget(p4)

        # Page 5: Defect Glossary
        p5 = QWidget()
        p5_lay = QVBoxLayout(p5)
        p5_lay.setContentsMargins(0, 0, 0, 0)
        p5_lay.setSpacing(8)
        
        glossary_title = QLabel("Defect Glossary / Dictionary:")
        glossary_title.setFont(QFont(FONT_UI, 11, QFont.Weight.Bold))
        p5_lay.addWidget(glossary_title)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        
        scroll_content = QWidget()
        scroll_lay = QVBoxLayout(scroll_content)
        scroll_lay.setContentsMargins(0, 0, 0, 0)
        scroll_lay.setSpacing(10)
        
        defects = [
            ("Missing Hole", "A required drill hole is absent. Typically a drilling toolpath or process fault."),
            ("Mouse Bite", "An edge erosion on a copper trace. Decreases trace cross-section and changes impedance."),
            ("Open Circuit", "A physical break/cut in a copper trace that completely cuts off electrical signal flow."),
            ("Short Circuit", "An accidental solder/copper bridge between traces, causing current leakage or short."),
            ("Excess Solder", "Too much solder deposited, risking solder bridges and masking structural faults."),
            ("Insufficient Solder", "Too little solder deposited, causing mechanically weak joints prone to cracking."),
            ("Cold Solder Joint", "Granular, dull joint due to inadequate melting or quick cooling; poor electrical connection.")
        ]
        
        for title, desc in defects:
            item_lbl = QLabel(f"• <b>{title}</b>: {desc}")
            item_lbl.setWordWrap(True)
            item_lbl.setFont(QFont(FONT_UI, 10))
            item_lbl.setProperty("secondary", "true")
            scroll_lay.addWidget(item_lbl)
            
        scroll_lay.addStretch(1)
        scroll.setWidget(scroll_content)
        p5_lay.addWidget(scroll)
        self.stack.addWidget(p5)

        layout.addWidget(self.stack, stretch=1)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFrameShadow(QFrame.Shadow.Sunken)
        div.setStyleSheet(f"background-color: {palette['BORDER']}; max-height: 1px;")
        layout.addWidget(div)

        # Navigation Bar
        nav_lay = QHBoxLayout()
        self.indicator_lbl = QLabel("Step 1 of 4")
        self.indicator_lbl.setObjectName("StepIndicator")
        nav_lay.addWidget(self.indicator_lbl)
        nav_lay.addStretch(1)

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.clicked.connect(self._on_prev)
        nav_lay.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("PrimaryAction")
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.clicked.connect(self._on_next)
        nav_lay.addWidget(self.next_btn)

        layout.addLayout(nav_lay)

    def _on_prev(self):
        curr = self.stack.currentIndex()
        if curr > 0:
            self.stack.setCurrentIndex(curr - 1)
            self._update_navigation()

    def _on_next(self):
        curr = self.stack.currentIndex()
        if curr < self.stack.count() - 1:
            self.stack.setCurrentIndex(curr + 1)
            self._update_navigation()
        else:
            self.accept()

    def _update_navigation(self):
        curr = self.stack.currentIndex()
        total = self.stack.count()
        self.indicator_lbl.setText(f"Step {curr + 1} of {total}")
        self.prev_btn.setEnabled(curr > 0)
        if curr == total - 1:
            self.next_btn.setText("Get Started")
        else:
            self.next_btn.setText("Next")
