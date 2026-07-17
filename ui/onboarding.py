from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect, QRectF, QObject, QSettings
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath, QPen, QBrush

class OnboardingPopover(QFrame):
    next_clicked = pyqtSignal()
    back_clicked = pyqtSignal()
    skip_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("OnboardingPopover")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Title
        self.title_label = QLabel()
        self.title_label.setObjectName("OnboardingTitle")
        self.title_label.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        
        # Description
        self.desc_label = QLabel()
        self.desc_label.setObjectName("OnboardingDesc")
        self.desc_label.setFont(QFont("Inter", 9))
        self.desc_label.setWordWrap(True)
        layout.addWidget(self.desc_label)
        
        # Horizontal layout for control buttons & progress indicator
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(8)
        
        # Progress Label
        self.progress_label = QLabel()
        self.progress_label.setObjectName("OnboardingProgress")
        self.progress_label.setFont(QFont("Inter", 8, QFont.Weight.Medium))
        bottom_layout.addWidget(self.progress_label)
        
        bottom_layout.addStretch(1)
        
        # Skip Link
        self.skip_btn = QPushButton("Skip Tour")
        self.skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.skip_btn.setObjectName("OnboardingSkipBtn")
        self.skip_btn.setFlat(True)
        self.skip_btn.clicked.connect(self.skip_clicked)
        bottom_layout.addWidget(self.skip_btn)
        
        # Back Button
        self.back_btn = QPushButton("Back")
        self.back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_btn.setObjectName("OnboardingBackBtn")
        self.back_btn.clicked.connect(self.back_clicked)
        bottom_layout.addWidget(self.back_btn)
        
        # Next Button
        self.next_btn = QPushButton("Next")
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.setObjectName("OnboardingNextBtn")
        self.next_btn.clicked.connect(self.next_clicked)
        bottom_layout.addWidget(self.next_btn)
        
        layout.addLayout(bottom_layout)

    def set_step_data(self, title: str, description: str, current: int, total: int):
        self.title_label.setText(title)
        self.desc_label.setText(description)
        self.progress_label.setText(f"{current} of {total}")
        
        # Hide Back button on step 1
        self.back_btn.setVisible(current > 1)
        if current == total:
            self.next_btn.setText("Finish")
        else:
            self.next_btn.setText("Next")


class OnboardingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.target_widget = None
        self.popover = OnboardingPopover(self)
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def set_target(self, widget):
        self.target_widget = widget
        self.update()
        if widget and widget.isVisible():
            target_rect = widget.rect()
            global_pos = widget.mapToGlobal(target_rect.topLeft())
            overlay_local_pos = self.mapFromGlobal(global_pos)
            target_in_overlay = QRect(overlay_local_pos, target_rect.size())
            self.reposition_popover(target_in_overlay)
        else:
            self.popover.hide()

    def reposition_popover(self, target_in_overlay: QRect):
        popover_width = 280
        popover_height = 140
        self.popover.setFixedSize(popover_width, popover_height)
        
        # Prefer placing it directly below the target centered
        x = target_in_overlay.center().x() - popover_width // 2
        y = target_in_overlay.bottom() + 12
        
        # Keep popover within overlay boundaries
        if x < 10:
            x = 10
        if x + popover_width > self.width() - 10:
            x = self.width() - popover_width - 10
            
        if y + popover_height > self.height() - 10:
            # Shift above target if running off screen
            y = target_in_overlay.top() - popover_height - 12
            if y < 10:
                y = 10
                # Fallback to horizontal positioning if vertical doesn't fit
                if target_in_overlay.right() + popover_width + 12 < self.width() - 10:
                    x = target_in_overlay.right() + 12
                    y = target_in_overlay.center().y() - popover_height // 2
                elif target_in_overlay.left() - popover_width - 12 > 10:
                    x = target_in_overlay.left() - popover_width - 12
                    y = target_in_overlay.center().y() - popover_height // 2

        self.popover.move(QPoint(int(x), int(y)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.target_widget:
            self.set_target(self.target_widget)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        path = QPainterPath()
        path.addRect(QRectF(self.rect()))
        
        if self.target_widget and self.target_widget.isVisible():
            target_rect = self.target_widget.rect()
            global_pos = self.target_widget.mapToGlobal(target_rect.topLeft())
            overlay_local_pos = self.mapFromGlobal(global_pos)
            target_in_overlay = QRect(overlay_local_pos, target_rect.size())
            
            hole_path = QPainterPath()
            hole_path.addRoundedRect(QRectF(target_in_overlay), 6.0, 6.0)
            
            mask_path = path.subtracted(hole_path)
            painter.fillPath(mask_path, QColor(7, 11, 14, 200))
            
            # Draw sharp electric blue focus outline
            painter.setPen(QPen(QColor(0, 210, 255), 2))
            painter.drawRoundedRect(QRectF(target_in_overlay), 6.0, 6.0)
        else:
            painter.fillPath(path, QColor(7, 11, 14, 200))


class OnboardingController(QObject):
    tour_finished = pyqtSignal()

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.overlay = OnboardingOverlay(main_window)
        
        self.current_step = 0
        self.steps = []
        
        self.overlay.popover.next_clicked.connect(self.next_step)
        self.overlay.popover.back_clicked.connect(self.back_step)
        self.overlay.popover.skip_clicked.connect(self.finish_tour)
        
        self.overlay.hide()

    def init_steps(self):
        self.steps = [
            {
                "target": self.main_window.nav_sidebar,
                "title": "Navigation Sidebar",
                "description": "Switch tabs to load board files, capture frames from USB inspection cameras, reanalyze, and view telemetry diagnostics."
            },
            {
                "target": self.main_window.inspect_widget,
                "title": "PCB Viewport View",
                "description": "Examine detection overlay logs, zoom into sub-resolutions, and analyze identified bounding boxes."
            },
            {
                "target": self.main_window.analytics_dashboard,
                "title": "Telemetry Repair Advisor",
                "description": "Complete repair checklists, diagnose classifications (short/open), and mark false positive instances."
            },
            {
                "target": self.main_window.side_panel,
                "title": "Configuration Drawer",
                "description": "Configure image optimization sliders, scan local USB vision devices, and switch AI inference engine weights."
            }
        ]

    def start_tour(self):
        self.init_steps()
        self.current_step = 0
        self.overlay.setGeometry(self.main_window.centralWidget().rect())
        self.overlay.show()
        self.overlay.raise_()
        self.show_step()

    def show_step(self):
        if 0 <= self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            target = step["target"]
            
            # Auto-expand/collapse configurations panel when active
            if target == self.main_window.side_panel:
                self.main_window.side_panel.show_panel()
            else:
                self.main_window.side_panel.hide_panel()
                
            self.overlay.set_target(target)
            self.overlay.popover.set_step_data(
                step["title"],
                step["description"],
                self.current_step + 1,
                len(self.steps)
            )
            self.overlay.popover.show()
            self.overlay.update()

    def next_step(self):
        self.current_step += 1
        if self.current_step >= len(self.steps):
            self.finish_tour()
        else:
            self.show_step()

    def back_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.show_step()

    def finish_tour(self):
        self.overlay.hide()
        self.main_window.side_panel.hide_panel()
        
        settings = QSettings("CIRCA", "VisionStudio")
        settings.setValue("hasRunOnboarding", True)
        
        self.tour_finished.emit()
