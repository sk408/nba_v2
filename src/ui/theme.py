"""Modern Broadcast Theme QSS stylesheet for PySide6 GUI."""

GLOBAL_STYLESHEET = """
/* ---- Global Modern Broadcast Theme ---- */
QWidget {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Oswald', 'Segoe UI', sans-serif;
    font-size: 14px;
}

QMainWindow {
    background-color: #0b0f19;
}

/* ---- Tab Widget ---- */
QTabWidget::pane {
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(20, 30, 45, 0.8);
    border-radius: 4px;
}

QTabBar::tab {
    background: rgba(20, 30, 45, 0.8);
    color: #94a3b8;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: 700;
    font-size: 14px;
    text-transform: uppercase;
}

QTabBar::tab:selected {
    background: rgba(20, 30, 45, 0.95);
    color: #00e5ff;
    border-bottom: 2px solid #00e5ff;
}

QTabBar::tab:hover {
    background: rgba(30, 45, 65, 0.9);
    color: #e2e8f0;
}

/* ---- Buttons ---- */
QPushButton {
    background-color: rgba(30, 45, 65, 0.8);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: 700;
    text-transform: uppercase;
    min-height: 32px;
}

QPushButton:hover {
    background-color: rgba(59, 130, 246, 0.8);
    border-color: #00e5ff;
    color: #00e5ff;
}

QPushButton:pressed {
    background-color: rgba(29, 78, 216, 0.9);
}

QPushButton:disabled {
    background-color: rgba(15, 25, 35, 0.5);
    color: #64748b;
    border-color: transparent;
}

QPushButton[class="danger"] {
    background-color: rgba(239, 68, 68, 0.8);
}

QPushButton[class="danger"]:hover {
    background-color: rgba(220, 38, 38, 0.9);
    border-color: #ff4d4f;
    color: #ffffff;
}

QPushButton[class="outline"] {
    background-color: transparent;
    border: 1px solid #475569;
    color: #94a3b8;
}

QPushButton[class="outline"]:hover {
    border-color: #00e5ff;
    color: #00e5ff;
    background-color: rgba(0, 229, 255, 0.1);
}

QPushButton[class="success"] {
    background-color: rgba(34, 197, 94, 0.8);
}

QPushButton[class="success"]:hover {
    background-color: rgba(22, 163, 74, 0.95);
    border-color: #22c55e;
    color: #ffffff;
}

/* ---- Input Fields ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: rgba(20, 30, 45, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    padding: 6px 10px;
    color: #e2e8f0;
    min-height: 32px;
    font-weight: 600;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #00e5ff;
    background-color: rgba(20, 30, 45, 0.95);
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #0b0f19;
    border: 1px solid #00e5ff;
    selection-background-color: rgba(59, 130, 246, 0.5);
    color: #e2e8f0;
}

/* ---- Tables ---- */
QTableWidget, QTableView {
    background-color: rgba(11, 15, 25, 0.6);
    alternate-background-color: rgba(20, 30, 45, 0.6);
    gridline-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    selection-background-color: rgba(0, 229, 255, 0.2);
    selection-color: #00e5ff;
    font-family: 'Segoe UI', sans-serif;
}

QHeaderView::section {
    background-color: rgba(20, 30, 45, 0.8);
    color: #94a3b8;
    padding: 6px 10px;
    border: none;
    border-bottom: 2px solid #00e5ff;
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
}

QTableWidget::item:hover {
    background-color: rgba(0, 229, 255, 0.06);
}

/* ---- Scroll Bars ---- */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    min-height: 32px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(0, 229, 255, 0.5);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    min-width: 32px;
}

QScrollBar::handle:horizontal:hover {
    background: rgba(0, 229, 255, 0.5);
}

/* ---- Labels ---- */
QLabel {
    color: #e2e8f0;
    background: transparent;
}

QLabel[class="header"] {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding-bottom: 6px;
    border-bottom: 2px solid rgba(0, 229, 255, 0.4);
}

QLabel[class="subheader"] {
    font-size: 14px;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
}

QLabel[class="muted"] {
    color: #64748b;
    font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
}

QLabel[class="accent"] {
    color: #00e5ff;
    font-weight: 700;
}

QLabel[class="success"] {
    color: #22c55e;
}

QLabel[class="danger"] {
    color: #ef4444;
}

QLabel[class="warning"] {
    color: #f59e0b;
}

/* ---- Text Edit (Log Output) ---- */
QTextEdit, QPlainTextEdit {
    background-color: rgba(11, 15, 25, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    padding: 8px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
    color: #94a3b8;
}

/* ---- Group Box ---- */
QGroupBox {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    margin-top: 12px;
    padding: 16px 12px 12px;
    font-weight: 700;
    color: #ffffff;
    background: rgba(20, 30, 45, 0.4);
    text-transform: uppercase;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #00e5ff;
}

/* ---- Progress Bar ---- */
QProgressBar {
    background-color: rgba(20, 30, 45, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    height: 6px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #00e5ff;
    border-radius: 3px;
}

/* ---- Splitter ---- */
QSplitter::handle {
    background: rgba(255, 255, 255, 0.1);
    width: 2px;
    height: 2px;
}

/* ---- Status Bar ---- */
QStatusBar {
    background-color: #0b0f19;
    color: #94a3b8;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    padding: 4px 12px;
    font-size: 12px;
    font-family: 'Segoe UI', sans-serif;
}

/* ---- Info Cards & Collapsible Sections ---- */
#infoCard {
    background: rgba(20, 30, 45, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 8px;
}

#collapsibleBar {
    background: rgba(20, 30, 45, 0.8);
    border-radius: 4px;
}

/* ---- Tool Tip ---- */
QToolTip {
    background-color: rgba(20, 30, 45, 0.95);
    color: #ffffff;
    border: 1px solid #00e5ff;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}

/* ---- Menu ---- */
QMenu {
    background-color: rgba(20, 30, 45, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: rgba(0, 229, 255, 0.2);
    color: #00e5ff;
}

/* ---- Check Box ---- */
QCheckBox {
    color: #e2e8f0;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    background: rgba(20, 30, 45, 0.8);
}

QCheckBox::indicator:checked {
    background: #00e5ff;
    border-color: #00e5ff;
}

/* ---- Broadcast Glass Cards ---- */
QFrame[class="broadcast-card"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(30, 41, 59, 0.9), stop:1 rgba(15, 23, 42, 0.95));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 12px;
}

QFrame[class="broadcast-card"]:hover {
    border-color: rgba(0, 229, 255, 0.3);
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(30, 41, 59, 0.95), stop:1 rgba(20, 30, 50, 0.98));
}

/* ---- Team Color Accent Frames ---- */
QFrame[class="team-panel-home"], QFrame[class="team-panel-away"] {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 8px;
}

/* ---- Confidence Progress Bar ---- */
QProgressBar[class="confidence"] {
    border: none;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.05);
    text-align: center;
    color: #e2e8f0;
    font-size: 11px;
    font-weight: 700;
    min-height: 20px;
}

QProgressBar[class="confidence"]::chunk {
    border-radius: 4px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ef4444, stop:0.4 #f59e0b, stop:0.7 #22c55e, stop:1 #00e5ff);
}

/* ---- Live Badge ---- */
QLabel[class="live-badge"] {
    background: #ef4444;
    color: white;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
}

/* ---- Edge Indicator ---- */
QLabel[class="edge-positive"] {
    color: #22c55e;
    font-weight: 700;
}

QLabel[class="edge-negative"] {
    color: #ef4444;
    font-weight: 700;
}

/* ---- Semantic Card Panels ---- */
QFrame[class="card-panel"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
}

QFrame[class="card-panel-danger"] {
    background: #1e293b;
    border: 2px solid #ef4444;
    border-radius: 8px;
    padding: 16px;
}

/* ---- Semantic Text Classes ---- */
QLabel[class="text-primary"] {
    color: #e2e8f0;
    font-size: 13px;
}

QLabel[class="text-secondary"] {
    color: #94a3b8;
    font-size: 13px;
}

QLabel[class="text-hint"] {
    color: #64748b;
    font-size: 11px;
}

QLabel[class="section-title"] {
    color: #e2e8f0;
    font-size: 14px;
    font-weight: 700;
}

QLabel[class="section-title-danger"] {
    color: #ef4444;
    font-size: 14px;
    font-weight: 700;
}

/* ---- Stat Value (large accent numbers) ---- */
QLabel[class="stat-value"] {
    font-size: 26px;
    font-weight: 700;
    color: #00e5ff;
}

QLabel[class="stat-label"] {
    font-size: 10px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
}

/* ---- Button Variants ---- */
QPushButton[class="primary"] {
    background-color: #2563eb;
    color: white;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 600;
}

QPushButton[class="primary"]:hover {
    background-color: #1d4ed8;
}

QPushButton[class="warn"] {
    background-color: #d97706;
    color: white;
    font-weight: 700;
}

QPushButton[class="warn"]:hover {
    background-color: #b45309;
}

/* ---- Toggle Link Button ---- */
QPushButton[class="link"] {
    text-align: left;
    background: transparent;
    color: #00e5ff;
    font-weight: bold;
    border: none;
    padding: 0px;
}

QPushButton[class="link"]:hover {
    color: #00b8cc;
}

/* ---- Terminal / Log Output ---- */
QTextEdit[class="terminal"] {
    background-color: #000000;
    color: #4ade80;
    border: 1px solid #1e293b;
    border-radius: 4px;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 12px;
}

/* ---- Dog Pick Badges ---- */
QLabel[class="badge-dog"] {
    background: #f59e0b;
    color: #000000;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
}

QLabel[class="badge-dog-outside"] {
    background: #475569;
    color: #e2e8f0;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 700;
}

QLabel[class="badge-status"] {
    background: rgba(100, 116, 139, 0.15);
    color: #94a3b8;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
}

/* ---- VS Separator ---- */
QLabel[class="vs-label"] {
    color: #64748b;
    font-family: 'Oswald', 'Segoe UI', sans-serif;
    font-size: 18px;
    font-weight: 700;
}

/* ---- Step Indicator States (QLabel + QFrame) ---- */
QLabel[class="step-pending"], QFrame[class="step-pending"] {
    background: #1e293b;
    border: 2px solid #475569;
    border-radius: 16px;
    color: #94a3b8;
}

QLabel[class="step-active"], QFrame[class="step-active"] {
    background: rgba(0, 229, 255, 0.15);
    border: 2px solid #00e5ff;
    border-radius: 16px;
    color: #00e5ff;
}

QLabel[class="step-done"], QFrame[class="step-done"] {
    background: rgba(34, 197, 94, 0.15);
    border: 2px solid #22c55e;
    border-radius: 16px;
    color: #22c55e;
}

QLabel[class="step-skipped"], QFrame[class="step-skipped"] {
    background: rgba(100, 116, 139, 0.15);
    border: 2px solid #64748b;
    border-radius: 16px;
    color: #64748b;
}

QLabel[class="step-error"], QFrame[class="step-error"] {
    background: rgba(239, 68, 68, 0.15);
    border: 2px solid #ef4444;
    border-radius: 16px;
    color: #ef4444;
}

/* Child labels inside step indicators inherit step state colors */
QFrame[class="step-pending"] QLabel { color: #94a3b8; }
QFrame[class="step-active"] QLabel { color: #ffffff; font-weight: bold; }
QFrame[class="step-done"] QLabel { color: #d1fae5; font-weight: bold; }
QFrame[class="step-skipped"] QLabel { color: #94a3b8; font-style: italic; }
QFrame[class="step-error"] QLabel { color: #fee2e2; font-weight: bold; }

/* ---- Overnight / Indigo Button ---- */
QPushButton[class="indigo"] {
    background-color: #4f46e5;
    color: white;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 600;
}

QPushButton[class="indigo"]:hover {
    background-color: #4338ca;
}

QPushButton[class="indigo"]:disabled {
    background-color: rgba(79, 70, 229, 0.3);
    color: #64748b;
}
"""

def apply_card_shadow(widget, level="md"):
    """Apply a drop shadow to a widget. Levels: sm, md, lg."""
    from PySide6.QtWidgets import QGraphicsDropShadowEffect
    from PySide6.QtGui import QColor
    effect = QGraphicsDropShadowEffect(widget)
    if level == "sm":
        effect.setBlurRadius(4)
        effect.setOffset(0, 1)
        effect.setColor(QColor(0, 0, 0, 77))
    elif level == "lg":
        effect.setBlurRadius(32)
        effect.setOffset(0, 8)
        effect.setColor(QColor(0, 0, 0, 128))
    else:  # md
        effect.setBlurRadius(12)
        effect.setOffset(0, 4)
        effect.setColor(QColor(0, 0, 0, 102))
    widget.setGraphicsEffect(effect)


def setup_theme(widget):
    """Apply the theme to a given widget or QApplication, adjusting for OLED mode."""
    from src.config import get as get_setting
    oled_mode = get_setting("oled_mode", False)
    
    stylesheet = GLOBAL_STYLESHEET
    if oled_mode:
        # Replace main dark backgrounds with pure black for OLED
        stylesheet = stylesheet.replace("#0b0f19", "#000000")
        stylesheet = stylesheet.replace("rgba(20, 30, 45, 0.8)", "#000000")
        stylesheet = stylesheet.replace("rgba(20, 30, 45, 0.95)", "#000000")
        stylesheet = stylesheet.replace("rgba(20, 30, 45, 0.6)", "#000000")
        stylesheet = stylesheet.replace("rgba(20, 30, 45, 0.4)", "#000000")
        stylesheet = stylesheet.replace("rgba(11, 15, 25, 0.6)", "#000000")
        stylesheet = stylesheet.replace("rgba(11, 15, 25, 0.8)", "#000000")
        stylesheet = stylesheet.replace("rgba(15, 25, 35, 0.5)", "#000000")
        stylesheet = stylesheet.replace("rgba(30, 45, 65, 0.8)", "#000000")
        stylesheet = stylesheet.replace("rgba(30, 45, 65, 0.9)", "#000000")
        # Replace hardcoded dark gray colors
        stylesheet = stylesheet.replace("#1e293b", "#000000")
        stylesheet = stylesheet.replace("#0f172a", "#000000")

    widget.setStyleSheet(stylesheet)
