"""NBA Fundamentals V2 -- Desktop Application."""

import sys
sys.path.insert(0, ".")

from PySide6.QtWidgets import QApplication

from src.bootstrap import setup_logging, bootstrap, shutdown


def main():
    setup_logging()

    app = QApplication(sys.argv)

    # Show splash
    from src.ui.splash import SplashScreen
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # Bootstrap (splash.set_status accepts msg + optional progress float)
    splash.set_status("Initializing...", 0.0)
    bootstrap(status_callback=splash.set_status, enable_daily_automation=True)

    # Main window (imported after bootstrap so DB etc. are ready)
    from src.ui.main_window import MainWindow
    window = MainWindow()

    # Crossfade from splash to main window (handles close + show internally)
    splash.start_linger(window, duration_ms=0)

    code = app.exec()
    shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
