"""
Global exception handler and crash logger.
"""
import sys
import time
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox

def setup_global_exception_handler():
    """Installs a global exception hook to catch unhandled crashes, write to crash_log.txt, and show a Qt dialog."""
    def exception_hook(exc_type, exc_value, exc_tb):
        err_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(f"\n[CRASH DETECTED]\n{err_msg}\n")
        try:
            with open("crash_log.txt", "a", encoding="utf-8") as f:
                f.write(f"\n--- CRASH LOG [{time.strftime('%Y-%m-%d %H:%M:%S')}] ---\n")
                f.write(err_msg)
                f.write("\n" + "="*60 + "\n")
        except Exception:
            pass

        app = QApplication.instance()
        if app is not None:
            try:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("Errore Imprevisto (Crash Log)")
                msg_box.setText(f"<b>Si è verificato un errore imprevisto:</b><br><br><code>{exc_value}</code><br><br>I dettagli sono stati registrati in <code>crash_log.txt</code>.")
                msg_box.setDetailedText(err_msg)
                msg_box.exec_()
            except Exception:
                pass
        else:
            sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = exception_hook
