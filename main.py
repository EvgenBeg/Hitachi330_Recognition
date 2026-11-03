import tkinter as tk
import ctypes
import sys

from gui import MainApp
from camera import Camera
from config import Config, ensure_files_exist


def enable_dpi_awareness():
    """Включает DPI-awareness в Windows.
    Без этого cv2.selectROI и cv2.imshow показывают чёрное окно
    на мониторах с масштабированием (125%, 150%, 175%, 200%)."""
    if sys.platform != 'win32':
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


if __name__ == "__main__":
    enable_dpi_awareness()

    ensure_files_exist()

    root = tk.Tk()
    config = Config()

    source_type = config.get('camera', 'source_type', default='webcam')
    source = config.get('camera', 'source', default=0)
    rotate_code = config.get_camera_rotate_code()

    camera = Camera(
        source_type=source_type,
        source=source,
        rotate_code=rotate_code,
        prefer_mjpeg=config.get('camera', 'prefer_mjpeg', default=True),
        capture_width=config.get('camera', 'capture', 'width', default=1920),
        capture_height=config.get('camera', 'capture', 'height', default=1080),
        capture_fps=config.get('camera', 'capture', 'fps', default=30),
    )
    app = MainApp(root, camera, config)
    root.mainloop()