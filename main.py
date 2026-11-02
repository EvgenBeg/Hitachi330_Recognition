import tkinter as tk
from gui import MainApp
from camera import Camera
from config import Config

if __name__ == "__main__":
    root = tk.Tk()
    config = Config()

    source_type = config.get('camera', 'source_type', default='webcam')
    source = config.get('camera', 'source', default=0)
    rotate_code = config.get_camera_rotate_code()

    camera = Camera(source_type=source_type, source=source, rotate_code=rotate_code)
    app = MainApp(root, camera, config)
    root.mainloop()