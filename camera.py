import cv2
import time

class Camera:
    def __init__(self, source_type='webcam', source=0, rotate_code=None):
        self.source_type = source_type
        self.source = source
        self.rotate_code = rotate_code
        self.cap = None
        self.connected = False

    def connect(self):
        self.release()
        try:
            if self.source_type == 'webcam':
                self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
            elif self.source_type == 'ip':
                self.cap = cv2.VideoCapture(str(self.source))
            else:
                raise ValueError("Неизвестный тип источника камеры")
            if not self.cap.isOpened():
                self.cap = None
                return False
            self.connected = True
            return True
        except Exception as e:
            print(f"Ошибка подключения камеры: {e}")
            self.cap = None
            self.connected = False
            return False

    def set_resolution(self, width, height):
        """Устанавливает разрешение видео."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def set_fps(self, fps):
        """Устанавливает частоту кадров видео."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FPS, int(fps))

    def read(self):
        if not self.connected or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        if self.rotate_code is not None:
            frame = cv2.rotate(frame, self.rotate_code)
        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.connected = False

    def reconnect(self, delay=3):
        print(f"Переподключение через {delay} сек...")
        time.sleep(delay)
        return self.connect()