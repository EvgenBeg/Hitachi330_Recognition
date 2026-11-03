import cv2
import time


class Camera:
    def __init__(self, source_type='webcam', source=0, rotate_code=None,
                 crop=None, prefer_mjpeg=True,
                 capture_width=1920, capture_height=1080, capture_fps=30):
        self.source_type = source_type
        self.source = source
        self.rotate_code = rotate_code
        self.crop = crop  # {'x': int, 'y': int, 'w': int, 'h': int} или None
        self.prefer_mjpeg = prefer_mjpeg
        self.capture_width = int(capture_width)
        self.capture_height = int(capture_height)
        self.capture_fps = int(capture_fps)
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

            # --- Применяем формат и разрешение из config.json ---
            if self.source_type == 'webcam' and self.prefer_mjpeg:
                # Порядок важен: сначала MJPG, потом разрешение,
                # иначе DSHOW сбросит размер на 640x480.
                self.cap.set(cv2.CAP_PROP_FOURCC,
                             cv2.VideoWriter_fourcc(*'MJPG'))

            if self.capture_width > 0 and self.capture_height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            if self.capture_fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, self.capture_fps)

            # Прогрев: DSHOW применяет новые параметры не сразу
            for _ in range(10):
                self.cap.read()
                time.sleep(0.02)

            self.connected = True
            return True
        except Exception as e:
            print(f"Ошибка подключения камеры: {e}")
            self.cap = None
            self.connected = False
            return False

    def read(self, apply_crop=True):
        if not self.connected or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        if self.rotate_code is not None:
            frame = cv2.rotate(frame, self.rotate_code)
        if apply_crop and self.crop:
            x, y = int(self.crop['x']), int(self.crop['y'])
            w, h = int(self.crop['w']), int(self.crop['h'])
            fh, fw = frame.shape[:2]
            x = max(0, min(x, fw - 1))
            y = max(0, min(y, fh - 1))
            w = max(1, min(w, fw - x))
            h = max(1, min(h, fh - y))
            frame = frame[y:y+h, x:x+w]
        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            time.sleep(0.3)
        self.connected = False

    def reconnect(self, delay=3):
        print(f"Переподключение через {delay} сек...")
        time.sleep(delay)
        return self.connect()