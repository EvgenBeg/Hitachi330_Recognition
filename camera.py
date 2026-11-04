import cv2
import time


class Camera:
    def __init__(self, source_type='webcam', source=0, rotate_code=None, crop=None,
                 capture_width=0, capture_height=0, capture_fps=0):
        self.source_type = source_type
        self.source = source
        self.rotate_code = rotate_code
        self.crop = crop
        self.capture_width = int(capture_width) if capture_width else 0
        self.capture_height = int(capture_height) if capture_height else 0
        self.capture_fps = int(capture_fps) if capture_fps else 0
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

            applied = False
            if self.capture_width > 0 and self.capture_height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
                applied = True
            if self.capture_fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, self.capture_fps)
                applied = True

            if applied:
                # Короткий прогрев: 3 попытки по 50 мс.
                # read() не блокирует надолго — DSHOW возвращает False,
                # если кадра нет. Если за 150 мс не пришло — не страшно,
                # update_video_frame подхватит кадры, как только они пойдут.
                for _ in range(3):
                    ret, frm = self.cap.read()
                    if ret and frm is not None and frm.size > 0:
                        break
                    time.sleep(0.05)

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
            frame = self._apply_crop(frame)
        return True, frame

    def _apply_crop(self, frame):
        """Применяет crop с масштабированием относительно ref_width/ref_height.
        Это позволяет crop корректно работать после смены разрешения камеры."""
        fh, fw = frame.shape[:2]
        c = self.crop
        ref_w = int(c.get('ref_width', fw) or fw)
        ref_h = int(c.get('ref_height', fh) or fh)
        sx = fw / ref_w if ref_w > 0 else 1.0
        sy = fh / ref_h if ref_h > 0 else 1.0

        x = int(c['x'] * sx)
        y = int(c['y'] * sy)
        w = int(c['w'] * sx)
        h = int(c['h'] * sy)

        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = max(1, min(w, fw - x))
        h = max(1, min(h, fh - y))
        return frame[y:y+h, x:x+w]

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