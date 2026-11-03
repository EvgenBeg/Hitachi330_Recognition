import cv2
import numpy as np
import json
import os
import math
import ctypes


# ---------- Вспомогательные классы и функции ----------
class DraggableRect:
    """Прямоугольник, который можно перетаскивать мышью (с опциональным углом)."""
    def __init__(self, x, y, w, h, angle=0, label=""):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.angle = angle
        self.label = label

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, img, color=(0, 255, 0), thickness=2):
        if self.angle == 0:
            cv2.rectangle(img, (self.x, self.y),
                          (self.x + self.w, self.y + self.h), color, thickness)
        else:
            pts = self.get_corners()
            cv2.polylines(img, [pts], True, color, thickness)
        if self.label:
            cv2.putText(img, self.label, (self.x, self.y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def get_corners(self):
        rad = math.radians(self.angle)
        corners = [
            (self.x, self.y),
            (self.x + self.w * math.cos(rad), self.y + self.w * math.sin(rad)),
            (self.x + self.w * math.cos(rad) - self.h * math.sin(rad),
             self.y + self.w * math.sin(rad) + self.h * math.cos(rad)),
            (self.x - self.h * math.sin(rad), self.y + self.h * math.cos(rad))
        ]
        return np.array(corners, dtype=np.int32)


def _apply_rotation(frame, rotate_code):
    """Применяет поворот к кадру, если задан rotate_code."""
    if rotate_code is not None:
        frame = cv2.rotate(frame, rotate_code)
    return frame


def _get_screen_size():
    """Возвращает размер рабочего стола с учётом DPI и мультимонитора.
    Tk даёт правильные физические пиксели, ctypes — только для fallback."""
    try:
        import tkinter as tk
        _r = tk.Tk()
        _r.withdraw()
        w = _r.winfo_screenwidth()
        h = _r.winfo_screenheight()
        _r.destroy()
        return w, h
    except Exception:
        try:
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            return 1920, 1080


def _fit_window_size(frame_w, frame_h, screen_w, screen_h, fraction=0.8):
    """Возвращает размер окна с сохранением пропорций кадра."""
    max_w = int(screen_w * fraction)
    max_h = int(screen_h * fraction)
    scale = min(max_w / frame_w, max_h / frame_h)
    win_w = max(1, int(frame_w * scale))
    win_h = max(1, int(frame_h * scale))
    return win_w, win_h


def load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Сохранено в {filename}")


# ---------- Общий блок: фиксированный масштаб отображения ----------
def _compute_display_scale(frame_w, frame_h, fraction=0.75,
                            min_win_w=900, min_win_h=600):
    """Считает display_scale/display_w/display_h/scale_x/scale_y.
    WINDOW_AUTOSIZE + фиксированный масштаб = предсказуемые координаты мыши."""
    screen_w, screen_h = _get_screen_size()
    max_display_w = int(screen_w * fraction)
    max_display_h = int(screen_h * fraction)

    display_scale = min(max_display_w / frame_w, max_display_h / frame_h)
    if frame_w * display_scale < min_win_w:
        display_scale = min_win_w / frame_w
    if frame_h * display_scale < min_win_h:
        display_scale = max(display_scale, min_win_h / frame_h)
    # Не больше окна ни при каких условиях
    display_scale = min(display_scale,
                        max_display_w / frame_w,
                        max_display_h / frame_h)

    display_w = max(1, int(frame_w * display_scale))
    display_h = max(1, int(frame_h * display_scale))
    scale_x = display_w / frame_w
    scale_y = display_h / frame_h
    return display_scale, display_w, display_h, scale_x, scale_y


# ---------- Функция настройки HSV ----------
def color_calibration(camera_index=0, config_file='hsv_config.json',
                      rotate_code=None, crop=None):
    """Открывает окно с трекбарами для подбора HSV-диапазона."""
    init = {
        'H_MIN': 0, 'H_MAX': 179,
        'S_MIN': 100, 'S_MAX': 255,
        'V_MIN': 100, 'V_MAX': 255
    }
    if os.path.exists(config_file):
        saved = load_json(config_file)
        if saved:
            init.update(saved)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    # MJPG даёт стабильный кадр — тот же формат, что и в основной программе
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass

    # Читаем первый кадр для определения пропорций
    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка: не удалось получить кадр")
        cap.release()
        return
    first_frame = _apply_rotation(first_frame, rotate_code)
    if crop:
        x, y, w, h = crop['x'], crop['y'], crop['w'], crop['h']
        first_frame = first_frame[y:y+h, x:x+w]
    frame_h, frame_w = first_frame.shape[:2]

    display_scale, display_w, display_h, scale_x, scale_y = \
        _compute_display_scale(frame_w, frame_h)

    cv2.namedWindow('Mask', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Settings')

    def nothing(x):
        pass

    cv2.createTrackbar('H_MIN', 'Settings', init['H_MIN'], 179, nothing)
    cv2.createTrackbar('H_MAX', 'Settings', init['H_MAX'], 179, nothing)
    cv2.createTrackbar('S_MIN', 'Settings', init['S_MIN'], 255, nothing)
    cv2.createTrackbar('S_MAX', 'Settings', init['S_MAX'], 255, nothing)
    cv2.createTrackbar('V_MIN', 'Settings', init['V_MIN'], 255, nothing)
    cv2.createTrackbar('V_MAX', 'Settings', init['V_MAX'], 255, nothing)

    print("Клавиши: 's' - сохранить, 'q' - выйти")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _apply_rotation(frame, rotate_code)
        if crop:
            frame = frame[crop['y']:crop['y']+crop['h'],
                          crop['x']:crop['x']+crop['w']]

        h_min = cv2.getTrackbarPos('H_MIN', 'Settings')
        h_max = cv2.getTrackbarPos('H_MAX', 'Settings')
        s_min = cv2.getTrackbarPos('S_MIN', 'Settings')
        s_max = cv2.getTrackbarPos('S_MAX', 'Settings')
        v_min = cv2.getTrackbarPos('V_MIN', 'Settings')
        v_max = cv2.getTrackbarPos('V_MAX', 'Settings')

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Масштабируем до размера окна
        interp = cv2.INTER_AREA if display_scale < 1.0 else cv2.INTER_LINEAR
        display_bgr = cv2.resize(mask_bgr, (display_w, display_h),
                                 interpolation=interp)

        # Текст рисуем ПОСЛЕ resize, подбираем font_scale под ширину окна
        text = "Press: q - quit, s - save config"
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_fs = 0.7
        (tw, th), bl = cv2.getTextSize(text, font, base_fs, 2)
        max_tw = display_w - 30
        if tw > max_tw and max_tw > 0:
            font_scale = max(0.30, base_fs * max_tw / tw)
        else:
            font_scale = base_fs
        thickness = 2 if font_scale >= 0.5 else 1

        (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = 10, th + 10
        cv2.rectangle(display_bgr, (x - 5, y - th - 5),
                      (x + tw + 5, y + bl + 5), (0, 255, 255), -1)
        cv2.putText(display_bgr, text, (x, y), font, font_scale,
                    (0, 0, 255), thickness)

        cv2.imshow('Mask', display_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('й'):
            break
        elif key == ord('s') or key == ord('ы'):
            config = {
                'H_MIN': h_min, 'H_MAX': h_max,
                'S_MIN': s_min, 'S_MAX': s_max,
                'V_MIN': v_min, 'V_MAX': v_max
            }
            save_json(config, config_file)

    cap.release()
    cv2.destroyAllWindows()


# ---------- Мастер настройки ROI и сегментов ----------
def segment_calibration(camera_index=0, segment_file='segment_config.json',
                        roi_file='roi_config.json', rotate_code=None, crop=None):
    """Запускает мастер настройки областей ROI и сегментов."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    # Тот же MJPG, что в основной программе — иначе разрешение «съедет»
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # прогрев, чтобы DSHOW применил формат
        for _ in range(10):
            cap.read()
    except Exception:
        pass

    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось получить кадр")
        cap.release()
        return
    frame = _apply_rotation(frame, rotate_code)
    if crop:
        frame = frame[crop['y']:crop['y']+crop['h'],
                      crop['x']:crop['x']+crop['w']]
    frame_h, frame_w = frame.shape[:2]

    # Размер квадратиков сегментов (визуальный) — объявлен ДО чтения конфига,
    # чтобы подхватить сохранённое значение из прошлой сессии.
    seg_size = 9
    current_seg_size = seg_size

    # ==== Фиксированный масштаб отображения ====
    display_scale, display_w, display_h, scale_x, scale_y = \
        _compute_display_scale(frame_w, frame_h)

    # Загружаем существующую конфигурацию или создаём новую
    if os.path.exists(segment_file):
        config = load_json(segment_file)
    else:
        config = {
            'rois': [
                {'x': int(frame_w*0.1), 'y': int(frame_h*0.3),
                 'w': int(frame_w*0.3), 'h': int(frame_h*0.4),
                 'digits': [{'segments': [{'x':0.0,'y':0.0} for _ in range(7)]}
                            for _ in range(4)],
                 'point': {'x':0.0, 'y':0.0}},
                {'x': int(frame_w*0.6), 'y': int(frame_h*0.3),
                 'w': int(frame_w*0.3), 'h': int(frame_h*0.4),
                 'digits': [{'segments': [{'x':0.0,'y':0.0} for _ in range(7)]}
                            for _ in range(4)],
                 'point': {'x':0.0, 'y':0.0}}
            ]
        }
        for roi in config['rois']:
            roi['digits'][0]['segments'] = [
                {'x':0.3,'y':0.1}, {'x':0.8,'y':0.25}, {'x':0.8,'y':0.65},
                {'x':0.3,'y':0.8}, {'x':0.1,'y':0.65}, {'x':0.1,'y':0.25},
                {'x':0.3,'y':0.45}
            ]
            roi['point'] = {'x':0.85, 'y':0.85}

    # Приводим сохранённые ROI к текущему кадру
    if 'rois' in config:
        for _r in config['rois']:
            _r['w'] = max(10, min(int(_r['w']), frame_w))
            _r['h'] = max(10, min(int(_r['h']), frame_h))
            _r['x'] = max(0, min(int(_r['x']), frame_w - _r['w']))
            _r['y'] = max(0, min(int(_r['y']), frame_h - _r['h']))

    # Восстанавливаем сохранённый визуальный размер квадратиков
    if '_seg_size' in config:
        current_seg_size = max(5, min(int(config['_seg_size']), 40))
        seg_size = current_seg_size

    current_roi = 0
    step = 0
    total_steps = 8

    # Переменные для перетаскивания
    current_rects = []
    dragging = None
    drag_offset_x = 0
    drag_offset_y = 0
    group_drag = False
    group_start_x = 0
    group_start_y = 0

    base_offsets = None

    settings_window = 'Settings'
    cv2.namedWindow(settings_window)

    roi_w_track = 'ROI W'
    roi_h_track = 'ROI H'
    seg_size_track = 'Seg size'

    def create_roi_trackbars(roi):
        roi['w'] = max(10, min(int(roi['w']), frame_w))
        roi['h'] = max(10, min(int(roi['h']), frame_h))
        cv2.createTrackbar(roi_w_track, settings_window, roi['w'],
                           frame_w, lambda x: None)
        cv2.createTrackbar(roi_h_track, settings_window, roi['h'],
                           frame_h, lambda x: None)

    def create_seg_size_trackbar():
        """Создаёт ползунок размера квадратиков. Вызывается ТОЛЬКО на шаге 1."""
        cv2.createTrackbar(seg_size_track, settings_window,
                           int(current_seg_size), 40, lambda x: None)
        cv2.setTrackbarMin(seg_size_track, settings_window, 5)

    def clear_trackbars():
        cv2.destroyWindow(settings_window)
        cv2.namedWindow(settings_window)

    def rel_to_abs(roi, rel_x, rel_y):
        return int(roi['x'] + rel_x * roi['w']), int(roi['y'] + rel_y * roi['h'])

    def abs_to_rel(roi, abs_x, abs_y):
        return (abs_x - roi['x']) / roi['w'], (abs_y - roi['y']) / roi['h']

    def init_digit(roi, digit_idx):
        if digit_idx == 0:
            return
        base_segments = roi['digits'][0]['segments']
        digit = roi['digits'][digit_idx]
        for seg, base_seg in zip(digit['segments'], base_segments):
            seg['x'] = base_seg['x']
            seg['y'] = base_seg['y']

    def setup_step():
        nonlocal current_rects, dragging, group_drag
        current_rects = []
        dragging = None
        group_drag = False
        clear_trackbars()
        roi = config['rois'][current_roi]

        if step == 0:  # Настройка ROI
            create_roi_trackbars(roi)

        elif step == 1:  # Настройка формы сегментов
            create_seg_size_trackbar()
            cx, cy = frame_w // 2, frame_h // 2
            if base_offsets is None:
                # Смещения в долях размера кадра
                rel_offsets = [
                    (-0.10, -0.15), (0.05, -0.10), (0.05, 0.05), (-0.10, 0.10),
                    (-0.20, 0.05), (-0.20, -0.10), (-0.10, -0.05)
                ]
                offsets_px = [(int(frame_w * rx), int(frame_h * ry))
                              for rx, ry in rel_offsets]
            else:
                max_dx = max(abs(dx) for dx, _ in base_offsets) or 1
                max_dy = max(abs(dy) for _, dy in base_offsets) or 1
                fit_x = min(1.0, (frame_w * 0.25) / max_dx)
                fit_y = min(1.0, (frame_h * 0.25) / max_dy)
                offsets_px = [(int(dx * fit_x), int(dy * fit_y))
                              for dx, dy in base_offsets]

            for idx, (dx, dy) in enumerate(offsets_px):
                current_rects.append(DraggableRect(cx + dx, cy + dy,
                                                   current_seg_size,
                                                   current_seg_size,
                                                   label=str(idx + 1)))

        elif step in [2, 3, 4, 5]:  # Цифры 1-4
            digit_idx = step - 2
            if digit_idx == 0:
                digit = roi['digits'][0]
            else:
                init_digit(roi, digit_idx)
                digit = roi['digits'][digit_idx]
            for idx, seg in enumerate(digit['segments']):
                abs_x, abs_y = rel_to_abs(roi, seg['x'], seg['y'])
                current_rects.append(DraggableRect(abs_x, abs_y,
                                                   current_seg_size,
                                                   current_seg_size,
                                                   label=str(idx + 1)))
            group_drag = True

        elif step == 6:  # Точка
            pt = roi['point']
            abs_x, abs_y = rel_to_abs(roi, pt['x'], pt['y'])
            current_rects.append(DraggableRect(abs_x, abs_y,
                                               current_seg_size,
                                               current_seg_size, label='P'))

        elif step == 7:  # Проверка
            for digit in roi['digits']:
                for idx, seg in enumerate(digit['segments']):
                    abs_x, abs_y = rel_to_abs(roi, seg['x'], seg['y'])
                    current_rects.append(DraggableRect(abs_x, abs_y,
                                                       current_seg_size,
                                                       current_seg_size,
                                                       label=str(idx + 1)))
            pt = roi['point']
            abs_x, abs_y = rel_to_abs(roi, pt['x'], pt['y'])
            current_rects.append(DraggableRect(abs_x, abs_y,
                                               current_seg_size,
                                               current_seg_size, label='P'))
            group_drag = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, drag_offset_x, drag_offset_y, group_drag
        nonlocal group_start_x, group_start_y

        orig_x = int(x / scale_x)
        orig_y = int(y / scale_y)

        roi = config['rois'][current_roi]

        if event == cv2.EVENT_LBUTTONDOWN:
            if step == 0 and roi['x'] <= orig_x <= roi['x'] + roi['w'] \
                    and roi['y'] <= orig_y <= roi['y'] + roi['h']:
                dragging = 'roi'
                drag_offset_x = orig_x - roi['x']
                drag_offset_y = orig_y - roi['y']
            else:
                for rect in current_rects:
                    if rect.contains(orig_x, orig_y):
                        if group_drag:
                            dragging = 'group'
                            group_start_x = orig_x
                            group_start_y = orig_y
                        else:
                            dragging = rect
                            drag_offset_x = orig_x - rect.x
                            drag_offset_y = orig_y - rect.y
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging == 'roi':
                new_x = orig_x - drag_offset_x
                new_y = orig_y - drag_offset_y
                roi['x'] = max(0, min(new_x, frame_w - roi['w']))
                roi['y'] = max(0, min(new_y, frame_h - roi['h']))
            elif dragging == 'group':
                dx = orig_x - group_start_x
                dy = orig_y - group_start_y
                group_start_x = orig_x
                group_start_y = orig_y
                for rect in current_rects:
                    rect.x += dx
                    rect.y += dy
            elif dragging is not None:
                dragging.x = orig_x - drag_offset_x
                dragging.y = orig_y - drag_offset_y
                if step == 0:
                    roi['w'] = max(10, min(
                        cv2.getTrackbarPos(roi_w_track, settings_window), frame_w))
                    roi['h'] = max(10, min(
                        cv2.getTrackbarPos(roi_h_track, settings_window), frame_h))
                    dragging.w = roi['w']
                    dragging.h = roi['h']

        elif event == cv2.EVENT_LBUTTONUP:
            if dragging == 'roi':
                roi['w'] = max(10, min(
                    cv2.getTrackbarPos(roi_w_track, settings_window), frame_w))
                roi['h'] = max(10, min(
                    cv2.getTrackbarPos(roi_h_track, settings_window), frame_h))
                roi['x'] = max(0, min(roi['x'], frame_w - roi['w']))
                roi['y'] = max(0, min(roi['y'], frame_h - roi['h']))
            elif dragging is not None and step in [1, 2, 3, 4, 5, 6, 7]:
                if step == 1:
                    cx = sum(r.x for r in current_rects) / len(current_rects)
                    cy = sum(r.y for r in current_rects) / len(current_rects)
                    base_offsets = [(r.x - cx, r.y - cy) for r in current_rects]
                    for i, rect in enumerate(current_rects):
                        rel_x, rel_y = abs_to_rel(roi, rect.x, rect.y)
                        roi['digits'][0]['segments'][i]['x'] = rel_x
                        roi['digits'][0]['segments'][i]['y'] = rel_y
                elif step in [2, 3, 4, 5]:
                    digit_idx = step - 2
                    digit = roi['digits'][digit_idx]
                    for i, rect in enumerate(current_rects):
                        rel_x, rel_y = abs_to_rel(roi, rect.x, rect.y)
                        digit['segments'][i]['x'] = rel_x
                        digit['segments'][i]['y'] = rel_y
                elif step == 6:
                    rel_x, rel_y = abs_to_rel(roi, current_rects[0].x,
                                              current_rects[0].y)
                    roi['point']['x'] = rel_x
                    roi['point']['y'] = rel_y
                elif step == 7:
                    idx = 0
                    for digit in roi['digits']:
                        for i in range(7):
                            rel_x, rel_y = abs_to_rel(roi, current_rects[idx].x,
                                                      current_rects[idx].y)
                            digit['segments'][i]['x'] = rel_x
                            digit['segments'][i]['y'] = rel_y
                            idx += 1
                    rel_x, rel_y = abs_to_rel(roi, current_rects[-1].x,
                                              current_rects[-1].y)
                    roi['point']['x'] = rel_x
                    roi['point']['y'] = rel_y
            dragging = None

    cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Video', mouse_callback)

    setup_step()
    print(f"ROI {current_roi+1}, шаг {step+1}. "
          f"Клавиши: 'p' - продолжить, 'q' - выйти")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _apply_rotation(frame, rotate_code)
        if crop:
            frame = frame[crop['y']:crop['y']+crop['h'],
                          crop['x']:crop['x']+crop['w']]
        roi = config['rois'][current_roi]

        # Ползунок размера квадратиков существует только на шаге 1.
        # На остальных шагах используем «запомненный» current_seg_size.
        if step == 1:
            try:
                seg_size = max(5, cv2.getTrackbarPos(seg_size_track,
                                                    settings_window))
                current_seg_size = seg_size
            except cv2.error:
                pass

        if step == 0:
            roi['w'] = max(10, min(
                cv2.getTrackbarPos(roi_w_track, settings_window), frame_w))
            roi['h'] = max(10, min(
                cv2.getTrackbarPos(roi_h_track, settings_window), frame_h))
            roi['x'] = max(0, min(roi['x'], frame_w - roi['w']))
            roi['y'] = max(0, min(roi['y'], frame_h - roi['h']))
            cv2.rectangle(frame, (roi['x'], roi['y']),
                          (roi['x']+roi['w'], roi['y']+roi['h']),
                          (50, 255, 50), 2)
        else:
            cv2.rectangle(frame, (roi['x'], roi['y']),
                          (roi['x']+roi['w'], roi['y']+roi['h']),
                          (50, 255, 50), 2)

        # Синхронизируем размер отрисовки с «запомненным» current_seg_size
        for rect in current_rects:
            rect.w = current_seg_size
            rect.h = current_seg_size
            rect.draw(frame, (0, 255, 0), 1)

        # Масштабируем кадр под окно
        interp = cv2.INTER_AREA if display_scale < 1.0 else cv2.INTER_LINEAR
        display_frame = cv2.resize(frame, (display_w, display_h),
                                   interpolation=interp)

        # Текст рисуем на уже уменьшенном кадре
        text = f"ROI {current_roi+1}, Step {step+1}. Press: p - continue, q - quit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_fs = 0.7
        (tw, th), bl = cv2.getTextSize(text, font, base_fs, 2)
        max_tw = display_w - 30
        if tw > max_tw and max_tw > 0:
            font_scale = max(0.30, base_fs * max_tw / tw)
        else:
            font_scale = base_fs
        thickness = 2 if font_scale >= 0.5 else 1

        (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = 10, th + 10
        cv2.rectangle(display_frame, (x - 5, y - th - 5),
                      (x + tw + 5, y + bl + 5), (0, 255, 255), -1)
        cv2.putText(display_frame, text, (x, y), font, font_scale,
                    (0, 0, 0), thickness)

        cv2.imshow('Video', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('й'):
            break
        elif key == ord('p') or key == ord('з'):
            step += 1
            if step >= total_steps:
                if current_roi == 0:
                    current_roi = 1
                    step = 0
                    base_offsets = None
                    print("Переходим ко второму ROI")
                else:
                    # Сохраняем визуальный размер сегментов,
                    # чтобы превью в «Видео» рисовало квадратики того же размера
                    config['_seg_size'] = int(current_seg_size)
                    save_json(config, segment_file)
                    roi_config = {
                        'roi1': {'x': config['rois'][0]['x'],
                                 'y': config['rois'][0]['y'],
                                 'w': config['rois'][0]['w'],
                                 'h': config['rois'][0]['h'],
                                 'scale': 1.0, 'angle': 0},
                        'roi2': {'x': config['rois'][1]['x'],
                                 'y': config['rois'][1]['y'],
                                 'w': config['rois'][1]['w'],
                                 'h': config['rois'][1]['h'],
                                 'scale': 1.0, 'angle': 0}
                    }
                    save_json(roi_config, roi_file)
                    print("Настройка завершена")
                    break
            setup_step()
            print(f"ROI {current_roi+1}, шаг {step+1}. 'p' - продолжить")

    cap.release()
    cv2.destroyAllWindows()


# ---------- Точка входа (для отладки) ----------
if __name__ == "__main__":
    segment_calibration(0)