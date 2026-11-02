import cv2
import numpy as np
import json
import os
import math

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

    def draw(self, img, color=(0,255,0), thickness=2):
        if self.angle == 0:
            cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), color, thickness)
        else:
            pts = self.get_corners()
            cv2.polylines(img, [pts], True, color, thickness)
        if self.label:
            cv2.putText(img, self.label, (self.x, self.y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def get_corners(self):
        rad = math.radians(self.angle)
        corners = [
            (self.x, self.y),
            (self.x + self.w*math.cos(rad), self.y + self.w*math.sin(rad)),
            (self.x + self.w*math.cos(rad) - self.h*math.sin(rad),
             self.y + self.w*math.sin(rad) + self.h*math.cos(rad)),
            (self.x - self.h*math.sin(rad), self.y + self.h*math.cos(rad))
        ]
        return np.array(corners, dtype=np.int32)

def _apply_rotation(frame, rotate_code):
    """Применяет поворот к кадру, если задан rotate_code."""
    if rotate_code is not None:
        frame = cv2.rotate(frame, rotate_code)
    return frame

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Сохранено в {filename}")

# ---------- Функция настройки HSV ----------
def color_calibration(camera_index=0, config_file='hsv_config.json', rotate_code=None):
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

    cv2.namedWindow('Mask')
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

        # Преобразуем маску в BGR, чтобы putText рисовал цветную надпись
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Подпись поверх маски
        cv2.putText(mask_bgr, "Press: q - quit, s - save config",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

        cv2.imshow('Mask', mask_bgr)

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
def segment_calibration(camera_index=0, segment_file='segment_config.json', roi_file='roi_config.json', rotate_code=None):
    """Запускает мастер настройки областей ROI и сегментов."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось получить кадр")
        return
    frame = _apply_rotation(frame, rotate_code)
    frame_h, frame_w = frame.shape[:2]

    # Загружаем существующую конфигурацию или создаём новую
    if os.path.exists(segment_file):
        config = load_json(segment_file)
    else:
        config = {
            'rois': [
                {'x': int(frame_w*0.1), 'y': int(frame_h*0.3), 'w': int(frame_w*0.3), 'h': int(frame_h*0.4),
                 'digits': [{'segments': [{'x':0.0,'y':0.0} for _ in range(7)]} for _ in range(4)],
                 'point': {'x':0.0, 'y':0.0}},
                {'x': int(frame_w*0.6), 'y': int(frame_h*0.3), 'w': int(frame_w*0.3), 'h': int(frame_h*0.4),
                 'digits': [{'segments': [{'x':0.0,'y':0.0} for _ in range(7)]} for _ in range(4)],
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

    def create_roi_trackbars(roi):
        cv2.createTrackbar(roi_w_track, settings_window, roi['w'], frame_w, lambda x: None)
        cv2.createTrackbar(roi_h_track, settings_window, roi['h'], frame_h, lambda x: None)

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
            # Не добавляем DraggableRect — ROI рисуется вручную в основном цикле

        elif step == 1:  # Настройка формы сегментов
            if base_offsets is None:
                cx, cy = frame_w//2, frame_h//2
                positions = [
                    (cx-50, cy-60), (cx+20, cy-40), (cx+20, cy+20), (cx-50, cy+40),
                    (cx-80, cy+20), (cx-80, cy-40), (cx-50, cy-20)
                ]
                for idx, (seg, (px, py)) in enumerate(zip(roi['digits'][0]['segments'], positions)):
                    current_rects.append(DraggableRect(px, py, 9, 9, label=str(idx+1)))
            else:
                cx, cy = frame_w//2, frame_h//2
                for idx, (dx, dy) in enumerate(base_offsets):
                    current_rects.append(DraggableRect(cx+dx, cy+dy, 9, 9, label=str(idx+1)))

        elif step in [2, 3, 4, 5]:  # Цифры 1-4
            digit_idx = step - 2
            if digit_idx == 0:
                digit = roi['digits'][0]
            else:
                init_digit(roi, digit_idx)
                digit = roi['digits'][digit_idx]
            for idx, seg in enumerate(digit['segments']):
                abs_x, abs_y = rel_to_abs(roi, seg['x'], seg['y'])
                current_rects.append(DraggableRect(abs_x, abs_y, 9, 9, label=str(idx+1)))
            group_drag = True

        elif step == 6:  # Точка
            pt = roi['point']
            abs_x, abs_y = rel_to_abs(roi, pt['x'], pt['y'])
            current_rects.append(DraggableRect(abs_x, abs_y, 9, 9, label='P'))

        elif step == 7:  # Проверка
            for digit in roi['digits']:
                for idx, seg in enumerate(digit['segments']):
                    abs_x, abs_y = rel_to_abs(roi, seg['x'], seg['y'])
                    current_rects.append(DraggableRect(abs_x, abs_y, 9, 9, label=str(idx+1)))
            pt = roi['point']
            abs_x, abs_y = rel_to_abs(roi, pt['x'], pt['y'])
            current_rects.append(DraggableRect(abs_x, abs_y, 9, 9, label='P'))
            group_drag = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, drag_offset_x, drag_offset_y, group_drag, group_start_x, group_start_y
        roi = config['rois'][current_roi]

        if event == cv2.EVENT_LBUTTONDOWN:
            if step == 0 and roi['x'] <= x <= roi['x']+roi['w'] and roi['y'] <= y <= roi['y']+roi['h']:
                dragging = 'roi'
                drag_offset_x = x - roi['x']
                drag_offset_y = y - roi['y']
            else:
                for rect in current_rects:
                    if rect.contains(x, y):
                        if group_drag:
                            dragging = 'group'
                            group_start_x = x
                            group_start_y = y
                        else:
                            dragging = rect
                            drag_offset_x = x - rect.x
                            drag_offset_y = y - rect.y
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging == 'roi':
                roi['x'] = x - drag_offset_x
                roi['y'] = y - drag_offset_y
            elif dragging == 'group':
                dx = x - group_start_x
                dy = y - group_start_y
                group_start_x = x
                group_start_y = y
                for rect in current_rects:
                    rect.x += dx
                    rect.y += dy
            elif dragging is not None:
                dragging.x = x - drag_offset_x
                dragging.y = y - drag_offset_y
                if step == 0:
                    roi['w'] = cv2.getTrackbarPos(roi_w_track, settings_window)
                    roi['h'] = cv2.getTrackbarPos(roi_h_track, settings_window)
                    dragging.w = roi['w']
                    dragging.h = roi['h']

        elif event == cv2.EVENT_LBUTTONUP:
            if dragging == 'roi':
                roi['w'] = cv2.getTrackbarPos(roi_w_track, settings_window)
                roi['h'] = cv2.getTrackbarPos(roi_h_track, settings_window)
            elif dragging is not None and step in [1,2,3,4,5,6,7]:
                if step == 1:
                    cx = sum(r.x for r in current_rects) / len(current_rects)
                    cy = sum(r.y for r in current_rects) / len(current_rects)
                    base_offsets = [(r.x - cx, r.y - cy) for r in current_rects]
                    for i, rect in enumerate(current_rects):
                        rel_x, rel_y = abs_to_rel(roi, rect.x, rect.y)
                        roi['digits'][0]['segments'][i]['x'] = rel_x
                        roi['digits'][0]['segments'][i]['y'] = rel_y
                elif step in [2,3,4,5]:
                    digit_idx = step - 2
                    digit = roi['digits'][digit_idx]
                    for i, rect in enumerate(current_rects):
                        rel_x, rel_y = abs_to_rel(roi, rect.x, rect.y)
                        digit['segments'][i]['x'] = rel_x
                        digit['segments'][i]['y'] = rel_y
                elif step == 6:
                    rel_x, rel_y = abs_to_rel(roi, current_rects[0].x, current_rects[0].y)
                    roi['point']['x'] = rel_x
                    roi['point']['y'] = rel_y
                elif step == 7:
                    idx = 0
                    for digit in roi['digits']:
                        for i in range(7):
                            rel_x, rel_y = abs_to_rel(roi, current_rects[idx].x, current_rects[idx].y)
                            digit['segments'][i]['x'] = rel_x
                            digit['segments'][i]['y'] = rel_y
                            idx += 1
                    rel_x, rel_y = abs_to_rel(roi, current_rects[-1].x, current_rects[-1].y)
                    roi['point']['x'] = rel_x
                    roi['point']['y'] = rel_y
            dragging = None

    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', mouse_callback)

    setup_step()
    print(f"ROI {current_roi+1}, шаг {step+1}. Клавиши: 'p' - продолжить, 'q' - выйти")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _apply_rotation(frame, rotate_code)
        roi = config['rois'][current_roi]

        if step == 0:
            roi['w'] = cv2.getTrackbarPos(roi_w_track, settings_window)
            roi['h'] = cv2.getTrackbarPos(roi_h_track, settings_window)
            # Синий прямоугольник ROI (только на шаге настройки ROI)
            cv2.rectangle(frame, (roi['x'], roi['y']),
                          (roi['x']+roi['w'], roi['y']+roi['h']), (50, 255, 50), 2)
        else:
            # Тонкая серая рамка ROI как ориентир
            cv2.rectangle(frame, (roi['x'], roi['y']),
                          (roi['x']+roi['w'], roi['y']+roi['h']), (50, 255, 50), 2)

        # Рисуем сегменты (на шаге 0 их нет)
        for rect in current_rects:
            rect.draw(frame, (0, 255, 0), 1)

        text = f"ROI {current_roi+1}, Step {step+1}. Press: p - continue, q - quit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_color = (0, 0, 0)          # чёрный (BGR)
        bg_color = (0, 255, 255)        # светло-голубой фон (BGR)
        padding = 5                     # отступ внутри фона

        # Узнаём размер текста
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Верхний левый угол текста
        x, y = 10, 30

        # Прямоугольник фона: (x - padding, y - text_h - padding) до (x + text_w + padding, y + baseline + padding)
        cv2.rectangle(frame,
                      (x - padding, y - text_h - padding),
                      (x + text_w + padding, y + baseline + padding),
                      bg_color, -1)   # -1 = заливка

        # Текст поверх фона
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

        cv2.imshow('Video', frame)

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
                    save_json(config, segment_file)
                    roi_config = {
                        'roi1': {'x': config['rois'][0]['x'], 'y': config['rois'][0]['y'],
                                 'w': config['rois'][0]['w'], 'h': config['rois'][0]['h'],
                                 'scale': 1.0, 'angle': 0},
                        'roi2': {'x': config['rois'][1]['x'], 'y': config['rois'][1]['y'],
                                 'w': config['rois'][1]['w'], 'h': config['rois'][1]['h'],
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
    # color_calibration(0)
    segment_calibration(0)