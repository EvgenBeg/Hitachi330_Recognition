import json
import os
import cv2
import numpy as np
import math

def load_json(filename):
    """Загружает JSON-файл. Если файл не существует, возвращает None."""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка загрузки {filename}: {e}")
        return None

def save_json(data, filename):
    """Сохраняет данные в JSON-файл с отступами."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Сохранено в {filename}")
        return True
    except Exception as e:
        print(f"Ошибка сохранения {filename}: {e}")
        return False

def rotate_image(img, angle_deg):
    """Поворачивает изображение на заданный угол (0, 90, 180, 270)."""
    if angle_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def rotate_point(px, py, cx, cy, angle_deg):
    """Поворачивает точку (px,py) вокруг центра (cx,cy) на angle_deg градусов."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = px - cx
    dy = py - cy
    new_x = cx + dx * cos_a - dy * sin_a
    new_y = cy + dx * sin_a + dy * cos_a
    return new_x, new_y

def scale_point(px, py, cx, cy, scale):
    """Масштабирует точку (px,py) относительно центра (cx,cy)."""
    return cx + (px - cx) * scale, cy + (py - cy) * scale

def bgr_to_photoimage(frame, target_width=None, target_height=None):
    """Конвертирует BGR кадр OpenCV в PhotoImage для Tkinter."""
    import tkinter as tk
    from PIL import Image, ImageTk

    if target_width and target_height:
        frame = cv2.resize(frame, (target_width, target_height))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(image=img)

def create_video_writer(filename, fps, frame_size):
    """Создаёт объект VideoWriter для записи видео."""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)