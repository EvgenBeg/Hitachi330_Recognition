import cv2
import numpy as np


def apply_hsv_filter(bgr_img, lower_hsv, upper_hsv):
    """
    Применяет HSV-фильтр к изображению.
    Возвращает бинарную маску: белые пиксели там, где цвет попал в диапазон.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    return mask


def preprocess_roi(roi_img, lower_hsv, upper_hsv):
    """
    Получает цветное изображение ROI, выделяет целевые сегменты,
    возвращает бинарное изображение, где сегменты БЕЛЫЕ на ЧЁРНОМ фоне.
    """
    mask = apply_hsv_filter(roi_img, lower_hsv, upper_hsv)
    # Инвертируем маску, чтобы целевые объекты стали чёрными
    processed = cv2.bitwise_not(mask)
    # Порог для чёткой бинаризации
    _, processed = cv2.threshold(processed, 128, 255, cv2.THRESH_BINARY)
    # Возвращаем сегменты в белый цвет (важно для is_segment_on)
    processed = cv2.bitwise_not(processed)
    return processed


def is_segment_on(binary_img, x, y, size=9, threshold=0.30):
    """
    Проверяет, активирован ли сегмент (квадрат размером size x size),
    расположенный в координатах (x,y) на бинарном изображении.
    Возвращает True, если доля белых пикселей превышает threshold.
    """
    h, w = binary_img.shape[:2]
    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))
    x2 = min(w, x + size)
    y2 = min(h, y + size)

    region = binary_img[y:y2, x:x2]
    if region.size == 0:
        return False

    white_ratio = np.mean(region > 0)
    return white_ratio > threshold