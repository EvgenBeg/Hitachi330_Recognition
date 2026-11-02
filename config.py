import json
import os
import sys

def _app_dir():
    """Возвращает папку, где лежат ресурсы (для .exe — временная папка PyInstaller)."""
    if getattr(sys, 'frozen', False):
        # Запущено как .exe: ресурсы распакованы в sys._MEIPASS
        return sys._MEIPASS
    else:
        # Обычный запуск из Python
        return os.path.dirname(os.path.abspath(__file__))

class Config:
    """
    Класс для управления конфигурацией приложения.
    Загружает и сохраняет настройки в JSON-файл.
    """

    DEFAULT_CONFIG = {
        "camera": {
            "source_type": "webcam",
            "source": 0,
            "rotate": 90,
            "fps": 30,
            "resolution": "640x480"
        },
        "paths": {
            "segment_config": "segment_config.json",
            "roi_config": "roi_config.json",
            "save_directory": os.getcwd()
        },
        "hsv": {
            "lower": [0, 100, 158],
            "upper": [179, 255, 255]
        },
        "recognition": {
            "fill_threshold_digit": 0.30,
            "fill_threshold_point": 0.20,
            "segment_size": 9,
            "wavelength_jump_threshold": 1.0,
            "min_wavelength": 187.0
        },
        "gui": {
            "base_width": 1920,
            "base_height": 1080,
            "font_size": 24
        }
    }

    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.data = self.load()

    def load(self):
        """Загружает конфигурацию из файла, при отсутствии использует значения по умолчанию."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # Рекурсивно обновляем значения по умолчанию, чтобы не потерять новые ключи
                return self._merge(self.DEFAULT_CONFIG.copy(), loaded)
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")
        return self.DEFAULT_CONFIG.copy()

    def save(self):
        """Сохраняет текущую конфигурацию в файл."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            print(f"Конфигурация сохранена в {self.config_file}")
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")

    def _merge(self, base, updates):
        """Рекурсивно объединяет словари, дополняя base значениями из updates."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge(base[key], value)
            else:
                base[key] = value
        return base

    # ---------- Удобные методы доступа ----------
    def get(self, *keys, default=None):
        """Получает значение по цепочке ключей. Пример: get('camera', 'source')"""
        node = self.data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def set(self, *keys, value):
        """Устанавливает значение по цепочке ключей. Пример: set('camera', 'source', value=1)"""
        node = self.data
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

    # ---------- Специализированные методы для камеры ----------
    def get_camera_rotate_code(self):
        """Преобразует угол поворота из конфига в код OpenCV."""
        angle = self.get('camera', 'rotate', default=0)
        import cv2
        if angle == 90:
            return cv2.ROTATE_90_CLOCKWISE
        elif angle == 180:
            return cv2.ROTATE_180
        elif angle == 270:
            return cv2.ROTATE_90_COUNTERCLOCKWISE
        return None

    def get_hsv_ranges(self):
        """Возвращает нижний и верхний HSV пороги как кортеж numpy массивов."""
        import numpy as np
        lower = self.get('hsv', 'lower')
        upper = self.get('hsv', 'upper')
        return np.array(lower), np.array(upper)