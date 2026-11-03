import json
import os
import sys

def _exe_dir():
    """Возвращает папку, где лежит .exe (или .py при обычном запуске)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

# ---------- Данные по умолчанию (взяты из ваших файлов) ----------
DEFAULT_HSV_CONFIG = {
    "H_MIN": 54,
    "H_MAX": 179,
    "S_MIN": 114,
    "S_MAX": 255,
    "V_MIN": 100,
    "V_MAX": 255
}

DEFAULT_ROI_CONFIG = {
    "roi1": {
        "x": 61, "y": 110, "w": 338, "h": 132, "scale": 1.0, "angle": 0
    },
    "roi2": {
        "x": 40, "y": 346, "w": 410, "h": 130, "scale": 1.0, "angle": 0
    }
}

DEFAULT_SEGMENT_CONFIG = {
    "rois": [
        {
            "x": 61, "y": 110, "w": 338, "h": 132,
            "digits": [
                {
                    "segments": [
                        {"x": 0.28402366863905326, "y": 0.2878787878787879},
                        {"x": 0.3609467455621302, "y": 0.45454545454545453},
                        {"x": 0.30177514792899407, "y": 0.5681818181818182},
                        {"x": 0.23372781065088757, "y": 0.4772727272727273},
                        {"x": 0.257396449704142, "y": 0.11363636363636363},
                        {"x": 0.1804733727810651, "y": 0.6136363636363636},
                        {"x": 0.1804733727810651, "y": 0.32575757575757575}
                    ]
                },
                {
                    "segments": [
                        {"x": 0.4822485207100592, "y": 0.3333333333333333},
                        {"x": 0.5591715976331361, "y": 0.5},
                        {"x": 0.5, "y": 0.6136363636363636},
                        {"x": 0.4319526627218935, "y": 0.5227272727272727},
                        {"x": 0.4556213017751479, "y": 0.1590909090909091},
                        {"x": 0.378698224852071, "y": 0.6590909090909091},
                        {"x": 0.378698224852071, "y": 0.3712121212121212}
                    ]
                },
                {
                    "segments": [
                        {"x": 0.7011834319526628, "y": 0.2878787878787879},
                        {"x": 0.7781065088757396, "y": 0.45454545454545453},
                        {"x": 0.7189349112426036, "y": 0.5681818181818182},
                        {"x": 0.650887573964497, "y": 0.4772727272727273},
                        {"x": 0.6745562130177515, "y": 0.11363636363636363},
                        {"x": 0.5976331360946746, "y": 0.6136363636363636},
                        {"x": 0.5976331360946746, "y": 0.32575757575757575}
                    ]
                },
                {
                    "segments": [
                        {"x": 0.8254437869822485, "y": 0.3712121212121212},
                        {"x": 0.9023668639053254, "y": 0.5378787878787878},
                        {"x": 0.8431952662721893, "y": 0.6515151515151515},
                        {"x": 0.7751479289940828, "y": 0.5606060606060606},
                        {"x": 0.7988165680473372, "y": 0.19696969696969696},
                        {"x": 0.7218934911242604, "y": 0.696969696969697},
                        {"x": 0.7218934911242604, "y": 0.4090909090909091}
                    ]
                }
            ],
            "point": {"x": 0.650887573964497, "y": 0.6666666666666666}
        },
        {
            "x": 40, "y": 346, "w": 410, "h": 130,
            "digits": [
                {
                    "segments": [
                        {"x": 0.17560975609756097, "y": 0.3923076923076923},
                        {"x": 0.23170731707317074, "y": 0.5307692307692308},
                        {"x": 0.3121951219512195, "y": 0.5615384615384615},
                        {"x": 0.1902439024390244, "y": 0.2},
                        {"x": 0.2073170731707317, "y": 0.2692307692307692},
                        {"x": 0.12926829268292683, "y": 0.6153846153846154},
                        {"x": 0.25609756097560976, "y": 0.33076923076923076}
                    ]
                },
                {
                    "segments": [
                        {"x": 0.3, "y": 0.45384615384615384},
                        {"x": 0.35609756097560974, "y": 0.5923076923076923},
                        {"x": 0.43658536585365854, "y": 0.6230769230769231},
                        {"x": 0.3146341463414634, "y": 0.26153846153846155},
                        {"x": 0.33170731707317075, "y": 0.33076923076923076},
                        {"x": 0.25365853658536586, "y": 0.676923076923077},
                        {"x": 0.3804878048780488, "y": 0.3923076923076923}
                    ]
                },
                {
                    "segments": [
                        {"x": 0.5024390243902439, "y": 0.35384615384615387},
                        {"x": 0.5585365853658537, "y": 0.49230769230769234},
                        {"x": 0.6390243902439025, "y": 0.5230769230769231},
                        {"x": 0.5170731707317073, "y": 0.16153846153846155},
                        {"x": 0.5341463414634147, "y": 0.23076923076923078},
                        {"x": 0.4560975609756098, "y": 0.5769230769230769},
                        {"x": 0.5829268292682926, "y": 0.2923076923076923}
                    ]
                },
                {
                    "segments": [
                        {"x": 0.7853658536585366, "y": 0.5307692307692308},
                        {"x": 0.8414634146341463, "y": 0.6692307692307692},
                        {"x": 0.9219512195121952, "y": 0.7},
                        {"x": 0.8, "y": 0.3384615384615385},
                        {"x": 0.8170731707317073, "y": 0.4076923076923077},
                        {"x": 0.7390243902439024, "y": 0.7538461538461538},
                        {"x": 0.8658536585365854, "y": 0.46923076923076923}
                    ]
                }
            ],
            "point": {"x": 0.6731707317073171, "y": 0.6538461538461539}
        }
    ]
}

def ensure_files_exist():
    """Создаёт файлы конфигурации с настройками по умолчанию, если их нет."""
    base = _exe_dir()
    files = {
        "hsv_config.json": DEFAULT_HSV_CONFIG,
        "segment_config.json": DEFAULT_SEGMENT_CONFIG,
        "roi_config.json": DEFAULT_ROI_CONFIG
    }
    for filename, default_data in files.items():
        filepath = os.path.join(base, filename)
        if not os.path.exists(filepath):
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(default_data, f, indent=4)
                print(f"Создан файл по умолчанию: {filepath}")
            except Exception as e:
                print(f"Не удалось создать {filename}: {e}")

class Config:
    DEFAULT_CONFIG = {
        "camera": {
            "source_type": "webcam",
            "source": 0,
            "rotate": 0,
            "crop": None,
            "prefer_mjpeg": True,
            "capture": {
                "width": 1920,
                "height": 1080,
                "fps": 30
            }
        },
        "paths": {
            "segment_config": "segment_config.json",
            "roi_config": "roi_config.json",
            "hsv_config": "hsv_config.json",
            "save_directory": "C:/Users/Evgenii/Desktop/Спектрофотометр_данные"
        },
        "hsv": {
            "lower": [54, 114, 100],
            "upper": [179, 255, 255]
        },
        "recognition": {
            "fill_threshold_digit": 0.3,
            "fill_threshold_point": 0.2,
            "segment_size": 9,
            "wavelength_jump_threshold": 5.0,
            "min_wavelength": 200.0
        },
        "gui": {
            "base_width": 1920,
            "base_height": 1080,
            "font_size": 24
        }
    }

    def __init__(self, config_file="config.json"):
        # Всегда используем полный путь к папке с exe/py
        self.config_file = os.path.join(_exe_dir(), config_file)
        self.data = self.load()
        # Если файла не было, создаём его с настройками по умолчанию
        if not os.path.exists(self.config_file):
            self.save()

    def load(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                return self._merge(self.DEFAULT_CONFIG.copy(), loaded)
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")
        return self.DEFAULT_CONFIG.copy()

    def save(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            print(f"Конфигурация сохранена в {self.config_file}")
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")

    def _merge(self, base, updates):
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge(base[key], value)
            else:
                base[key] = value
        return base

    def get(self, *keys, default=None):
        node = self.data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def set(self, *keys, value):
        node = self.data
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

    def get_camera_rotate_code(self):
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
        import numpy as np
        lower = self.get('hsv', 'lower')
        upper = self.get('hsv', 'upper')
        return np.array(lower), np.array(upper)

    def get_absolute_path(self, key):
        """Возвращает абсолютный путь к файлу конфигурации."""
        path = self.get('paths', key)
        if not path:
            return None
        if not os.path.isabs(path):
            return os.path.join(_exe_dir(), path)
        return path