# segments.py

# Таблица кодов цифр для 7-сегментного индикатора.
# Порядок сегментов: A B C D E F G
# Каждый символ кода: '1' = сегмент включён, '0' = выключен
DIGIT_CODES = {
    '0000000': '',   # нет цифры
    '0101000': '1',
    '1110110': '2',
    '1111100': '3',
    '0111001': '4',
    '1011101': '5',
    '0011111': '6',
    '1101000': '7',
    '1111111': '8',
    '1111001': '9',
    '1101111': '0'
}

def code_from_states(states):
    """
    Преобразует список состояний семи сегментов (True/False или 1/0) в строку кода.
    """
    return ''.join('1' if s else '0' for s in states)

def digit_from_code(code):
    """
    По строке кода из 7 символов возвращает цифру или '?' если код неизвестен.
    """
    return DIGIT_CODES.get(code, '?')

def digit_from_states(states):
    """
    По списку состояний семи сегментов возвращает распознанную цифру.
    states: список из 7 элементов (True/False или 1/0)
    """
    code = code_from_states(states)
    return digit_from_code(code)

def point_is_on(binary_img, x, y, size=9, threshold=0.20):
    """
    Проверяет, горит ли точка (квадрат size x size) в позиции (x,y) на бинарном изображении.
    Возвращает True, если доля белых пикселей превышает threshold.
    """
    import numpy as np
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