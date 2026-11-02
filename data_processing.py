import numpy as np
from collections import defaultdict

def average_duplicates(wavelengths, transmittances):
    """
    Усредняет пропускание для одинаковых длин волн.
    Принимает списки wavelengths и transmittances.
    Возвращает два отсортированных списка (wl_unique, tr_avg).
    """
    data = defaultdict(list)
    for wl, tr in zip(wavelengths, transmittances):
        data[wl].append(tr)

    wl_unique = sorted(data.keys())
    tr_avg = [np.mean(data[wl]) for wl in wl_unique]
    return wl_unique, tr_avg

def median_filter(wavelengths, transmittances, window_size=5):
    """
    Применяет медианный фильтр к списку пропускания.
    Длина волны остаётся без изменений.
    """
    if len(transmittances) < window_size:
        return wavelengths, transmittances
    half = window_size // 2
    filtered = []
    for i in range(len(transmittances)):
        start = max(0, i - half)
        end = min(len(transmittances), i + half + 1)
        filtered.append(np.median(transmittances[start:end]))
    return wavelengths, filtered

def remove_wavelength_jumps(wavelengths, transmittances, threshold=1.0):
    """
    Удаляет точки, где длина волны изменяется скачком более чем на threshold нм.
    Возвращает очищенные списки.
    """
    if not wavelengths:
        return [], []
    clean_wl = [wavelengths[0]]
    clean_tr = [transmittances[0]]
    last_wl = wavelengths[0]
    for wl, tr in zip(wavelengths[1:], transmittances[1:]):
        if abs(wl - last_wl) <= threshold:
            clean_wl.append(wl)
            clean_tr.append(tr)
            last_wl = wl
        # иначе пропускаем точку
    return clean_wl, clean_tr

def process_spectrum(wavelengths, transmittances,
                     jump_threshold=1.0, median_window=5):
    """
    Полный цикл постобработки:
    1. Удаляет скачки длины волны.
    2. Усредняет дубликаты по длине волны.
    3. Применяет медианный фильтр.
    Возвращает финальные списки (wl, tr).
    """
    wl_clean, tr_clean = remove_wavelength_jumps(wavelengths, transmittances, jump_threshold)
    wl_unique, tr_avg = average_duplicates(wl_clean, tr_clean)
    wl_final, tr_final = median_filter(wl_unique, tr_avg, median_window)
    return wl_final, tr_final

def load_csv(filename):
    """Загружает CSV с колонками Wavelength и Transmittance."""
    data = np.genfromtxt(filename, delimiter=',', names=True)
    return data['Wavelength'].tolist(), data['Transmittance'].tolist()

def save_csv(filename, wavelengths, transmittances):
    """Сохраняет обработанные данные в CSV."""
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Wavelength', 'Transmittance'])
        for wl, tr in zip(wavelengths, transmittances):
            writer.writerow([f"{wl:.1f}", f"{tr:.1f}"])