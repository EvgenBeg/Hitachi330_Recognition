import cv2
import numpy as np
import json
import os
import csv
from datetime import datetime
import time
from collections import defaultdict, deque

from camera import Camera
from color import preprocess_roi, is_segment_on
from segments import digit_from_states, point_is_on


def load_json(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден")
    with open(filename, 'r') as f:
        return json.load(f)


def _compute_roi_scale(segment_config, frame_w, frame_h):
    """Коэффициенты масштабирования ROI из ref-разрешения в текущий кадр.
    Нужны, чтобы ROI, сохранённые при 1920×1080, корректно работали при 640×480
    и наоборот. Тот же подход, что в gui._roi_scale."""
    saved_w = int(segment_config.get('_ref_width', 0) or 0)
    saved_h = int(segment_config.get('_ref_height', 0) or 0)
    if saved_w > 0 and saved_h > 0 and (saved_w != frame_w or saved_h != frame_h):
        return frame_w / saved_w, frame_h / saved_h
    return 1.0, 1.0


def recognize_digit(binary_frame, roi, digit_idx, roi_cfg,
                    threshold=0.30, size=9):
    states = []
    for seg in roi['digits'][digit_idx]['segments']:
        abs_x = roi['x'] + seg['x'] * roi['w']
        abs_y = roi['y'] + seg['y'] * roi['h']
        cx = roi['x'] + roi['w'] / 2
        cy = roi['y'] + roi['h'] / 2
        scale = roi_cfg.get('scale', 1.0)
        angle = roi_cfg.get('angle', 0)
        x = cx + (abs_x - cx) * scale
        y = cy + (abs_y - cy) * scale
        import math
        rad = math.radians(angle)
        dx = x - cx
        dy = y - cy
        x = cx + dx * math.cos(rad) - dy * math.sin(rad)
        y = cy + dx * math.sin(rad) + dy * math.cos(rad)
        state = is_segment_on(binary_frame, x, y, size=size, threshold=threshold)
        states.append(state)
    return digit_from_states(states)


def recognize_point(binary_frame, roi, roi_cfg, threshold=0.20, size=9):
    if 'point' not in roi:
        return False
    abs_x = roi['x'] + roi['point']['x'] * roi['w']
    abs_y = roi['y'] + roi['point']['y'] * roi['h']
    cx = roi['x'] + roi['w'] / 2
    cy = roi['y'] + roi['h'] / 2
    scale = roi_cfg.get('scale', 1.0)
    angle = roi_cfg.get('angle', 0)
    x = cx + (abs_x - cx) * scale
    y = cy + (abs_y - cy) * scale
    import math
    rad = math.radians(angle)
    dx = x - cx
    dy = y - cy
    x = cx + dx * math.cos(rad) - dy * math.sin(rad)
    y = cy + dx * math.sin(rad) + dy * math.cos(rad)
    return point_is_on(binary_frame, x, y, size=size, threshold=threshold)


def _get_ordinate_name(mode):
    """Возвращает название столбца ординат в зависимости от режима."""
    if mode == 'Transmission':
        return 'Transmittance, %'
    elif mode == 'Reflection':
        return 'Reflectance, %'
    elif mode == 'Absorption':
        return 'Optical density'
    else:
        return 'Value'


def _write_metadata(writer, mode, scan_speed, scale_min, scale_max, comment):
    """Записывает метаданные измерения и заголовок таблицы в CSV."""
    date_str = datetime.now().strftime("%d.%m.%Y")
    writer.writerow([f"Date: {date_str}"])
    writer.writerow([f"Mode: {mode}"])
    writer.writerow([f"Scan speed: {scan_speed} nm/min"])
    writer.writerow([f"Scale: {scale_min} - {scale_max} %"])
    writer.writerow([f"Comment: {comment}"])
    writer.writerow([])  # пустая строка перед таблицей
    writer.writerow(['Wavelength, nm', _get_ordinate_name(mode)])


def start_recognition(camera, segment_config, roi_config, hsv_lower, hsv_upper,
                      save_directory, spectrum_name,
                      wavelength_jump_threshold=3.0,
                      min_wavelength=200.0,
                      max_wavelength=2500.0,
                      min_transmittance=0.0,
                      max_transmittance=105.0,
                      fill_threshold_digit=0.30, fill_threshold_point=0.10,
                      mode='Transmission', scan_speed=60.0,
                      scale_min=0.0, scale_max=100.0, comment='',
                      stop_event=None, data_callback=None, frame_callback=None):
    """
    Запускает процесс распознавания и записи спектра.

    Важно: roi_config / segment_config содержат абсолютные координаты ROI
    в разрешении, при котором калибровали (_ref_width/_ref_height).
    Если камера сейчас отдаёт другое разрешение, координаты масштабируются
    автоматически, чтобы распознавание читало те же участки сцены.
    """
    roi1 = segment_config['rois'][0]
    roi2 = segment_config['rois'][1]
    roi1_cfg = roi_config['roi1']
    roi2_cfg = roi_config['roi2']

    # ---- Размер сегмента из мастера калибровки ----
    seg_size = int(segment_config.get('_seg_size', 9))
    seg_size = max(1, min(seg_size, 40))
    print(f"Размер сегмента (из мастера): {seg_size}")

    # ---- Первое чтение кадра: нужно для определения разрешения и масштаба ----
    ret, first_frame = camera.read()
    if not ret or first_frame is None:
        print("Не удалось получить первый кадр — прерываем измерение")
        return None
    fh, fw = first_frame.shape[:2]
    print(f"Разрешение кадра: {fw}x{fh}")

    # ---- Коэффициенты масштабирования ROI под текущее разрешение ----
    rx, ry = _compute_roi_scale(segment_config, fw, fh)
    if rx != 1.0 or ry != 1.0:
        print(f"ROI масштабируются: rx={rx:.3f}, ry={ry:.3f} "
              f"(ref={segment_config.get('_ref_width')}x"
              f"{segment_config.get('_ref_height')} → {fw}x{fh})")

    # Применяем масштаб к ROI. Дальше работаем с уже масштабированными
    # координатами — recognize_digit/recognize_point получают их как есть.
    def scale_roi(roi, roi_cfg):
        roi['x'] = int(roi_cfg['x'] * rx)
        roi['y'] = int(roi_cfg['y'] * ry)
        roi['w'] = int(roi_cfg['w'] * rx)
        roi['h'] = int(roi_cfg['h'] * ry)

    scale_roi(roi1, roi1_cfg)
    scale_roi(roi2, roi2_cfg)

    # Масштабируем и seg_size под текущее разрешение (если ref != cur)
    seg_size_rx = max(1, int(round(seg_size * rx)))
    seg_size_ry = max(1, int(round(seg_size * ry)))
    # is_segment_on использует квадрат size×size, берём среднее
    seg_size_scaled = max(1, (seg_size_rx + seg_size_ry) // 2)
    if seg_size_scaled != seg_size:
        print(f"Размер сегмента после масштабирования: {seg_size_scaled}")

    raw_data = []
    processed_data = defaultdict(lambda: deque(maxlen=5))
    last_wavelength = None
    start_wavelength = None
    consecutive_jumps = 0
    rejected_count = 0

    csv_path = os.path.join(save_directory, f"{spectrum_name}.csv")
    raw_csv_path = os.path.join(save_directory, f"{spectrum_name}_raw.csv")

    print(f"Начало измерения. Mode={mode}, Scan speed={scan_speed} nm/min, "
          f"Scale={scale_min}-{scale_max} %, Comment='{comment}'")
    print(f"Обработанные данные: {csv_path}")
    print(f"Сырые данные: {raw_csv_path}")
    print(f"Диапазон длины волны: [{min_wavelength}, {max_wavelength}] нм")

    # Первый кадр уже получен — используем его как стартовый,
    # чтобы не терять данные.
    pending_frame = first_frame

    while True:
        if stop_event is not None and stop_event.is_set():
            print("Остановлено пользователем")
            break

        if pending_frame is not None:
            frame = pending_frame
            pending_frame = None
        else:
            ret, frame = camera.read()
            if not ret:
                print("Не удалось получить кадр, попытка переподключения...")
                if not camera.reconnect():
                    print("Не удалось переподключиться к камере")
                    break
                continue

        if frame_callback:
            frame_callback(frame.copy())

        binary_frame = preprocess_roi(frame, hsv_lower, hsv_upper)

        # ---- Длина волны (ROI1) ----
        wavelength_digits = []
        for i in range(4):
            digit = recognize_digit(binary_frame, roi1, i, roi1_cfg,
                                    fill_threshold_digit,
                                    size=seg_size_scaled)
            wavelength_digits.append(digit)
        point_on_wl = recognize_point(binary_frame, roi1, roi1_cfg,
                                      fill_threshold_point,
                                      size=seg_size_scaled)

        wavelength = ""
        if all(d != '?' and d != '' for d in wavelength_digits):
            wavelength = ''.join(wavelength_digits)
            if point_on_wl:
                wavelength = wavelength[:-1] + '.' + wavelength[-1]

        # ---- Пропускание (ROI2) ----
        transmittance_digits = []
        for i in range(4):
            digit = recognize_digit(binary_frame, roi2, i, roi2_cfg,
                                    fill_threshold_digit,
                                    size=seg_size_scaled)
            transmittance_digits.append(digit)
        point_on_tr = recognize_point(binary_frame, roi2, roi2_cfg,
                                      fill_threshold_point,
                                      size=seg_size_scaled)

        while transmittance_digits and transmittance_digits[0] == '':
            transmittance_digits.pop(0)

        transmittance = ""
        if transmittance_digits and all(d != '?' for d in transmittance_digits):
            if point_on_tr:
                transmittance = ''.join(transmittance_digits[:-1]) + '.' + transmittance_digits[-1]
            else:
                transmittance = ''.join(transmittance_digits)

        if wavelength and transmittance:
            try:
                wl_val = float(wavelength)
                tr_val = float(transmittance)
            except ValueError:
                continue

            # ===== 1. ОСТАНОВКА: нижняя граница =====
            if wl_val <= min_wavelength:
                print(f"Достигнута нижняя граница {min_wavelength} нм "
                      f"(λ={wl_val:.1f}). Завершение.")
                break

            # ===== 2. Проверка верхней границы =====
            if wl_val > max_wavelength:
                rejected_count += 1
                print(f"λ={wl_val:.1f} выше {max_wavelength}, пропуск")
                continue

            # ===== 3. Проверка пропускания =====
            if not (min_transmittance <= tr_val <= max_transmittance):
                rejected_count += 1
                print(f"T={tr_val:.1f} вне [{min_transmittance}, {max_transmittance}], пропуск")
                continue

            # ===== 4. Первое успешное измерение =====
            if start_wavelength is None:
                start_wavelength = wl_val
                last_wavelength = wl_val
                print(f"Начало измерения: λ={wl_val:.1f} нм")

                raw_data.append((wl_val, tr_val))
                processed_data[wl_val].append(tr_val)
                if data_callback:
                    wl_sorted = sorted(processed_data.keys(), reverse=True)
                    tr_medians = [float(np.median(list(processed_data[w]))) for w in wl_sorted]
                    data_callback(wl_val, tr_val, wl_sorted, tr_medians)
                continue

            # ===== 5. Фильтр "измерение идёт вниз" =====
            ASCENT_TOLERANCE = 2.0

            if wl_val > start_wavelength + ASCENT_TOLERANCE:
                rejected_count += 1
                print(f"λ={wl_val:.1f} выше стартовой {start_wavelength:.1f}, "
                      f"пропуск (ошибка распознавания)")
                continue

            if wl_val > last_wavelength + ASCENT_TOLERANCE:
                rejected_count += 1
                print(f"λ={wl_val:.1f} выше предыдущей {last_wavelength:.1f}, "
                      f"пропуск (ошибка распознавания)")
                continue

            if last_wavelength - wl_val > wavelength_jump_threshold:
                consecutive_jumps += 1
                if consecutive_jumps < 3:
                    rejected_count += 1
                    print(f"λ={wl_val:.1f} резко ниже {last_wavelength:.1f}, "
                          f"пропуск ({consecutive_jumps}/3)")
                    continue
            else:
                consecutive_jumps = 0

            # ===== 6. Обновляем last и сохраняем =====
            last_wavelength = wl_val
            raw_data.append((wl_val, tr_val))
            processed_data[wl_val].append(tr_val)

            if data_callback:
                wl_sorted = sorted(processed_data.keys(), reverse=True)
                tr_medians = [float(np.median(list(processed_data[w]))) for w in wl_sorted]
                data_callback(wl_val, tr_val, wl_sorted, tr_medians)

            print(f"[{datetime.now().isoformat(timespec='seconds')}] λ={wl_val:.1f}, T={tr_val:.1f}")

        time.sleep(0.01)

    # ---- Сохранение RAW ----
    print(f"Сохранение сырых данных в {raw_csv_path}...")
    with open(raw_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        _write_metadata(writer, mode, scan_speed, scale_min, scale_max, comment)
        for wl, tr in raw_data:
            writer.writerow([f"{wl:.1f}", f"{tr:.1f}"])

    # ---- Сохранение PROCESSED ----
    print(f"Сохранение обработанных данных в {csv_path}...")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        _write_metadata(writer, mode, scan_speed, scale_min, scale_max, comment)
        for wl in sorted(processed_data.keys(), reverse=True):
            median_tr = np.median(list(processed_data[wl]))
            writer.writerow([f"{wl:.1f}", f"{median_tr:.1f}"])

    print(f"Готово. RAW: {len(raw_data)}, PROCESSED: {len(processed_data)}, "
          f"отклонено: {rejected_count}")
    return csv_path