import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import numpy as np
import os
import time
import cv2
import threading

from camera import Camera
from config import Config
from utils import load_json, bgr_to_photoimage
from recognition import start_recognition
import calibration


class MainApp:
    def __init__(self, root, camera, config):
        self.root = root
        self.root.title("Спектрофотометр")

        self.camera = camera
        self.config = config

        self.base_width = config.get('gui', 'base_width', default=1920)
        self.base_height = config.get('gui', 'base_height', default=1080)
        self.font_size = config.get('gui', 'font_size', default=24)

        # ---- Размер и положение окна: половина экрана, по центру ----
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = screen_w // 2
        win_h = screen_h // 2
        min_w, min_h = 800, 600
        if win_w < min_w:
            win_w = min_w
        if win_h < min_h:
            win_h = min_h
        pos_x = (screen_w - win_w) // 2
        pos_y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")
        self.root.minsize(min_w, min_h)

        self.pages = {}
        self.current_page = None
        self.tab_buttons = {}

        self.save_directory = config.get('paths', 'save_directory', default=os.getcwd())

        self.video_running = False
        self.video_after_id = None
        self.measurement_running = False
        self.measurement_stop_event = threading.Event()
        self.measurement_thread = None

        # Данные графика
        self.measurement_wavelengths = []
        self.measurement_transmittances = []
        self.raw_wavelengths = []
        self.raw_transmittances = []

        self.segment_config = load_json(config.get_absolute_path('segment_config'))
        self.roi_config = load_json(config.get_absolute_path('roi_config'))

        self.create_tab_buttons()
        self.create_pages()
        self.show_page("measurement")
        self.root.bind("<Configure>", self.on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._no_frame_count = 0
        self._update_camera_buttons()

    # ---------- Масштабирование ----------
    def on_resize(self, event):
        if event.widget is not self.root:
            return
        if not hasattr(self, 'measurement_page_created'):
            return
        try:
            if not self.roi1_frame.winfo_exists():
                return
        except Exception:
            return
        self.update_measurement_layout()

    def get_scale(self):
        current_width = self.root.winfo_width()
        current_height = self.root.winfo_height()
        scale_x = current_width / self.base_width
        scale_y = current_height / self.base_height
        return scale_x, scale_y

    # ---------- Вкладки ----------
    def create_tab_buttons(self):
        top_frame = tk.Frame(self.root, bg='lightgray')
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tabs = [
            ("measurement", "Измерение"),
            ("video", "Видео"),
            ("data_view", "Просмотр данных"),
            ("settings", "Настройки"),
        ]
        for key, text in tabs:
            btn = tk.Button(top_frame, text=text,
                            command=lambda k=key: self.show_page(k),
                            font=('Arial', 12))
            btn.pack(side=tk.LEFT, padx=5, pady=5)
            self.tab_buttons[key] = btn

    def create_pages(self):
        self.page_container = tk.Frame(self.root)
        self.page_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        measurement_page = tk.Frame(self.page_container)
        self.pages["measurement"] = measurement_page
        self.setup_measurement_page(measurement_page)
        self.measurement_page_created = True

        video_page = tk.Frame(self.page_container)
        self.pages["video"] = video_page
        self.setup_video_page(video_page)

        data_view_page = tk.Frame(self.page_container)
        self.pages["data_view"] = data_view_page
        self.setup_data_view_page(data_view_page)

        settings_page = tk.Frame(self.page_container)
        self.pages["settings"] = settings_page
        self.setup_settings_page(settings_page)

    def show_page(self, page_name):
        if self.current_page is not None:
            self.pages[self.current_page].pack_forget()
        self.pages[page_name].pack(fill=tk.BOTH, expand=True)
        self.current_page = page_name

        for name, btn in self.tab_buttons.items():
            if name == page_name:
                btn.config(relief=tk.SUNKEN, bg='white')
            else:
                btn.config(relief=tk.RAISED, bg='lightgray')

    # ---------- Страница измерения ----------
    def setup_measurement_page(self, page):
        self.measurement_wavelengths = []
        self.measurement_transmittances = []
        self.raw_wavelengths = []
        self.raw_transmittances = []

        self.measurement_top_panel = tk.Frame(page, height=150)
        self.measurement_top_panel.pack(fill=tk.X, padx=10, pady=5)
        self.measurement_top_panel.pack_propagate(False)

        self.roi1_frame = tk.Frame(self.measurement_top_panel, bg='black')
        self.roi1_label = tk.Label(self.roi1_frame, text="ROI 1", bg='black', fg='white')
        self.roi1_label.pack(expand=True)

        self.roi2_frame = tk.Frame(self.measurement_top_panel, bg='black')
        self.roi2_label = tk.Label(self.roi2_frame, text="ROI 2", bg='black', fg='white')
        self.roi2_label.pack(expand=True)

        self.name_frame = tk.Frame(self.measurement_top_panel)
        self.name_label = tk.Label(self.name_frame, text="Название спектра:", font=('Arial', self.font_size))
        self.name_label.pack(anchor='w')
        self.spectrum_name_entry = tk.Entry(self.name_frame, width=30, font=('Arial', self.font_size))
        self.spectrum_name_entry.pack(anchor='w')

        self.start_measurement_btn = tk.Button(self.measurement_top_panel, text="Scan",
                                               command=self.start_measurement,
                                               font=('Arial', self.font_size), bg='lightgreen')
        self.stop_measurement_btn = tk.Button(self.measurement_top_panel, text="Stop",
                                              command=self.stop_measurement,
                                              font=('Arial', self.font_size), bg='lightcoral',
                                              state=tk.DISABLED)

        chart_controls = tk.Frame(page)
        chart_controls.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        graph_frame = tk.Frame(chart_controls)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.measurement_fig = Figure(figsize=(6, 4), dpi=100)
        self.measurement_ax = self.measurement_fig.add_subplot(111)
        self.measurement_ax.set_xlabel("Длина волны, нм")
        self.measurement_ax.set_ylabel("Коэффициент пропускания / отражения")
        self.measurement_ax.grid(True)
        self.measurement_canvas = FigureCanvasTkAgg(self.measurement_fig, master=graph_frame)
        self.measurement_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        controls_frame = tk.LabelFrame(chart_controls, text="Параметры измерения",
                                       font=('Arial', 12))
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        tk.Label(controls_frame, text="Mode:", font=('Arial', 12)).grid(
            row=0, column=0, sticky='w', padx=5, pady=8)
        self.mode_var = tk.StringVar(value='Transmission')
        mode_menu = tk.OptionMenu(controls_frame, self.mode_var,
                                  'Transmission', 'Reflection', 'Absorption')
        mode_menu.config(font=('Arial', 12), width=14)
        mode_menu.grid(row=0, column=1, sticky='w', padx=5, pady=8)

        tk.Label(controls_frame, text="Scan speed:", font=('Arial', 12)).grid(
            row=1, column=0, sticky='w', padx=5, pady=8)
        speed_frame = tk.Frame(controls_frame)
        speed_frame.grid(row=1, column=1, sticky='w', padx=5, pady=8)
        self.scan_speed_var = tk.StringVar(value='60')
        tk.Entry(speed_frame, textvariable=self.scan_speed_var, width=6,
                 font=('Arial', 12)).pack(side=tk.LEFT)
        tk.Label(speed_frame, text=" nm/min", font=('Arial', 12)).pack(side=tk.LEFT)

        tk.Label(controls_frame, text="Scale:", font=('Arial', 12)).grid(
            row=2, column=0, sticky='w', padx=5, pady=8)
        scale_frame = tk.Frame(controls_frame)
        scale_frame.grid(row=2, column=1, sticky='w', padx=5, pady=8)
        self.scale_min_var = tk.StringVar(value='0')
        self.scale_max_var = tk.StringVar(value='100')
        tk.Entry(scale_frame, textvariable=self.scale_min_var, width=5,
                 font=('Arial', 12)).pack(side=tk.LEFT)
        tk.Label(scale_frame, text=" - ", font=('Arial', 12)).pack(side=tk.LEFT)
        tk.Entry(scale_frame, textvariable=self.scale_max_var, width=5,
                 font=('Arial', 12)).pack(side=tk.LEFT)
        tk.Label(scale_frame, text=" %", font=('Arial', 12)).pack(side=tk.LEFT)

        tk.Label(controls_frame, text="Comment:", font=('Arial', 12)).grid(
            row=3, column=0, sticky='w', padx=5, pady=8)
        self.comment_var = tk.StringVar(value='')
        tk.Entry(controls_frame, textvariable=self.comment_var, width=22,
                 font=('Arial', 12)).grid(row=3, column=1, sticky='w', padx=5, pady=8)

        tk.Label(controls_frame, text="Graph update:", font=('Arial', 12)).grid(
            row=4, column=0, sticky='w', padx=5, pady=(40, 8))
        self.graph_update_var = tk.StringVar(value='1')
        graph_menu = tk.OptionMenu(controls_frame, self.graph_update_var,
                                   '1', '2', '5', '10', '20', '50')
        graph_menu.config(font=('Arial', 12), width=6)
        graph_menu.grid(row=4, column=1, sticky='w', padx=5, pady=(40, 8))

        tk.Label(controls_frame, text="обновлять каждый N-й замер",
                 font=('Arial', 9), fg='gray').grid(
            row=5, column=1, sticky='w', padx=5, pady=(0, 8))

        self.graph_update_counter = 0

        self.update_measurement_layout()

    def update_measurement_layout(self):
        scale_x, scale_y = self.get_scale()
        ROI_WIDTH = 390 * scale_x
        ROI_HEIGHT = 130 * scale_y
        ROI1_X = 5 * scale_x
        ROI1_Y = 20 * scale_y
        ROI2_X = 410 * scale_x
        ROI2_Y = 20 * scale_y
        NAME_X = 900 * scale_x
        NAME_Y = 30 * scale_y
        BUTTON_X = 1500 * scale_x
        BUTTON_Y = 60 * scale_y
        STOP_X = 1700 * scale_x

        self.roi1_frame.place(x=ROI1_X, y=ROI1_Y, width=ROI_WIDTH, height=ROI_HEIGHT)
        self.roi2_frame.place(x=ROI2_X, y=ROI2_Y, width=ROI_WIDTH, height=ROI_HEIGHT)
        self.name_frame.place(x=NAME_X, y=NAME_Y)
        self.start_measurement_btn.place(x=BUTTON_X, y=BUTTON_Y)
        self.stop_measurement_btn.place(x=STOP_X, y=BUTTON_Y)

        font_size = int(self.font_size * min(scale_x, scale_y))
        self.name_label.config(font=('Arial', font_size))
        self.spectrum_name_entry.config(font=('Arial', font_size))
        self.start_measurement_btn.config(font=('Arial', font_size))
        self.stop_measurement_btn.config(font=('Arial', font_size))

    def start_measurement(self):
        if self.measurement_running:
            return
        if not self.camera.connected:
            if not self.camera.connect():
                messagebox.showerror("Ошибка", "Не удалось подключиться к камере")
                return
        spectrum_name = self.spectrum_name_entry.get().strip()
        if not spectrum_name:
            messagebox.showerror("Ошибка", "Введите название спектра")
            return

        mode = self.mode_var.get()
        try:
            scan_speed = float(self.scan_speed_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Scan speed должно быть числом")
            return
        try:
            scale_min = float(self.scale_min_var.get())
            scale_max = float(self.scale_max_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Scale должно содержать числа")
            return
        comment = self.comment_var.get()

        self.measurement_wavelengths = []
        self.measurement_transmittances = []
        self.raw_wavelengths = []
        self.raw_transmittances = []
        self.graph_update_counter = 0
        self.update_measurement_plot()

        hsv_lower, hsv_upper = self.config.get_hsv_ranges()

        self.measurement_stop_event.clear()
        self.measurement_thread = threading.Thread(
            target=self._measurement_worker,
            args=(spectrum_name, hsv_lower, hsv_upper,
                  mode, scan_speed, scale_min, scale_max, comment)
        )
        self.measurement_thread.start()

        self.measurement_running = True
        self.start_measurement_btn.config(state=tk.DISABLED)
        self.stop_measurement_btn.config(state=tk.NORMAL)

    def _measurement_worker(self, spectrum_name, hsv_lower, hsv_upper,
                            mode, scan_speed, scale_min, scale_max, comment):
        def frame_callback(frame):
            self.root.after(0, self.update_measurement_roi, frame)

        def data_callback(wl_raw, tr_raw, wl_proc, tr_proc):
            self.root.after(0, self.add_measurement_data,
                            wl_raw, tr_raw, wl_proc, tr_proc)

        error_message = None
        try:
            start_recognition(
                camera=self.camera,
                segment_config=self.segment_config,
                roi_config=self.roi_config,
                hsv_lower=hsv_lower,
                hsv_upper=hsv_upper,
                save_directory=self.save_directory,
                spectrum_name=spectrum_name,
                wavelength_jump_threshold=self.config.get('recognition', 'wavelength_jump_threshold', default=3.0),
                min_wavelength=self.config.get('recognition', 'min_wavelength', default=200.0),
                fill_threshold_digit=self.config.get('recognition', 'fill_threshold_digit', default=0.30),
                fill_threshold_point=self.config.get('recognition', 'fill_threshold_point', default=0.10),
                mode=mode,
                scan_speed=scan_speed,
                scale_min=scale_min,
                scale_max=scale_max,
                comment=comment,
                stop_event=self.measurement_stop_event,
                data_callback=data_callback,
                frame_callback=frame_callback
            )
        except Exception as exc:
            error_message = str(exc)

        if error_message:
            self.root.after(0, lambda msg=error_message: messagebox.showerror("Ошибка", msg))
        self.root.after(0, self.measurement_finished)

    def measurement_finished(self):
        self.measurement_running = False
        self.start_measurement_btn.config(state=tk.NORMAL)
        self.stop_measurement_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Готово", "Измерение завершено")

    def stop_measurement(self):
        self.measurement_stop_event.set()

    def add_measurement_data(self, wl_raw, tr_raw, wl_proc, tr_proc):
        self.raw_wavelengths.append(wl_raw)
        self.raw_transmittances.append(tr_raw)
        self.measurement_wavelengths = list(wl_proc)
        self.measurement_transmittances = list(tr_proc)

        try:
            n = int(self.graph_update_var.get())
        except (ValueError, AttributeError):
            n = 1
        if n < 1:
            n = 1

        self.graph_update_counter += 1
        if self.graph_update_counter % n == 0:
            self.update_measurement_plot()

    def update_measurement_plot(self):
        self.measurement_ax.clear()

        if self.raw_wavelengths:
            self.measurement_ax.plot(
                self.raw_wavelengths, self.raw_transmittances,
                'o', markersize=3, color='#1fa80a', alpha=0.6,
                label='raw', zorder=1
            )

        if self.measurement_wavelengths:
            self.measurement_ax.plot(
                self.measurement_wavelengths, self.measurement_transmittances,
                '-', linewidth=1.5, color='red', label='processed', zorder=2
            )

        self.measurement_ax.set_xlabel("Длина волны, нм")
        self.measurement_ax.set_ylabel("Коэффициент пропускания / отражения")
        self.measurement_ax.grid(True)
        if self.raw_wavelengths or self.measurement_wavelengths:
            self.measurement_ax.legend(loc='best')
        self.measurement_canvas.draw()

    def update_measurement_roi(self, frame):
        if frame is None:
            return
        if self.segment_config and self.roi_config:
            roi1 = self.segment_config['rois'][0]
            roi2 = self.segment_config['rois'][1]
            x1, y1 = int(roi1['x']), int(roi1['y'])
            w1, h1 = int(roi1['w']), int(roi1['h'])
            x2, y2 = int(roi2['x']), int(roi2['y'])
            w2, h2 = int(roi2['w']), int(roi2['h'])
            h_frame, w_frame = frame.shape[:2]

            x1 = max(0, x1); y1 = max(0, y1)
            w1 = min(w1, w_frame - x1); h1 = min(h1, h_frame - y1)
            x2 = max(0, x2); y2 = max(0, y2)
            w2 = min(w2, w_frame - x2); h2 = min(h2, h_frame - y2)

            if w1 > 0 and h1 > 0:
                roi1_img = frame[y1:y1+h1, x1:x1+w1]
                self.roi1_photo = bgr_to_photoimage(roi1_img, self.roi1_frame.winfo_width(), self.roi1_frame.winfo_height())
                self.roi1_label.config(image=self.roi1_photo)
            if w2 > 0 and h2 > 0:
                roi2_img = frame[y2:y2+h2, x2:x2+w2]
                self.roi2_photo = bgr_to_photoimage(roi2_img, self.roi2_frame.winfo_width(), self.roi2_frame.winfo_height())
                self.roi2_label.config(image=self.roi2_photo)

    # ---------- Видео ----------
    def setup_video_page(self, page):
        camera_panel = tk.Frame(page)
        camera_panel.pack(fill=tk.X, padx=10, pady=5)

        row1 = tk.Frame(camera_panel)
        row1.pack(fill=tk.X, pady=2)

        tk.Label(row1, text="Тип камеры:").pack(side=tk.LEFT, padx=5)
        self.camera_type_var = tk.StringVar(value=self.config.get('camera', 'source_type', default='webcam'))
        tk.Radiobutton(row1, text="Веб-камера", variable=self.camera_type_var,
                       value='webcam', command=self.on_camera_type_changed).pack(side=tk.LEFT)
        tk.Radiobutton(row1, text="IP-камера", variable=self.camera_type_var,
                       value='ip', command=self.on_camera_type_changed).pack(side=tk.LEFT)

        tk.Label(row1, text="Индекс / URL:").pack(side=tk.LEFT, padx=10)
        self.camera_source_entry = tk.Entry(row1, width=25)
        self.camera_source_entry.insert(0, str(self.config.get('camera', 'source', default=0)))
        self.camera_source_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(row1, text="Поворот:").pack(side=tk.LEFT, padx=10)
        self.rotate_var = tk.StringVar(value=str(self.config.get('camera', 'rotate', default=0)))
        rotate_menu = tk.OptionMenu(row1, self.rotate_var, "0", "90", "180", "270",
                                    command=self.on_rotate_changed)
        rotate_menu.pack(side=tk.LEFT)

        self.connect_camera_btn = tk.Button(row1, text="Подключить",
                                            command=self.reconnect_camera,
                                            font=('Arial', 12))
        self.connect_camera_btn.pack(side=tk.LEFT, padx=10)

        self.stop_camera_btn = tk.Button(row1, text="Стоп",
                                         command=self.stop_video,
                                         font=('Arial', 12),
                                         bg='lightcoral', state=tk.NORMAL)
        self.stop_camera_btn.pack(side=tk.LEFT, padx=5)

        self.camera_status_label = tk.Label(row1, text="Статус: не подключено", font=('Arial', 12))
        self.camera_status_label.pack(side=tk.LEFT, padx=10)

        # --- Row 3: X / Y / W / H + Свойства драйвера ---
        row3 = tk.Frame(camera_panel)
        row3.pack(fill=tk.X, pady=2)

        self.crop_enabled_var = tk.BooleanVar(
            value=bool(self.config.get('camera', 'crop', default=None))
        )
        tk.Checkbutton(row3, text="Crop включён:",
                       variable=self.crop_enabled_var,
                       command=self.on_crop_toggle,
                       font=('Arial', 12)).pack(side=tk.LEFT, padx=5)

        tk.Label(row3, text="X:", font=('Arial', 12)).pack(side=tk.LEFT)
        self.crop_x_var = tk.StringVar(value="0")
        tk.Entry(row3, textvariable=self.crop_x_var, width=6,
                 font=('Arial', 12)).pack(side=tk.LEFT)

        tk.Label(row3, text="Y:", font=('Arial', 12)).pack(side=tk.LEFT)
        self.crop_y_var = tk.StringVar(value="0")
        tk.Entry(row3, textvariable=self.crop_y_var, width=6,
                 font=('Arial', 12)).pack(side=tk.LEFT)

        tk.Label(row3, text="W:", font=('Arial', 12)).pack(side=tk.LEFT)
        self.crop_w_var = tk.StringVar(value="640")
        tk.Entry(row3, textvariable=self.crop_w_var, width=6,
                 font=('Arial', 12)).pack(side=tk.LEFT)

        tk.Label(row3, text="H:", font=('Arial', 12)).pack(side=tk.LEFT)
        self.crop_h_var = tk.StringVar(value="480")
        tk.Entry(row3, textvariable=self.crop_h_var, width=6,
                 font=('Arial', 12)).pack(side=tk.LEFT)

        tk.Button(row3, text="Применить", command=self.apply_crop,
                  font=('Arial', 11), bg='lightblue').pack(side=tk.LEFT, padx=5)
        tk.Button(row3, text="Сбросить", command=self.reset_crop_area,
                  font=('Arial', 11)).pack(side=tk.LEFT, padx=5)

        tk.Button(row3, text="Свойства драйвера",
                  command=self.open_driver_settings,
                  font=('Arial', 11), bg='lightyellow').pack(side=tk.LEFT, padx=(20, 5))

        # --- Row 4: стрелки перемещения + кнопки размера ---
        row4 = tk.Frame(camera_panel)
        row4.pack(fill=tk.X, pady=2)

        tk.Label(row4, text="Позиция:", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)
        tk.Button(row4, text="←", width=2, font=('Arial', 12),
                  command=lambda: self.move_crop(-10, 0)).pack(side=tk.LEFT)
        tk.Button(row4, text="→", width=2, font=('Arial', 12),
                  command=lambda: self.move_crop(10, 0)).pack(side=tk.LEFT)
        tk.Button(row4, text="↑", width=2, font=('Arial', 12),
                  command=lambda: self.move_crop(0, -10)).pack(side=tk.LEFT)
        tk.Button(row4, text="↓", width=2, font=('Arial', 12),
                  command=lambda: self.move_crop(0, 10)).pack(side=tk.LEFT)

        tk.Label(row4, text="Масштаб:", font=('Arial', 12)).pack(side=tk.LEFT, padx=(20, 5))
        tk.Button(row4, text="+", width=2, font=('Arial', 12),
                  command=lambda: self.resize_crop(-10)).pack(side=tk.LEFT)
        tk.Button(row4, text="-", width=2, font=('Arial', 12),
                  command=lambda: self.resize_crop(10)).pack(side=tk.LEFT)

        self.show_full_frame_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row4, text="Показать полный кадр",
                       variable=self.show_full_frame_var,
                       font=('Arial', 12)).pack(side=tk.LEFT, padx=(20, 5))

        self.crop_info_var = tk.StringVar(value=self._get_crop_text())
        tk.Label(row4, textvariable=self.crop_info_var,
                 font=('Arial', 11), fg='gray').pack(side=tk.LEFT, padx=10)

        # --- Видео ---
        self.video_label = tk.Label(page, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Загружаем сохранённый crop в поля ввода
        self.load_crop_to_ui()

    def _get_crop_text(self):
        crop = self.config.get('camera', 'crop', default=None)
        if not crop:
            return "весь кадр"
        return f"x={crop['x']}, y={crop['y']}, w={crop['w']}, h={crop['h']}"

    def _get_text_crop(self, crop):
        return f"x={crop['x']}, y={crop['y']}, w={crop['w']}, h={crop['h']}"

    def load_crop_to_ui(self):
        """Загружает сохранённый crop в поля ввода."""
        crop = self.config.get('camera', 'crop', default=None)
        if crop:
            self.crop_x_var.set(str(crop['x']))
            self.crop_y_var.set(str(crop['y']))
            self.crop_w_var.set(str(crop['w']))
            self.crop_h_var.set(str(crop['h']))
            self.crop_enabled_var.set(True)

    def on_crop_toggle(self):
        if not self.crop_enabled_var.get():
            self.config.set('camera', 'crop', value=None)
            self.config.save()
            self.camera.crop = None
            self.crop_info_var.set("весь кадр")
        else:
            self.apply_crop()

    def apply_crop(self):
        try:
            x = int(self.crop_x_var.get())
            y = int(self.crop_y_var.get())
            w = int(self.crop_w_var.get())
            h = int(self.crop_h_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "X, Y, W, H должны быть целыми числами")
            return
        if w < 10 or h < 10:
            messagebox.showerror("Ошибка", "W и H должны быть не меньше 10")
            return
        x = max(0, x); y = max(0, y)
        crop = {'x': x, 'y': y, 'w': w, 'h': h}
        self.config.set('camera', 'crop', value=crop)
        self.config.save()
        self.camera.crop = crop
        self.crop_enabled_var.set(True)
        self.crop_x_var.set(str(x))
        self.crop_y_var.set(str(y))
        self.crop_info_var.set(self._get_text_crop(crop))

    def move_crop(self, dx, dy):
        try:
            x = int(self.crop_x_var.get()) + dx
            y = int(self.crop_y_var.get()) + dy
        except ValueError:
            return
        self.crop_x_var.set(str(max(0, x)))
        self.crop_y_var.set(str(max(0, y)))
        self.apply_crop()

    def resize_crop(self, delta):
        try:
            w = int(self.crop_w_var.get()) + delta
            h = int(self.crop_h_var.get()) + delta
        except ValueError:
            return
        self.crop_w_var.set(str(max(10, w)))
        self.crop_h_var.set(str(max(10, h)))
        self.apply_crop()

    def reset_crop_area(self):
        self.config.set('camera', 'crop', value=None)
        self.config.save()
        self.camera.crop = None
        self.crop_enabled_var.set(False)
        self.crop_info_var.set("весь кадр")

    def on_camera_type_changed(self):
        if self.camera_type_var.get() == 'webcam':
            self.camera_source_entry.delete(0, tk.END)
            self.camera_source_entry.insert(0, "0")
        else:
            self.camera_source_entry.delete(0, tk.END)
            self.camera_source_entry.insert(0, "http://")

    def on_rotate_changed(self, value):
        angle = int(value)
        self.config.set('camera', 'rotate', value=angle)
        self.config.save()
        if angle == 90:
            self.camera.rotate_code = cv2.ROTATE_90_CLOCKWISE
        elif angle == 180:
            self.camera.rotate_code = cv2.ROTATE_180
        elif angle == 270:
            self.camera.rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        else:
            self.camera.rotate_code = None

    def reconnect_camera(self):
        if self.measurement_running:
            messagebox.showwarning(
                "Внимание",
                "Идёт измерение. Нажмите Stop или дождитесь окончания."
            )
            return

        self._stop_video_loop()

        source_type = self.camera_type_var.get()
        source = self.camera_source_entry.get()
        if source_type == 'webcam':
            try:
                source = int(source)
            except ValueError:
                messagebox.showerror("Ошибка", "Индекс веб-камеры должен быть числом")
                return

        self.camera.source_type = source_type
        self.camera.source = source
        self.camera.rotate_code = self.config.get_camera_rotate_code()
        self.camera.crop = self.config.get('camera', 'crop', default=None)

        ok = self.camera.connect()
        if not ok:
            # DSHOW иногда не отпускает устройство мгновенно — пробуем ещё раз
            self.camera_status_label.config(text="Статус: повторная попытка...")
            self.root.update_idletasks()
            time.sleep(0.8)
            ok = self.camera.connect()

        if ok:
            self.camera_status_label.config(text="Статус: подключено")
            self.video_running = True
            self._no_frame_count = 0
            self.update_video_frame()

            self.config.set('camera', 'source_type', value=source_type)
            self.config.set('camera', 'source', value=source)
            self.config.save()
        else:
            self.camera_status_label.config(text="Статус: ошибка подключения")
            self.video_running = False

        self._update_camera_buttons()

    def start_video(self):
        if self.video_running and self.video_after_id is not None:
            return
        if not self.camera.connected:
            if not self.camera.connect():
                self.camera_status_label.config(text="Статус: не удалось подключить")
                return
            self.camera_status_label.config(text="Статус: подключено")
        self.video_running = True
        self.update_video_frame()

    def update_video_frame(self):
        if not self.video_running:
            self.video_after_id = None
            return

        crop = self.config.get('camera', 'crop', default=None)
        crop_active = crop is not None and self.crop_enabled_var.get()
        show_full = bool(self.show_full_frame_var.get())
        apply_crop_now = crop_active and not show_full

        ret, frame = self.camera.read(apply_crop=apply_crop_now)

        if not ret or frame is None:
            self._no_frame_count += 1
            if self._no_frame_count == 100:
                self.camera_status_label.config(text="Статус: нет кадров, переподключаю...")
                try:
                    self.camera.release()
                    self.camera.connect()
                    self.camera_status_label.config(text="Статус: подключено")
                except Exception as e:
                    print(f"Автопереподключение не удалось: {e}")
                self._no_frame_count = 0
            self.video_after_id = self.root.after(30, self.update_video_frame)
            return

        self._no_frame_count = 0
        fh, fw = frame.shape[:2]

        # Затемнение вокруг crop — только в режиме «полный кадр»
        if crop_active and show_full:
            cx, cy = int(crop['x']), int(crop['y'])
            cw, ch = int(crop['w']), int(crop['h'])
            x2 = min(cx + cw, fw)
            y2 = min(cy + ch, fh)
            cx = max(0, cx); cy = max(0, cy)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 0), -1)
            overlay[cy:y2, cx:x2] = frame[cy:y2, cx:x2]
            frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
            cv2.rectangle(frame, (cx, cy), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Crop {cw}x{ch}", (cx + 5, max(cy + 20, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ROI-прямоугольники, сегменты и точка
        if self.segment_config and self.roi_config:
            roi1 = self.segment_config['rois'][0]
            roi2 = self.segment_config['rois'][1]
            if apply_crop_now:
                off_x, off_y = 0, 0
            else:
                off_x = int(crop['x']) if crop_active else 0
                off_y = int(crop['y']) if crop_active else 0

            # Размер квадратиков сегментов — из segment_config (или 9 по умолчанию)
            seg_size = int(self.segment_config.get('_seg_size', 9))
            seg_size = max(5, min(seg_size, 40))

            cv2.rectangle(frame,
                          (off_x + roi1['x'], off_y + roi1['y']),
                          (off_x + roi1['x'] + roi1['w'],
                           off_y + roi1['y'] + roi1['h']),
                          (255, 0, 0), 2)
            cv2.rectangle(frame,
                          (off_x + roi2['x'], off_y + roi2['y']),
                          (off_x + roi2['x'] + roi2['w'],
                           off_y + roi2['y'] + roi2['h']),
                          (0, 255, 0), 2)

            for roi_data in (roi1, roi2):
                for digit in roi_data['digits']:
                    for seg in digit['segments']:
                        x = int(off_x + roi_data['x']
                                + seg['x'] * roi_data['w'])
                        y = int(off_y + roi_data['y']
                                + seg['y'] * roi_data['h'])
                        cv2.rectangle(frame, (x, y),
                                      (x + seg_size, y + seg_size),
                                      (0, 255, 255), 1)

                # Точка (десятичный разделитель)
                if 'point' in roi_data:
                    px = int(off_x + roi_data['x']
                             + roi_data['point']['x'] * roi_data['w'])
                    py = int(off_y + roi_data['y']
                             + roi_data['point']['y'] * roi_data['h'])
                    cv2.rectangle(frame, (px, py),
                                  (px + seg_size, py + seg_size),
                                  (0, 128, 255), 1)   # оранжевый — чтобы отличать

        # Масштаб под окно: fill = зум-эффект для обрезанного кадра
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w > 1 and label_h > 1:
            scale = min(label_w / fw, label_h / fh)
            new_w = max(1, int(fw * scale))
            new_h = max(1, int(fh * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.video_after_id = self.root.after(30, self.update_video_frame)

    def _stop_video_loop(self):
        self.video_running = False
        if self.video_after_id is not None:
            try:
                self.root.after_cancel(self.video_after_id)
            except Exception:
                pass
            self.video_after_id = None

    def on_close(self):
        self._stop_video_loop()
        try:
            self.measurement_stop_event.set()
        except Exception:
            pass
        if self.measurement_thread is not None and self.measurement_thread.is_alive():
            self.measurement_thread.join(timeout=2.0)
        try:
            self.camera.release()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def stop_video(self):
        self._stop_video_loop()
        try:
            self.camera.release()
        except Exception:
            pass
        self.camera_status_label.config(text="Статус: остановлено")
        self._update_camera_buttons()

    def _update_camera_buttons(self):
        if self.video_running:
            self.stop_camera_btn.config(state=tk.NORMAL)
        else:
            self.stop_camera_btn.config(state=tk.DISABLED)

    def open_driver_settings(self):
        """Открывает системный диалог свойств драйвера DirectShow.
        Там доступны все параметры, которые поддерживает камера:
        экспозиция, яркость, контраст, баланс белого, зум, фокус и т.п."""
        if not self.camera.connected or self.camera.cap is None:
            messagebox.showwarning(
                "Камера не подключена",
                "Сначала нажмите «Подключить», чтобы открыть камеру."
            )
            return
        try:
            self.camera.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        except Exception as e:
            messagebox.showerror(
                "Ошибка",
                f"Не удалось открыть свойства драйвера: {e}"
            )

    # ---------- Данные ----------
    def setup_data_view_page(self, page):
        button_panel = tk.Frame(page)
        button_panel.pack(fill=tk.X, padx=10, pady=5)

        self.add_graph_btn = tk.Button(button_panel, text="Добавить график из файла",
                                       command=self.add_graph_from_file,
                                       font=('Arial', 12))
        self.add_graph_btn.pack(side=tk.LEFT, padx=5)

        self.remove_graph_btn = tk.Button(button_panel, text="Удалить выбранный график",
                                          command=self.remove_selected_graph,
                                          font=('Arial', 12))
        self.remove_graph_btn.pack(side=tk.LEFT, padx=5)

        self.graph_listbox = tk.Listbox(page, width=40, height=5)
        self.graph_listbox.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.metadata_frame = tk.LabelFrame(page, text="Информация о спектре",
                                            font=('Arial', 12))
        self.metadata_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

        self.metadata_label = tk.Label(self.metadata_frame, text="Файл не выбран",
                                       font=('Arial', 11), justify=tk.LEFT, anchor='w')
        self.metadata_label.pack(fill=tk.X, padx=10, pady=5)

        self.graph_listbox.bind('<<ListboxSelect>>', self.on_graph_selected)

        self.data_view_fig = Figure(figsize=(10, 5), dpi=100)
        self.data_view_ax = self.data_view_fig.add_subplot(111)
        self.data_view_ax.set_xlabel("Длина волны, нм")
        self.data_view_ax.set_ylabel("Коэффициент пропускания / отражения")
        self.data_view_ax.grid(True)

        self.data_view_canvas = FigureCanvasTkAgg(self.data_view_fig, master=page)
        self.data_view_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                                   padx=10, pady=5)

        self.loaded_graphs = {}

    def parse_spectrum_file(self, file_path):
        metadata = {
            'Date': '',
            'Mode': '',
            'Scan speed': '',
            'Scale': '',
            'Comment': ''
        }
        wavelengths = []
        transmittances = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data_start = None
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('wavelength'):
                data_start = i + 1
                break

        if data_start is None:
            data_start = 0

        if data_start > 0:
            for line in lines[:data_start - 1]:
                for key in metadata.keys():
                    prefix = f"{key}:"
                    if line.strip().startswith(prefix):
                        metadata[key] = line.strip()[len(prefix):].strip()
                        break

        for line in lines[data_start:]:
            parts = line.strip().split(';')
            if len(parts) < 2:
                continue
            try:
                wl = float(parts[0])
                tr = float(parts[1])
                wavelengths.append(wl)
                transmittances.append(tr)
            except ValueError:
                continue

        return wavelengths, transmittances, metadata

    def add_graph_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            wavelengths, transmittances, metadata = self.parse_spectrum_file(file_path)
            if not wavelengths:
                messagebox.showwarning("Предупреждение",
                                       "В файле не найдено числовых данных")
                return
            graph_name = os.path.basename(file_path)
            self.loaded_graphs[graph_name] = (wavelengths, transmittances, metadata)
            self.graph_listbox.insert(tk.END, graph_name)
            self.redraw_data_view()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}")

    def remove_selected_graph(self):
        selection = self.graph_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        graph_name = self.graph_listbox.get(index)
        if graph_name in self.loaded_graphs:
            del self.loaded_graphs[graph_name]
        self.graph_listbox.delete(index)
        self.metadata_label.config(text="Файл не выбран")
        self.redraw_data_view()

    def on_graph_selected(self, event):
        selection = self.graph_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        graph_name = self.graph_listbox.get(index)
        data = self.loaded_graphs.get(graph_name)
        if not data:
            return
        metadata = data[2]
        info_lines = [
            f"Файл: {graph_name}",
            f"Date: {metadata.get('Date', '—')}",
            f"Mode: {metadata.get('Mode', '—')}",
            f"Scan speed: {metadata.get('Scan speed', '—')}",
            f"Scale: {metadata.get('Scale', '—')}",
            f"Comment: {metadata.get('Comment', '—')}",
        ]
        self.metadata_label.config(text="\n".join(info_lines))

    def redraw_data_view(self):
        self.data_view_ax.clear()
        self.data_view_ax.set_xlabel("Длина волны, нм")
        self.data_view_ax.set_ylabel("Коэффициент пропускания / отражения")
        self.data_view_ax.grid(True)
        for name, (wl, tr, _meta) in self.loaded_graphs.items():
            self.data_view_ax.plot(wl, tr, label=name)
        if self.loaded_graphs:
            self.data_view_ax.legend()
        self.data_view_canvas.draw()

    # ---------- Настройки ----------
    def setup_settings_page(self, page):
        settings_frame = tk.Frame(page)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        btn1 = tk.Button(settings_frame, text="Настройка областей распознавания",
                         command=self.open_roi_config,
                         font=('Arial', 12), width=30, height=2)
        btn1.pack(pady=10)

        btn2 = tk.Button(settings_frame, text="Настройка фильтрации цвета",
                         command=self.open_color_config,
                         font=('Arial', 12), width=30, height=2)
        btn2.pack(pady=10)

        thr_frame = tk.LabelFrame(settings_frame, text="Чувствительность сегментов",
                                  font=('Arial', 12))
        thr_frame.pack(pady=15, fill=tk.X, padx=20)

        row1 = tk.Frame(thr_frame)
        row1.pack(fill=tk.X, pady=5, padx=10)
        tk.Label(row1, text="Порог для цифр (0.0–1.0):", font=('Arial', 12)).pack(side=tk.LEFT)
        self.fill_digit_var = tk.StringVar(
            value=str(self.config.get('recognition', 'fill_threshold_digit', default=0.30))
        )
        tk.Entry(row1, textvariable=self.fill_digit_var, width=10,
                 font=('Arial', 12)).pack(side=tk.LEFT, padx=10)

        row2 = tk.Frame(thr_frame)
        row2.pack(fill=tk.X, pady=5, padx=10)
        tk.Label(row2, text="Порог для точки (0.0–1.0):", font=('Arial', 12)).pack(side=tk.LEFT)
        self.fill_point_var = tk.StringVar(
            value=str(self.config.get('recognition', 'fill_threshold_point', default=0.10))
        )
        tk.Entry(row2, textvariable=self.fill_point_var, width=10,
                 font=('Arial', 12)).pack(side=tk.LEFT, padx=10)

        tk.Button(thr_frame, text="Сохранить пороги", command=self.save_thresholds,
                  font=('Arial', 12), bg='lightblue').pack(pady=10)

        save_frame = tk.Frame(settings_frame)
        save_frame.pack(pady=20)

        self.save_dir_button = tk.Button(save_frame, text="Выбрать папку сохранения",
                                         command=self.choose_save_directory,
                                         font=('Arial', 12), width=30, height=2)
        self.save_dir_button.pack(pady=5)

        self.save_dir_label = tk.Label(save_frame, text=f"Текущая папка: {self.save_directory}",
                                       font=('Arial', 12), wraplength=600)
        self.save_dir_label.pack(pady=5)

    def save_thresholds(self):
        try:
            digit = float(self.fill_digit_var.get())
            point = float(self.fill_point_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Пороги должны быть числами")
            return
        if not (0.0 <= digit <= 1.0) or not (0.0 <= point <= 1.0):
            messagebox.showerror("Ошибка", "Пороги должны быть в диапазоне 0.0–1.0")
            return
        self.config.set('recognition', 'fill_threshold_digit', value=digit)
        self.config.set('recognition', 'fill_threshold_point', value=point)
        self.config.save()
        messagebox.showinfo("Готово", "Пороги сохранены")

    def choose_save_directory(self):
        directory = filedialog.askdirectory(initialdir=self.save_directory)
        if directory:
            self.save_directory = directory
            self.save_dir_label.config(text=f"Текущая папка: {self.save_directory}")
            self.config.set('paths', 'save_directory', value=directory)
            self.config.save()

    def open_roi_config(self):
        rotate_code = self.config.get_camera_rotate_code()
        crop = self.config.get('camera', 'crop', default=None)
        seg_path = self.config.get_absolute_path('segment_config')
        roi_path = self.config.get_absolute_path('roi_config')
        calibration.segment_calibration(
            camera_index=self.config.get('camera', 'source', default=0),
            segment_file=seg_path, roi_file=roi_path,
            rotate_code=rotate_code, crop=crop
        )
        self.segment_config = load_json(seg_path)
        self.roi_config = load_json(roi_path)

    def open_color_config(self):
        rotate_code = self.config.get_camera_rotate_code()
        crop = self.config.get('camera', 'crop', default=None)
        hsv_path = self.config.get_absolute_path('hsv_config')
        calibration.color_calibration(
            camera_index=self.config.get('camera', 'source', default=0),
            config_file=hsv_path,
            rotate_code=rotate_code, crop=crop
        )
        hsv_config = load_json(hsv_path)
        if hsv_config:
            self.config.set('hsv', 'lower',
                            value=[hsv_config['H_MIN'], hsv_config['S_MIN'], hsv_config['V_MIN']])
            self.config.set('hsv', 'upper',
                            value=[hsv_config['H_MAX'], hsv_config['S_MAX'], hsv_config['V_MAX']])
            self.config.save()