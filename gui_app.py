import os, time, threading, logging, json, base64
import cv2
import numpy as np
import flet as ft
import joblib
from typing import List, Tuple

# Reutilizar extract_features del módulo principal si existe
try:
    from SistemaDPVOL import extract_features
except Exception:
    # Fallback mínimo
    def extract_features(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.flatten()

DEFAULT_MODEL_PATH = os.path.join(os.getcwd(), 'models', 'parking_model.pkl')
DEFAULT_ROIS_PATH = os.path.join(os.getcwd(), 'rois.json')
DEFAULT_VIDEO_FOLDER = os.path.join(os.getcwd(), 'pruebavideo')
FRAME_DISPLAY_SECONDS = 2.0  # valor inicial, se puede ajustar con slider
PLACEHOLDER_PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAjv1P3QAAAABJRU5ErkJggg=="  # 1x1 png transparente

class ParkingVideoPlayer:
    def __init__(self, model_path: str, rois_path: str, folder: str, update_fn):
        self.model_path = model_path
        self.rois_path = rois_path
        self.folder = folder
        self.update_fn = update_fn  # callback para enviar frame y texto
        self._stop = threading.Event()
        self._thread = None
        self.clf = None
        self.rois = None
        self.images: List[str] = []
        self.paused = False
        self.current_index = 0
        self.frame_interval = FRAME_DISPLAY_SECONDS  # segundos entre frames ajustable desde UI

    def load_resources(self):
        # Modelo
        if not os.path.exists(self.model_path):
            logging.warning(f"Modelo no encontrado: {self.model_path}")
        else:
            try:
                self.clf = joblib.load(self.model_path)
            except Exception:
                logging.exception("Error cargando modelo")
        # ROIs
        if not os.path.exists(self.rois_path):
            logging.warning(f"ROIs no encontradas: {self.rois_path}")
            self.rois = []
        else:
            try:
                with open(self.rois_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.rois = data.get('rois', [])
            except Exception:
                logging.exception("Error leyendo rois.json")
                self.rois = []
        # Imágenes
        if not os.path.isdir(self.folder):
            logging.warning(f"Carpeta de video no existe: {self.folder}")
        else:
            files = [f for f in os.listdir(self.folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            files.sort()
            self.images = [os.path.join(self.folder, f) for f in files]
            if not self.images:
                logging.warning("No se encontraron imágenes en pruebavideo")

    def start(self):
        if self._thread and self._thread.is_alive():
            # si estaba pausado, continuar
            self.paused = False
            return
        self._stop.clear()
        self.paused = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def _run_loop(self):
        if not self.images:
            self.update_fn(None, "Sin imágenes para reproducir.")
            return
        while not self._stop.is_set():
            if self.paused:
                time.sleep(0.1)
                continue
            img_path = self.images[self.current_index]
            frame = cv2.imread(img_path)
            if frame is None:
                self.update_fn(None, f"No se pudo leer {os.path.basename(img_path)}")
            else:
                annotated, summary = self._annotate_frame(frame.copy())
                # convertir a PNG base64 para Flet Image (o usar ruta temporal)
                try:
                    ok, buf = cv2.imencode('.png', annotated)
                    if ok:
                        b64 = base64.b64encode(buf).decode('utf-8')
                        self.update_fn(b64, summary)
                    else:
                        logging.error("cv2.imencode falló")
                        self.update_fn(None, summary)
                except Exception:
                    logging.exception("Error codificando frame")
                    self.update_fn(None, summary)
            elapsed = 0.0
            interval = max(0.2, float(self.frame_interval))
            while elapsed < interval and not self._stop.is_set() and not self.paused:
                time.sleep(0.1)
                elapsed += 0.1
            if not self.paused:
                self.current_index = (self.current_index + 1) % len(self.images)

    def _annotate_frame(self, frame):
        free_count = 0
        occ_count = 0
        if not self.rois:
            return frame, "Sin ROIs"
        if self.clf is None:
            return frame, "Modelo no cargado"
        for r in self.rois:
            if len(r) != 4:
                continue
            x, y, w, h = r
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            try:
                crop_r = cv2.resize(crop, (64,64))
            except Exception:
                continue
            feats = extract_features(crop_r).reshape(1,-1)
            try:
                pred = int(self.clf.predict(feats)[0])
            except Exception:
                pred = 1
            color = (0,0,255) if pred == 0 else (255,0,0)
            label = 'Libre' if pred == 0 else 'Ocupado'
            if pred == 0:
                free_count += 1
            else:
                occ_count += 1
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
        summary = f"Libres={free_count} Ocupados={occ_count}"
        return frame, summary


def main(page: ft.Page):
    page.title = "Sistema de Detección de Espacios Libres y Ocupados en Estacionamientos mediante Visión Artificial"
    page.window_width = 900
    page.window_height = 720
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO
    # Tema oscuro estilizado
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#121212"
    page.fonts = {}
    # AppBar
    page.appbar = ft.AppBar(title=ft.Text("Sistema de Detección de Espacios Libres y Ocupados en Estacionamientos mediante Visión Artificial"), center_title=True, bgcolor="#1E1E1E")

    status_text = ft.Text(value="Listo", size=14)
    # Eliminado chip de resumen para quitar cuadro blanco bajo el slider
    image_control = ft.Image(width=640, height=480, fit=ft.ImageFit.CONTAIN, src_base64=PLACEHOLDER_PX)

    player = ParkingVideoPlayer(DEFAULT_MODEL_PATH, DEFAULT_ROIS_PATH, DEFAULT_VIDEO_FOLDER,
                                update_fn=lambda data, summary: update_frame(data, summary))
    player.load_resources()

    def update_frame(data: str, summary: str):
        if data is not None:
            image_control.src_base64 = data
        # Mostrar resumen directamente en status_text
        status_text.value = summary
        page.update()

    def start_clicked(e):
        player.start()
        status_text.value = "Reproduciendo..."
        page.update()

    def pause_clicked(e):
        player.pause()
        status_text.value = "Pausado"
        page.update()

    def continue_clicked(e):
        player.resume()
        status_text.value = "Reproduciendo..."
        page.update()

    def stop_clicked(e):
        status_text.value = "Detenido"
        player.stop()
        page.update()

    # Slider para velocidad (intervalo entre frames)
    def slider_changed(e):
        player.frame_interval = speed_slider.value
        page.update()

    speed_slider = ft.Slider(min=0.5, max=5.0, divisions=9, value=2.0, label="{value}s", on_change=slider_changed)
    speed_label = ft.Text("Intervalo (s):")

    controls = ft.Row([
        ft.ElevatedButton(text="Iniciar", on_click=start_clicked, icon=ft.Icon(name="play_arrow")),
        ft.ElevatedButton(text="Pausar", on_click=pause_clicked, icon=ft.Icon(name="pause")),
        ft.ElevatedButton(text="Continuar", on_click=continue_clicked, icon=ft.Icon(name="play_circle")),
        ft.ElevatedButton(text="Apagar", on_click=stop_clicked, icon=ft.Icon(name="stop")),
        status_text
    ], alignment=ft.MainAxisAlignment.CENTER, spacing=12)

    speed_row = ft.Row([speed_label, speed_slider], alignment=ft.MainAxisAlignment.CENTER)

    # Flet versión instalada no provee ft.colors; usar código hex directo
    video_container = ft.Container(
        content=image_control,
        padding=8,
        bgcolor="#222222",  # fondo oscuro neutro
        border_radius=8,
        alignment=ft.alignment.center,
        width=660,
        height=500,
    )

    meta_info = ft.Row([
        ft.Text(f"Modelo: {os.path.basename(DEFAULT_MODEL_PATH)}"),
        ft.Text(f"ROIs: {os.path.basename(DEFAULT_ROIS_PATH)}"),
        ft.Text(f"Frames: {len(os.listdir(DEFAULT_VIDEO_FOLDER)) if os.path.isdir(DEFAULT_VIDEO_FOLDER) else 0}")
    ], alignment=ft.MainAxisAlignment.CENTER, spacing=25)

    controls_card = ft.Card(content=ft.Container(
        content=ft.Column([
            controls,
            speed_row
        ], spacing=15, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        padding=20,
        bgcolor="#1E1E1E",
        border_radius=12
    ))

    page.add(
        ft.Column([
            video_container,
            controls_card,
            meta_info,
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20)
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    ft.app(target=main)
