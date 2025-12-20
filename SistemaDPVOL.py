"""
Sistema de Detección de Espacios Libres y Ocupados en Estacionamientos mediante Visión Artificial

Este sistema utiliza técnicas de procesamiento de imágenes y aprendizaje automático para detectar y clasificar espacios de estacionamiento como libres u ocupados.
"""

import cv2
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
import random
import os
import json
import logging
import threading
import sys
import time
import csv
import shutil
import re
from datetime import datetime
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

def load_reference_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"La imagen de referencia no se encontró en {path}")
    ref_img = cv2.imread(path)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    # usar umbral adaptativo para mayor robustez ante variaciones de iluminación
    ref_mask = cv2.adaptiveThreshold(gray_ref, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    # aplicar opening para reducir ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ref_mask = cv2.morphologyEx(ref_mask, cv2.MORPH_OPEN, kernel)
    return ref_mask

def detect_occupancy(frame, ref_mask):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_mask = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    diff = cv2.absdiff(ref_mask, frame_mask)
    # limpiar ruido con morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    occupied_spaces = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Umbral para ignorar ruido
            occupied_spaces += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return frame, occupied_spaces


def extract_features(image):
    # Extrae características HOG de la imagen en escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

def augment_image(img):
    """Genera versiones aumentadas simples de la imagen (brillo, rotación, flip)."""
    outs = []
    try:
        # brillo/contraste
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        outs.append(bright)
        # flip horizontal
        outs.append(cv2.flip(img, 1))
        # pequeña rotación
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), random.uniform(-7, 7), 1)
        rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        outs.append(rot)
    except Exception:
        logging.exception("Error en augment_image")
    return outs


def load_dataset(dataset_path, augment=False, augment_factor=2):
    X = []
    y = []
    # Soportar convenciones de nombres: 'free'/'occupied' o carpetas que contengan 'vacio'/'lleno'
    # Normalize class directories to check
    entries = os.listdir(dataset_path) if os.path.exists(dataset_path) else []
    allowed_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for entry in entries:
        entry_path = os.path.join(dataset_path, entry)
        if not os.path.isdir(entry_path):
            continue
        key = entry.lower()
        if key in ('free', 'occupied'):
            label = 0 if key == 'free' else 1
        else:
            # fallback: infer by folder name containing 'vacio' or 'lleno'
            if 'vacio' in key or 'vac' in key:
                label = 0
            elif 'lleno' in key or 'ocup' in key:
                label = 1
            else:
                # desconocido -> tratar como carpeta con subfolders (ej. Mañana/Tarde/Noche)
                label = None
        if label is not None:
            # leer imágenes dentro de la carpeta (incluye subcarpetas)
            for root, _, files in os.walk(entry_path):
                for fname in files:
                    if os.path.splitext(fname)[1].lower() not in allowed_exts:
                        continue
                    fpath = os.path.join(root, fname)
                    img = cv2.imread(fpath)
                    if img is None:
                        logging.warning(f"No se pudo leer imagen {fpath}")
                        continue
                    try:
                        img = cv2.resize(img, (64, 64))
                    except Exception:
                        logging.exception(f"Error redimensionando imagen {fpath}")
                        continue
                    features = extract_features(img)
                    X.append(features)
                    y.append(label)
                    # augment
                    if augment:
                        for i, aimg in enumerate(augment_image(img)):
                            if i >= augment_factor:
                                break
                            try:
                                aimg_r = cv2.resize(aimg, (64, 64))
                                features_a = extract_features(aimg_r)
                                X.append(features_a)
                                y.append(label)
                            except Exception:
                                logging.exception(f"Error augmentando imagen {fpath}")
        else:
            # intentar recorrer subcarpetas (MaÃ±ana/Tarde/Noche)
                for sub in os.listdir(entry_path):
                    subdir = os.path.join(entry_path, sub)
                    if not os.path.isdir(subdir):
                        continue
                    for root, _, files in os.walk(subdir):
                        for fname in files:
                            if os.path.splitext(fname)[1].lower() not in allowed_exts:
                                continue
                            fpath = os.path.join(root, fname)
                            img = cv2.imread(fpath)
                            if img is None:
                                logging.warning(f"No se pudo leer imagen {fpath}")
                                continue
                            try:
                                img = cv2.resize(img, (64, 64))
                            except Exception:
                                logging.exception(f"Error redimensionando imagen {fpath}")
                                continue
                            features = extract_features(img)
                            # inferir etiqueta por el nombre de la subcarpeta o la carpeta padre
                            sub_lower = sub.lower()
                            entry_lower = entry.lower()
                            combined = f"{sub_lower} {entry_lower}"
                            if 'vacio' in combined or 'vac' in combined or 'free' in combined:
                                label_infer = 0
                            elif 'lleno' in combined or 'ocup' in combined or 'occupied' in combined or 'busy' in combined:
                                label_infer = 1
                            else:
                                # si no se puede inferir, omitir (use DEBUG para no spamear la consola)
                                logging.debug(f"Omitiendo imagen porque no se pudo inferir etiqueta: {fpath}")
                                continue
                            X.append(features)
                            y.append(label_infer)
                            if augment:
                                for i, aimg in enumerate(augment_image(img)):
                                    if i >= augment_factor:
                                        break
                                    try:
                                        aimg_r = cv2.resize(aimg, (64, 64))
                                        features_a = extract_features(aimg_r)
                                        X.append(features_a)
                                        y.append(label_infer)
                                    except Exception:
                                        logging.exception(f"Error augmentando imagen {fpath}")
    return np.array(X), np.array(y)


class Spinner:
    """Simple spinner thread to indicate progress during long operations."""
    def __init__(self, message='Processing '):
        self._stop = threading.Event()
        self.thread = None
        self.message = message

    def _spin(self):
        chars = '|/-\\'
        idx = 0
        sys.stdout.write(self.message)
        sys.stdout.flush()
        while not self._stop.is_set():
            sys.stdout.write(chars[idx % len(chars)])
            sys.stdout.flush()
            time.sleep(0.2)
            sys.stdout.write('\b')
            idx += 1
        sys.stdout.write('\n')

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join()

def train_model(dataset_path, out_path, augment=False, cv_folds=0, min_required=5, report_path=None, test_size=0.2):
    # Robustez: validar test_size
    try:
        if not (0.05 <= float(test_size) < 0.9):
            logging.warning(f"test_size {test_size} fuera de rango, usando 0.2")
            test_size = 0.2
    except Exception:
        logging.warning("test_size inválido, usando 0.2")
        test_size = 0.2
    logging.info(f"Cargando dataset desde {dataset_path} ...")
    X, y = load_dataset(dataset_path, augment=augment)
    logging.info(f"Total de muestras: {len(y)}")
    if len(y) == 0:
        logging.error("No se encontraron imágenes para entrenar. Revisa la estructura de carpetas.")
        return
    # comprobar número de muestras por clase
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    logging.info(f"Distribución por clase: {class_counts}")
    for cls, cnt in class_counts.items():
        if cnt < min_required:
            logging.error(f"Clase {cls} tiene solo {cnt} muestras. Se requieren >= {min_required} para entrenar.")
            return

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    except Exception:
        logging.exception("Error en train_test_split; usando split simple no estratificado")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    logging.info("Entrenando modelo (Pipeline: scaler + SVM) ...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(max_iter=10000, class_weight='balanced'))
    ])

    if cv_folds and cv_folds > 1:
        logging.info(f"Usando GridSearchCV con {cv_folds}-folds para buscar mejor C")
        param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
        grid = GridSearchCV(pipeline, param_grid, cv=cv_folds, scoring='f1', n_jobs=-1)
        # mostrar indicador mientras se ejecuta GridSearchCV
        spinner = Spinner('Buscando mejores parámetros (GridSearchCV) ')
        spinner.start()
        try:
            grid.fit(X_train, y_train)
        finally:
            spinner.stop()
        pipeline = grid.best_estimator_
        logging.info(f"Mejor parámetro encontrado: {grid.best_params_}")
    else:
        # mostrar indicador mientras se entrena
        spinner = Spinner('Entrenando modelo (SVM) ')
        spinner.start()
        try:
            pipeline.fit(X_train, y_train)
        finally:
            spinner.stop()
    y_pred = pipeline.predict(X_test)
    logging.info("Reporte de clasificación:")
    report = classification_report(y_test, y_pred, target_names=['free', 'occupied'], output_dict=True)
    logging.info('\n' + classification_report(y_test, y_pred, target_names=['free', 'occupied']))
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f"Matriz de confusión:\n{cm}")
    # si se pidió guardar reporte, escribir JSON
    if report_path:
        try:
            with open(report_path, 'w', encoding='utf-8') as rf:
                json.dump({'classification_report': report, 'confusion_matrix': cm.tolist()}, rf, ensure_ascii=False, indent=2)
            logging.info(f"Reporte de evaluación guardado en {report_path}")
        except Exception:
            logging.exception(f"No se pudo guardar el reporte en {report_path}")
    # asegurar carpeta de salida
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipeline, out_path)
    logging.info(f"Modelo guardado en {out_path}")


def select_rois(frame):
    print("Selecciona las regiones de interés (ROIs) para cada espacio de estacionamiento. Presiona ENTER después de cada selección y ESC cuando termines.")
    rois = cv2.selectROIs("Selecciona ROIs", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Selecciona ROIs")
    return rois

def save_rois(rois, path):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'rois': rois.tolist()}, f, ensure_ascii=False, indent=2)
        logging.info(f"ROIs guardadas en {path}")
    except Exception:
        logging.exception(f"No se pudo guardar ROIs en {path}")

def load_rois(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data.get('rois', []), dtype=int)

def detect_and_mark_spaces(frame, rois, clf):
    # soporte de suavizado temporal: pred_buffers es una lista de deque por cada ROI
    free_count = 0
    occupied_count = 0
    for idx, roi in enumerate(rois):
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            logging.warning(f"ROI vacía o fuera de frame: {roi}")
            continue
        crop_resized = cv2.resize(crop, (64, 64))
        features = extract_features(crop_resized).reshape(1, -1)
        try:
            pred = clf.predict(features)[0]
        except Exception:
            logging.exception("Error al predecir una ROI; marcando como ocupado por seguridad.")
            pred = 1
        # Usuario quiere: ocupado = azul, libre = rojo
        # OpenCV usa BGR: azul=(255,0,0), rojo=(0,0,255)
        color = (0, 0, 255) if pred == 0 else (255, 0, 0)
        label = "Libre" if pred == 0 else "Ocupado"
        if pred == 0:
            free_count += 1
        else:
            occupied_count += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame, free_count, occupied_count


def read_camera_csv(csv_path):
    """Lee un csv de cámara con columnas SlotId,X,Y,W,H y devuelve dict slotid->(x,y,w,h).
    Las coordenadas en los CSV están en referencia a 2592x1944 y se escalarán más tarde.
    """
    slots = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                try:
                    sid = int(row.get('SlotId') or row.get('slotid') or row.get('slot'))
                    x = int(row.get('X') or row.get('x'))
                    y = int(row.get('Y') or row.get('y'))
                    w = int(row.get('W') or row.get('w'))
                    h = int(row.get('H') or row.get('h'))
                    slots[sid] = (x, y, w, h)
                except Exception:
                    logging.debug(f"Fila inválida en {csv_path}: {row}")
    except Exception:
        logging.exception(f"No se pudo leer CSV de cámara: {csv_path}")
    return slots


def build_dataset_from_full_images(metadata_csv, full_image_root, cameras_csv_dir, out_dataset_dir, resize=(64, 64)):
    """Construye un dataset de parches etiquetados (free/occupied) a partir de los full frames
    usando el CSV de metadata (CNRPark+EXT.csv) y los CSVs por cámara con las bounding boxes.

    metadata_csv: ruta al CNRPark+EXT.csv
    full_image_root: ruta a FULL_IMAGE_1000x750
    cameras_csv_dir: ruta donde están camera1.csv..camera9.csv
    out_dataset_dir: carpeta destino donde se crearán subcarpetas 'free' y 'occupied'
    """
    if not os.path.exists(metadata_csv):
        logging.error(f"Metadata CSV no encontrado: {metadata_csv}")
        return
    if not os.path.exists(full_image_root):
        logging.error(f"Full image root no encontrado: {full_image_root}")
        return

    # preparar carpeta destino
    free_dir = os.path.join(out_dataset_dir, 'free')
    occ_dir = os.path.join(out_dataset_dir, 'occupied')
    os.makedirs(free_dir, exist_ok=True)
    os.makedirs(occ_dir, exist_ok=True)

    # leer metadata CSV y crear un mapa por (cam, slot, date) -> list of (time_minutes, label)
    label_entries = {}
    re_cam = re.compile(r'camera\s*(\d+)', re.IGNORECASE)
    re_c0 = re.compile(r'C0?(\d{1,2})', re.IGNORECASE)
    re_date_dash = re.compile(r'(\d{4}-\d{2}-\d{2})')
    re_date_compact = re.compile(r'(\d{8})')
    re_time_colon = re.compile(r'(\d{1,2}[.:]\d{2})')
    try:
        with open(metadata_csv, 'r', encoding='utf-8') as mf:
            reader = csv.DictReader(mf)
            for row in reader:
                imgurl = (row.get('image_url') or row.get('image') or '')
                occ = row.get('occupancy') or row.get('label') or row.get('occup') or row.get('occupancy')
                try:
                    label = int(occ)
                except Exception:
                    continue
                if not imgurl:
                    continue
                s = imgurl.replace('\\', '/').lstrip('./')
                cam = None
                slot = None
                date = None
                time_min = None
                # camera id
                m = re_cam.search(s)
                if m:
                    try:
                        cam = int(m.group(1))
                    except Exception:
                        cam = None
                if cam is None:
                    m = re_c0.search(s)
                    if m:
                        try:
                            cam = int(m.group(1))
                        except Exception:
                            cam = None
                # date
                m = re_date_dash.search(s)
                if m:
                    date = m.group(1)
                else:
                    m = re_date_compact.search(s)
                    if m:
                        date = m.group(1)
                # time
                m = re_time_colon.search(s)
                if m:
                    tstr = m.group(1).replace('.', ':')
                    try:
                        dt = datetime.strptime(tstr, '%H:%M')
                        time_min = dt.hour * 60 + dt.minute
                    except Exception:
                        pass
                # also try HHMM groups
                if time_min is None:
                    # search 4-digit groups
                    for g in re.findall(r'(\d{4})', s):
                        # skip those that look like dates (YYYY)
                        if len(g) == 4:
                            hh = int(g[:2])
                            mm = int(g[2:])
                            if 0 <= hh < 24 and 0 <= mm < 60:
                                time_min = hh * 60 + mm
                                break
                # slot id: try pattern _C0N_SLOT or final numeric token
                m = re.search(r'_C0?\d+_(\d+)', s)
                if m:
                    try:
                        slot = int(m.group(1))
                    except Exception:
                        slot = None
                if slot is None:
                    # last numeric token before extension
                    toks = os.path.basename(s).split('_')
                    if toks:
                        last = toks[-1]
                        digits = ''.join([c for c in last if c.isdigit()])
                        if digits:
                            try:
                                slot = int(digits)
                            except Exception:
                                slot = None
                # normalize compact date to YYYY-MM-DD
                if date and len(date) == 8:
                    try:
                        date = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
                    except Exception:
                        pass
                if cam and slot and date and time_min is not None:
                    key = (cam, slot, date)
                    label_entries.setdefault(key, []).append((time_min, label))
    except Exception:
        logging.exception(f"Error leyendo metadata CSV {metadata_csv}")

    # sort time lists
    for k in label_entries:
        label_entries[k].sort()
    logging.info(f"Entradas etiquetadas en metadata (cam,slot,date): {len(label_entries)}")

    # leer bounding boxes de cada cámara
    cam_slots = {}
    for i in range(1, 10):
        csv_name = os.path.join(cameras_csv_dir, f'camera{i}.csv')
        if os.path.exists(csv_name):
            cam_slots[i] = read_camera_csv(csv_name)

    # escala de coordenadas desde 2592x1944 a 1000x750
    sx = 1000.0 / 2592.0
    sy = 750.0 / 1944.0

    # recorrer full images
    count_saved = 0
    full_images_checked = 0
    for weather in os.listdir(full_image_root):
        weather_dir = os.path.join(full_image_root, weather)
        if not os.path.isdir(weather_dir):
            continue
        for date in os.listdir(weather_dir):
            date_dir = os.path.join(weather_dir, date)
            if not os.path.isdir(date_dir):
                continue
            for camdir in os.listdir(date_dir):
                cam_path = os.path.join(date_dir, camdir)
                if not os.path.isdir(cam_path):
                    continue
                # camdir expected like 'camera6'
                try:
                    cam_id = int(''.join([c for c in camdir if c.isdigit()]))
                except Exception:
                    continue
                slots = cam_slots.get(cam_id, {})
                for fname in os.listdir(cam_path):
                    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    full_images_checked += 1
                    if full_images_checked % 200 == 0:
                        logging.info(f"Procesadas {full_images_checked} imágenes (guardadas {count_saved} parches)")
                    # filename like 2015-11-22_0947.jpg or 2015-11-22_09.47.jpg
                    name_noext = os.path.splitext(fname)[0]
                    # buscar parte de tiempo
                    toks = name_noext.split('_')
                    time_str = None
                    if len(toks) >= 2:
                        t = toks[1]
                        time_str = t.replace('.', '')
                        if len(time_str) == 3:
                            time_str = '0' + time_str
                    if not time_str:
                        # si no hay tiempo, saltar
                        continue
                    full_img_path = os.path.join(cam_path, fname)
                    try:
                        img = cv2.imread(full_img_path)
                        if img is None:
                            continue
                    except Exception:
                        continue
                    for slotid, (x, y, w, h) in slots.items():
                        sx_i = int(x * sx)
                        sy_i = int(y * sy)
                        sw_i = int(w * sx)
                        sh_i = int(h * sy)
                        crop = img[sy_i:sy_i+sh_i, sx_i:sx_i+sw_i]
                        if crop is None or crop.size == 0:
                            continue

                        # parse frame time into minutes
                        try:
                            ft = time_str
                            if ':' in ft or '.' in ft:
                                ft = ft.replace('.', ':')
                                dt = datetime.strptime(ft, '%H:%M')
                                frame_min = dt.hour * 60 + dt.minute
                            else:
                                if len(ft) == 3:
                                    ft = '0' + ft
                                frame_min = int(ft[:2]) * 60 + int(ft[2:])
                        except Exception:
                            frame_min = None

                        # try to find label using label_entries built from metadata
                        label = None
                        key = (cam_id, slotid, date)
                        candidates = label_entries.get(key, [])
                        # try compact date variant
                        if not candidates and '-' in date:
                            key2 = (cam_id, slotid, date.replace('-', ''))
                            candidates = label_entries.get(key2, [])
                        # try any date for cam+slot
                        if not candidates:
                            for k2 in label_entries.keys():
                                if k2[0] == cam_id and k2[1] == slotid:
                                    candidates = label_entries.get(k2, [])
                                    if candidates:
                                        break

                        if candidates and frame_min is not None:
                            # choose nearest time
                            best = None
                            best_d = None
                            for tmin, lab in candidates:
                                dmin = abs(tmin - frame_min)
                                if best is None or dmin < best_d:
                                    best = lab
                                    best_d = dmin
                            if best is not None and best_d is not None and best_d <= 20:
                                label = best
                        elif candidates and frame_min is None:
                            label = candidates[0][1]

                        if label is None:
                            # no label found -> skip quietly
                            logging.debug(f"Sin etiqueta para cam{cam_id} slot{slotid} fecha {date} time {time_str}")
                            continue

                        out_folder = free_dir if label == 0 else occ_dir
                        try:
                            crop_r = cv2.resize(crop, resize)
                            out_name = f'cam{cam_id}_{date}_{time_str}_slot{slotid}.jpg'
                            out_path = os.path.join(out_folder, out_name)
                            cv2.imwrite(out_path, crop_r)
                            count_saved += 1
                        except Exception:
                            logging.exception(f"No se pudo guardar parche para {full_img_path} slot {slotid}")
    logging.info(f"Parches guardados: {count_saved} (en {out_dataset_dir})")
    return


def main():
    # comportamiento: entrenar automáticamente con dataset detectado y luego abrir la cámara
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    # Si hay el conjunto FULL_IMAGE_1000x750 y el CSV de metadata, ofrecer construir parches
    cwd = os.getcwd()
    possible_full = None
    # intentar detectar rutas típicas dentro del proyecto
    candidates = [
        os.path.join(cwd, 'CNR-EXT_FULL_IMAGE_1000x750', 'FULL_IMAGE_1000x750'),
        os.path.join(cwd, 'CNR-EXT_FULL_IMAGE_1000x750'),
        os.path.join(cwd, 'CNR-EXT_FULL_IMAGE_1000x750', 'FULL_IMAGE_1000x750'),
        os.path.join(cwd, 'CNR-EXT_FULL_IMAGE_1000x750', 'FULL_IMAGE_1000x750')
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isdir(p):
            possible_full = p
            break
    metadata_csv = os.path.join(cwd, 'CNRPark+EXT.csv')
    cameras_csv_dir = os.path.join(cwd, 'CNR-EXT_FULL_IMAGE_1000x750')
    if possible_full and os.path.exists(metadata_csv) and os.path.isdir(cameras_csv_dir):
        try:
            build_ans = input(f"Se detectó FULL_IMAGE dataset en {possible_full} y metadata {metadata_csv}. ¿Generar dataset de parches etiquetados y guardarlo en ./dataset ? [Y/n]: ").strip().lower()
        except Exception:
            build_ans = 'y'
        if build_ans in ('', 'y', 'yes'):
            out_dataset = os.path.join(cwd, 'dataset')
            # limpiar dataset previo si existe
            if os.path.exists(out_dataset):
                try:
                    shutil.rmtree(out_dataset)
                except Exception:
                    logging.exception(f"No se pudo borrar carpeta previa {out_dataset}")
            logging.info('Construyendo dataset de parches desde full frames (esto puede tardar) ...')
            build_dataset_from_full_images(metadata_csv, possible_full, cameras_csv_dir, out_dataset, resize=(64, 64))

    # detectar dataset automáticamente: buscar carpetas con 'vacio'/'lleno' o una carpeta 'dataset' en el cwd
    entries = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d))]
    entries_lower = [d.lower() for d in entries]
    dataset_found = False
    has_vacio = any('vacio' in e for e in entries_lower)
    has_lleno = any('lleno' in e for e in entries_lower)

    if has_vacio and has_lleno:
        dataset_path = cwd
        dataset_found = True
        logging.info(f"Dataset detectado en {dataset_path} (contiene carpetas con 'vacio' y 'lleno')")
    else:
        # fallback a carpeta 'dataset'
        fallback = os.path.join(cwd, 'dataset')
        if os.path.exists(fallback) and os.path.isdir(fallback):
            dataset_path = fallback
            dataset_found = True
            logging.info(f"Usando carpeta de dataset: {dataset_path}")

    if not dataset_found:
        logging.error('No se encontró un dataset automáticamente. Crea carpetas que contengan "vacio" y "lleno", o una carpeta "dataset" con subcarpetas free/occupied.')
        return

    # intentar cargar configuración previa (ej. preferencia de augmentation)
    config_path = os.path.join(cwd, 'config.json')
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as cf:
                config = json.load(cf)
            logging.info(f"Configuración cargada desde {config_path}")
        except Exception:
            logging.exception(f"No se pudo leer {config_path}; se usarán valores por defecto.")

    # Mostrar conteo por clase y pedir confirmación
    # Hacer un conteo rápido de archivos (rápido) antes de extraer features (costoso)
    def quick_count_images(path):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        total = 0
        if not os.path.exists(path):
            return 0
        for root, _, files in os.walk(path):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    total += 1
        return total

    # contar rápidamente en dataset/free y dataset/occupied si existen
    free_dir = os.path.join(dataset_path, 'free')
    occ_dir = os.path.join(dataset_path, 'occupied')
    count_free = quick_count_images(free_dir)
    count_occ = quick_count_images(occ_dir)
    logging.info(f"Conteo rápido de archivos: free={count_free}, occupied={count_occ}")

    try:
        answer = input('Deseas continuar con el entrenamiento (esto extraerá features y puede tardar)? [Y/n]: ').strip().lower()
    except Exception:
        answer = 'y'
    if answer not in ('', 'y', 'yes'):
        logging.info('Entrenamiento cancelado por el usuario.')
        return

    # ahora sí cargar dataset (operación costosa)
    X_count, y_count = load_dataset(dataset_path, augment=False)
    unique, counts = np.unique(y_count, return_counts=True)
    class_counts = dict(zip(unique, counts))
    logging.info(f"Se encontraron las siguientes cantidades por clase: {class_counts}")

    # preguntar por augmentation (por defecto sí)
    # usar preferencia guardada si existe
    aug_default = True
    if 'augment' in config:
        aug_default = bool(config.get('augment'))
    prompt = 'Activar data augmentation durante el entrenamiento? [Y/n]: '
    if not aug_default:
        prompt = 'Activar data augmentation durante el entrenamiento? [y/N]: '
    try:
        aug_ans = input(prompt).strip().lower()
    except Exception:
        aug_ans = ''
    if aug_ans == '':
        augment_flag = aug_default
    else:
        augment_flag = True if aug_ans in ('y', 'yes') else False

    # entrenar y guardar modelo
    out_path = os.path.join(cwd, 'models', 'parking_model.pkl')
    logging.info('Iniciando entrenamiento automático... (esto puede tardar)')
    report_path = os.path.join(cwd, 'report.json')
    train_model(dataset_path, out_path, augment=augment_flag, cv_folds=0, min_required=5, report_path=report_path)

    # guardar configuración con preferencia de augmentation y última ruta de dataset/modelo
    cfg = {
        'augment': bool(augment_flag),
        'last_dataset': dataset_path,
        'last_model': out_path,
        'last_report': report_path
    }
    try:
        with open(config_path, 'w', encoding='utf-8') as cf:
            json.dump(cfg, cf, ensure_ascii=False, indent=2)
        logging.info(f"Configuración guardada en {config_path}")
    except Exception:
        logging.exception(f"No se pudo guardar la configuración en {config_path}")

    # cargar modelo y ejecutar cámara
    if not os.path.exists(out_path):
        logging.error(f"El modelo no se encontró en {out_path} después del entrenamiento.")
        return
    try:
        clf = joblib.load(out_path)
    except Exception:
        logging.exception(f"No se pudo cargar el modelo {out_path}")
        return

    # En lugar de usar la cámara en tiempo real, usar una imagen de prueba fija llamada 'prueba.jpg'
    test_img_path = os.path.join(cwd, 'prueba.jpg')
    if not os.path.exists(test_img_path):
        logging.error(f"Imagen de prueba no encontrada en {test_img_path}")
        return
    # Comprobación robusta de cv2.imread
    if not hasattr(cv2, 'imread'):
        logging.error('cv2.imread no disponible. Reinstala opencv-python. Abortando parte de prueba.')
        return
    frame = cv2.imread(test_img_path)
    if frame is None:
        logging.error(f"No se pudo leer la imagen de prueba: {test_img_path}")
        return

    # Cargar ROIs desde rois.json (debe existir en el cwd)
    rois_path = os.path.join(cwd, 'rois.json')
    if not os.path.exists(rois_path):
        logging.error(f"No se encontró {rois_path}. Coloca las ROIs en ese archivo para evaluar la imagen de prueba.")
        return
    try:
        rois = load_rois(rois_path)
        logging.info(f"ROIs cargadas desde {rois_path} ({len(rois)} regiones)")
    except Exception:
        logging.exception(f"No se pudieron cargar ROIs desde {rois_path}")
        return

    logging.info(f"Evaluando imagen de prueba: {test_img_path} con {len(rois)} ROIs")
    processed_frame, free_count, occupied_count = detect_and_mark_spaces(frame.copy(), rois, clf)

    # Además imprimir etiqueta por ROI
    per_roi = []
    # preparar carpeta de debug para crops
    debug_dir = os.path.join(cwd, 'debug_crops')
    os.makedirs(debug_dir, exist_ok=True)
    for idx, roi in enumerate(rois):
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            per_roi.append((idx, None, None))
            continue
        crop_resized = cv2.resize(crop, (64, 64))
        features = extract_features(crop_resized).reshape(1, -1)
        score = None
        try:
            if hasattr(clf, 'decision_function'):
                try:
                    score = float(clf.decision_function(features)[0])
                except Exception:
                    score = None
            elif hasattr(clf, 'predict_proba'):
                try:
                    prob = clf.predict_proba(features)[0]
                    # probability of class 1 (occupied)
                    score = float(prob[1])
                except Exception:
                    score = None
        except Exception:
            score = None
        try:
            pred = int(clf.predict(features)[0])
        except Exception:
            pred = 1
        label_str = 'Libre' if pred == 0 else 'Ocupado'
        per_roi.append((idx, label_str, score))
        # guardar crop para inspección con nombre que incluye pred y score (si existe)
        try:
            score_s = f"_{score:.3f}" if score is not None else ''
            out_fname = f'roi{idx}_pred{pred}{score_s}.jpg'
            out_path = os.path.join(debug_dir, out_fname)
            cv2.imwrite(out_path, crop_resized)
        except Exception:
            logging.exception(f"No se pudo guardar crop de ROI {idx} en {debug_dir}")

    # Imprimir resultados
    print(f"Resultado imagen de prueba: Libres={free_count}, Ocupados={occupied_count}")
    for idx, lab, score in per_roi:
        print(f"ROI {idx}: {lab}" + (f" (score: {score:.3f})" if score is not None else ""))

    out_path = os.path.join(cwd, 'prueba_out.jpg')
    try:
        cv2.imwrite(out_path, processed_frame)
        logging.info(f"Imagen con resultados guardada en {out_path}")
    except Exception:
        logging.exception(f"No se pudo guardar la imagen de salida en {out_path}")

if __name__ == '__main__':
    main()
