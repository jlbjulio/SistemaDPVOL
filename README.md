Sistema de Detección de Espacios Libres y Ocupados en Estacionamientos mediante Visión Artificial

## 1. Presentación del Proyecto (Resumen Ejecutivo)

Objetivo: Automatizar la estimación de disponibilidad de plazas en un estacionamiento a partir de imágenes fijas (multicámara) o secuencias simuladas, clasificando cada plaza como Libre u Ocupada.

Valor: Permite integrar conteo de plazas en sistemas de gestión (paneles, apps) sin hardware adicional (sensores) usando solo visión artificial. El prototipo logra >95% de exactitud en pruebas internas.

## 2. Arquitectura General

Flujo principal (script `SistemaDPVOL.py`):

1. (Opcional) Construcción automática de dataset de parches desde frames completos usando:
   - Metadata `CNRPark+EXT.csv` (ocupancy + slot_id + timestamp).
   - CSV por cámara (`camera1.csv` … `camera9.csv`) con bounding boxes originales (2592x1944) re‑escaladas a 1000x750.
2. Detección del dataset existente (`dataset/free`, `dataset/occupied`) o carpetas equivalentes "vacio" / "lleno".
3. Conteo rápido por clase y confirmación usuario.
4. Entrenamiento del modelo (Pipeline: `StandardScaler` + `LinearSVC`) sobre vectores HOG (64x64 px por parche).
5. Data augmentation opcional (brillo/contraste, flip horizontal, rotación leve).
6. Generación de `report.json` (classification_report + confusion_matrix) y guardado del modelo en `models/parking_model.pkl`.
7. Evaluación sobre una imagen fija de prueba (`prueba.jpg`) usando ROIs almacenadas en `rois.json`.

Visualización interactiva (script `gui_app.py`):

1. Carga modelo + ROIs + carpeta `pruebavideo`.
2. Reproduce imágenes secuenciales como pseudo‑video (loop) con overlay (rojo=Libre, azul=Ocupado según convención actual del proyecto).
3. Controles: Iniciar / Pausar / Continuar / Apagar + slider para intervalo de fotogramas.

```
	   FULL FRAMES + METADATA
		    │
	    (build_dataset opcional)
		    ↓
	dataset/free  dataset/occupied
		    │
	┌──────── Feature Extraction (HOG 64x64) ───────┐
	│                                              │
    StandardScaler + LinearSVC (entrenamiento)         │
	│                                              │
      parking_model.pkl ──► Evaluación (report.json)    │
	│                                              │
	└────────► GUI Flet (visualización ROIs) ◄─────┘
```

## 3. Modelo y Tipo de Entrenamiento

- Clasificador: `LinearSVC` (SVM lineal) con `class_weight='balanced'` para compensar posibles desbalances.
- Representación: Histogram of Oriented Gradients (HOG) con parámetros (orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)).
- Preprocesamiento: Redimensionado uniforme a 64x64, conversión a gris, extracción HOG, escalado con `StandardScaler`.
- Augmentations: variación de brillo/contraste, flip horizontal, rotación aleatoria (-7° a +7°). Control interactivo almacenado en `config.json`.
- Validación: Split estratificado train/test (`train_test_split` con `test_size` configurable). Opcional GridSearchCV para hiperparámetro C (no activado por defecto para reducir tiempo).

## 4. Dataset y Fuentes de Datos

1. Manual / estático: Carpetas `dataset/free` y `dataset/occupied` (o nombres equivalentes con "vacio" / "lleno"), imágenes ya recortadas o seleccionadas.
2. Automático (en desarrollo): Función `build_dataset_from_full_images` que recorre directorios por clima y fecha dentro de `FULL_IMAGE_1000x750`, usa bounding boxes de `cameraN.csv` y etiqueta cada parche buscando el timestamp más cercano (±20 min) en la metadata global `CNRPark+EXT.csv`.
3. ROIs operativas: `rois.json` contiene regiones seleccionadas manualmente sobre una imagen de referencia para la fase de inferencia/visualización.

Etiquetas:

- 0 = Libre (folder `free`)
- 1 = Ocupado (folder `occupied`)

## 5. Métricas de Evaluación (Ejemplo real de `report.json`)

Accuracy ≈ 95.65%
Clase free: Precision 0.94, Recall 0.96, F1 0.95
Clase occupied: Precision 0.97, Recall 0.95, F1 0.96
Confusion Matrix (resumida): Altas tasas de acierto con bajo false positive/false negative relativo.

Interpretación: El modelo distingue correctamente la mayoría de ocupaciones; los errores principales provienen de variaciones de iluminación, reflejos o árboles parcialmente cubriendo plazas.

## 6. Interfaz (Flet GUI)

Archivo: `gui_app.py`.
Características:

- Tema oscuro, AppBar y controles agrupados.
- Playback loop con slider de intervalo (0.5–5 s).
- Overlay por ROI con etiqueta textual y colores (rojo libre / azul ocupado).
- Manejo de errores: si falta modelo/ROIs/carpeta, muestra mensajes en lugar de fallar.

## 7. Uso Rápido (Comandos PowerShell)

Crear entorno e instalar dependencias:

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Entrenar (flujo automático):

```powershell
python SistemaDPVOL.py
```

Después del proceso tendrás: `models/parking_model.pkl`, `report.json`, `config.json`, y si existían ROIs, se utilizarán sobre `prueba.jpg`.

Abrir interfaz de visualización:

```powershell
python gui_app.py
```

Coloca tus frames secuenciales en `pruebavideo/` para el loop.

## 8. Archivo Clave y Funciones Principales

`SistemaDPVOL.py`:

- `load_dataset` / `extract_features` (HOG)
- `train_model` (Pipeline + split estratificado)
- `build_dataset_from_full_images` (etiquetado automático por metadata — todavía se debe calibrar para aumentar la tasa de matching)
- `detect_and_mark_spaces` (inferencias ROI por ROI)

`gui_app.py`:

- Clase `ParkingVideoPlayer` (hilo de reproducción)
- Overlay dinámico y controles de usuario.

## 9. Limitaciones Actuales / Trabajo Futuro

- Matching de metadata a frames puede producir pocos parches (ajustar tolerancia temporal, normalización de nombres, mejorar parse de slot_id).
- Falta integración de un detector refinado de plaza (actualmente dependemos de ROIs fijas).
- No se aplica post‑procesamiento temporal (suavizado por historial de predicciones) en la GUI.
- Podría migrarse a un modelo CNN ligero (e.g. MobileNet) para robustez ante iluminación.

## 10. Guión de Presentación (Versión Explicada 5–7 min)

Objetivo del guión: que puedas convertir cada bloque en 1–2 diapositivas claras, entendibles para público general pero manteniendo rigor básico.

1. (0:00–0:50) Introducción

   - Problema: muchos estacionamientos no muestran disponibilidad en tiempo real; usuarios pierden tiempo buscando plaza.
   - Propuesta: usar sólo imágenes (cámaras ya instaladas) para contar plazas libres y ocupadas automáticamente.
   - Beneficio inmediato: mejor experiencia, posible reducción de congestión interna.

2. (0:50–1:40) Datos

   - Fuente base: conjunto público CNRPark+EXT (múltiples días, clima variado, 9 cámaras).
   - Archivos CSV por cámara contienen la posición (rectángulo) de cada plaza.
   - Metadata global (CNRPark+EXT.csv) indica si una plaza estaba libre u ocupada en distintos momentos.
   - ROIs: “Regiones de Interés” = rectángulos sobre cada plaza; permiten recortar sólo la zona relevante.
   - Construcción: recortar cada plaza y etiquetarla como libre/ocupada para formar carpetas `free/` y `occupied/`.

3. (1:40–2:30) Modelo

   - Qué hace: aprende diferencias visuales entre una plaza vacía (patrón de piso) y una ocupada (forma/color de auto).
   - Técnica: extraemos un resumen numérico de la imagen (HOG: dirección de bordes) y entrenamos un SVM lineal.
   - Razones de elección: rápido, interpretable, funciona bien con imágenes pequeñas (64x64) y hardware estándar.
   - Pequeño aumento de datos: variaciones de brillo, rotación ligera y espejo horizontal para robustez.

4. (2:30–3:30) Pipeline

   - Paso 1: Detectar si ya existe dataset organizado (`dataset/free` y `dataset/occupied`).
   - Paso 2: (Opcional) Generar dataset desde frames completos usando CSVs y metadata (matching por tiempo ±20 min).
   - Paso 3: Extraer características HOG de cada recorte (convertir imagen → vector de números).
   - Paso 4: Entrenar modelo (escalado + SVM) y separar parte para prueba (validación interna).
   - Paso 5: Guardar modelo (`parking_model.pkl`) y reporte (`report.json`).
   - Paso 6: Usar ROIs sobre una imagen o secuencia y clasificar cada plaza en vivo.

5. (3:30–4:30) Resultados

   - Métricas reales: ~95% de exactitud; altas precisiones y recalls en ambas clases.
   - Interpretación simple: el sistema casi siempre distingue correctamente un auto de una plaza vacía.
   - Errores típicos: sombras fuertes, reflejos en piso mojado, ramas/árboles tapando parcialmente.
   - Mensaje clave: rendimiento sólido sin técnicas profundas más pesadas.

6. (4:30–5:20) Interfaz (Demo)

   - Aplicación Flet: reproduce imágenes como video (loop) para simular flujo temporal.
   - Controles: iniciar, pausar, continuar, apagar y ajustar intervalo entre frames.
   - Overlay: cada plaza coloreada y etiquetada directamente (rojo = libre, azul = ocupado según convención inicial del prototipo).
   - Valor en demo: no depende de cámara física en vivo; fácil de mostrar a stakeholders.

7. (5:20–6:10) Conclusión

   - Síntesis: datos públicos + recortes por plaza + modelo clásico = solución eficiente y portable.
   - Impacto potencial: panel de disponibilidad, app de guiado, integración con sistemas de cobro.
   - Cierre: "Transformamos imágenes en información accionable de ocupación en tiempo casi real."

8. (6:10–7:00) Preguntas
   - Reservar último minuto para aclaraciones sobre precisión, ampliación a video real, integración futura.

Frase final sugerida: "Con un enfoque ligero (HOG + SVM) y una interfaz accesible demostramos que la visión artificial puede mejorar la gestión de estacionamientos sin infraestructura adicional."

## 11. Créditos

Proyecto académico 2025. Integrantes: Lara, Julio; Batista, Joseph; Alvarado, Alex.

## 12. Licencia / Uso

Prototipo educativo. Ajustar antes de uso productivo (validación legal de cámaras, privacidad, calibración fina de ROIs).

---

Si necesitas versión resumida para diapositivas, puedo generar un extracto adicional.
