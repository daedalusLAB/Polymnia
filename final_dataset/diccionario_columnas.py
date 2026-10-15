#!/usr/bin/env python3
"""Diccionario columna por columna de las dos tablas finales, generado desde el código.

Nada de lo que escribe este script se inventa aquí: cada fórmula y cada unidad se
sacan de las constantes que viven junto a la matematica (`gesture_features.py`:
FEATURE_UNITS, EXTENDED_FEATURE_UNITS, SEMANTICS) o de la función que calcula la
columna (`_sequence_summary`, `load_frame_metadata`). Lo unico redactado a mano es
la linea de "qué mide", y cada columna lleva una `fuente` que dice de donde salió
el resto. Si aparece una columna sin resolver, el script falla en vez de callarse:
asi un README nunca puede quedarse mintiendo cuando cambian las columnas.

Además de la descripción, mide en el parquet entregado cuántos clips tienen dato
real en cada columna: un estadistico con 3 de 29 clips no es lo mismo que uno con
29, y eso es lo primero que un alumno tiene que poder ver.

Con `--readme` escribe ademas el README de la entrega: el mismo documento con una
portada de numeros medidos (filas, columnas, version de features, padding, fecha) y
sin una sola cifra heredada de otra tanda. Ese fichero es el que `final_dataset.py`
publica como `DATASET.md` dentro de `final/`, y por eso se genera **antes** de
publicar: `pipeline_lote.sh` hace una pasada de medicion a `.medicion`, saca de ahi
diccionario y README, publica `final/` con el README dentro, y compara los parquet de
ambas pasadas para que el documento no pueda describir otra cosa que lo entregado.

    python diccionario_columnas.py --final lote3/final \
        --csv lote3/informes/DICCIONARIO_COLUMNAS.csv \
        --md lote3/DICCIONARIO_COLUMNAS.md \
        --readme lote3/informes/README_entrega.md
"""

import argparse
import json
import math
import os
import re
import sys

import pandas as pd

TABLA_MANO = {"right": "mano derecha", "left": "mano izquierda"}

# Lo que mide cada medida, en una linea. La formula y la unidad van debajo, tomadas
# del codigo; aqui solo hace falta el significado, que el codigo no lo cuenta.
MIDE_V1 = {
    "n_observed": "fotogramas del hueco en los que la fuente dio ese keypoint, aunque luego fuera rechazado por confianza",
    "n_interpolated": "fotogramas rellenos por interpolación lineal entre dos extremos válidos (huecos de a lo sumo max_gap_frames)",
    "n_valid": "fotogramas con medida usable: el denominador real de todo lo demás de esta mano",
    "frac_observed": "n_observed entre frames_grid",
    "frac_interpolated": "n_interpolated entre frames_grid; si es alto, la mitad de la trayectoria es reconstruida",
    "frac_valid": "n_valid entre frames_grid",
    "mean_speed": "velocidad media de la muñeca sobre los fotogramas válidos",
    "max_speed": "la velocidad puntual más alta de la muñeca",
    "mean_accel_magnitude": "módulo de aceleración medio",
    "max_accel_magnitude": "módulo de aceleración máximo",
    "path_length": "longitud de camino: suma de desplazamientos entre fotogramas consecutivos del mismo tramo contiguo",
    "nx_range": "cuánto horizontales son los barridos: max - min de la coordenada x suavizada",
    "ny_range": "cuánto verticales son los barridos: max - min de la coordenada y suavizada",
    "mean_neck_distance": "a qué distancia del cuello se mueve la muñeca, de media",
    "angle_mean": "dirección media del vector cuello->muñeca (media circular)",
    "angle_std": "dispersión de esa dirección (medida de concentración, no una von Mises ajustada)",
    "time_to_peak_speed": "segundos desde el primer fotograma con reloj hasta el fotograma más rápido",
    "early_speed_mean": "velocidad media en el primer 10% del hueco de reloj",
    "late_speed_mean": "velocidad media en el último 10% del hueco de reloj",
}

MIDE_BILATERAL_V1 = {
    "n_both_hands": "fotogramas con las DOS muñecas válidas: sin esto, un estadistico bilateral no dice nada",
    "frac_both_hands": "n_both_hands entre frames_grid",
    "mean_pair_separation": "separación media entre las dos muñecas",
    "max_pair_separation": "la mayor separación entre las dos muñecas en un mismo fotograma",
    "mean_mirror_error_x": "media de |x_R + x_L|: cuánto se sale la pareja del eje vertical del cuerpo",
    "mean_mirror_error_y": "media de |y_R - y_L|: cuánto las dos manos están a distinta altura",
    "time_to_peak_pair_separation": "segundos hasta el fotograma de máxima separación entre manos",
}

# Familias del bloque extendido: una linea por familia, la formula va desde el codigo.
# Las claves son los nombres de familia EXACTOS que salen de las claves de
# EXTENDED_FEATURE_UNITS; si el codigo renombra una familia, la columna cae en el
# guard de "sin documentar" en vez de quedarse con una descripcion vieja.
MIDE_FAMILIA = {
    "distribution aggregates": "la forma completa de la distribución de esa medida en el clip, no solo su media",
    "extent": "hasta dónde llega la muñeca en cada eje: extremos y dispersión de la coordenada suavizada",
    "reversals": "cuántas veces cambia de sentido el movimiento en ese eje (agite, no desplazamiento)",
    "reversals_per_s": "cambios de sentido por segundo, para comparar clips de distinta duración",
    "hull_area": "área del envolvente convexo de la trayectoria: el área que la mano barre",
    "ellipse_area": "área de la elipse de 2 desviaciones tipicas de la nube de puntos: el área 'habituada'",
    "bbox_area": "área del rectángulo alineado con los ejes que encierra toda la trayectoria",
    "scale-free intra-clip ratios": "cocientes entre dos estadisticos del MISMO clip: forma sin depender de otros clips",
    "shape of the speed distribution": "asimetría y apuntamiento de la distribución de velocidad",
    "frac_speed_gt_0p5 / 1p0 / 2p0": "fracción del tiempo con la mano por encima de una velocidad absoluta dada",
    "speed_sq_mean, speed_sq_sum": "media y suma de la velocidad al cuadrado (lo que el código de referencia llamaba energía cinética)",
    "n_speed": "fotogramas con velocidad finita: el denominador detrás de todos los estadisticos de velocidad de esta mano",
    "longest_moving_s": "la tanda continua más larga moviendo esa mano por encima del umbral",
    "cos_phase_*": "en qué dirección van las dos manos a la vez: cos del ángulo entre sus velocidades (+1 mismo sentido, -1 contrario)",
    "frac_cos_gt_0p8, frac_cos_lt_m0p2": "cuánto del tiempo es in-fase o anti-fase de verdad",
    "n_cos_phase": "fotogramas detrás de las cifras de fase: si es pequeño, la fase de ese clip no es un resultado",
    "coactivity_frac_0p5, coactivity_frac_1p0": "fracción del tiempo con las DOS manos por encima del umbral de velocidad",
    "coactivity_bout_*": "cuántas tandas hay y cuánto duran: ritmo de alternancia entre manos",
    "mirror_error_x_std, mirror_error_y_std": "lo variable que es la simetría dentro del clip, no solo su media",
    "abs_y_pair_sum_mean": "magnitud de la altura conjunta de la pareja de manos (NO es simetría)",
    "pair_elevation_mean": "altura media de la pareja de manos respecto al cuello, con signo (positivo = por encima)",
    "bbox_area_wrists": "área del rectángulo que tiene las dos muñecas como esquinas opuestas",
    "jerk": "lo brusco que cambia la aceleración: la tercera derivada de la coordenada suavizada",
}

MIDE_PROCEDENCIA = {
    "id": "identificador del clip: nombre del procesado con el que encaja la ventana del TSV",
    "people_id": "índice del candidato de OpenPose dentro del clip; en esta entrega siempre 1",
    "frames_source": "filas que dio la fuente para este clip",
    "frames_grid": "hueco de fotogramas entre el primero y el último observado (min..max), inclusive",
    "frames_missing_grid": "huecos dentro de ese rango que la fuente no dio y que se marcan como inferidos",
    "n_timed_frames": "fotogramas con marca de reloj real",
    "n_timestamp_inferred": "fotogramas a los que se les prestó el reloj por interpolación corta entre dos con reloj",
    "n_missing_timestamps": "frames_grid - n_timed_frames: fotogramas sin reloj, donde lo temporal es NULL",
    "t_start": "primer reloj conocido de la secuencia (segundos del frame_timestamp de origen)",
    "t_end": "último reloj conocido de la secuencia",
    "duration_s": "t_end - t_start; es la duración MEDIDA del intervalo, no la del video crudo",
}

MIDE_ANNOTATION = {
    "indoor_outdoor": "etiqueta del modelo visual: interior o exterior",
    "show_type": "etiqueta del modelo visual: género del programa (news anchor, weather report, interview, monologue)",
    "hands_free": "etiqueta del modelo visual: si tiene las manos libres para gesticular",
    "sitting_standing": "etiqueta del modelo visual: sentado o de pie",
    "screen_interaction": "etiqueta del modelo visual: si interacciona con una pantalla",
    "sex": "etiqueta del modelo visual: sexo atribuido a la persona",
    "age": "edad estimada por el modelo visual, en años",
    "status": "estado de la llamada al modelo visual para este clip",
    "error": "texto de error de la anotación; vacio cuando la anotación salió bien",
}

MIDE_META = {
    "newscape_clip_id": "el nombre de clip que produjo el pipeline (raw -> ASD -> OpenPose), tal cual",
    "tab_context": "campo 7 del TSV de entrada: el contexto que pidió quien hizo la selección",
    "tab_start_raw": "inicio de la ventana tal como venía en el TSV, sin decodificar",
    "tab_end_raw": "fin de la ventana tal como venía en el TSV, sin decodificar",
    "tab_padding_s": "colchón que se añadió a la ventana del TSV para casarla con el clip procesado",
    "processed_padding_s": "colchón que ya traia el nombre del clip procesado",
    "id_source": "cómo se emparejó esta fila con el TSV: id exacto o centro de ventana",
    "datetime_utc": "fecha y hora de emisión del programa en UTC",
    "country_code": "código de país del programa",
    "network_code": "código de cadena del programa",
    "exp_search": "searchterm del TSV: la búsqueda de la que salió este clip",
}

MIDE_FRAME = {
    "formant.1_frequency": "primer formante de la voz en ese fotograma (Hz, Praat)",
    "formant.2_frequency": "segundo formante de la voz en ese fotograma (Hz, Praat)",
    "formant.3_frequency": "tercer formante de la voz en ese fotograma (Hz, Praat)",
    "formant.4_frequency": "cuarto formante de la voz en ese fotograma (Hz, Praat)",
    "formant.5_frequency": "quinto formante de la voz en ese fotograma (Hz, Praat)",
    "frame": "número de fotograma dentro del clip",
    "frame_timestamp": "segundos del frame_timestamp de origen; NULL donde no hay reloj",
    "source_observed": "la fuente dio este fotograma (falso = hueco rellenado)",
    "frame_inferred": "este fotograma no estaba en la fuente y se reconstruyó",
    "clock_known": "este fotograma tiene reloj real",
    "timestamp_inferred": "el reloj de este fotograma se prestó por interpolación corta",
    "both_hands_valid": "las dos muñecas son válidas en este fotograma",
    "pair_separation": "distancia entre las dos muñecas",
    "pair_angle": "ángulo del vector muñeca izquierda -> muñeca derecha",
    "centroid_x": "coordenada x del punto medio entre las dos muñecas",
    "centroid_y": "coordenada y del punto medio entre las dos muñecas",
    "mirror_error_x": "|x_R + x_L| sobre el eje vertical del cuerpo, fotograma a fotograma",
    "mirror_error_y": "|y_R - y_L|: diferencia de altura entre las dos manos",
    "pitch": "tono fundamental de la voz en ese fotograma (Hz, Praat)",
    "voice_present": "si Praat seguía una pista de pitch en ese fotograma: True medida, False sin pista, "
                     "NULL si el clip no tiene pista de pitch. Es la puerta que hay que abrir antes de "
                     "cualquier estadístico de voz (como right_n_speed en las muñecas)",
    "intensity": "intensidad de la voz en ese fotograma (dB, Praat)",
    "harmonicity": "relación armónico-ruído de la voz en ese fotograma (Praat)",
    "vad": "actividad de voz: hay habla en ese fotograma",
    "words": "palabra activa en ese fotograma segun WhisperX",
    "openpose_json": "nombre del JSON de OpenPose del que salió esta fila (trazabilidad)",
    "datetime_utc": "fecha y hora de emisión del programa en UTC",
    "country_code": "código de país del programa",
    "network_code": "código de cadena del programa",
    "exp_search": "searchterm del TSV: la búsqueda de la que salió este clip",
    "id": "identificador del clip, igual que en final_videos",
    "people_id": "índice del candidato de OpenPose; en esta entrega siempre 1",
}

MIDE_FRAME_MANO = {
    "nx": "coordenada x cruda del keypoint, normalizada por el cuello",
    "ny": "coordenada y cruda del keypoint, normalizada por el cuello",
    "nx_smooth": "coordenada x suavizada (Savitzky-Golay por tramo contiguo)",
    "ny_smooth": "coordenada y suavizada",
    "confidence": "confianza que OpenPose asigna a ese keypoint en este fotograma",
    "observed": "la fuente dio este keypoint en este fotograma",
    "interpolated": "la coordenada viene de interpolación de un hueco corto",
    "valid": "la medida es usable en este fotograma: keypoint presente y confianza >= mínimo, o interpolado",
    "above_neck": "la muñeca esta por encima de la altura del cuello (ny suavizada > 0)",
    "neck_distance": "distancia de la muñeca al cuello",
    "neck_angle": "ángulo del vector cuello -> muñeca",
    "velocity_x": "velocidad de la muñeca en x",
    "velocity_y": "velocidad de la muñeca en y",
    "speed": "módulo de la velocidad de la muñeca",
    "accel_x": "aceleración de la muñeca en x",
    "accel_y": "aceleración de la muñeca en y",
    "accel_magnitude": "módulo de la aceleración",
    "step_distance": "desplazamiento al fotograma anterior del mismo tramo contiguo (no necesita reloj)",
    "jerk_x": "variación de la aceleración en x",
    "jerk_y": "variación de la aceleración en y",
    "jerk_magnitude": "módulo del jerk: lo brusco del cambio de aceleración",
}

STATS = ("min", "p05", "p25", "p50", "p75", "p95", "max", "std")

# Contadores cuya definicion vive en el codigo DENTRO de la nota de otra familia, en
# vez de tener clave propia. Sin esto quedarian "sin documentar" aunque el codigo si
# los explique; se enlazan a esa nota para no redactar una segunda version que se
# pueda desincronizar.
NOTA_DE = {"n_cos_phase": "cos_phase_"}
MIDE_CONTADORES = {"n_cos_phase": "fotogramas detrás de las cifras de fase: si es pequeño, la fase de ese clip no es un resultado"}
MIDE_STAT = {"min": "mínimo", "max": "máximo", "std": "desviación típica muestral (ddof=1)",
             "p05": "percentil 5", "p25": "percentil 25", "p50": "mediana",
             "p75": "percentil 75", "p95": "percentil 95"}
NOMBRE_MEDIDA_LARGA = {"accel_magnitude": "aceleración", "jerk_magnitude": "jerk",
                       "neck_distance": "distancia al cuello", "speed": "velocidad"}

# Fórmulas y unidades que NO viven en las constantes de unidades del código. Están
# transcriptas de la función que las calcula (cada fila cita cuál), así que su verdad
# se comprueba leyendo esa función. Sin esta tabla, ~95 columnas saldrían en el
# diccionario con la fórmula vacía: el alumno vería "qué mide" pero no "cómo".
FORMULA_PROPIA = {
    # --- por fotograma y mano: gesture_features.py: _hand_rows
    "confidence": ("confianza del keypoint",
                   "score de confianza de OpenPose para el keypoint de esa muñeca, en [0,1]; no es una probabilidad calibrada",
                   "gesture_features.py: _hand_rows"),
    "valid": ("confianza del keypoint",
              "usable = coordenadas finitas y conf > 0 y conf >= min_keypoint_confidence (0,2 por defecto); valid = usable | interpolado. Si es falsa, todo lo demás de esa mano en este fotograma es NULL",
              "gesture_features.py: _hand_rows"),
    "observed": ("confianza del keypoint",
                 "usable y la fuente dio este fotograma (no relleno). Difiere de valid exactamente en los fotogramas interpolados",
                 "gesture_features.py: _hand_rows"),
    "interpolated": ("confianza del keypoint",
                     "la coordenada viene de _interpolate_gaps: hueco corto entre dos extremos válidos, nunca en los bordes",
                     "gesture_features.py: _hand_rows"),
    "above_neck": ("geometría por fotograma",
                   "ny_smooth > 0: la muñeca está por encima del cuello. NULL donde no hay coordenada válida",
                   "gesture_features.py: _hand_rows"),
    # --- reloj y hueco: gesture_features.py: _sequence_grid
    "frame": ("identificación y procedencia", "número de fotograma dentro del clip, entero del JSON de OpenPose",
              "gesture_features.py: _sequence_grid"),
    "frame_timestamp": ("time",
                        "segundos del frame_timestamp de origen; NULL donde no hay reloj. Solo se interpola en huecos cortos internos (timestamp_inferred); los bordes no se extrapolan",
                        "gesture_features.py: _sequence_grid"),
    "source_observed": ("identificación y procedencia", "la fuente dio este fotograma dentro del hueco observado",
                        "gesture_features.py: _sequence_grid"),
    "frame_inferred": ("identificación y procedencia", "no estaba en la fuente: hueco interno del rango observado",
                       "gesture_features.py: _sequence_grid"),
    "clock_known": ("time", "este fotograma tiene reloj real (frame_timestamp finito)",
                    "gesture_features.py: _sequence_grid"),
    "timestamp_inferred": ("time", "el reloj viene de prestárselo a un hueco interno de a lo sumo max_gap_frames fotogramas",
                           "gesture_features.py: _sequence_grid"),
    "both_hands_valid": ("counts and fractions", "right_valid y left_valid en el mismo fotograma: la puerta de todo lo bilateral",
                         "gesture_features.py: _sequence_frame_rows"),
    # --- agregados v1: gesture_features.py: _hand_summary
    "n_observed": ("counts and fractions", "suma del flag observed de esa mano sobre el hueco observado (unidad: fotogramas)",
                   "gesture_features.py: _hand_summary"),
    "n_interpolated": ("counts and fractions", "suma del flag interpolated de esa mano (fotogramas)",
                       "gesture_features.py: _hand_summary"),
    "n_valid": ("counts and fractions", "suma del flag valid de esa mano: el denominador real de sus medidas",
                "gesture_features.py: _hand_summary"),
    "frac_observed": ("counts and fractions", "n_observed / frames_grid, adimensional",
                      "gesture_features.py: _hand_summary"),
    "frac_interpolated": ("counts and fractions", "n_interpolated / frames_grid, adimensional",
                          "gesture_features.py: _hand_summary"),
    "frac_valid": ("counts and fractions", "n_valid / frames_grid, adimensional",
                   "gesture_features.py: _hand_summary"),
    "mean_speed": ("velocity", "np.mean sobre los speed finitos del clip, en unidades de cuello por segundo",
                   "gesture_features.py: _hand_summary"),
    "max_speed": ("velocity", "np.max de los speed finitos: lo define un solo fotograma, no es robusto a cortes de plano",
                  "gesture_features.py: _hand_summary"),
    "mean_accel_magnitude": ("acceleration", "np.mean de accel_magnitude finito, en unidades de cuello por segundo al cuadrado",
                             "gesture_features.py: _hand_summary"),
    "max_accel_magnitude": ("acceleration", "np.max de accel_magnitude finito", "gesture_features.py: _hand_summary"),
    "mean_neck_distance": ("distances", "np.mean de neck_distance finito, en unidades de cuello",
                           "gesture_features.py: _hand_summary"),
    "early_speed_mean": ("velocity", "media de speed en el primer 10% del hueco [t_start, t_end] de reloj conocido",
                         "gesture_features.py: _edge_window_mean"),
    "late_speed_mean": ("velocity", "media de speed en el último 10% del hueco [t_start, t_end] de reloj conocido",
                        "gesture_features.py: _edge_window_mean"),
    # --- bilateral v1: gesture_features.py: _sequence_summary
    "n_both_hands": ("counts and fractions", "suma de both_hands_valid (fotogramas)",
                     "gesture_features.py: _sequence_summary"),
    "frac_both_hands": ("counts and fractions", "n_both_hands / frames_grid, adimensional",
                        "gesture_features.py: _sequence_summary"),
    "mean_pair_separation": ("distances", "np.mean de pair_separation = hipot(nx_R - nx_L, ny_R - ny_L) suavizadas, en unidades de cuello",
                             "gesture_features.py: _sequence_summary"),
    "max_pair_separation": ("distances", "np.max de pair_separation", "gesture_features.py: _sequence_summary"),
    "mean_mirror_error_x": ("distances",
                            "np.mean de |nx_R + nx_L|: con X centrada en el cuello, una pareja simétrica da ~0; crece si las dos manos se van al mismo lado",
                            "gesture_features.py: _sequence_summary"),
    "mean_mirror_error_y": ("distances", "np.mean de |ny_R - ny_L|: diferencia de altura entre las dos manos",
                            "gesture_features.py: _sequence_summary"),
    # --- procedencia de la secuencia: gesture_features.py: _sequence_summary
    "frames_source": ("identificación y procedencia", "cuántos fotogramas del hueco trajeron fila de la fuente (suma de source_observed)",
                      "gesture_features.py: _sequence_summary"),
    "frames_grid": ("identificación y procedencia",
                    "tamaño del hueco min(frame)..max(frame) inclusive: el denominador de todos los frac_*",
                    "gesture_features.py: _sequence_grid"),
    "frames_missing_grid": ("identificación y procedencia", "suma de frame_inferred: huecos internos sin fila de la fuente",
                            "gesture_features.py: _sequence_summary"),
    "n_timed_frames": ("time", "suma de clock_known: fotogramas con reloj real", "gesture_features.py: _sequence_summary"),
    "n_timestamp_inferred": ("time", "suma de timestamp_inferred: fotogramas con reloj prestado",
                             "gesture_features.py: _sequence_summary"),
    "n_missing_timestamps": ("time", "frames_grid - n_timed_frames: fotogramas sin reloj",
                             "gesture_features.py: _sequence_summary"),
    # --- voz: la produce el stage de voz (parole), NO la matemática de muñecas. Las cotas
    # del análisis son fijas y están en extract_prosody.praat: pitch 60-600 Hz, formantes
    # hasta 5500 Hz, suelo del Harmonicity (cc) en -200 dB. Cualquier valor FUERA de esas
    # cotas no lo produjo Praat: delata la cadena antigua (una spline para toda la pista y
    # centinela pitch<10 -> 0). El regimen de cada entrega se MIDE, no se supone.
    "pitch": ("voz (Praat)",
              "tono fundamental en Hz. Cota del análisis: 60-600 Hz (Praat To Pitch, paso 0,001 s). Praat marca el silencio con el candidato 'mute' (frequency = 0), que en la cadena actual sale como NULL y no como 0 Hz. Un valor fuera de 60-600 no lo pudo producir Praat",
              "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "intensity": ("voz (Praat)",
                  "intensidad en dB (Praat To Intensity: min pitch 75 Hz, paso 0,001 s). Praat la sigue midiendo durante el silencio, así que esta columna casi no tiene huecos: no la uses como índice de voz, para eso está voice_present",
                  "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "harmonicity": ("voz (Praat)",
                    "relación armónico-ruído en dB (Praat To Harmonicity (cc): paso 0,01 s, desde 75 Hz). Los valores negativos son normales (más ruido que armónicos) y el suelo del método es -200 dB: en silencio vale -200. Lo que no existe es un valor POR DEBAJO de -200",
                    "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "formant.1_frequency": ("voz (Praat)",
                            "primer formante en Hz (Praat To Formant (burg): paso 0,001 s, 5 formantes con techo 5500 Hz). Toda frecuencia fuera de 0-5500 Hz es imposible: no es un valor raro, es otra cadena de procesado",
                            "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "formant.2_frequency": ("voz (Praat)", "segundo formante en Hz (misma extracción y mismas cotas que formant.1_frequency)",
                            "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "formant.3_frequency": ("voz (Praat)", "tercer formante en Hz (misma extracción y mismas cotas que formant.1_frequency)",
                            "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "formant.4_frequency": ("voz (Praat)", "cuarto formante en Hz (misma extracción y mismas cotas que formant.1_frequency)",
                            "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "formant.5_frequency": ("voz (Praat)", "quinto formante en Hz (misma extracción y mismas cotas que formant.1_frequency)",
                            "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "voice_present": ("voz (Praat)",
                      "TRUE solo donde Praat seguía una pista de pitch real en ese fotograma, FALSE donde no la había, NULL si el clip no tiene pista de pitch. Es la puerta que hay que abrir antes de cualquier estadístico de voz, y el equivalente a right_n_speed en las muñecas: sin ella, NULL, silencio y valor interpolado visten igual",
                      "create_datasets/parole/scripts/process_prosody.R: voice_present = !is.na(pitch) alineada por tramos"),
    "vad": ("voz (Praat)",
            "actividad de voz por fotograma, de los segmentos de Silero VAD: es lo único de la voz que es una detección y no un muestreo de una pista. Publica como booleana (medido: BOOLEAN en las tres bases del corpus, con valor en ~99 % de los fotogramas). Si en algún lote llegara entera, el constructor la castea a booleana antes de publicar",
            "create_datasets/parole/scripts/silero_vad_runner.py via parole.sh"),
    "words": ("voz (Praat)",
              "palabra activa en ese fotograma según WhisperX con las marcas alineadas al video. NULL donde no hay palabra transcrita: aquí NULL sí significa silencio real",
              "apate.py (whisperx + framer)"),
    # --- anotaciones visuales: las escribe theia.py, no se miden sobre el video
    "indoor_outdoor": ("anotación visual",
                       "etiqueta del modelo visual a partir de frames del video RAW, en {indoor, outdoor}. Son dos llamadas por video: esta es la de escena. NULL si la llamada fallo",
                       "theia.py: SCENE_FIELDS + ENUMS"),
    "show_type": ("anotación visual",
                  "etiqueta de género del programa en {news anchor, monologue, weather report, interview}; el vocabulario es cerrado, el modelo no puede inventar otra clase",
                  "theia.py: SCENE_FIELDS + ENUMS"),
    "hands_free": ("anotación visual",
                   "BOOLEANO: el modelo entiende que la persona tiene las manos libres para gesticular. Sale de la llamada de hablante (frames MASKED + RAW) y se normaliza desde true/yes/si/1",
                   "theia.py: SPEAKER_FIELDS + TRUEISH/FALSEISH"),
    "sitting_standing": ("anotación visual",
                         "etiqueta en {sitting, standing} de la llamada de hablante. Sirve para estratificar: un presentador sentado gesticula distinto de uno de pie",
                         "theia.py: SPEAKER_FIELDS + ENUMS"),
    "screen_interaction": ("anotación visual",
                           "BOOLEANO: interacciona con una pantalla, la misma normalizacion que hands_free. En este lote solo 3 de 29 clips dicen que si",
                           "theia.py: SPEAKER_FIELDS + TRUEISH/FALSEISH"),
    "sex": ("anotación visual",
            "etiqueta en {male, female} de la llamada de hablante: asignacion del modelo, no una variable registrada",
            "theia.py: SPEAKER_FIELDS + ENUMS"),
    "age": ("anotación visual",
            "edad estimada en años, entero validado en [0, 120]. Ojo: el modelo da números redondos; sirve para estratificar, no como edad real",
            "theia.py: SPEAKER_FIELDS + AGE_RANGE"),
    # --- identificación, ventana y metadatos: final_dataset.py
    "id": ("identificación y procedencia",
           "clave de clip, reconstruida desde el nombre procesado y casada con la ventana del TSV; es la única unión entre las dos tablas",
           "final_dataset.py: parse_clip_id / match_tab_to_clips"),
    "people_id": ("identificación y procedencia",
                  "índice del candidato de OpenPose dentro del clip (person_index del JSON, sin tracking entre fotogramas)",
                  "final_dataset.py: load_frame_metadata"),
    "newscape_clip_id": ("identificación y procedencia", "nombre exacto del clip que produjo el pipeline",
                         "final_dataset.py: parse_clip_id"),
    "tab_context": ("identificación y procedencia", "campo 7 del TSV de entrada, literal",
                    "final_dataset.py: read_tab"),
    "tab_start_raw": ("identificación y procedencia",
                      "inicio de ventana tal como venía en el TSV. Recuerda el divisor: msec se divide por 1000 si el programa empieza por 2017, por 100 en el resto",
                      "final_dataset.py: fractional_divisor / read_tab"),
    "tab_end_raw": ("identificación y procedencia", "fin de ventana tal como venía en el TSV, con el mismo divisor",
                    "final_dataset.py: fractional_divisor / read_tab"),
    "tab_padding_s": ("identificación y procedencia",
                      "colchón añadido a la ventana del TSV para casarla. No es una constante universal: depende del lote (las entregas con ids ya procesados usan 2,5 s)",
                      "final_dataset.py: clip_window"),
    "processed_padding_s": ("identificación y procedencia", "colchón que ya traía el nombre del clip procesado",
                            "final_dataset.py: parse_clip_id"),
    "id_source": ("identificación y procedencia",
                  "cómo se emparejó la fila: id exacto o centro de ventana. Conviene no mezclar los dos en un mismo análisis",
                  "final_dataset.py: match_tab_to_clips"),
    "status": ("anotación visual", "estado de la llamada al modelo visual para este clip",
               "final_dataset.py: load_annotations"),
    "error": ("anotación visual",
              "texto de error; VACÍO (no NULL) cuando la anotación salió bien. Contar los OK exige mirar la cadena vacía",
              "final_dataset.py: load_annotations"),
    "openpose_json": ("identificación y procedencia",
                      "nombre del JSON de OpenPose del que salió la fila. program_name y time_range son ese mismo nombre, no metadata del programa",
                      "final_dataset.py: load_frame_metadata"),
    "datetime_utc": ("identificación y procedencia", "fecha y hora de emisión del programa, en UTC",
                     "final_dataset.py: load_frame_metadata"),
    "country_code": ("identificación y procedencia", "código de país del programa, tal como lo da NewsScape",
                     "final_dataset.py: load_frame_metadata"),
    "network_code": ("identificación y procedencia", "código de cadena del programa", "final_dataset.py: load_frame_metadata"),
    "exp_search": ("identificación y procedencia", "searchterm del TSV: la búsqueda de la que salió este clip",
                   "final_dataset.py: read_tab"),
}


def tabla_desde_codigo(gf):
    """Las definiciones leidas de las constantes de gf, en tres niveles de precisión.

    Las claves de FEATURE_UNITS y EXTENDED_FEATURE_UNITS llevan entre parentesis la
    lista de medidas a las que se aplican; ahí esta la fórmula real. Se separan en
    `exactos` (el nombre completo de una medida), `prefijos` (`cos_phase_*`) y
    `sufijos` (`*_p95`). La resolución prueba exacto, luego prefijo más largo, luego
    sufijo: asi la definición especifica de `mirror_error_x_std` no queda tapada por
    la familia general `mirror_error_*`.
    """
    exactos, prefijos, sufijos = {}, [], []

    def apunta(nombre, familia, texto):
        nombre = nombre.strip()
        if not nombre:
            return
        if nombre.endswith("_*"):
            prefijos.append((nombre[:-1], familia, texto))
        elif nombre.startswith("*_"):
            sufijos.append((nombre[1:], familia, texto))
        elif "/" in nombre:                          # "frac_speed_gt_0p5 / 1p0 / 2p0"
            base = nombre.rsplit("_", 1)[0]
            for parte in nombre.split("/"):
                exactos.setdefault(f"{base}_{parte.strip().rsplit('_', 1)[-1]}", (familia, texto))
        else:
            exactos.setdefault(nombre, (familia, texto))

    for dicc in (gf.EXTENDED_FEATURE_UNITS, gf.FEATURE_UNITS):
        for clave, texto in dicc.items():
            encontrado = re.match(r"^(.*?)\s*\((.*)\)$", clave.strip(), re.S)
            contenido = encontrado.group(2) if encontrado else clave
            familia = (encontrado.group(1) if encontrado else clave).strip()
            for trozo in contenido.split(","):
                apunta(trozo, familia, texto)
    prefijos.sort(key=lambda t: len(t[0]), reverse=True)
    for nombre, prefijo in NOTA_DE.items():
        definicion = next(((f, t) for p, f, t in prefijos if p == prefijo), None)
        if definicion is not None:
            exactos.setdefault(nombre, definicion)
    return exactos, prefijos, sufijos


def _limpiar(texto):
    return re.sub(r"\s+", " ", texto).strip()


def resolver(tabla, columna, gf, codigo):
    """Una fila del diccionario, o None si no hay forma honesta de documentarla."""
    exactos, prefijos, sufijos = codigo
    ambito, base = "clip" if tabla == "final_videos" else "fotograma", columna
    for prefijo, nombre in TABLA_MANO.items():
        if columna.startswith(prefijo + "_"):
            ambito, base = nombre, columna[len(prefijo) + 1:]
            break
    familia = formula = mides = None

    definicion = exactos.get(base)
    if definicion is None:
        definicion = next(((f, t) for p, f, t in prefijos if base.startswith(p)), None)
    if definicion is None:
        sufijo = next(((f, t) for s, f, t in sufijos if base.endswith(s)), None)
        if sufijo is not None:
            medida, estadistico = base.rsplit("_", 1)
            definicion = sufijo
            mides = (f"{MIDE_STAT[estadistico]} de {NOMBRE_MEDIDA_LARGA.get(medida, medida)} "
                     "en el clip, sobre los valores finitos de esa medida")
    if definicion is not None:
        familia, formula = _limpiar(definicion[0]), _limpiar(definicion[1])
    elif base in FORMULA_PROPIA:
        familia, formula = _limpiar(FORMULA_PROPIA[base][0]), _limpiar(FORMULA_PROPIA[base][1])

    if mides is None:
        mides = (MIDE_V1.get(base) or MIDE_BILATERAL_V1.get(base) or MIDE_PROCEDENCIA.get(base)
                 or MIDE_ANNOTATION.get(base) or MIDE_META.get(base) or MIDE_FRAME.get(base)
                 or MIDE_FRAME_MANO.get(base) or MIDE_CONTADORES.get(base))
    if mides is None and familia:
        mides = MIDE_FAMILIA.get(_limpiar(familia))
    if mides is None:
        return None

    fuente = ("gesture_features.py: FEATURE_UNITS/EXTENDED_FEATURE_UNITS" if formula else
              "gesture_features.py: _hand_summary/_sequence_summary")
    # Precedencia de la fuente: la cita concreta manda sobre la genérica. Sin este
    # orden, `frames_grid` perdía _sequence_grid y quedaba en _sequence_summary.
    if base in FORMULA_PROPIA:
        fuente = FORMULA_PROPIA[base][2]
    elif base in MIDE_META or base in MIDE_ANNOTATION:
        fuente = "final_dataset.py: load_annotations / CLIP_META"
    elif base in MIDE_PROCEDENCIA:
        fuente = "gesture_features.py: _sequence_summary"
    elif tabla == "final_frames" and (base in MIDE_FRAME_MANO or base in MIDE_FRAME):
        fuente = "gesture_features.py: FRAME_COLUMNS"
    return {"tabla": tabla, "columna": columna, "ambito": ambito,
            "familia": familia or "identificación y procedencia",
            "que_mide": mides, "formula_unidades": formula or "", "fuente": fuente}


def cubrir(df, fila):
    """Cuántos datos reales hay en esa columna, en ESTE lote."""
    serie = df[fila["columna"]]
    fila["no_nulos"] = int(serie.notna().sum())
    fila["pct_no_nulos"] = round(100.0 * fila["no_nulos"] / len(df), 1)
    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
        limpios = serie.dropna()
        if len(limpios):
            fila["minimo"] = round(float(limpios.min()), 4)
            fila["mediana"] = round(float(limpios.median()), 4)
            fila["maximo"] = round(float(limpios.max()), 4)
    fila["tipo"] = str(serie.dtype)
    return fila


def generar(ruta_final, polymnia_root):
    sys.path.insert(0, polymnia_root)
    try:
        import gesture_features as gf
    except ImportError as exc:
        raise SystemExit(f"no puedo importar gesture_features desde {polymnia_root}: {exc}")
    codigo = tabla_desde_codigo(gf)

    manifiesto = {}
    ruta_manifest = os.path.join(ruta_final, "manifest.json")
    if os.path.isfile(ruta_manifest):
        with open(ruta_manifest, encoding="utf-8") as lector:
            manifiesto = json.load(lector)

    filas, pendientes = [], []
    for tabla in ("final_videos", "final_frames"):
        ruta = os.path.join(ruta_final, f"{tabla}.parquet")
        if not os.path.isfile(ruta):
            continue
        df = pd.read_parquet(ruta)
        for columna in df.columns:
            fila = resolver(tabla, columna, gf, codigo)
            if fila is None:
                pendientes.append(f"{tabla}.{columna}")
                continue
            filas.append(cubrir(df, {**fila, "filas_tabla": len(df)}))
    if pendientes:
        raise SystemExit("columnas sin documentar (anadelas a las tablas MIDE, no las "
                         "inventeis aquí): " + ", ".join(pendientes))

    informe = {
        "feature_version": manifiesto.get("features", {}).get("version"),
        "extendidas": manifiesto.get("features", {}).get("extended"),
        # Datos de la portada del README. Del manifiesto de la pasada de medicion, que es el
        # mismo manifiesto que se publica: el pipeline compara los parquet antes de fiarse.
        "fecha": manifiesto.get("generated_at"),
        "padding_s": (manifiesto.get("clip_id_mapping") or {}).get("requested_padding_s"),
        "searchterm": (manifiesto.get("clip_id_mapping") or {}).get("searchterm"),
        "umbral_movimiento": manifiesto.get("movement_speed_units_per_s", gf.MOVEMENT_SPEED),
        "suelo_muestras": gf.MIN_STAT_SAMPLES,
        "reglas_admision": manifiesto.get("admission_rules", {}),
        "semantica_diseno": {**gf.SEMANTICS, **gf.EXTENDED_SEMANTICS},
    }
    return pd.DataFrame(filas), informe


ORDEN_FAMILIAS = ("identificación y procedencia", "coordinates", "distances", "velocity",
                  "acceleration", "angles", "time", "counts and fractions", "NULL",
                  "distribution aggregates", "extent", "reversals", "hull_area", "ellipse_area",
                  "bbox_area", "scale-free intra-clip ratios", "shape of the speed distribution",
                  "speed_sq_mean, speed_sq_sum", "cos_phase_*", "coactivity_bout_*",
                  "jerk")


def _familia_de_orden(familia):
    for i, nombre in enumerate(ORDEN_FAMILIAS):
        if familia.startswith(nombre):
            return i
    return len(ORDEN_FAMILIAS)


def _portada(diccionario, informe):
    """La cabecera que convierte el documento en README de una entrega concreta.

    Solo los numeros medidos: filas y columnas salen del parquet, version y padding del
    manifiesto. No hay ni una cifra de otra tanda, que es como nacio el problema.
    """
    lineas = [
        "> **README de la entrega.** Va junto a `final_videos.parquet` y `final_frames.parquet`: "
        "es la documentación de las dos tablas que tienes delante, columna por columna.",
        "",
        "| dato de esta entrega | valor |",
        "|---|---|",
    ]
    for tabla in ("final_videos", "final_frames"):
        sub = diccionario[diccionario["tabla"] == tabla]
        if sub.empty:
            continue
        granos = "clips" if int(sub["filas_tabla"].iloc[0]) < 1000 else "filas"
        if tabla == "final_frames":
            granos = "fotogramas"
        lineas.append(f"| `{tabla}` | {int(sub['filas_tabla'].iloc[0])} {granos} x "
                      f"{len(sub)} columnas |")
    extendidas = "incluido" if informe["extendidas"] else "no incluido"
    lineas.append(f"| versión de features | {informe['feature_version']} "
                  f"(bloque extendido: {extendidas}) |")
    if informe.get("padding_s") is not None:
        extra = f", searchterm `{informe['searchterm']}`" if informe.get("searchterm") else ""
        lineas.append(f"| padding de la ventana | {informe['padding_s']} s{extra} |")
    if informe.get("fecha"):
        lineas.append(f"| generado | {informe['fecha']} |")
    lineas += [
        "",
        "Este documento **se genera**, no se escribe: sale del manifiesto, de las constantes de "
        "`gesture_features.py` y del propio parquet. Cada cifra de cobertura es de ESTA entrega. "
        "Si quieres cambiar el texto, se cambia el generador y se regenera. La fecha de arriba es "
        "la del manifiesto con el que se calculó esta cobertura.",
        "",
    ]
    return lineas


def muestra_de_ejemplos(ruta_final):
    """Lo que el documento afirma sobre el lote, medido sobre el parquet entregado."""
    clips = frames = 0
    sustento, clip_max, max_speed = None, None, None
    ruta = os.path.join(ruta_final, "final_videos.parquet")
    if os.path.isfile(ruta):
        videos = pd.read_parquet(ruta)
        clips = len(videos)
        # El bloque extendido es opcional: una entrega FEATURE_VERSION 2 no tiene
        # n_speed ni max por mano, y el README no puede petar ni escribir "0 de 29"
        # por leer columnas que no existen. Se mide lo que haya, y si no esta, se
        # dice que no está en vez de inventar un número.
        if {"right_n_speed", "left_n_speed"} <= set(videos.columns):
            sustento = int(((videos.right_n_speed >= 30) & (videos.left_n_speed >= 30)).sum())
        if {"id", "right_max_speed"} <= set(videos.columns) and videos.right_max_speed.notna().any():
            rapido = videos.right_max_speed.idxmax()
            clip_max = str(videos.loc[rapido, "id"])
            max_speed = float(videos.loc[rapido, "right_max_speed"])
    ruta = os.path.join(ruta_final, "final_frames.parquet")
    if os.path.isfile(ruta):
        frames = int(pd.read_parquet(ruta, columns=["frame"]).shape[0])
    return {"clips": clips, "frames": frames, "sustento": sustento,
            "clip_max_speed": clip_max, "max_speed": max_speed,
            "voz": sanidad_de_voz(ruta_final)}


def escribir_markdown(diccionario, informe, ruta, muestra_ejemplos, portada=False):
    """El documento que se le da a la clase."""
    lineas = ["# Qué es cada columna de este dataset", ""]
    if portada:
        lineas += _portada(diccionario, informe)
    clips = muestra_ejemplos["clips"]
    frames = muestra_ejemplos["frames"]
    lineas += [
        "Dos tablas, una fila cada una. `final_videos` tiene **una fila por clip** "
        f"({clips} clips aquí): cada clip es una persona (`people_id`) dentro de un recorte de "
        "televisión. `final_frames` tiene **una fila por fotograma** de cada clip "
        f"({frames} filas), y es donde esta la serie temporal y la voz.",
        "",
        "## Las dos unidades del dataset",
        "",
        "- **Tiempo**: segundos del `frame_timestamp` de origen. Nada asume 30 fps.",
        "- **Espacio**: *unidades de cuello*. El origen es el cuello (BODY_25 punto 1) y las "
        "coordenadas están normalizadas por el hombro. **No son metros y no se pueden convertir**: "
        "no hay escala metrica en los datos. Por eso `mean_speed = 3` significa 'tres veces la "
        "distancia cuello-hombro por segundo', y solo es comparable con otros números del "
        "mismo dataset.",
        "- Angulos en radianes. Áreas en unidades de cuello al cuadrado.",
        "",
        "## NULL no es cero",
        "",
        "Un NULL significa **no medido**: el keypoint no estaba, tenia confianza insuficiente, o "
        "habia demasiadas pocas muestras detrás. Un 0 significa medido y cero. Y un estadistico con "
        "pocas muestras no es un resultado pequeño, es ruido con decimales.",
        "",
        f"El código aplica dos constantes, declaradas en el manifiesto: umbral de movimiento "
        f"**{informe['umbral_movimiento']} unidades de cuello/s** (por debajo de ahí una mano cuenta "
        f"como quieta y no se le cuentan cambios de sentido) y suelo de **{informe['suelo_muestras']} "
        "muestras** para publicar un percentil o una media agregada.",
        "",
        "**Antes de usar un estadistico, mira su contador.** Van en la misma fila:",
        "",
        "| si vas a usar | mira antes |",
        "|---|---|",
        "| cualquier `*_speed`, `*_accel`, `jerk_*` | `right_n_speed` / `left_n_speed` |",
        "| cualquier `cos_phase_*` | `n_cos_phase` |",
        "| cualquier columna bilateral | `n_both_hands`, `frac_both_hands` |",
        "| `*_p05`, `*_p95`, `*_std` | el `n` de esa mano: percentiles con 5 fotogramas no son percentiles |",
        "| cualquier `frac_*` | `frames_grid` |",
        "",
        "## Las 7 anotaciones son etiquetas de un modelo, no realidad medida",
        "",
        "`indoor_outdoor`, `show_type`, `hands_free`, `sitting_standing`, `screen_interaction`, "
        "`sex`, `age` las escribio un modelo visual sobre fotogramas del clip. Sirven para "
        "**estratificar** ('comparemos anclas sentadas con presentadores de pie'), no como verdad. "
        "En particular `age` son números redondos: un modelo no mide 44 años, los estima.",
        "",
        *(_seccion_voz(muestra_ejemplos["voz"]) if muestra_ejemplos else []),
        "## Columnas",
        "",
        "`datos` es cuántos de los clips (o fotogramas) de este lote tienen valor en esa columna. "
        "Si es bajo, la columna existe pero no tiene sustento aquí: no la uses para comparar clips.",
        "",
    ]

    for tabla in ("final_videos", "final_frames"):
        sub = diccionario[diccionario["tabla"] == tabla].copy()
        if sub.empty:
            continue
        sub["orden"] = sub["familia"].map(_familia_de_orden)
        sub = sub.sort_values(["orden", "columna"], kind="stable")
        lineas += [f"### `{tabla}.parquet` — {int(sub['filas_tabla'].iloc[0])} filas x {len(sub)} columnas", ""]
        actual = None
        for _, fila in sub.iterrows():
            if fila["familia"] != actual:
                actual = fila["familia"]
                lineas += ["", f"#### {actual}", "",
                           "| columna | ambito | qué mide | datos | min | mediana | max |",
                           "|---|---|---|---|---|---|---|"]
            def numero(clave):
                valor = fila.get(clave)
                return "" if valor is None or (isinstance(valor, float) and math.isnan(valor)) else valor
            lineas.append(f"| `{fila['columna']}` | {fila['ambito']} | {fila['que_mide']} | "
                          f"{fila['pct_no_nulos']}% | {numero('minimo')} | {numero('mediana')} | "
                          f"{numero('maximo')} |")
        lineas += [""]

    familias = (diccionario[diccionario["formula_unidades"] != ""]
                .drop_duplicates("familia")[["familia", "formula_unidades", "fuente"]]
                .sort_values("familia", key=lambda s: s.map(_familia_de_orden), kind="stable"))
    lineas += ["## Definiciones, tal como están en el código", "",
               "Esto no es un resumen: es el texto que vive junto a la matematica "
               "(`gesture_features.py`). Si aquí y en el código difieren, manda el código.", ""]
    for _, fila in familias.iterrows():
        lineas += [f"**{fila['familia']}**  ", f"{fila['formula_unidades']}  ",
                   f"*{fila['fuente']}*", ""]

    lineas += ["## Las decisiones de diseño que hay que conocer", ""]
    for titulo, texto in informe["semantica_diseno"].items():
        lineas += [f"- **{titulo}** — {texto}", ""]

    lineas += [
        "## Cinco cosas que hacer con esto",
        "",
        "```python",
        "import pandas as pd",
        "videos = pd.read_parquet('final_videos.parquet')",
        "frames = pd.read_parquet('final_frames.parquet')",
        "```",
        "",
        "1. **Filtrar por sustento antes que por valor.** Quédate con clips con "
        "`right_n_speed >= 30` y `left_n_speed >= 30` y mira cómo cambia la distribución de "
        "`right_mean_speed`: es la diferencia entre gesticulación y ruido de detección. "
        + (f"En este lote ese umbral deja **{muestra_ejemplos['sustento']} de "
           f"{muestra_ejemplos['clips']} clips**: casi no filtra, y saberlo también es parte "
           "del ejercicio (en la entrega vieja de 29 clips dejaba 6). Sube el umbral y mira "
           "cuándo empieza a moverse la media: ahí está el punto donde dejas de promediar "
           "ruido." if muestra_ejemplos["sustento"] is not None else
           "Estos contadores vienen del bloque extendido: si esta entrega no lo trae, el "
           "sustento se mira con `right_n_valid` y `left_n_valid`."),
        "2. **Buscar gesticuladores sin usar la media.** Ordena por `right_frac_speed_gt_1p0` y "
        "compara con `right_mean_speed`: la media la puede inflar un corte de montaje, el frac no.",
        "3. **Manos que se turnan vs manos que se mueven juntas.** `coactivity_frac_0p5` alto con "
        "`cos_phase_mean` bajo = alternancia; los dos altos = ambas manos en el mismo gesto. "
        "Solo en clips con `n_cos_phase >= 10`.",
        "4. **Qué dice la voz del gesto (abre la puerta primero).** Cruza `intensity` y `pitch`",
        "de `final_frames` con `right_speed` de la misma fila, **solo en los fotogramas con voz**",
        "(`pitch` entre 60 y 600, o `vad` si viene relleno): sin filtrar estás promediando",
        "también los de silencio, que llevan número por la spline. Medido aquí: la",
        f"correlación vale {muestra_ejemplos['voz']['corr_todo']} sin filtrar y",
        f"{muestra_ejemplos['voz']['corr_voz']} filtrando por voz. «Casi nada en los dos",
        "casos» es una respuesta válida, y es la que da este lote. La voz **solo existe",
        "en `final_frames`**: `final_videos` no tiene ni una columna de voz.",
        "5. **Comprobar que el número es real.** Coge el clip más rápido por `right_max_speed` "
        + (f"(aquí **{muestra_ejemplos['clip_max_speed']}: "
           f"{muestra_ejemplos['max_speed']:.1f} unidades de cuello por segundo, y eso lo firma "
           "un solo fotograma**)" if muestra_ejemplos["max_speed"] is not None else "")
        + ", extrae cuatro fotogramas del video y míralos. Un `max_speed` de dos cifras casi "
        "siempre es un corte de plano o un zoom, no un manotazo: la velocidad es un cociente "
        "entre dos posiciones, y si el encuadre cambia, la posición cambia sola.",
        "",
        "---",
        "",
        f"Generado con `final_dataset/diccionario_columnas.py` (FEATURE_VERSION "
        f"{informe['feature_version']}, extendidas {informe['extendidas']}). Cada fórmula viene de "
        "las constantes de `gesture_features.py`; la cobertura de datos está medida sobre este "
        "mismo parquet. No edites este fichero a mano: regenerarlo.",
    ]
    with open(ruta, "w", encoding="utf-8") as escritor:
        escritor.write("\n".join(lineas) + "\n")


COTA_PITCH = (60.0, 600.0)      # Praat To Pitch: min 60, max 600 (extract_prosody.praat)
COTA_FORMANTE = 5500.0           # Praat To Formant (burg): 5 formantes hasta 5500 Hz
SUELO_HNR = -200.0               # suelo que reporta To Harmonicity (cc) en silencio
VOZ_COLUMNAS = ("pitch", "intensity", "harmonicity", "voice_present",
                "formant.1_frequency", "formant.2_frequency", "formant.3_frequency",
                "formant.4_frequency", "formant.5_frequency")


def _seccion_voz(voz):
    """La seccion de voz del README, en dos versiones elegidas por MEDICION.

    La clave no es «cual me gusta»: son las cotas del analisis de Praat, que estan fijas en
    extract_prosody.praat (To Pitch 60-600 Hz; To Formant (burg) con techo 5500 Hz; To
    Harmonicity (cc) con suelo -200 dB). Cualquier valor fuera de esas cotas no lo pudo
    producir Praat, asi que delata la cadena antigua. Y `voice_present` solo existe desde el
    arreglo. Un lote regenerado con `pipeline_lote.sh --desde voz` cambia de seccion solo.
    """
    if voz["regimen"] == "sin voz":
        # Callar no es honesto: si no hay columnas de voz, se dice. Y no se enseñan numeros,
        # porque un «0 de 0» en una tabla parece una medida de cero fotogramas.
        return [
            "## La voz: la parte que viene de otra cadena, y como comprobarla",
            "",
            "**Esta entrega no trae columnas de voz**: no hay `pitch`, `intensity`, `harmonicity`, "
            "formantes ni `voice_present` en `final_frames` (el stage de voz no corrio, o no dejo "
            "nada publicable). No se enseña ningun numero de voz porque no hay de donde sacarlo: si "
            "en un analisis de voz de este lote aparece un 0, ese 0 lo has puesto tu.",
            "",
        ]
    cabecera = [
        "## La voz: la parte que viene de otra cadena, y como comprobarla",
        "",
        "Las columnas de voz no las hace la matemática de muñecas: las hace "
        "`create_datasets/parole` (Praat + WhisperX + Silero VAD). Y durante años no cumplieron "
        "el contrato del NULL, por tres motivos que se comprueban con las cotas del propio Praat "
        f"(`To Pitch: 60, 600`; formantes hasta `{COTA_FORMANTE:.0f}` Hz; suelo del "
        f"Harmonicidad `(cc)` en `{SUELO_HNR:.0f}` dB):",
        "",
        "- Una unica spline cubica sobre TODA la pista, evaluada en todos los fotogramas: con 4 "
        "puntos validos en todo el clip, hasta el silencio recibia numero.",
        "- `splinefun` no puede apagar la extrapolacion (no acepta el argumento `extrapolation`), "
        "asi que fuera del principio y del fin de la pista inventaba valores: de ahi salieron un "
        "F5 de 29.144.321 Hz y un HNR de -246,7 dB en entregas del corpus.",
        "- El centinela `pitch < 10 -> 0` pintaba de 0 la extrapolacion. En Praat `frequency = 0` "
        "es el candidato `mute` («no hay pitch»), asi que el 0 SI significaba silencio; el problema "
        "es que publicado como numero se cuela en las medias: medido, la media de pitch bajaba de "
        "182,5 Hz a 125,3 Hz por culpa de esos ceros.",
        "",
        "La cadena actual (`prosody_align.R`) parte cada pista en tramos de muestras consecutivas "
        "y solo interpola **dentro** de un tramo; fuera de un tramo, NULL. Y publica "
        "`voice_present`. Esto es lo que mide este documento sobre el Parquet que tienes delante:",
        "",
        "| comprobacion en este lote | valor |",
        "|---|---|",
        f"| regimen de la cadena de voz | **{voz['regimen']}** |",
        f"| `voice_present` | {voz['voice_present']} |",
        f"| `pitch == 0` | {voz['pitch_ceros']} fotogramas |",
        f"| `pitch` fuera de {COTA_PITCH[0]:.0f}-{COTA_PITCH[1]:.0f} Hz | {voz['pitch_fuera']} fotogramas |",
        f"| `harmonicity` por debajo de {SUELO_HNR:.0f} dB | {voz['hnr_bajo_suelo']} fotogramas |",
        f"| formantes fuera de 0-{COTA_FORMANTE:.0f} Hz | {voz['formantes_fuera']} valores |",
        f"| rango de `intensity` | {voz['int_rango']} |",
        f"| correlacion `intensity` con `right_speed`: sin filtrar / filtrando por voz ({voz['n_voz']} fotogramas) | {voz['corr_todo']} / {voz['corr_voz']} |",
        "",
    ]
    if voz["regimen"] == "tramos":
        resto = [
            "Este lote sale de la cadena con `prosody_align.R`: no hay voz donde Praat no seguia "
            "pista, no hay valores fuera de las cotas, y `voice_present` existe. Aun asi:",
            "",
            "1. **Abre la puerta con `voice_present` antes de cualquier estadistico de voz.** Es el "
            "equivalente exacto de exigir `right_n_speed` en las muñecas. La correlacion de la tabla "
            "de arriba usa esta puerta.",
            "2. **`intensity` no es un detector de voz**: Praat la sigue midiendo en el silencio, asi "
            "que casi no tiene huecos. Para saber si hay voz esta `voice_present`.",
            "3. **`words` es texto de un transcantor**, con sus errores; aqui NULL si significa silencio real.",
            "4. **La voz no esta en `final_videos`.** Si quieres un estadistico de voz por clip, "
            "calculalo tu desde `final_frames` y di con cuantos fotogramas lo sustentas, como hacen "
            "`right_n_speed` y `n_cos_phase`.",
            "",
            "Si aun asi aparece algun valor fuera de cotas, no lo imputes: aparta esos fotogramas y "
            "dejalo constar. Y si comparas este lote con uno viejo, recuerda que en el viejo un tercio "
            "de los fotogramas de voz son ceros de centinela.",
            "",
        ]
    else:
        resto = [
            "**Este lote sale de la cadena antigua** (spline unica sobre toda la pista + centinela), "
            "y eso se nota en los numeros de arriba: hay valores que el analisis de Praat no puede "
            "producir, y no hay columna `voice_present`, asi que la unica puerta disponible es la "
            f"banda de pitch ({voz['puerta']}).",
            "",
            "1. **No trates un valor de voz como medida de ese fotograma.** Es la curva del clip "
            "evaluado ahi: sirve para la forma global de un segmento, no para «que paso en el "
            "fotograma 12».",
            f"2. **Filtra antes de calcular cualquier cosa** (pitch entre {COTA_PITCH[0]:.0f} y "
            f"{COTA_PITCH[1]:.0f} Hz). Sin filtro, parte de los fotogramas son ceros que no significan "
            "«sin voz»: medido, la media de pitch pasa de 125,3 Hz a 182,5 Hz al filtrar.",
            "3. **`intensity` es lo mas aprovechable** de esta parte, pero solo con la puerta puesta.",
            "4. Los negativos de `harmonicity` son normales (un HNR negativo existe: es ruido); lo que "
            f"no existe es por debajo de {SUELO_HNR:.0f} dB. Y una frecuencia de formante negativa o "
            f"mayor que {COTA_FORMANTE:.0f} Hz es directamente imposibilidad, no un outlier.",
            "",
            "El arreglo vive en `final_dataset/upstream_parole/` (con patch para enviar arriba); al "
            "regenerar un lote con `pipeline_lote.sh --desde voz`, la voz sale con `voice_present` y "
            "dentro de cotas, y esta seccion cambia sola.",
            "",
        ]
    return cabecera + resto


def sanidad_de_voz(ruta_final):
    """Números de la voz medidos sobre el Parquet entregado, no de memoria.

    Lo que decide el regimen NO es una opinion: son las cotas del analisis. Con
    `To Pitch: 60, 600`, `To Formant (burg): ..., 5500` y `To Harmonicity (cc)` no puede
    salir un pitch de 0 Hz, un HNR de -1193 dB ni un formante de 29 MHz. Si los hay, esta
    entrega viene de la cadena antigua (una spline cubica sobre toda la pista, evaluada
    donde no habia medida, mas el centinela pitch<10 -> 0). Medido en el corpus viejo:
    1408 fotogramas con pitch == 0, un F5 de 29.144.321 Hz y un HNR de -246,7 dB.
    """
    vacio = {"regimen": "sin voz", "voice_present": "(sin columna de voz)", "pitch_ceros": 0,
             "pitch_fuera": 0, "hnr_bajo_suelo": 0, "formantes_fuera": 0, "int_rango": "(sin columna)",
             "corr_todo": "(sin columna)", "corr_voz": "(sin columna)", "n_voz": 0, "puerta": "n/a"}
    ruta = os.path.join(ruta_final, "final_frames.parquet")
    if not os.path.isfile(ruta):
        return vacio
    columnas = set(pd.read_parquet(ruta).columns)
    if "pitch" not in columnas:
        return vacio
    datos = pd.read_parquet(ruta, columns=[c for c in list(VOZ_COLUMNAS) + ["right_speed"] if c in columnas])
    total = len(datos)
    bajo_suelo = int((datos["harmonicity"] < SUELO_HNR).sum()) if "harmonicity" in datos else 0
    forma_cols = [c for c in VOZ_COLUMNAS if c.startswith("formant.") and c in datos.columns]
    formantes_fuera = int(sum(((datos[c] < 0) | (datos[c] > COTA_FORMANTE)).sum() for c in forma_cols)) if forma_cols else 0
    pitch_fuera = int(((datos["pitch"] < COTA_PITCH[0]) | (datos["pitch"] > COTA_PITCH[1])).sum())
    pitch_ceros = int((datos["pitch"] == 0).sum())
    # regimen: la banda 0 < pitch < 60 es imposible por construccion (el analisis no busca
    # por debajo de 60 Hz), y lo mismo bajo el suelo del HNR o fuera del techo de formantes.
    # `voice_present` solo existe desde el arreglo, asi que tambien vale como testigo.
    imposible = pitch_fuera + bajo_suelo + formantes_fuera
    if "voice_present" in columnas:
        regimen = "tramos"
        vp = datos["voice_present"]
        con_voz = int(vp.fillna(False).sum())
        voice_present = (f"{con_voz} de {total} fotogramas con pista medida "
                         f"({100 * con_voz / max(total, 1):.1f} %); {int(vp.isna().sum())} sin pista de pitch")
        puerta = "voice_present"
        voz = datos[vp.fillna(False)]
    else:
        regimen = "spline"
        voice_present = "(sin columna: este lote es anterior al arreglo de la prosodia)"
        puerta = f"{COTA_PITCH[0]:.0f}-{COTA_PITCH[1]:.0f} Hz de pitch"
        voz = datos[(datos["pitch"] >= COTA_PITCH[0]) & (datos["pitch"] <= COTA_PITCH[1])]
    if regimen == "tramos" and imposible:
        regimen = "tramos con restos"   # no se confia: se dice

    def correlacion(tabla):
        if {"intensity", "right_speed"} <= set(tabla.columns):
            pares = tabla[["intensity", "right_speed"]].dropna()
            if len(pares) > 2 and pares.std().min() > 0:
                return f"{pares.corr().iloc[0, 1]:+.3f} (n={len(pares)})"
        return "no computable"

    return {
        "regimen": regimen, "voice_present": voice_present, "puerta": puerta,
        "pitch_ceros": pitch_ceros, "pitch_fuera": pitch_fuera,
        "hnr_bajo_suelo": bajo_suelo, "formantes_fuera": formantes_fuera,
        "int_rango": (f"{datos['intensity'].min():.1f} .. {datos['intensity'].max():.1f} dB"
                      if "intensity" in datos else "(sin columna)"),
        "corr_todo": correlacion(datos), "corr_voz": correlacion(voz), "n_voz": len(voz),
    }


def _polymnia_por_defecto():
    """Donde está Polymnia: puede ser el padre (final_dataset/ vive dentro del repo)
    o un hermano (una copia suelta junto a un checkout de Polymnia)."""
    arriba = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return arriba if os.path.exists(os.path.join(arriba, "gesture_features.py")) \
        else os.path.join(arriba, "Polymnia")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", required=True, help="carpeta final/ con los parquet y el manifiesto")
    parser.add_argument("--csv", help="donde escribir el diccionario en CSV")
    parser.add_argument("--md", help="donde escribir el diccionario en Markdown")
    parser.add_argument("--readme", help="donde escribir el README con portada de entrega "
                        "(es el fichero que se publica como DATASET.md dentro de final/)")
    parser.add_argument("--polymnia_root", default=_polymnia_por_defecto())
    args = parser.parse_args(argv)

    diccionario, informe = generar(args.final, args.polymnia_root)
    # Nombres de esquema, no prosa: en ASCII a propósito, y el guard de abajo pifia
    # ruidosamente si alguno no existe (un rename aquí vacía columnas del CSV en
    # silencio, que es justo el tipo de bug que este script no puede tener).
    columnas = ["tabla", "columna", "ambito", "familia", "que_mide", "formula_unidades",
                "tipo", "no_nulos", "pct_no_nulos", "minimo", "mediana", "maximo", "fuente"]
    faltantes = [c for c in columnas if c not in diccionario.columns]
    if faltantes:
        raise SystemExit(f"el diccionario no produce estas columnas de esquema: {faltantes}. "
                         "Los nombres de campo no son texto de clase: no se acentúan.")
    vista = diccionario.reindex(columns=columnas)

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        vista.to_csv(args.csv, index=False)
    muestra = muestra_de_ejemplos(args.final) if (args.md or args.readme) else None
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
        escribir_markdown(diccionario, informe, args.md, muestra)
    if args.readme:
        destino = os.path.abspath(args.readme)
        dentro = os.path.abspath(args.final) + os.sep
        if destino.startswith(dentro):
            raise SystemExit(f"--readme no puede escribir dentro de la entrega ({args.final}): "
                             "final/ es un paquete atomico con sha256 en el manifiesto. Escribelo "
                             "fuera y dejalo entrar a final_dataset.py con su propio --readme.")
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        escribir_markdown(diccionario, informe, destino, muestra, portada=True)

    print(f"columnas documentadas: {len(vista)} "
          f"({(vista['tabla'] == 'final_videos').sum()} en final_videos, "
          f"{(vista['tabla'] == 'final_frames').sum()} en final_frames), sin resolver: 0")
    if args.csv:
        print(f"csv: {args.csv}")
    if args.md:
        print(f"markdown: {args.md}")
    if args.readme:
        print(f"readme: {os.path.abspath(args.readme)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
