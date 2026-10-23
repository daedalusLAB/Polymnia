#!/usr/bin/env python3
"""Dataset FINAL: tab de NewsScape + DuckDB procesado -> 2 tablas (parquet + CSV).

    python3 final_dataset.py --db_name ../entrega_1p/1p.duckdb \
        --tab clips.txt --output_folder ../entrega_1p_final --offset 2.0

Con `--extended` la tabla de vídeos añade el bloque de FEATURES_VERSION 3 (jerk,
agregados de distribución, forma, dirección y bimanuales). Es opt-in: sin la
bandera la salida es byte a byte la de la versión 2.

Dos tablas, cada una con una columna `id` que apunta al clip:

* **`final_frames`** - una fila por (id, people_id, frame): las `nx` / `ny` / `c`
  de las muñecas BODY_25 4 y 7 y del cuello 1, las features cinemáticas de
  `gesture_features.compute_gesture_features` y la metadata por fotograma
  (timestamp, prosodia, VAD, palabra activa, procedencia del JSON de OpenPose).
* **`final_videos`** - una fila por (id, people_id): las 7 anotaciones visuales
  de Theia, los agregados del clip completo y los campos del tab (identificador
  `t__...`, tiempos crudos, texto de contexto).

Cada tabla se escribe dos veces: `.parquet` (tipos y NULL fieles) y `.csv`
(NULL = campo vacío) para Excel/R. Se publica además `manifest.json` con el
esquema, las unidades, la semántica, los SHA-256 de entradas y salidas, y qué
filas del tab casaron con qué clips procesados.

Garantías (las mismas de `Polymnia/gestures.py`, que es quien hace la matemática):

* la base fuente se abre `read_only=True`; no se escribe, ni se altera, ni se
  reimporta nada;
* `--output_folder` no puede existir: se escribe en un directorio temporal
  contiguo y se publica con un único rename `RENAME_NOREPLACE`; un fallo no
  publica nada parcial;
* lo único nuevo aquí es el mapeo línea del tab -> `id` de clip y el ensamblado
  de tablas: ninguna feature se recalcula a mano.

El tab es el formato de `download_clips.py`: sin cabecera, separado por
tabuladores, `1=programa.txt  2=id NewsScape  3/4=start segundos/milis  5/6=end
segundos/milis  7..=aquí se usa el 7 como contexto`. El divisor fraccionario es
1000 si el nombre normalizado empieza por `2017` y 100 en caso contrario: es la
regla literal del descargador, no una garantía general para otros años.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import pandas as pd

def _polymnia_por_defecto():
    """Donde está Polymnia: puede ser el padre (final_dataset/ vive dentro del repo)
    o un hermano (una copia suelta junto a un checkout de Polymnia)."""
    arriba = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return arriba if os.path.exists(os.path.join(arriba, "gesture_features.py")) \
        else os.path.join(arriba, "Polymnia")

POLYMNIA_DEFAULT = _polymnia_por_defecto()
FRAMES_STEM, VIDEOS_STEM, MANIFEST = "final_frames", "final_videos", "manifest.json"
README_NAME = "DATASET.md"
STAGING_PREFIX = ".final-dataset-staging-"

# Constantes dentro de un clip. Se comprueba en la fuente: 1 valor distinto por id.
# `datetime` es TIMESTAMPTZ y el entorno no tiene pytz, así que sale como texto UTC.
CLIP_META = ("datetime", "country_code", "network_code", "exp_search")
# Repetidas en las ~137 filas de keypoint de cada fotograma: se agrupa antes de leer.
FRAME_PROSODY = ("pitch", "intensity", "harmonicity",
                 "formant.1_frequency", "formant.2_frequency", "formant.3_frequency",
                 "formant.4_frequency", "formant.5_frequency")
# Bandera de procedencia de la voz: TRUE solo donde Praat siguio una pista de pitch de
# verdad (lo produce Polymnia/create_datasets/parole/scripts/prosody_align.R). Es el
# companero de `right_n_speed` en la parte de voz: sin ella, NULL, silencio y valor
# interpolado vestian igual. Se publica SOLO si la fuente la trae: las entregas anteriores
# no la tienen y su `final_frames` tiene que seguir saliendo byte a byte igual.
FRAME_VOICE_FLAG = "voice_present"
# En esta entrega estos dos campos no traen metadata del programa: llevan el nombre del
# JSON de OpenPose de ESE fotograma y son byte a byte iguales entre si. Se conserva el
# primero como `openpose_json` por frame y se descarta el duplicado (consta en manifest).
FRAME_PROVENANCE = ("program_name", "time_range")
ANNOTATIONS = ("indoor_outdoor", "show_type", "hands_free", "sitting_standing",
               "screen_interaction", "sex", "age")


class FinalDatasetError(RuntimeError):
    """No se puede construir el dataset final (entrada ausente, ambigua o no publicable)."""


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _quote(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


# --------------------------------------------------------------- tab NewsScape ----

def fractional_divisor(program):
    """Regla literal de download_clips.py: 1000 para nombres que empiezan por 2017."""
    return 1000 if os.path.basename(program).startswith("2017") else 100


def clip_window(fields, offset):
    """(start, end) en segundos del clip, con `offset` de holgura a cada lado."""
    divisor = fractional_divisor(fields[0])
    start = round(int(fields[2]) + int(fields[3]) / divisor - float(offset), 3)
    end = round(int(fields[4]) + int(fields[5]) / divisor + float(offset), 3)
    return start, end


def clip_id(fields, offset, searchterm):
    """El `id` con el que el pipeline conoce a este clip, tal como lo forma download_clips.py."""
    program = os.path.basename(fields[0]).replace(".txt", "")
    start, end = clip_window(fields, offset)
    return f"{program}_{start}_{end}_{searchterm}"


def read_tab(path, offset, searchterm):
    """Una fila por línea del tab, con su `id` candidato y su contexto."""
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                raise FinalDatasetError(f"{path}:{number}: hacen falta al menos 6 campos separados "
                                        f"por tabuladores, hay {len(fields)}")
            try:
                start, end = clip_window(fields, offset)
            except ValueError as exc:
                raise FinalDatasetError(f"{path}:{number}: los campos 3-6 tienen que ser enteros "
                                        f"({exc})") from None
            if start >= end:
                raise FinalDatasetError(f"{path}:{number}: intervalo invertido o vacío "
                                        f"(start {start} >= end {end})")
            divisor = fractional_divisor(fields[0])
            rows.append({"id": clip_id(fields, offset, searchterm),
                         "newscape_clip_id": fields[1],
                         "program": os.path.basename(fields[0]).replace(".txt", ""),
                         "tab_start_raw": round(int(fields[2]) + int(fields[3]) / divisor, 3),
                         "tab_end_raw": round(int(fields[4]) + int(fields[5]) / divisor, 3),
                         "tab_padding_s": float(offset),
                         "context": fields[6].strip() if len(fields) > 6 else None})
    if not rows:
        raise FinalDatasetError(f"{path}: no tiene ninguna línea con datos")
    counts = pd.DataFrame(rows)["id"].value_counts()
    doubled = counts[counts > 1]
    if not doubled.empty:
        raise FinalDatasetError(f"{path}: dos líneas producen el mismo id de clip "
                                f"(e.g. {doubled.index[0]}); revisa los tiempos del tab")
    return pd.DataFrame(rows)


def parse_clip_id(clip, searchterm):
    """`programa_start_end_searchterm` -> (programa, start, end) o None si no es de esta forma.

    El centro de la ventana, (start + end) / 2, no depende de la holgura: anadir `pad` a
    un lado y restarlo al otro lo deja igual. Por eso se puede emparejar una linea del
    tab pensada con 2.0 s con un clip que se procesó con 2.5 s sin adivinar nada.
    """
    suffix = f"_{searchterm}"
    if not clip.endswith(suffix):
        return None
    head = clip[: -len(suffix)]
    fields = head.rsplit("_", 2)
    if len(fields) != 3:
        return None
    program, start, end = fields
    try:
        start, end = float(start), float(end)
    except ValueError:
        return None
    return program, start, end


def match_tab_to_clips(tab, clip_ids, searchterm):
    """Qué filas del tab corresponden a qué clips ya procesados.

    Primero el `id` exacto con la holgura pedida. Si no está, se empareja por programa y
    por el centro de la ventana, que es invariante a la holgura: así un tab pensado con
    2.0 s localiza un clip procesado con otra holgura y se reporta la holgura real de ese
    clip (`processed_padding_s`) en lugar de asignarlo a ciegas. Un centro que no es
    único, o un clip cuyo nombre no se puede parsear, queda como no emparejado.
    """
    available = {}
    for raw in clip_ids:
        clip = str(raw)
        parsed = parse_clip_id(clip, searchterm)
        if parsed is None:
            continue
        program, start, end = parsed
        available.setdefault((program, round((start + end) / 2, 6)), []).append((clip, start, end))
    exact = {str(c) for c in clip_ids}
    matched, unmatched = [], []
    for row in tab.itertuples(index=False):
        record = {"id": row.id, "newscape_clip_id": row.newscape_clip_id, "tab_context": row.context,
                  "tab_start_raw": row.tab_start_raw, "tab_end_raw": row.tab_end_raw,
                  "tab_padding_s": row.tab_padding_s}
        if row.id in exact:
            matched.append(record | {"id_source": "exact", "processed_padding_s": row.tab_padding_s})
            continue
        center = round((row.tab_start_raw + row.tab_end_raw) / 2, 6)
        candidates = available.get((row.program, center), [])
        if len(candidates) == 1:
            clip, start, end = candidates[0]
            padding = round(((end - start) - (row.tab_end_raw - row.tab_start_raw)) / 2, 3)
            record["id"] = clip
            matched.append(record | {"id_source": "window_center", "processed_padding_s": padding})
        else:
            unmatched.append({"id": row.id, "newscape_clip_id": row.newscape_clip_id,
                              "program": row.program,
                              "reason": ("no hay ningun clip procesado de este programa y esta ventana"
                                         if not candidates else
                                         f"{len(candidates)} clips procesados comparten programa y ventana")})
    return pd.DataFrame(matched), pd.DataFrame(unmatched)


def guarda_holgura(matched, offset, permite=False):
    """Se niega a publicar si los clips procesados estan cortados con otra holgura.

    `processed_padding_s` esta medida de verdad solo en los emparejamientos por centro de
    ventana: los de id exacto llevan, por construccion, la holgura pedida (el id se forma
    restando y sumando esa holgura), asi que ahi comparar no daria nunca discrepancia. Que
    aparezca un valor distinto significa que el tab de esta pasada no es el que se descargo:
    los fotogramas existen, pero desplazados `diferencia` segundos respecto a lo que se pide,
    y publicar la mezcla es peor que parar. La duracion del clip lo comprueba aparte, con
    ffprobe, en tools/verificar_holgura.py.
    """
    medidas = sorted({round(float(p), 3) for p in matched["processed_padding_s"]})
    torcidas = [p for p in medidas if abs(p - float(offset)) > 0.001]
    if not torcidas:
        return
    if permite:
        print(f"aviso: {torcidas} s por lado no es la holgura pedida ({offset} s). "
              f"Publicas con --allow-padding-mismatch, a tu cuenta.")
        return
    raise FinalDatasetError(
        f"los clips procesados estan cortados a {torcidas} s por lado y tu pides {offset} s. "
        "Encajan con el tab por centro de ventana, no por id exacto, asi que los fotogramas "
        "estan desplazados respecto a lo que pides. Reconstruye con "
        f"--offset {torcidas[0]:g} (no vuelve a bajar nada) o vuelve a descargar con la "
        "holgura que quieras. --allow-padding-mismatch lo publica igualmente, a tu cuenta.")


# -------------------------------------------------------------------- fuente ----

def open_read_only(db_path, table):
    import duckdb

    if not os.path.isfile(db_path):
        raise FinalDatasetError(f"base de datos no encontrada: {db_path}")
    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception as exc:
        raise FinalDatasetError(f"no se puede abrir {db_path} en solo lectura: {exc}") from None
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if table not in tables:
        con.close()
        raise FinalDatasetError(f"la tabla {table!r} no está en {db_path} "
                                f"(hay: {', '.join(tables) or 'ninguna'})")
    return con


def load_frame_metadata(con, table):
    """Una fila por (id, people_id, frame) con prosodia, VAD, palabra y procedencia.

    `multi_data` desnormaliza: esos valores vienen repetidos en cada fila de keypoint
    del fotograma, así que se agrupa primero y solo se usan agregados que no cambian
    nada si la repetición es fiel (`max`, `bool_or`, `any_value`).
    """
    descritor = {r[0]: str(r[1]).upper()
                 for r in con.execute(f"DESCRIBE {_quote(table)}").fetchall()}
    columns = set(descritor)

    def agg(column, fn="max"):
        """Agrega una columna que puede no existir, o tener otro tipo segun la base.

        `vad` es BOOLEAN en 1p.duckdb pero INTEGER en las bases que saca el pipeline de
        hoy, y `bool_or` solo acepta BOOLEAN (da Binder Error). TRY_CAST deja NULL en vez
        de inventarse un 0 donde no hay medida.
        """
        if column not in columns:
            return f"NULL AS {_quote(column)}"
        expr = _quote(column)
        if fn in ("bool_or", "bool_and") and descritor[column] != "BOOLEAN":
            expr = f"TRY_CAST({expr} AS BOOLEAN)"
        return f"{fn}({expr}) AS {_quote(column)}"

    prosody = [c for c in FRAME_PROSODY if c in columns]
    meta = [c for c in CLIP_META if c in columns]
    provenance = [c for c in FRAME_PROVENANCE if c in columns]
    parts = ["id", "CAST(people_id AS BIGINT) AS people_id", "frame",
             "max(frame_timestamp) AS frame_timestamp",
             "bool_and(frame_timestamp IS NULL) AS clock_absent",
             agg("vad", "bool_or"), agg("words", "max")]
    parts += [f"max({_quote(c)}) AS {_quote(c)}" for c in prosody]
    # La bandera de voz sale con bool_or, igual que `vad`: en la fuente larga el valor se
    # repite en las ~137 filas de keypoint del fotograma, y si todo es NULL debe quedar
    # NULL (no habia pista), no un FALSE que afirma silencio.
    if FRAME_VOICE_FLAG in columns:
        parts.append(agg(FRAME_VOICE_FLAG, "bool_or"))
        prosody = prosody + [FRAME_VOICE_FLAG]
    parts.append(f"any_value({_quote(provenance[0])}) AS openpose_json" if provenance
                 else "NULL AS openpose_json")
    clip_meta = []
    if "datetime" in meta:
        parts.append("any_value(CAST(datetime AS VARCHAR)) AS datetime_utc")
        clip_meta.append("datetime_utc")
        meta = [c for c in meta if c != "datetime"]
    parts += [f"any_value({_quote(c)}) AS {_quote(c)}" for c in meta]
    clip_meta += list(meta)
    frame = con.execute(f"SELECT {', '.join(parts)} FROM {_quote(table)} GROUP BY 1, 2, 3").df()

    conflicts = con.execute(f"SELECT count(*) FROM (SELECT id, frame FROM {_quote(table)} "
                            "GROUP BY 1, 2 HAVING count(DISTINCT frame_timestamp) > 1)").fetchone()[0]
    if conflicts:
        raise FinalDatasetError(f"{conflicts} fotogramas tienen mas de un frame_timestamp distinto: "
                                "la fuente es ambigua, no se publica")
    for column in CLIP_META:
        if column not in columns:
            continue
        varying = con.execute(f"SELECT count(*) FROM (SELECT id FROM {_quote(table)} GROUP BY 1 "
                              f"HAVING count(DISTINCT {_quote(column)}) > 1)").fetchone()[0]
        if varying:
            raise FinalDatasetError(f"la metadata de clip {column!r} cambia dentro del mismo clip en "
                                    f"{varying} clips: no se puede publicar a nivel de video")
    note = None
    if len(provenance) == 2:
        differing = con.execute(f"SELECT count(*) FROM {_quote(table)} WHERE "
                                f"{_quote(provenance[0])} IS DISTINCT FROM {_quote(provenance[1])}"
                                ).fetchone()[0]
        note = (f"{provenance[0]} y {provenance[1]} son el nombre del JSON de OpenPose del fotograma, "
                f"no metadata del programa; son identicos en {differing and 'todas'} las filas "
                f"(difieren en {differing}); se conserva {provenance[0]} como openpose_json por frame")
    return frame, prosody, clip_meta, note


def load_annotations(con):
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "video_annotations" not in tables:
        return pd.DataFrame(columns=["id"]), [], "no hay tabla video_annotations: sin anotaciones"
    columns = {r[0] for r in con.execute("DESCRIBE video_annotations").fetchall()}
    fields = [c for c in ANNOTATIONS + ("status", "error") if c in columns]
    frame = con.execute(f"SELECT id, {', '.join(_quote(f) for f in fields)} FROM video_annotations").df()
    missing = [c for c in ANNOTATIONS if c not in fields]
    note = (f"faltan anotaciones en video_annotations: {', '.join(missing)}" if missing else None)
    return frame, fields, note


# -------------------------------------------------------------------- ensamblado ----

ADMISSION_RULES = (
    "1. una columna solo existe si se puede medir en los frames validos del propio clip; si no, NULL"
    " (los unicos 0 honestos son contadores: n_speed, n_cos_phase)",
    "2. no entra ninguna columna que sea funcion determinista de otra publicada: por eso no hay "
    "*_iqr (p75-p25) ni path_ratio (displacement/path)",
    "3. no entra nada cuyo valor cambie al cambiar la lista de clips: eso descarto los 79 scores "
    "min-max/z-score del codigo de referencia, que normalizaban contra el conjunto",
    "4. toda columna lleva formula, unidades y referencia en el manifiesto, y una prueba con "
    "trayectoria conocida en Polymnia/tools/test_gesture_features.py",
    "5. un ratio intra-clip solo entra como cociente entre dos estadisticos del propio clip:"
    " normalizar un estadistico por si mismo del propio clip da una constante (max/max = 1,"
    " fraction sobre su propia mediana = 0.5)",
    "6. no hay bins espaciales: la distancia al cuello de una mano elevada mide entre 1,91 (p10) y"
    " 2,77 (max), asi que un anillo absoluto satura y un cuartil intra-clip es uniforme por"
    " construccion; se midio y se descarto, no se omitio",
    "7. el unico umbral absoluto del modulo es MOVEMENT_SPEED = 0,5 unidades de cuello/s (~p55 de la"
    " distribucion observada, mediana 0,38); sin movimiento por encima de el, las features de"
    " direccion y forma son NULL porque el polvo de coma flotante de np.gradient (~1e-15) no es un"
    " angulo ni una forma",
)


def build_provenance(path, polymnia_root):
    """What produced these numbers, as far as that can be known afterwards (#14).

    `path` is the JSON that `pipeline_lote.sh` writes in stage 0 with
    `tools/tool_provenance.py`. Embedding that file, instead of re-probing here, is the
    whole point: a 5000-clip lot runs for hours and a delivery built at the end would
    otherwise describe whatever is installed at report time. `delivery_code` is measured
    now, at build time, because the wrist math lives in this repo and a silently patched
    tree is not the commit it points at.

    Nothing here fails the delivery. A missing or unreadable file is recorded as a reason,
    so the manifest says "we do not know" instead of looking complete.
    """
    block = {"captured_from": os.path.abspath(path) if path else None,
             "captured_at": None, "tools": None, "delivery_code": None, "reason": None}
    if not path:
        block["reason"] = ("no --provenance file was given: this delivery records no tool versions. "
                           "pipeline_lote.sh writes informes/procedencia.json in stage 0")
        return block
    if not os.path.isfile(path):
        block["reason"] = f"--provenance file not found: {path}"
        return block
    try:
        with open(path, encoding="utf-8") as handle:
            tools = json.load(handle)
    except (OSError, ValueError) as exc:
        block["reason"] = f"--provenance file could not be read ({type(exc).__name__}): {exc}"
        return block
    if not isinstance(tools, dict):
        block["reason"] = f"--provenance file is not a JSON object: {path}"
        return block
    block["tools"] = tools
    block["captured_at"] = tools.get("recorded_at")
    return block


def delivery_code(polymnia_root):
    """Commit and dirty flag of the repo holding the wrist maths, measured at build time.

    The probe is loaded from the given repo by path, not imported by name: a caller that
    points at another checkout must get that checkout's identity, not whatever copy happens
    to be in sys.modules already.
    """
    import importlib.util

    probe = os.path.join(polymnia_root, "tools", "tool_provenance.py")
    if not os.path.isfile(probe):
        return {"polymnia": {"commit": None, "dirty": None,
                             "reason": f"tools/tool_provenance.py unavailable under {polymnia_root}"}}
    try:
        spec = importlib.util.spec_from_file_location("polymnia_tool_provenance", probe)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - provenance never breaks a delivery
        return {"polymnia": {"commit": None, "dirty": None,
                             "reason": f"tools/tool_provenance.py unavailable: {type(exc).__name__}: {exc}"}}
    return {"polymnia": module.git_identity(polymnia_root)}


def build(db_path, tab_path, output_folder, offset, searchterm, polymnia_root,
          table="multi_data", min_keypoint_confidence=0.2, max_gap_frames=3,
          window_size=7, poly_order=2, extended=False, readme=None,
          allow_padding_mismatch=False, provenance_path=None):
    """Construye las dos tablas y las publica de forma atómica en `output_folder`."""
    if os.path.exists(output_folder):
        raise FinalDatasetError(f"la carpeta de salida ya existe: {output_folder} "
                                "(se niega a sobrescribir datos existentes)")
    for name, path in (("db_name", db_path), ("tab", tab_path)):
        if not os.path.isfile(path):
            raise FinalDatasetError(f"{name} no encontrado: {path}")
    if not os.path.isdir(polymnia_root):
        raise FinalDatasetError(f"no encuentro el repo de Polymnia en {polymnia_root} (usa "
                                "--polymnia_root): aqui vive la unica matematica de munecas")
    sys.path.insert(0, polymnia_root)
    try:
        import gesture_features as gf
        import gestures
    except ImportError as exc:
        raise FinalDatasetError(f"no se pueden importar los modulos de Polymnia desde "
                                f"{polymnia_root}: {exc}") from None

    provenance = build_provenance(provenance_path, polymnia_root)
    provenance["delivery_code"] = delivery_code(polymnia_root)

    tab = read_tab(tab_path, offset, searchterm)
    con = open_read_only(db_path, table)
    try:
        clip_ids = [r[0] for r in con.execute(f"SELECT DISTINCT id FROM {_quote(table)}").fetchall()]
        matched, unmatched = match_tab_to_clips(tab, clip_ids, searchterm)
        if matched.empty:
            # El mismo sintoma aparece cuando el tab SI esta procesado pero el `searchterm`
            # no es el termino que lleva en el nombre de fichero con el que se proceso: los
            # ids se recomponen como `<basename>_<sec>_<sec>_<searchterm>`, de modo que NADA
            # casa. Y "nada casa" es exactamente como se ve un lote al que le faltan etapas,
            # asi que el mensaje viejo mandaba a repetir download->theia (ya hechos) en vez
            # de mirar el flag. Costo real de aquel diagnostico equivocado: media hora y una
            # entrega apartada. De ahi que se impriman el termino usado y dos ids de muestra.
            ejemplo = "  |  ".join(str(c) for c in clip_ids[:2]) or "(la tabla no tiene ids)"
            raise FinalDatasetError(
                f"ningun clip de este tab casa con los {len(clip_ids)} ids de la tabla "
                f"{_quote(table)} usando --searchterm='{searchterm}'.\n"
                f"  ids en la base (muestra): {ejemplo}\n"
                "  Si esos ids terminan en un termino distinto al de arriba, ESE es el fallo: "
                "pasa el --searchterm con el que se descargo el lote (el defecto es "
                "'beginning_to_end', y no es el termino con el que se descargo casi nada).\n"
                "  Si de verdad no hay nada procesado, falta la parte del pipeline: download -> "
                "asd -> filter -> argos -> voice -> merge -> theia. Un tab solo aporta que clips "
                "pedir, no fotogramas ni keypoints.")
        guarda_holgura(matched, offset, permite=allow_padding_mismatch)
        keep = set(matched["id"])
        frame_meta, prosody, clip_meta, provenance_note = load_frame_metadata(con, table)
        annotations, annotation_fields, annotation_note = load_annotations(con)
        source_rows = int(con.execute(f"SELECT count(*) FROM {_quote(table)}").fetchone()[0])
        long_df = gestures.load_pose_keypoints(db_path, table)
    finally:
        con.close()

    wide, clock_stats = gestures.pivot_wrists(long_df, min_keypoint_confidence)
    wide["people_id"] = wide["people_id"].astype("int64")
    wide = wide[wide["id"].isin(keep)].reset_index(drop=True)
    origin = gestures.verify_neck_origin(wide)
    config = gf.GestureConfig(min_keypoint_confidence=min_keypoint_confidence,
                              max_gap_frames=max_gap_frames, window_size=window_size,
                              poly_order=poly_order)
    frame_features, sequences = gf.compute_gesture_features(wide, config, extended=extended)

    frame_meta = frame_meta[frame_meta["id"].isin(keep)]
    # Las dos lecturas del reloj (filas de pose vs. agregado por fotograma) tienen que decir
    # lo mismo: si no, la fuente es ambigua y no se publica una tabla con dos versiones.
    stamps = frame_features[["id", "people_id", "frame", "frame_timestamp"]].merge(
        frame_meta[["id", "people_id", "frame", "frame_timestamp"]].rename(
            columns={"frame_timestamp": "metadata_frame_timestamp"}),
        on=["id", "people_id", "frame"], how="inner", validate="one_to_one")
    both = stamps["frame_timestamp"].notna() & stamps["metadata_frame_timestamp"].notna()
    unequal = both & ((stamps["frame_timestamp"] - stamps["metadata_frame_timestamp"]).abs() > 1e-6)
    partial = stamps["frame_timestamp"].notna() ^ stamps["metadata_frame_timestamp"].notna()
    if unequal.any() or partial.any():
        raise FinalDatasetError(
            f"el frame_timestamp no coincide entre las filas de pose y el agregado por fotograma "
            f"({int(unequal.sum())} valores distintos, {int(partial.sum())} presentes solo en un lado)")
    frames = frame_features.merge(frame_meta.drop(columns=["frame_timestamp", "clock_absent"]),
                                  on=["id", "people_id", "frame"], how="left",
                                  validate="one_to_one" if not frame_meta.empty else None)
    frames = frames.merge(matched[["id", "newscape_clip_id"]], on="id", how="left",
                          validate="many_to_one")

    annotations = annotations[annotations["id"].isin(keep)]
    videos = sequences.merge(annotations, on="id", how="left",
                          validate="one_to_one" if not annotations.empty else None)
    videos = videos.merge(matched[["id", "newscape_clip_id", "tab_context", "tab_start_raw",
                                  "tab_end_raw", "tab_padding_s", "processed_padding_s",
                                  "id_source"]], on="id", how="left",
                          validate="one_to_one")
    if clip_meta:
        clip_level = frame_meta.groupby(["id", "people_id"], sort=False)[clip_meta].first().reset_index()
        videos = videos.merge(clip_level, on=["id", "people_id"], how="left", validate="one_to_one")

    parent = os.path.dirname(os.path.abspath(output_folder)) or "."
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=parent)
    try:
        written = {}
        for stem, table_frame in ((FRAMES_STEM, frames), (VIDEOS_STEM, videos)):
            table_frame.to_parquet(os.path.join(staging, f"{stem}.parquet"), index=False)
            table_frame.to_csv(os.path.join(staging, f"{stem}.csv"), index=False,
                               na_rep="", date_format="iso")
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "readme": README_NAME if readme else None,
            "inputs": {"db_name": os.path.abspath(db_path), "db_sha256": sha256(db_path),
                       "tab": os.path.abspath(tab_path), "tab_sha256": sha256(tab_path),
                       "table": table, "source_rows": source_rows},
            "clip_id_mapping": {
                "requested_padding_s": float(offset), "searchterm": searchterm,
                "rule": ("basename(txt)_{start_sec+start_msec/div - pad}_{end_sec+end_msec/div + pad}"
                         f"_{searchterm}; div=1000 si el nombre empieza por 2017, si no 100"),
                "tab_rows": int(len(tab)), "matched": int(len(matched)),
                "matched_exact_id": int((matched["id_source"] == "exact").sum()) if len(matched) else 0,
                "matched_by_window_center": int((matched["id_source"] == "window_center").sum())
                if len(matched) else 0,
                "processed_padding_s": sorted(set(matched["processed_padding_s"])) if len(matched) else [],
                "unmatched": unmatched.to_dict("records")},
            "features": {"version": gf.feature_version(extended), "extended": bool(extended),
                         "config": config.as_dict(),
                         "clock_stats": clock_stats, "origin": origin,
                         "movement_speed_units_per_s": gf.MOVEMENT_SPEED,
                         "units": gf.FEATURE_UNITS, "semantics": gf.SEMANTICS,
                         **({"admission_rules": list(ADMISSION_RULES),
                             "extended_units": gf.EXTENDED_FEATURE_UNITS,
                             "extended_semantics": gf.EXTENDED_SEMANTICS,
                             "extended_columns": list(gf.EXTENDED_SEQUENCE_MEASURES),
                             "extended_dropped_from_reference_code": {
                                 "set_dependent_scores": "79 columnas *_norm/*_score: normalizaban "
                                     "min-max contra el conjunto completo de clips (leakage y valor "
                                     "dependiente del lote)",
                                 "qcut_rings_and_zones": "54 columnas de cuartiles del propio clip + "
                                     "zonas/cuadrantes: uniformes por construccion (ring_entropy max "
                                     "= ln 4) y sin anclaje fisico",
                                 "coordination_coef": "rho = +0,983 con min(rapidez media dcha, izq): "
                                     "medida de intensidad conjunta, no de coordinacion; sustituida "
                                     "por cos_phase_* y coactivity_*",
                                 "gesture_area_circle": "pi*(d/2)^2 es monotona en pair_separation: "
                                     "misma informacion con nombre de area",
                                 "hands_horizontal_symmetry": "ya estaba publicada como mirror_error_x "
                                     "(misma formula |x_R + x_L|)",
                                 "hands_vertical_symmetry": "|y_R + y_L| no es simetria sino elevacion: "
                                     "entra renombrada como abs_y_pair_sum_mean y pair_elevation_mean",
                                 "time_to_peak_sep": "columna placeholder sin calcular en el codigo "
                                     "de referencia"}} if extended else {})},
            "tables": {
                FRAMES_STEM: {"grain": "una fila por (id, people_id, frame)",
                              "rows": int(len(frames)), "columns": list(frames.columns),
                              "files": [f"{FRAMES_STEM}.parquet", f"{FRAMES_STEM}.csv"],
                              "prosody_columns": list(prosody),
                              "clip_metadata_columns": [c for c in clip_meta if c in frames.columns]},
                VIDEOS_STEM: {"grain": "una fila por (id, people_id)",
                              "rows": int(len(videos)), "columns": list(videos.columns),
                              "files": [f"{VIDEOS_STEM}.parquet", f"{VIDEOS_STEM}.csv"],
                              "annotation_columns": list(annotation_fields),
                              "clip_metadata_columns": [c for c in clip_meta if c in videos.columns],
                              "annotation_note": annotation_note,
                              "provenance_note": provenance_note}},
            "null_policy": ("NULL = no medido (keypoint ausente, confianza baja, pocos samples "
                            "conectados o fotograma sin reloj). NULL no es cero ni ausencia de gesto."),
            "platform": {"python": sys.version.split()[0], "os": platform.platform()},
            "provenance": provenance,
        }
        manifest_path = os.path.join(staging, MANIFEST)
        if readme:
            if not os.path.isfile(readme):
                raise FinalDatasetError(f"el --readme no existe: {readme}")
            shutil.copyfile(readme, os.path.join(staging, README_NAME))
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)
        manifest["output_sha256"] = {name: sha256(os.path.join(staging, name))
                                     for name in sorted(os.listdir(staging))
                                     if name != MANIFEST}
        # A manifest cannot carry its own digest: hashing it here would record the
        # bytes from before this block existed, which can never match the file on
        # disk. The self-entry is declared absent instead of being a wrong hash.
        manifest["output_sha256_note"] = ("output_sha256 cubre todos los ficheros publicados menos "
                                          "manifest.json: un manifiesto no puede contener el digest "
                                          "de si mismo")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)
        written = manifest["output_sha256"]
        gestures._atomic_rename(staging, output_folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    manifest["output_sha256"] = written
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db_name", required=True, help="DuckDB del pipeline, se abre en solo lectura")
    parser.add_argument("--tab", required=True, help="tab de NewsScape (sin cabecera, >=6 campos)")
    parser.add_argument("--output_folder", required=True, help="carpeta a crear; si existe, se niega")
    parser.add_argument("--offset", type=float, default=2.0,
                        help="holgura en segundos anadida a cada lado del clip (por defecto 2.0)")
    parser.add_argument("--searchterm", default="beginning_to_end",
                        help="etiqueta que el pipeline puso en el nombre del clip")
    parser.add_argument("--polymnia_root", default=POLYMNIA_DEFAULT,
                        help="repo Polymnia, de donde se importa la matematica de munecas")
    parser.add_argument("--table_name", default="multi_data", help="tabla larga de pose")
    parser.add_argument("--min-keypoint-confidence", dest="min_keypoint_confidence", type=float,
                        default=0.2, help="confianza minima de muneca por fotograma (inclusive)")
    parser.add_argument("--max-gap-frames", dest="max_gap_frames", type=int, default=3,
                        help="huecos interiores de hasta N frames se interpolan y se marcan; 0 desactiva")
    parser.add_argument("--window-size", dest="window_size", type=int, default=7,
                        help="ventana Savitzky-Golay impar por tramo contiguo; 1 desactiva el suavizado")
    parser.add_argument("--poly-order", dest="poly_order", type=int, default=2,
                        help="orden del polinomio de suavizado, menor que la ventana")
    parser.add_argument("--extended", action="store_true",
                        help="anadir a final_videos el bloque extendido (FEATURE_VERSION 3): jerk, "
                             "distribucion, forma, direccion y bimanuales. Sin la bandera la salida "
                             "es identica a la version 2")
    parser.add_argument("--readme", default=None,
                        help="markdown a publicar dentro de la carpeta (se hashea en el manifiesto)")
    parser.add_argument("--allow-padding-mismatch", dest="allow_padding_mismatch", action="store_true",
                        help="publicar aunque la holgura real de los clips procesados no sea la de "
                             "--offset. Sin esta bandera se niega: los clips encajan por centro de "
                             "ventana, no por id, y publicar con holguras mezcladas desplaza las medidas")
    parser.add_argument("--provenance", default=None,
                        help="JSON written by tools/tool_provenance.py at run start (pipeline_lote.sh "
                             "stage 0: informes/procedencia.json). It is embedded in manifest.json so "
                             "the delivery says which Praat, OpenPose, R and weights produced it. "
                             "Without it the delivery records that no tool versions were captured")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = os.path.abspath(args.output_folder)
    try:
        manifest = build(args.db_name, args.tab, output, args.offset, args.searchterm,
                         os.path.abspath(args.polymnia_root), args.table_name,
                         args.min_keypoint_confidence, args.max_gap_frames,
                         args.window_size, args.poly_order, extended=args.extended,
                         readme=os.path.abspath(args.readme) if args.readme else None,
                         allow_padding_mismatch=args.allow_padding_mismatch,
                         provenance_path=args.provenance)
    except (FinalDatasetError, ValueError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    tables, mapping = manifest["tables"], manifest["clip_id_mapping"]
    print(f"✅ {output}")
    for stem in (FRAMES_STEM, VIDEOS_STEM):
        info = tables[stem]
        print(f"   {stem}: {info['rows']} filas x {len(info['columns'])} columnas")
    print(f"   features: version {manifest['features']['version']}"
          f" ({'extendidas' if manifest['features']['extended'] else 'base'}) | "
          f"umbral de movimiento {manifest['features']['movement_speed_units_per_s']} u/s")
    provenance = manifest.get("provenance") or {}
    if provenance.get("reason"):
        print(f"   ⚠️  provenance: {provenance['reason']}")
    else:
        print(f"   provenance: {provenance['captured_from']} captured at {provenance['captured_at']} "
              f"(embedded in manifest.json)")
    print(f"   tab: {mapping['tab_rows']} lineas -> {mapping['matched']} clips procesados "
          f"({mapping['matched_exact_id']} por id exacto, "
          f"{mapping['matched_by_window_center']} por centro de ventana), "
          f"{len(mapping['unmatched'])} sin procesar")
    print(f"   holgura pedida: {mapping['requested_padding_s']} s por lado | "
          f"holgura real de los clips procesados: {mapping['processed_padding_s']} s")
    for name, digest in manifest["output_sha256"].items():
        print(f"   {digest[:12]}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
