#!/usr/bin/env python3
"""Test del cargador de metadata por fotograma de `final_dataset.py`.

Cubre la parte que se rompio dos veces por culpa del esquema de la base y no de la
matematica: `final_dataset.py` lee la voz de la DuckDB que arma el pipeline, y ese esquema
no es estable entre entregas (`vad` era BOOLEAN en 1p.duckdb y salio INTEGER en una base
nueva; `voice_present` no existe hasta el arreglo de la prosodia). Las tres comprobaciones
de aqui son las que protegen el contrato NULL = no medido en la parte de voz.

    Polymnia/.venv/bin/python final_dataset/test_final_dataset_voz.py
"""
import os
import sys
import tempfile

import duckdb
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from final_dataset import load_frame_metadata  # noqa: E402

FAIL = []


def check(nombre, bien, detalle=""):
    print(f"[{'ok ' if bien else 'FAIL'}] {nombre}" + ("" if bien else f"  {detalle}"))
    if not bien:
        FAIL.append(nombre)


def estado(serie):
    """[True, False, None] sin el ruido de tipos de pandas (np.True_ y pd.NA)."""
    return [None if pd.isna(v) else bool(v) for v in serie]


def base(columnas, filas):
    """DuckDB en memoria con una tabla tipo `multi_data` (larga, un bloque por fotograma)."""
    con = duckdb.connect(":memory:")
    definicion = ", ".join(f'"{c}" {t}' for c, t in columnas)
    con.execute(f"CREATE TABLE multi_data ({definicion})")
    con.executemany("INSERT INTO multi_data VALUES (" + ", ".join(["?"] * len(filas[0])) + ")", filas)
    return con


COLUMNAS_BASE = [("id", "VARCHAR"), ("people_id", "INTEGER"), ("frame", "INTEGER"),
                 ("frame_timestamp", "DOUBLE"), ("vad", "INTEGER"), ("words", "VARCHAR"),
                 ("pitch", "DOUBLE"), ("program_name", "VARCHAR"), ("time_range", "VARCHAR")]


def _filas(voz_por_fotograma):
    """Dos filas por fotograma, como en la fuente real (la voz se repite en cada keypoint)."""
    filas = []
    for frame, voz in voz_por_fotograma.items():
        for punto in (0, 1):
            filas.append(["clipX", 1, frame, 1.0 * frame, 1, "hola", voz, "archivo.json", "archivo.json"])
    return filas


def test_sin_voice_present_no_aparece_la_columna():
    """Una base vieja (sin el arreglo) no puede ganar una columna que no existe."""
    con = base(COLUMNAS_BASE, _filas({1: 120.0, 2: None}))
    frame, prosody, _, _ = load_frame_metadata(con, "multi_data")
    check("una fuente sin voice_present no publica la columna",
          "voice_present" not in frame.columns, list(frame.columns))
    check("la lista de prosodia de la fuente vieja no la menciona",
          "voice_present" not in prosody, prosody)


def test_voice_present_con_bool_or_y_todo_null():
    """TRUE si algun bloque del fotograma la tiene, FALSE sin pista, NULL si no hay pista de pitch."""
    columnas = COLUMNAS_BASE + [("voice_present", "BOOLEAN")]
    filas = []
    for frame, (voz, presente) in {1: (120.0, True), 2: (None, False), 3: (None, None)}.items():
        for _punto in (0, 1):
            filas.append(["clipX", 1, frame, 1.0 * frame, 1, "hola", voz,
                          "archivo.json", "archivo.json", presente])
    con = base(columnas, filas)
    frame, prosody, _, _ = load_frame_metadata(con, "multi_data")
    orden = frame.sort_values("frame").reset_index(drop=True)
    check("voice_present se publica y sale en la lista de prosodia",
          "voice_present" in frame.columns and "voice_present" in prosody, prosody)
    check("los tres estados se distinguen: medida / sin pista / sin columna de pitch",
          estado(orden.voice_present) == [True, False, None], estado(orden.voice_present))
    check("agrupar no duplica fotogramas", len(orden) == 3, len(orden))


def test_vad_entera_no_peta_y_conserva_el_valor():
    """`vad` llega INTEGER en las bases nuevas: bool_or tiene que castear sin inventarse ceros."""
    columnas = [("id", "VARCHAR"), ("people_id", "INTEGER"), ("frame", "INTEGER"),
                ("frame_timestamp", "DOUBLE"), ("vad", "INTEGER"), ("words", "VARCHAR"),
                ("pitch", "DOUBLE"), ("program_name", "VARCHAR"), ("time_range", "VARCHAR")]
    filas = []
    for frame, vad in {1: 1, 2: 0, 3: None}.items():
        filas.append(["clipX", 1, frame, 1.0 * frame, vad, "hola", 120.0,
                      "archivo.json", "archivo.json"])
    con = base(columnas, filas)
    frame, _, _, _ = load_frame_metadata(con, "multi_data")
    orden = frame.sort_values("frame").reset_index(drop=True)
    check("vad entera no revienta el constructor y no convierte NULL en False",
          estado(orden.vad) == [True, False, None], estado(orden.vad))


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[fn]()
        except Exception as e:  # noqa: BLE001
            check(f"{fn} (excepción)", False, repr(e))
    print("\nTOTAL:", "FAIL " + ", ".join(FAIL) if FAIL else "todo OK")
    sys.exit(1 if FAIL else 0)
