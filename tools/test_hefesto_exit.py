#!/usr/bin/env python3
"""Descarte por clip y umbral en `hefesto.py`.

`hefesto.py` mergea un Parquet por clip en la tabla `multi_data`. Antes, cualquier
discrepancia (cuatro causas distintas) contaba en el mismo contador y terminaba con
`sys.exit(1)`: un solo clip sin voz tumbaba el lote entero, con sus horas de OpenPose y
las etapas de anotacion y publicacion.

Politica nueva (INVENTARIO_FALLOS_LOTES.md §3): **descartar el clip, no el lote**, pero
solo para las causas que significan "no hay voz que alinear" - y siempre con el motivo
escrito y con un umbral que pare el lote cuando el patron no es un hablante mudo sino una
extraccion rota.

Orden de los casos:

  1. trio completo: sale 0 y sus filas estan en la tabla
  2. falta la prosodia: sale 0, el clip NO entra, se reporta con motivo `prosody_missing`
     y queda escrito en el fichero de descartes
  3. prosodia con 0 filas: motivo `prosody_empty`
  4. umbral: 2 descartes de 10 clips (20 %) > 5 % => sale distinto de 0
  5. umbral desactivado (`--max-discard-pct 0`) => sale 0 aunque descarte
  6. todo descartado (0 clips utilizables) => sale distinto de 0 aunque el umbral este apagado
  7. `--strict-discard` recupera la conducta vieja para depurar (un descarte => exit 1)
  8. fallo de insercion (no es "clip mudo", es un bug) => sale distinto de 0
  9. bucket sin carpeta de parquets: sigue saliendo 0 (create_dataset recorre varios buckets)
"""
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEFESTO = os.path.join(REPO, "hefesto.py")
FRAMES = [1, 2, 3]

FAIL = []


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def monta_lote(base, n=1, sin_prosodia=(), prosodia_vacia=()):
    """`n` clips con tres frames cada uno, en el layout que espera hefesto.

    `sin_prosodia` y `prosodia_vacia` son indices de clip (0..n-1) a los que se les
    quita el parquet de prosodia o se les deja con 0 filas.
    """
    sin_prosodia = set(sin_prosodia)
    prosodia_vacia = set(prosodia_vacia)
    ruta = os.path.join(base, "dataset", "1_persons")
    os.makedirs(os.path.join(ruta, "parquet_files"), exist_ok=True)
    os.makedirs(os.path.join(ruta, "parquet_speech_analysis"), exist_ok=True)
    os.makedirs(os.path.join(ruta, "whisperx_framer_output"), exist_ok=True)
    for i in range(n):
        clip = f"clip_{i}"
        pd.DataFrame({
            "frame": [f - 1 for f in FRAMES],
            "id": [clip] * len(FRAMES),
            "datetime": ["2026-01-01 00:00:00.%06d" % (f * 1000) for f in FRAMES],
        }).to_parquet(os.path.join(ruta, "parquet_files", f"{clip}.parquet"))
        if i in prosodia_vacia:
            pd.DataFrame({"Frame": [], "pitch": []}).to_parquet(
                os.path.join(ruta, "parquet_speech_analysis", f"{clip}.parquet"))
        elif i not in sin_prosodia:
            pd.DataFrame({"Frame": FRAMES, "pitch": [110.0, 115.0, 120.0]}).to_parquet(
                os.path.join(ruta, "parquet_speech_analysis", f"{clip}.parquet"))
        with open(os.path.join(ruta, "whisperx_framer_output", f"{clip}.json"), "w") as fh:
            json.dump([{"frame_number": f, "words": ["hola"], "scores": [0.9]} for f in FRAMES], fh)
    return base


def corre(lote, db, *extra):
    return subprocess.run([sys.executable, HEFESTO, "-i", lote, "-n", "1", "-d", db, *extra],
                          capture_output=True, text=True)


def filas(db):
    try:
        import duckdb
    except ImportError:
        return None
    if not os.path.isfile(db):
        return -1
    con = duckdb.connect(db, read_only=True)
    try:
        return int(con.execute("SELECT count(*) FROM multi_data").fetchone()[0])
    except Exception:
        return -1
    finally:
        con.close()


def reporte(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return [linea.rstrip("\n").split("\t") for linea in fh if linea.strip()]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # 1. todo presente: sale 0 y las filas estan
        lote = monta_lote(os.path.join(tmp, "completo"))
        db = os.path.join(tmp, "a.duckdb")
        p = corre(lote, db)
        check("un clip completo sale 0", p.returncode == 0, f"codigo {p.returncode}: {p.stdout[-300:]}")
        check("las 3 filas llegan a multi_data", filas(db) == 3, f"filas {filas(db)}")

        # 2. falta la prosodia: se descarta el clip, NO el lote (1 de 40 = 2,5 % < 5 %)
        lote = monta_lote(os.path.join(tmp, "sin_prosodia"), n=40, sin_prosodia={0})
        db = os.path.join(tmp, "b.duckdb")
        rep = os.path.join(tmp, "b.tsv")
        p = corre(lote, db, "--discard-report", rep)
        check("un clip sin prosodia no tumba el lote", p.returncode == 0,
              f"codigo {p.returncode}: {p.stdout[-300:]}")
        check("ese clip no entra en la base", filas(db) == 39 * len(FRAMES), f"filas {filas(db)}")
        check("la discrepancia sigue contada", "Total discrepancies: 1" in p.stdout, p.stdout[-300:])
        check("el motivo queda escrito en la salida", "prosody_missing=1" in p.stdout, p.stdout[-300:])
        filas_rep = reporte(rep) or []
        check("el reporte de descartes tiene cabecera y una fila",
              len(filas_rep) == 2, f"{len(filas_rep)} filas")
        if len(filas_rep) == 2:
            check("el reporte nombra el clip", filas_rep[1][0] == "clip_0", filas_rep[1])
            check("el reporte lleva el motivo", filas_rep[1][1] == "prosody_missing", filas_rep[1])

        # 3. prosodia con 0 filas: otro motivo, mismo tratamiento
        lote = monta_lote(os.path.join(tmp, "prosodia_vacia"), n=40, prosodia_vacia={0})
        db = os.path.join(tmp, "c.duckdb")
        p = corre(lote, db, "--discard-report", os.path.join(tmp, "c.tsv"))
        check("prosodia vacia se descarta con su motivo",
              p.returncode == 0 and "prosody_empty=1" in p.stdout, p.stdout[-300:])

        # 4. umbral: 2 de 10 (20 %) > 5 %
        lote = monta_lote(os.path.join(tmp, "umbral"), n=10, sin_prosodia={3, 7})
        db = os.path.join(tmp, "d.duckdb")
        p = corre(lote, db, "--discard-report", os.path.join(tmp, "d.tsv"))
        check("20 % de descartes para el lote", p.returncode != 0,
              f"salio {p.returncode} con 20 % > 5 %")
        check("y lo dice con números", "20.0 %" in p.stdout or "20 %" in p.stdout, p.stdout[-300:])
        check("los 8 buenos si entraron", filas(db) == 8 * len(FRAMES), f"filas {filas(db)}")

        # 5. umbral apagado
        p = corre(lote, os.path.join(tmp, "e.duckdb"), "--max-discard-pct", "0")
        check("con --max-discard-pct 0 el umbral no para", p.returncode == 0,
              f"codigo {p.returncode}: {p.stdout[-300:]}")

        # 6. todo descartado: nunca se publica un lote vacio
        lote = monta_lote(os.path.join(tmp, "todo_malo"), sin_prosodia={0})
        p = corre(lote, os.path.join(tmp, "f.duckdb"), "--max-discard-pct", "0")
        check("si no sobrevive ningun clip, sale distinto de 0", p.returncode != 0,
              f"codigo {p.returncode}")

        # 7. conducta vieja disponible para depurar
        lote = monta_lote(os.path.join(tmp, "estricto"), sin_prosodia={0})
        p = corre(lote, os.path.join(tmp, "g.duckdb"), "--strict-discard")
        check("--strict-discard recupera el exit 1", p.returncode != 0,
              f"codigo {p.returncode}")

        # 8. fallo de insercion: eso no es un clip mudo, es un bug
        lote = monta_lote(os.path.join(tmp, "insert"))
        db_como_dir = os.path.join(tmp, "db_es_un_dir")
        os.makedirs(db_como_dir)
        p = corre(lote, db_como_dir, "--max-discard-pct", "0")
        check("un fallo de insercion sigue abortando", p.returncode != 0,
              f"codigo {p.returncode}: {p.stdout[-300:]}")
        check("y no se confunde con un descarte",
              "Errores fatales: 1" in p.stdout and "prosody_missing=1" not in p.stdout,
              p.stdout[-300:])

        # 9. bucket vacio: no es un error
        p = corre(os.path.join(tmp, "sin_parquets"), os.path.join(tmp, "h.duckdb"))
        check("un bucket sin carpeta de parquets sigue saliendo 0", p.returncode == 0,
              f"codigo {p.returncode}")
        check("y lo dice clar", "does not exist" in p.stdout, p.stdout[-300:])

    print()
    if FAIL:
        print(f"TOTAL: FAIL {', '.join(FAIL)}")
        return 1
    print("TOTAL: todo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
