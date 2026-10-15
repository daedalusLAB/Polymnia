#!/usr/bin/env python3
"""Exit code of `hefesto.py`.

`hefesto.py` merges one Parquet per clip into the `multi_data` table and counts
"discrepancies" - files whose rows never made it into the database. Before this test the
process still returned 0 in that case, so a caller that checks the exit code (`create_dataset.py`,
or any batch script that continues into pose/gesture features) proceeds with a database that
is silently missing clips.

Order of the cases:

  1. a complete trio (pose + prosody + words) returns 0 and its rows are in the table
  2. a missing prosody file returns non-zero - and the discrepancy is still *reported* in
     the output, the exit code does not replace the message
  3. an absent `parquet_files` folder stays 0: `create_dataset.py` walks several
     `<n>_persons` buckets and an empty bucket is a normal, expected outcome
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


def monta_lote(base, con_prosodia=True, con_parquets=True):
    """One clip, three frames, in the folder layout hefesto expects."""
    ruta = os.path.join(base, "dataset", "1_persons")
    if con_parquets:
        os.makedirs(os.path.join(ruta, "parquet_files"), exist_ok=True)
        pd.DataFrame({
            "frame": [f - 1 for f in FRAMES],
            "id": ["clip_x"] * len(FRAMES),
            "datetime": ["2026-01-01 00:00:00.%06d" % (f * 1000) for f in FRAMES],
        }).to_parquet(os.path.join(ruta, "parquet_files", "clip_x.parquet"))
    if con_prosodia:
        os.makedirs(os.path.join(ruta, "parquet_speech_analysis"), exist_ok=True)
        pd.DataFrame({"Frame": FRAMES, "pitch": [110.0, 115.0, 120.0]}).to_parquet(
            os.path.join(ruta, "parquet_speech_analysis", "clip_x.parquet"))
    os.makedirs(os.path.join(ruta, "whisperx_framer_output"), exist_ok=True)
    with open(os.path.join(ruta, "whisperx_framer_output", "clip_x.json"), "w") as fh:
        json.dump([{"frame_number": f, "words": ["hola"], "scores": [0.9]} for f in FRAMES], fh)
    return base


def corre(lote, db):
    return subprocess.run([sys.executable, HEFESTO, "-i", lote, "-n", "1", "-d", db],
                          capture_output=True, text=True)


def filas(db):
    try:
        import duckdb
    except ImportError:
        return None
    if not os.path.exists(db):
        return -1
    con = duckdb.connect(db, read_only=True)
    try:
        return int(con.execute("SELECT count(*) FROM multi_data").fetchone()[0])
    except Exception:
        return -1
    finally:
        con.close()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # 1. todo presente: sale 0 y las filas estan
        lote = monta_lote(os.path.join(tmp, "completo"))
        db = os.path.join(tmp, "a.duckdb")
        p = corre(lote, db)
        check("un clip completo sale 0", p.returncode == 0, f"codigo {p.returncode}: {p.stdout[-200:]}")
        check("las 3 filas llegan a multi_data", filas(db) == 3, f"filas {filas(db)}")

        # 2. falta la prosodia: ese clip no entra, y hay que notarlo
        lote = monta_lote(os.path.join(tmp, "sin_prosodia"), con_prosodia=False)
        db = os.path.join(tmp, "b.duckdb")
        p = corre(lote, db)
        check("una discrepancia sale con codigo distinto de 0", p.returncode != 0,
              "salio 0 con filas sin insertar")
        check("la discrepancia sigue contada en la salida", "Total discrepancies: 1" in p.stdout,
              p.stdout[-200:])
        check("el fichero con problemas sigue listado", "clip_x" in p.stdout, p.stdout[-200:])
        check("no se crea una base a medias", filas(db) == -1, f"filas {filas(db)}")

        # 3. bucket vacio: no es un error
        p = corre(os.path.join(tmp, "sin_parquets"), os.path.join(tmp, "c.duckdb"))
        check("un bucket sin carpeta de parquets sigue saliendo 0", p.returncode == 0,
              f"codigo {p.returncode}")
        check("y lo dice clar", "does not exist" in p.stdout, p.stdout[-200:])

    print()
    if FAIL:
        print(f"TOTAL: FAIL {', '.join(FAIL)}")
        return 1
    print("TOTAL: todo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
