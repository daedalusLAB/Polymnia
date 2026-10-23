#!/usr/bin/env python3
"""`tools/check_annotations.py`: la guarda que impide publicar sin anotaciones.

Casos:

  1. 2 filas buenas + 1 con error  => exit 0 y cuentas correctas
  2. todas con error               => exit 1 (es el fallo de FREQ028_4 y FREQ029)
  3. tabla vacia                   => exit 1
  4. tabla que no existe           => exit 1
  5. base de datos que no existe   => exit 1
  6. `error` NULL cuenta como anotacion buena (theia deja cadena vacia, no NULL)
"""
import os
import subprocess
import sys
import tempfile

import duckdb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(REPO, "tools", "check_annotations.py")
FAIL = []


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def base(tmp, nombre, filas):
    db = os.path.join(tmp, nombre)
    con = duckdb.connect(db)
    if filas is not None:
        con.execute("CREATE TABLE video_annotations(id VARCHAR, error VARCHAR)")
        for clip, error in filas:
            con.execute("INSERT INTO video_annotations VALUES (?, ?)", [clip, error])
    con.close()
    return db


def corre(db):
    return subprocess.run([sys.executable, CHECK, "--db", db], capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # 1. mezcla: hay anotaciones, la etapa vale
        db = base(tmp, "ok.duckdb", [("a", ""), ("b", None), ("c", "429 Too Many Requests")])
        p = corre(db)
        check("con anotaciones sale 0", p.returncode == 0, f"codigo {p.returncode}: {p.stdout}")
        check("cuenta bien anotados y errores", "anotados=2 error=1 total=3" in p.stdout, p.stdout)

        # 2. todas con error: es el fallo del endpoint
        db = base(tmp, "todas_error.duckdb", [("a", "403 Forbidden"), ("b", "500 boom")])
        p = corre(db)
        check("sin una sola anotacion sale 1", p.returncode == 1, f"codigo {p.returncode}")
        check("y lo dice con numeros", "anotados=0 error=2 total=2" in p.stdout, p.stdout)

        # 3. tabla vacia
        db = base(tmp, "vacia.duckdb", [])
        p = corre(db)
        check("tabla vacia sale 1", p.returncode == 1, f"codigo {p.returncode}")

        # 4. tabla que no existe
        db = base(tmp, "sin_tabla.duckdb", None)
        p = corre(db)
        check("tabla ausente sale 1", p.returncode == 1, f"codigo {p.returncode}")
        check("y nombra la tabla", "video_annotations" in p.stdout, p.stdout)

        # 5. base que no existe
        p = corre(os.path.join(tmp, "no_existe.duckdb"))
        check("base ausente sale 1", p.returncode == 1, f"codigo {p.returncode}")

    print()
    if FAIL:
        print(f"TOTAL: FAIL {', '.join(FAIL)}")
        return 1
    print("TOTAL: todo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
