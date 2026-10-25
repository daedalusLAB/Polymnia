#!/usr/bin/env python3
"""`download_clips.py --searchterm`: qué nombres valen (y que el inválido falle, no calle).

El searchterm se pega al nombre de cada clip (`..._<searchterm>.mp4`), así que no puede llevar
separadores de ruta ni espacios. Hasta ahora `-` se rechazaba, pero las campañas de NewsScape lo
llevan en el nombre (`ayesha_ago_d-time`), y el pipeline lo deduce del nombre de la carpeta: un
lote con guion se paraba en la descarga con "El término de búsqueda debe ser..." y **exit 0**.

Casos:
  1. alfanumérico y `_` valen
  2. `-` también vale (campañas ayesha)
  3. vacío, espacios, `/`, `.` y comillas no valen
  4. un searchterm inválido hace que el script salga distinto de 0 (antes salía 0) y no cree
     la carpeta de salida
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import download_clips as dc  # noqa: E402

FAIL = []


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def main():
    for bueno in ("freq029", "ayesha_ago_d_time", "ayesha_ago_d-time", "A1_"):
        check(f"acepta {bueno!r}", dc.searchterm_valido(bueno))
    for malo in ("", " ", "a/b", "a b", "..", "a;b", "a'b"):
        check(f"rechaza {malo!r}", not dc.searchterm_valido(malo))

    with tempfile.TemporaryDirectory() as tmp:
        tab = os.path.join(tmp, "t.txt")
        with open(tab, "w", encoding="utf-8") as fh:
            fh.write("clip.txt\tid\t0\t0\t1\t0\t\n")
        salida = os.path.join(tmp, "out")
        p = subprocess.run([sys.executable, os.path.join(REPO, "download_clips.py"),
                            "--txt_file", tab, "--searchterm", "a/b",
                            "--output_dir", salida],
                           capture_output=True, text=True, cwd=tmp)
        check("searchterm inválido sale distinto de 0", p.returncode != 0,
              f"codigo {p.returncode}: {p.stdout[-200:]}")
        check("no crea la carpeta de salida", not os.path.isdir(salida))
        check("y dice por qué", "término de búsqueda" in (p.stdout + p.stderr).lower(),
              (p.stdout + p.stderr)[-200:])

    print()
    if FAIL:
        print(f"TOTAL: FAIL {', '.join(FAIL)}")
        return 1
    print("TOTAL: todo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
