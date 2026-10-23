#!/usr/bin/env python3
"""`tools/filtrar_fallos_descarga.py`: el informe de fallos de UN lote, no de todos.

`download_clips.py` escribe `failed_downloads.log` en el cwd del repo (`uvrun` la fija),
asi que todas las descargas -también dos lotes en paralelo- appenden al mismo fichero. El
pipeline sacaba el delta por offset de bytes: con dos descargas simultáneas cada lote
copiaba los fallos del otro a su `informes/failed_downloads_nuevos.log`.

Ahora se filtra por lote (`_<searchterm>.mp4`, que va en el nombre del clip) y por hora de
inicio de la etapa. Casos:

  1. solo las líneas del lote pedido
  2. `--desde` deja fuera lo que ya estaba en el log de pasadas anteriores
  3. sin `--desde` se queda con todas las del lote
  4. log inexistente o sin coincidencias => fichero vacío y exit 0
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILTRO = os.path.join(REPO, "tools", "filtrar_fallos_descarga.py")
FAIL = []

LINEA = ("{ts},123 - WARNING - Fallo en {clip}_{a}_{b}_freq{suf}.mp4 "
         "(intento 1). Código: 28\n")


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def lineas(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return [l for l in fh if l.strip()]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "failed_downloads.log")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(LINEA.format(ts="2026-10-20 09:00:00", clip="2017-01-01_0000_US_X",
                                  a="10.5", b="14.5", suf="002"))
            fh.write(LINEA.format(ts="2026-10-24 10:01:02", clip="2017-01-02_0000_US_X",
                                  a="20.5", b="24.5", suf="002"))
            fh.write(LINEA.format(ts="2026-10-24 10:01:03", clip="2017-01-03_0000_US_X",
                                  a="30.5", b="34.5", suf="007"))
            fh.write(LINEA.format(ts="2026-10-24 10:01:04", clip="2017-01-04_0000_US_X",
                                  a="40.5", b="44.5", suf="002"))

        def corre(*extra):
            out = os.path.join(tmp, "out.tsv")
            if os.path.exists(out):
                os.remove(out)
            r = subprocess.run([sys.executable, FILTRO, "--log", log,
                                "--searchterm", "freq002", "--out", out, *extra],
                               capture_output=True, text=True)
            return r, lineas(out)

        # 1+2. con --desde: solo las dos recientes de freq002
        r, filas = corre("--desde", "2026-10-24 10:00:00")
        check("sale 0", r.returncode == 0, r.stderr[-200:])
        check("se queda solo con las del lote", filas is not None and len(filas) == 2,
              f"{filas}")
        if filas:
            check("no mete el otro lote", all("freq007" not in l for l in filas), filas)
            check("no mete pasadas anteriores", all("09:00:00" not in l for l in filas), filas)

        # 3. sin --desde: las tres de freq002 (vieja incluida)
        r, filas = corre()
        check("sin --desde no filtra por hora", filas is not None and len(filas) == 3,
              f"{filas}")

        # 4a. log inexistente
        out = os.path.join(tmp, "vacio.tsv")
        r = subprocess.run([sys.executable, FILTRO, "--log", os.path.join(tmp, "no.log"),
                            "--searchterm", "freq002", "--out", out],
                           capture_output=True, text=True)
        check("log inexistente: exit 0 y fichero vacío",
              r.returncode == 0 and lineas(out) == [], r.stderr[-200:])

        # 4b. sin coincidencias
        r, filas = corre("--desde", "2026-10-25 00:00:00")
        check("sin coincidencias: fichero vacío", r.returncode == 0 and filas == [], f"{filas}")

    print()
    if FAIL:
        print(f"TOTAL: FAIL {', '.join(FAIL)}")
        return 1
    print("TOTAL: todo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
