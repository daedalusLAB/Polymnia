#!/usr/bin/env python3
"""Stage 1 de-duplicates the tab before `--limit` (and before downloading).

NewsScape tabs carry exact duplicate rows (~1 % of the lines: same program file, same clip
id, same start/end). Without de-duplicating, the download writes the same mp4 twice (one is
lost, `clips_descargados` comes out short) and stage 10 aborts with "dos líneas producen el
mismo id de clip", after all the expensive work is done.

Cases:

  1. a tab with 2 duplicated rows: only unique rows reach `tsv/efectivo.tsv`, the count is
     recorded in `informes/conteos.tsv`, and the duplicates are listed for auditing
  2. `--limit` is applied AFTER de-duplication, so "2 lines" means 2 unique clips
  3. a tab without duplicates still works and reports 0
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO, "final_dataset", "pipeline_lote.sh")
FAIL = []


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def linea(nombre, clip_id, start_s, start_ms, end_s, end_ms, texto):
    return f"{nombre}\t{clip_id}\t{start_s}\t{start_ms}\t{end_s}\t{end_ms}\t{texto}\n"


def monta_tsv(base, filas):
    os.makedirs(base, exist_ok=True)
    ruta = os.path.join(base, "tab.txt")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.writelines(filas)
    return ruta


def corre(base, *args):
    return subprocess.run(["bash", PIPELINE, *args],
                          cwd=base, capture_output=True, text=True)


def conteos(lote):
    ruta = os.path.join(lote, "informes", "conteos.tsv")
    datos = {}
    if os.path.isfile(ruta):
        for fila in open(ruta, encoding="utf-8"):
            partes = fila.rstrip("\n").split("\t")
            if len(partes) == 3:
                datos[(partes[0], partes[1])] = partes[2]
    return datos


def lineas(ruta):
    if not os.path.isfile(ruta):
        return None
    with open(ruta, encoding="utf-8") as fh:
        return [l for l in fh if l.strip()]


def main():
    a = linea("2017-01-01_0000_US_A.txt", "t__a", 10, 100, 10, 500, "hola")
    b = linea("2017-01-01_0000_US_B.txt", "t__b", 20, 200, 20, 600, "adios")
    c = linea("2017-01-02_0000_US_C.txt", "t__c", 30, 300, 30, 700, "buenas")

    with tempfile.TemporaryDirectory() as tmp:
        # 1. dos filas duplicadas (mismo fichero, id y tiempos) entre 5 lineas
        tab = monta_tsv(os.path.join(tmp, "dup"), [a, b, a, c, b])
        lote = os.path.join(tmp, "lote_dup")
        p = corre(tmp, "--tsv", tab, "--lote", lote, "--hasta", "tsv")
        check("un tab con duplicados no aborta", p.returncode == 0, p.stderr[-400:])
        efectivo = lineas(os.path.join(lote, "tsv", "efectivo.tsv")) or []
        check("efectivo se queda con las filas unicas", len(efectivo) == 3, f"{len(efectivo)}")
        datos = conteos(lote)
        check("conteos dice cuantas se descartaron",
              datos.get(("tsv", "lineas_duplicadas")) == "2",
              str(datos.get(("tsv", "lineas_duplicadas"))))
        check("lineas_tsv cuenta solo las unicas",
              datos.get(("tsv", "lineas_tsv")) == "3", str(datos.get(("tsv", "lineas_tsv"))))
        dup = lineas(os.path.join(lote, "informes", "duplicados_tab.tsv")) or []
        check("los duplicados quedan listados", len(dup) == 2, f"{len(dup)}")

        # 2. --limit se aplica DESPUES de deduplicar
        lote = os.path.join(tmp, "lote_limit")
        p = corre(tmp, "--tsv", tab, "--lote", lote, "--hasta", "tsv", "--limit", "2")
        efectivo = lineas(os.path.join(lote, "tsv", "efectivo.tsv")) or []
        check("--limit recorta filas unicas, no duplicadas", len(efectivo) == 2, f"{len(efectivo)}")
        datos = conteos(lote)
        check("y el conteo de duplicados sigue ahi",
              datos.get(("tsv", "lineas_duplicadas")) == "2",
              str(datos.get(("tsv", "lineas_duplicadas"))))

        # 3. tab sin duplicados
        tab = monta_tsv(os.path.join(tmp, "limpio"), [a, b, c])
        lote = os.path.join(tmp, "lote_limpio")
        p = corre(tmp, "--tsv", tab, "--lote", lote, "--hasta", "tsv")
        check("un tab sin duplicados sigue saliendo 0", p.returncode == 0, p.stderr[-400:])
        datos = conteos(lote)
        check("y reporta 0 duplicados", datos.get(("tsv", "lineas_duplicadas")) == "0",
              str(datos.get(("tsv", "lineas_duplicadas"))))
        check("sin duplicados no hay informe",
              not os.path.isfile(os.path.join(lote, "informes", "duplicados_tab.tsv")))

    print()
    if FAIL:
        print(f"TOTAL: FAIL {', '.join(FAIL)}")
        return 1
    print("TOTAL: todo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
