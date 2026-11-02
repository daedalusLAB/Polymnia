#!/usr/bin/env python3
"""`read_tab` y las filas repetidas del tab.

Los tabs de NewsScape traen filas duplicadas exactas (~1 %): mismo programa, mismo id de
clip y mismos tiempos. La etapa 1 ya las quita; este test fija lo que hace la etapa final
con un `efectivo.tsv` antiguo (un lote reanudado):

  1. fila repetida exacta -> se usa una vez (aviso, sin abortar)
  2. mismo id con filas distintas (mismo programa y tiempos, otro `t__...`) -> aborta:
     eso no es una repetición, es un tab mal montado
  3. mismos id de clip con tiempos distintos -> dos filas normales, sin aviso
"""
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "final_dataset"))
sys.path.insert(0, REPO)

import final_dataset as fd  # noqa: E402

FAIL = []


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def escribe(tmp, filas):
    ruta = os.path.join(tmp, "tab.txt")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("".join(filas))
    return ruta


def fila(programa, clip_id, start_s, start_ms, end_s, end_ms, texto="hola"):
    return f"{programa}\t{clip_id}\t{start_s}\t{start_ms}\t{end_s}\t{end_ms}\t{texto}\n"


def main():
    prog = "2017-01-01_0000_US_CNN_Show.txt"
    with tempfile.TemporaryDirectory() as tmp:
        # 1. repetida exacta
        tab = escribe(tmp, [fila(prog, "t__a", 10, 100, 10, 500),
                            fila(prog, "t__a", 10, 100, 10, 500)])
        df = fd.read_tab(tab, 2.0, "freqX")
        check("la fila repetida se usa una vez", len(df) == 1, f"{len(df)} filas")
        check("y el id es el esperado", df["id"].iloc[0].endswith("_freqX"), df["id"].iloc[0])

        # 2. mismo id de clip, filas distintas -> tab mal montado
        tab = escribe(tmp, [fila(prog, "t__a", 10, 100, 10, 500),
                            fila(prog, "t__x", 10, 100, 10, 500)])
        try:
            fd.read_tab(tab, 2.0, "freqX")
            check("dos filas distintas con el mismo id abortan", False, "no levantó error")
        except fd.FinalDatasetError as exc:
            check("dos filas distintas con el mismo id abortan", "mismo id" in str(exc), str(exc))

        # 3. tiempos distintos -> dos clips
        tab = escribe(tmp, [fila(prog, "t__a", 10, 100, 10, 500),
                            fila(prog, "t__a", 10, 200, 10, 600)])
        df = fd.read_tab(tab, 2.0, "freqX")
        check("tiempos distintos son dos clips", len(df) == 2, f"{len(df)} filas")

    print()
    if FAIL:
        print(f"TOTAL: FAIL {', '.join(FAIL)}")
        return 1
    print("TOTAL: todo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
