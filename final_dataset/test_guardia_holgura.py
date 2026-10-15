#!/usr/bin/env python3
"""Tests de la guardia de holgura de final_dataset.py.

La holgura (`--offset`) no se puede deducir del nombre del clip: el descargador escribe el
nombre con la holgura que us6, as que emparejar por id solo confirma lo que uno ya cree.
Cuando el tab de una pasada no es el que se descargo, `match_tab_to_clips` localiza el clip
por centro de ventana (invariante a la holgura) y reporta la holgura REAL. Este test cubre
que esa medida se usa para NEGARSE a publicar, y cuales son las dos salidas honestas: pedir
la holgura que esta en disco, o pasar el escape explícito.

Mide ademas que el camino feliz (tab y clips con la misma holgura) no se queja: una guardia
que salta siempre no es una guardia, es ruido.

Salida: una linea por comprobacion; exit 0 si todas pasan, 1 si alguna falla.
"""

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import final_dataset as fd  # noqa: E402

fallan = 0
hechas = 0


def chequea(nombre, condicion, detalle=""):
    global fallan, hechas
    hechas += 1
    if condicion:
        print(f"[ok ] {nombre}")
    else:
        fallan += 1
        print(f"[FALLO] {nombre}" + (f"  -- {detalle}" if detalle else ""))


def tab_de(dirs, filas, offset):
    """Escribe un TSV estilo NewsScape y lo lee con el lector de verdad del modulo."""
    ruta = os.path.join(dirs, "clips.txt")
    with open(ruta, "w", encoding="utf-8") as fh:
        for programa, inicio, fin in filas:
            d = 1000 if programa.startswith("2017") else 100
            fh.write(f"{programa}.txt\tt__abc\t{int(inicio)}\t{int(round((inicio % 1) * d))}\t"
                     f"{int(fin)}\t{int(round((fin % 1) * d))}\tcontexto\n")
    return fd.read_tab(ruta, offset, "prueba")


def clips_como_el_tab(tab, holgura_real):
    """Nombres de clip como los escribio el descargador con `holgura_real`."""
    return [f"{r.program}_{round(r.tab_start_raw - holgura_real, 3)}_"
            f"{round(r.tab_end_raw + holgura_real, 3)}_prueba" for r in tab.itertuples(index=False)]


def main():
    filas = [("2017-01-02_0600_US_CNN_The_Eighties", 900.000, 900.320),
             ("2017-03-03_0700_US_KABC_News", 400.004, 400.010)]

    with tempfile.TemporaryDirectory() as tmp:
        # tab pensado con 2.0 s, clips bajados realmente con 2.5 s
        tab = tab_de(tmp, filas, offset=2.0)
        emparejados, sin = fd.match_tab_to_clips(tab, clips_como_el_tab(tab, 2.5), "prueba")
        chequea("el tab de 2.0 localiza los clips de 2.5 por centro de ventana",
                len(emparejados) == 2 and set(emparejados["id_source"]) == {"window_center"},
                emparejados.to_string())
        medidas = sorted(set(emparejados["processed_padding_s"]))
        chequea("y reporta la holgura real, no la pedida", medidas == [2.5], medidas)

        try:
            fd.guarda_holgura(emparejados, 2.0)
            chequea("con holgura mezclada se niega a publicar", False, "no lanzo nada")
        except fd.FinalDatasetError as exc:
            texto = str(exc)
            chequea("con holgura mezclada se niega a publicar", True)
            chequea("el error dice la salida barata (reconstruir con la medida)",
                    "--offset 2.5" in texto and "--allow-padding-mismatch" in texto, texto[:200])

        salida = io.StringIO()
        try:
            with redirect_stdout(salida):
                fd.guarda_holgura(emparejados, 2.0, permite=True)
            chequea("--allow-padding-mismatch publica, pero deja aviso en la salida",
                    "aviso" in salida.getvalue())
        except fd.FinalDatasetError as exc:
            chequea("--allow-padding-mismatch publica, pero deja aviso en la salida",
                    False, f"lanzo: {exc}")

        try:
            fd.guarda_holgura(emparejados, 2.5)
            chequea("si pides la holgura que esta en disco, no hay nada que decir", True)
        except fd.FinalDatasetError as exc:
            chequea("si pides la holgura que esta en disco, no hay nada que decir",
                    False, f"lanzo: {exc}")

        # camino feliz: tab y clips con la misma holgura -> id exacto, y la guardia calla
        tab2 = tab_de(tmp, filas, offset=2.5)
        emparejados2, sin2 = fd.match_tab_to_clips(tab2, clips_como_el_tab(tab2, 2.5), "prueba")
        chequea("tab y clips con la misma holgura emparejan por id exacto",
                set(emparejados2["id_source"]) == {"exact"}, emparejados2["id_source"].tolist())
        try:
            fd.guarda_holgura(emparejados2, 2.5)
            chequea("el camino feliz no salta (si no, seria ruido, no guardia)", True)
        except fd.FinalDatasetError as exc:
            chequea("el camino feliz no salta (si no, seria ruido, no guardia)",
                    False, f"lanzo: {exc}")

        # el divisor del msec manda: un programa pre-2017 lleva centisegundos (/100)
        vieja = tab_de(tmp, [("2016-06-06_0600_US_MSNBC_Live", 500.00, 500.40)], offset=2.5)
        fila = vieja.iloc[0]
        chequea("programa pre-2017: el msec se divide por 100, no por 1000",
                abs(fila.tab_start_raw - 500.00) < 1e-6 and abs(fila.tab_end_raw - 500.40) < 1e-6,
                f"{fila.tab_start_raw}..{fila.tab_end_raw}")
        chequea("y su id lleva la holgura restada y sumada",
                fila.id.endswith("_497.5_502.9_prueba"), fila.id)

    parser = fd.build_parser()
    args = parser.parse_args(["--db_name", "d", "--tab", "t", "--output_folder", "o"])
    chequea("por defecto pide 2.0 s por lado", args.offset == 2.0, args.offset)
    chequea("por defecto la guardia esta activa", args.allow_padding_mismatch is False)
    args2 = parser.parse_args(["--db_name", "d", "--tab", "t", "--output_folder", "o",
                               "--allow-padding-mismatch"])
    chequea("la bandera existe y se activa", args2.allow_padding_mismatch is True)

    print(f"\nTOTAL: {hechas - fallan}/{hechas} OK")
    return 1 if fallan else 0


if __name__ == "__main__":
    sys.exit(main())
