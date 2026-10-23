#!/usr/bin/env python3
"""Tests de la REGLA DE ADMISIÓN de muñecas, que vive en dos sitios a la vez.

La criba (`wrist_prescreen.py`) escribe `munecas.csv`; `pipeline_lote.sh` lee ese CSV y
decide qué clips van a `videos/1_person`, que es lo que ve OpenPose. La regla se puede
romper sin tocar la otra mitad: si el pipeline filtra por una columna que la criba no
escribe, pandas suelta un AttributeError en mitad de una tanda de GPU; y si alguien
re-significa un flag existente, la admisión se ensancha o se estrecha sin que nadie lo
haya pedido — y el ancho de admisión es lo que decide qué clips hay en el dataset.

Aquí se prueba el contrato entre las dos mitades, no cada mitad por separado (eso ya lo
prueban `test_wrist_prescreen.py` y `test_auditoria_criba.py`). Sin YOLO ni GPU: la única
pasada de pipeline que se ejecuta es la de argument Parsing, que muere antes de tocar nada.

Casos que pinchan:
  1. best_wrist             -> una muñeca vale; un NaN no se traga la que sí se vio
  2. el CSV trae sp_any      -> sin columna no hay admisión por sp_any
  3. orden_de_fila           -> rankea la métrica que decide, y NaN va al final
  4. el fallback de pandas   -> equivale a best_wrist, con filas NaN incluidas
  5. --min-sp-both           -> se niega, con rc!=0 y SIN crear el lote
  6. --min-sp-any            -> existe y sale en --help
  7. el filtro del pipeline  -> admisión por sp_any, no por sp_both
"""
import os
import subprocess
import sys
import tempfile

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(AQUI)
PIPELINE = os.path.join(REPO, "final_dataset", "pipeline_lote.sh")
sys.path.insert(0, AQUI)

from wrist_prescreen import best_wrist, orden_de_fila, write_csv  # noqa: E402

FAIL = []
TOTAL = 0


def check(nombre, ok, detalle=""):
    global TOTAL
    TOTAL += 1
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def fila(video, sp_any=None, sp_both=None):
    nan = float("nan")
    return {"video": video, "sp_any": nan if sp_any is None else sp_any,
            "sp_both": nan if sp_both is None else sp_both}


# ------------------------------------------------------- la métrica que decide ----

def test_una_muneca_vale():
    """El motivo de la regla: ver UNA muñeca sí produce features de muñeca."""
    check("best_wrist toma la mejor", best_wrist(0.6667, 1.0) == 1.0)
    check("una muneca NaN no se traga la que si se vio", best_wrist(float("nan"), 0.2) == 0.2,
          str(best_wrist(float("nan"), 0.2)))
    check("las dos NaN siguen siendo NaN (no es un 0 disfrazado)",
          best_wrist(float("nan"), float("nan")) != best_wrist(float("nan"), float("nan")))
    check("0,0 no se confunde con sin dato", best_wrist(0.0, 0.0) == 0.0)
    # El clip que motivó el cambio, con sus numeros reales de FREQ025: sp_both 0,667 (fuera
    # de la regla vieja) pero una muneca perfecta. OpenPose le dio 91,9 % / 100 %.
    check("el KABC 1087 entra con sp_any y estaba fuera con sp_both",
          best_wrist(0.6667, 1.0) >= 0.6 and 0.6667 < 0.8)


def test_el_csv_de_la_criba_es_el_contrato():
    """El pipeline filtra por sp_any; si la criba no la escribe, la admisión no existe."""
    with tempfile.TemporaryDirectory() as td:
        ruta = os.path.join(td, "munecas.csv")
        write_csv([{"video": "a.mp4", "path": "/x/a.mp4", "reported_frames": 100,
                    "n_sampled": 15, "right": 0.5, "left": 0.1, "both": 0.1,
                    "sp_right": 0.6667, "sp_left": 1.0, "sp_both": 0.6667,
                    "sp_any": best_wrist(0.6667, 1.0), "max_people": 1, "status": "ok"}], ruta)
        with open(ruta, encoding="utf-8") as handle:
            cabecera = handle.readline().strip().split(",")
        check("munecas.csv trae la columna por la que se admite", "sp_any" in cabecera,
              str(cabecera))
        check("sp_both sigue en el CSV (ranking y evidencia de la regla anterior)",
              "sp_both" in cabecera, str(cabecera))
        d = pd.read_csv(ruta)
        check("el valor escrito es la mejor muneca, no otra cosa",
              float(d.loc[0, "sp_any"]) == 1.0, str(d.loc[0, "sp_any"]))


def test_el_orden_lo_marca_la_metrica_que_decide():
    filas = [fila("empataA", 0.0, 0.0), fila("kabc", 1.0, 0.6667), fila("kimmel", 0.9333, 0.9333),
             fila("corrupto", None, None), fila("empataB", 0.0, 0.0)]
    orden = [f["video"] for f in sorted(filas, key=orden_de_fila)]
    check("sp_any manda en el ranking (el KABC va primero)", orden[0] == "kabc", str(orden))
    check("un score NaN va al FINAL, no al principio (el corrupto no encabeza)",
          orden[-1] == "corrupto", str(orden))
    check("a igual sp_any desempata sp_both (mejor documentado antes)",
          orden[1] == "kimmel" and orden[2] == "empataA" and orden[3] == "empataB", str(orden))


def test_el_fallback_pandas_equivale_a_best_wrist():
    """pipeline_lote.sh deriva sp_any de sp_right/sp_left si el CSV es viejo.

    No se puede importar ese trozo de python (va dentro del heredoc del shell), así que se
    prueba la MISMA expresión contra la función de referencia. Si alguna de las dos cambia de
    significado, esto lo nota: dos definiciones de la métrica que admite son dos reglas.
    """
    nan = float("nan")
    d = pd.DataFrame({"sp_right": [0.6667, nan, nan, 0.0, 1.0], "sp_left": [1.0, 0.2, nan, 0.0, nan]})
    derivado = d[["sp_right", "sp_left"]].max(axis=1)
    esperado = [best_wrist(r, l) for r, l in zip(d.sp_right, d.sp_left)]
    obtenido = list(derivado)
    coincide = all((a != a and b != b) or a == b for a, b in zip(obtenido, esperado))
    check("el max(axis=1) del pipeline coincide con best_wrist fila por fila", coincide,
          f"{obtenido} vs {esperado}")
    check("una muneca NaN no borra la otra tampoco en pandas", float(derivado[1]) == 0.2,
          str(derivado[1]))
    check("las dos NaN siguen NaN en pandas (no se vuelven 0)", derivado[2] != derivado[2],
          str(derivado[2]))


# ----------------------------------------------------------------- los flags ----

def test_el_flag_viejo_se_niega():
    """`--min-sp-both` no se re-significa: aceptarlo ensancharía la admisión en silencio."""
    if not os.path.isfile(PIPELINE):
        check("existe pipeline_lote.sh que probar", False, PIPELINE)
        return
    with tempfile.TemporaryDirectory() as td:
        lote = os.path.join(td, "lote_no_deberia_existir")
        r = subprocess.run(["bash", PIPELINE, "--lote", lote, "--min-sp-both", "0.8"],
                           capture_output=True, text=True, timeout=60)
        salida = r.stdout + r.stderr
        check("--min-sp-both sale con rc!=0", r.returncode != 0, f"rc={r.returncode}")
        check("--min-sp-both explica la alternativa (no solo dice 'opcion desconocida')",
              "--min-sp-any" in salida, salida[:200])
        check("--min-sp-both no deja un lote a medias", not os.path.exists(lote), lote)


def test_el_flag_nuevo_existe():
    if not os.path.isfile(PIPELINE):
        check("existe pipeline_lote.sh que probar", False, PIPELINE)
        return
    r = subprocess.run(["bash", PIPELINE, "--help"], capture_output=True, text=True, timeout=60)
    check("--min-sp-any aparece en la ayuda", "--min-sp-any" in r.stdout, r.stdout[:200])
    check("la ayuda dice el defecto con el que se admite", "0.6" in r.stdout.split("--min-sp-any")[1][:200],
          "no se ve 0.6 junto a --min-sp-any")


def test_la_admision_del_pipeline_filtra_por_sp_any():
    """Pin del heredoc: que nadie vuelva a admitir por sp_both ni a leer una columna muerta.

    Es un pin textual, no un test de comportamiento: el bloque de admisión vive dentro del
    shell. Su comportamiento si se prueba de verdad (los 20 clips de FREQ025 admitiendo 5
    con la regla nueva), y este pin evita que un edit posterior lo deshaga sin querer.
    """
    with open(PIPELINE, encoding="utf-8") as handle:
        texto = handle.read()
    # Marcador exclusivo del bloque de admision: `OUT_DIR="$OUT"` a secas aparece antes (lo
    # usa la etapa del TSV), y partir por eso devolvía el bloque equivocado -> el check de
    # «no admite por sp_both» pasaba vacio. `MIN="$MIN_SP_ANY"` solo lo tiene la admision.
    marcador = 'OUT_DIR="$OUT" MIN="$MIN_SP_ANY"'
    bloque = texto.split(marcador, 1)[1].split("\nPY\n", 1)[0] if marcador in texto else ""
    check("hay un bloque de admision localizable", bool(bloque), "no se encuentra el marcador")
    check("el bloque capturado ES el de admision, no otra etapa", "munecas.csv" in bloque,
          bloque[:120])
    check("el pipeline admite por sp_any", "d.sp_any >= min_sp" in bloque, bloque[:200])
    check("el pipeline NO admite por sp_both", "d.sp_both >=" not in bloque, "sp_both sigue decidiendo")
    check("el bloque deriva sp_any si el CSV es viejo", '"sp_any" not in d.columns' in bloque)


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[fn]()
        except Exception as e:  # noqa: BLE001 - un test que peta es un fallo, no un silencio
            check(f"{fn} (excepcion)", False, repr(e))
    if FAIL:
        print(f"\nTOTAL: {TOTAL - len(FAIL)} / {TOTAL}   fallos: {', '.join(FAIL)}")
        sys.exit(1)
    print(f"\nTOTAL: {TOTAL} / {TOTAL}")
    sys.exit(0)
