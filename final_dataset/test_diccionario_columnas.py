#!/usr/bin/env python3
"""Tests de `diccionario_columnas.py`.

Lo que se prueba aquí no es el redactado: es que el diccionario **no pueda mentir por
silencio**. Su valor para una clase es que cada fórmula viene del código, y que si el
código gana una columna que nadie documentó, el generador se cae en vez de publicar un
README con un hueco. Orden de los casos:

  1. la definición específica gana a la familia general (esto estuvo roto: `mirror_error_*`
     de la familia `distances` tapaba la nota propia de `mirror_error_x_std`)
  2. los comodines del código resuelven: prefijo (`cos_phase_*`) y sufijo (`*_p05`)
  3. un contador sin clave propia (`n_cos_phase`) se enlaza a la nota donde vive
  4. una columna inventada NO se documenta: `resolver` devuelve None
  5. el texto sale literal de las constantes, no de una copia que se desincroniza
  6. sobre una entrega sintética: CSV con el esquema exacto, todas las columnas con
     fórmula, y el guard de columna desconocida petando con `sin documentar`
  7. el README que se publica como `DATASET.md` dentro de `final/` lleva portada con los numeros
     de la entrega medida (filas, columnas, version, padding, fecha del manifiesto), y el
     generador se niega a escribir dentro de la entrega que describe
  8. una entrega sin columna de voz lo confiesa (`(sin columna)`) en vez de enseñar ceros
"""
import io
import json
import os
import sys
import tempfile
import contextlib

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))


def _polymnia_por_defecto(aqui):
    """Polymnia es el padre (estamos dentro del repo) o un hermano (copia suelta)."""
    arriba = os.path.dirname(aqui)
    return arriba if os.path.exists(os.path.join(arriba, "gesture_features.py")) \
        else os.path.join(arriba, "Polymnia")


POLY = _polymnia_por_defecto(AQUI)
sys.path.insert(0, AQUI)
sys.path.insert(0, POLY)

import diccionario_columnas as dic  # noqa: E402
import gesture_features as gf        # noqa: E402

FAIL = []
CODIGO = dic.tabla_desde_codigo(gf)


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def fila(columna, tabla="final_videos"):
    return dic.resolver(tabla, columna, gf, CODIGO)


def test_la_definicion_especifica_gana_a_la_familia():
    f = fila("mirror_error_x_std")
    check("mirror_error_x_std usa su nota, no la familia distances",
          f is not None and "sample std" in f["formula_unidades"]
          and "NOT metres" not in f["formula_unidades"],
          "" if f else "sin resolver" if f is None else f["formula_unidades"][:120])
    sin_std = fila("right_speed_std")
    check("right_speed_std cae en la familia de distribución por sufijo",
          sin_std is not None and "percentile" not in sin_std["que_mide"]
          and "standard deviation" in sin_std["que_mide"],
          "" if sin_std else "sin resolver")


def test_los_comodines_del_codigo_resuelven():
    prefijo = fila("cos_phase_mean")
    check("cos_phase_mean resuelve por comodín de prefijo",
          prefijo is not None and prefijo["familia"] == "cos_phase_*",
          "" if prefijo else "sin resolver")
    sufijo = fila("right_speed_p05")
    check("right_speed_p05 resuelve por comodín de sufijo",
          sufijo is not None and "5th percentile" in sufijo["que_mide"]
          and "speed" in sufijo["que_mide"],
          "" if sufijo else "sin resolver")
    area = fila("left_bbox_area")
    check("left_bbox_area quita el prefijo de mano antes de buscar",
          area is not None and area["ambito"] == "left hand",
          "" if area else "sin resolver")


def test_un_contador_sin_clave_se_enlaza_a_su_nota():
    f = fila("n_cos_phase")
    check("n_cos_phase sale de la nota de cos_phase_ (no tiene clave propia)",
          f is not None and "n_cos_phase" in f["formula_unidades"],
          "" if f else "sin resolver")


def test_una_columna_inventada_no_se_documenta():
    check("una columna desconocida se queda sin resolver",
          fila("columna_inventada") is None)
    check("un prefijo de mano no fabrica una definición",
          fila("right_columna_inventada") is None)
    check("el prefijo de mano no tapa la búsqueda de la medida",
          fila("left_mean_speed") is not None)


def test_el_texto_es_literal_de_las_constantes():
    esperado = dic._limpiar(gf.EXTENDED_FEATURE_UNITS["hull_area"])
    f = fila("right_hull_area")
    check("hull_area cita la constante del código palabra por palabra",
          f is not None and f["formula_unidades"] == esperado,
          "" if f else "sin resolver")
    check("una columna de procedencia cita la función que la calcula",
          fila("frames_grid")["fuente"] == "gesture_features.py: _sequence_grid")


def _escribir_entrega(carpeta, columnas_videos, columnas_frames):
    pd.DataFrame(columnas_videos).to_parquet(os.path.join(carpeta, "final_videos.parquet"))
    pd.DataFrame(columnas_frames).to_parquet(os.path.join(carpeta, "final_frames.parquet"))
    with open(os.path.join(carpeta, "manifest.json"), "w", encoding="utf-8") as escritor:
        json.dump({"generated_at": "2026-10-14T11:23:49+00:00",
                   "features": {"version": "3", "extended": True},
                   "clip_id_mapping": {"requested_padding_s": 2.5, "searchterm": "prueba"},
                   "movement_speed_units_per_s": 0.5,
                   "admission_rules": {"una": "regla"}}, escritor)


def test_una_entrega_sintetica_sale_completa_y_el_guard_peta():
    videos = {"id": ["a", "b"], "people_id": [1, 1], "frames_grid": [100.0, 100.0],
              "right_mean_speed": [1.0, None], "right_n_speed": [40.0, 3.0],
              "cos_phase_mean": [0.5, None], "n_cos_phase": [22.0, 1.0],
              "indoor_outdoor": ["indoor", "indoor"], "age": [44, 30], "error": ["", ""]}
    frames = {"id": ["a", "b"], "frame": [1, 2], "right_speed": [1.0, 2.0],
              "right_confidence": [0.9, 0.1], "right_above_neck": [True, False],
              "pitch": [120.0, None], "vad": [1, 0]}
    with tempfile.TemporaryDirectory() as tmp:
        entrega = os.path.join(tmp, "final")
        os.makedirs(entrega)
        _escribir_entrega(entrega, videos, frames)
        csv = os.path.join(tmp, "d.csv")
        md = os.path.join(tmp, "d.md")
        with contextlib.redirect_stdout(io.StringIO()):
            dic.main(["--final", entrega, "--csv", csv, "--md", md,
                      "--polymnia_root", POLY])
        d = pd.read_csv(csv)
        check("el CSV sale con el esquema exacto y en orden",
              list(d.columns) == ["tabla", "columna", "ambito", "familia", "que_mide",
                                  "formula_unidades", "tipo", "no_nulos", "pct_no_nulos",
                                  "minimo", "mediana", "maximo", "fuente"], str(list(d.columns)))
        check("todas las columnas de la entrega salen documentadas", len(d) == 17, f"{len(d)} filas")
        check("ninguna columna se publica con la fórmula vacía",
              int(d.formula_unidades.isna().sum()) == 0,
              str(list(d[d.formula_unidades.isna()].columna)))
        check("el CSV distingue NULL de cero: right_mean_speed tiene 1 de 2",
              d.set_index("columna").loc["right_mean_speed", "pct_no_nulos"] == 50.0)
        check("el markdown menciona todas las columnas de la entrega",
              all(f"`{c}`" in open(md, encoding="utf-8").read()
                  for c in list(videos) + list(frames)))

        videos_rollos = dict(videos)
        videos_rollos["right_coseno_de_la_cabeza"] = [1.0, 2.0]
        _escribir_entrega(entrega, videos_rollos, frames)
        os.remove(csv)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                dic.main(["--final", entrega, "--csv", csv,
                          "--polymnia_root", POLY])
            check("una columna nueva sin documentar hace fallar al generador", False,
                  "salió 0: habría publicado un README incompleto")
        except SystemExit as salida:
            check("una columna nueva sin documentar hace fallar al generador",
                  "undocumented" in str(salida) and "right_coseno_de_la_cabeza" in str(salida),
                  str(salida)[:120])
        check("cuando falla no deja el CSV a medias escrito", not os.path.exists(csv))


def test_el_readme_lleva_portada_con_los_numeros_de_esta_entrega():
    """El README que se publica dentro de final/ trae los numeros de la entrega medida."""
    videos = {"id": ["a", "b"], "people_id": [1, 1], "frames_grid": [100.0, 100.0],
              "right_mean_speed": [1.0, 2.0], "right_n_speed": [40.0, 30.0],
              "left_n_speed": [40.0, 30.0], "right_max_speed": [1.0, 12.0]}
    frames = {"id": ["a", "b"], "frame": [1, 2], "right_speed": [1.0, 2.0],
              "pitch": [120.0, 0.0]}
    raiz = POLY
    with tempfile.TemporaryDirectory() as tmp:
        entrega = os.path.join(tmp, "final")
        os.makedirs(entrega)
        _escribir_entrega(entrega, videos, frames)
        readme = os.path.join(tmp, "informes", "README_entrega.md")
        with contextlib.redirect_stdout(io.StringIO()):
            dic.main(["--final", entrega, "--readme", readme, "--polymnia_root", raiz])
        texto = open(readme, encoding="utf-8").read()
        cabecera = texto[:1500]
        check("la portada dice filas y columnas de las dos tablas",
              "2 clips x 7 columns" in cabecera and "2 frames x 4 columns" in cabecera,
              [linea for linea in cabecera.splitlines() if "columns |" in linea])
        check("la portada trae la version de features y el padding del manifiesto",
              "3 (extended block: included)" in cabecera
              and "2.5 s, searchterm `prueba`" in cabecera
              and "2026-10-14T11:23:49+00:00" in cabecera,
              [linea for linea in cabecera.splitlines() if "features |" in linea or "padding" in linea])
        check("el README avisa de que se genera y no se edita a mano",
              "is generated" in cabecera and "regenerate" in cabecera)
        check("debajo de la portada sigue estando el diccionario",
              "`right_max_speed`" in texto and "Definitions, exactly as they are in the code" in texto)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                dic.main(["--final", entrega, "--readme", os.path.join(entrega, "DATASET.md"),
                          "--polymnia_root", raiz])
            check("un --readme dentro de final/ se rechaza", False, "lo escribio igualmente")
        except SystemExit as salida:
            check("un --readme dentro de final/ se rechaza (romperia el paquete hashado)",
                  "atomic" in str(salida), str(salida)[:90])


def test_un_readme_sin_voz_no_ensena_ceros():
    """Si la entrega no tiene columnas de voz, el README lo dice: no muestra '0 de 0'."""
    videos = {"id": ["a"], "people_id": [1], "frames_grid": [100.0], "right_mean_speed": [1.0]}
    frames = {"id": ["a"], "frame": [1], "right_speed": [1.0]}
    with tempfile.TemporaryDirectory() as tmp:
        entrega = os.path.join(tmp, "final")
        os.makedirs(entrega)
        _escribir_entrega(entrega, videos, frames)
        readme = os.path.join(tmp, "README.md")
        with contextlib.redirect_stdout(io.StringIO()):
            dic.main(["--final", entrega, "--readme", readme,
                      "--polymnia_root", POLY])
        texto = open(readme, encoding="utf-8").read()
        bloque = texto[texto.find("## The voice"):][:1400]
        check("sin columnas de voz el README lo dice en vez de enseñar ceros",
              "does not carry voice columns" in bloque and "0 of 0" not in bloque
              and "(no column)" not in bloque,
              [linea for linea in bloque.splitlines() if "|" in linea][:3])


def test_una_entrega_sin_bloque_extendido_no_peta():
    """Una entrega FEATURE_VERSION 2 no tiene n_speed ni max por mano: ni petar ni mentir."""
    videos = {"id": ["a"], "people_id": [1], "frames_grid": [100.0],
              "right_mean_speed": [1.0], "right_n_valid": [80.0], "left_n_valid": [70.0],
              "mean_pair_separation": [1.5], "age": [40]}
    frames = {"id": ["a"], "frame": [1], "right_speed": [1.0], "right_confidence": [0.8]}
    with tempfile.TemporaryDirectory() as tmp:
        entrega = os.path.join(tmp, "final")
        os.makedirs(entrega)
        _escribir_entrega(entrega, videos, frames)
        md = os.path.join(tmp, "d.md")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                dic.main(["--final", entrega, "--md", md,
                          "--polymnia_root", POLY])
        except Exception as e:  # noqa: BLE001
            check("una entrega sin bloque extendido se documenta sin petar", False, repr(e))
            return
        texto = open(md, encoding="utf-8").read()
        check("una entrega sin bloque extendido se documenta sin petar", True)
        check("sin contadores extendidos no inventa el numero de clips sustentados",
              "0 of 1 clips" not in texto and "right_n_valid" in texto,
              [linea for linea in texto.splitlines() if "support" in linea or "of 1 clips" in linea][:2])


def _readme_de(frames_extra, videos_extra=None):
    """Entrega sintetiva completa -> (texto del README, ruta del CSV)."""
    videos = {"id": ["a"], "people_id": [1], "frames_grid": [100.0], "right_mean_speed": [1.0]}
    videos.update(videos_extra or {})
    frames = {"id": ["a"] * 3, "frame": [1, 2, 3], "right_speed": [1.0, 2.0, 3.0]}
    frames.update(frames_extra)
    tmp = tempfile.mkdtemp()
    entrega = os.path.join(tmp, "final")
    os.makedirs(entrega)
    _escribir_entrega(entrega, videos, frames)
    readme, csv = os.path.join(tmp, "README.md"), os.path.join(tmp, "d.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        dic.main(["--final", entrega, "--csv", csv, "--readme", readme,
                  "--polymnia_root", POLY])
    return open(readme, encoding="utf-8").read(), csv


VOZ_ARREGLADA = {"pitch": [120.0, 130.0, None], "intensity": [60.0, 62.0, 58.0],
                 "harmonicity": [8.0, 9.0, -200.0], "formant.1_frequency": [500.0, 510.0, 495.0],
                 "voice_present": [True, True, False]}
VOZ_SPLINE = {"pitch": [120.0, 0.0, 900.0], "intensity": [60.0, -130.0, 58.0],
              "harmonicity": [8.0, -400.0, 7.0], "formant.1_frequency": [500.0, -12.0, 495.0]}


def test_un_lote_con_voice_present_se_documenta_como_cadena_nueva():
    """Con `voice_present` y todo dentro de cotas, el README habla de la cadena nueva."""
    texto, _ = _readme_de(VOZ_ARREGLADA)
    check("el regimen medido es la cadena con tramos",
          "| voice chain regime | **segments** |" in texto,
          [l for l in texto.splitlines() if "regime" in l][:1])
    check("la seccion nueva manda abrir la puerta con voice_present",
          "voice_present" in texto and "Open the door" in texto)
    check("la seccion nueva no se describe a si misma como cadena antigua",
          "comes out of the old chain" not in texto)
    check("el -200 dB de Praat no sale contado como valor imposible",
          "below -200 dB | 0 frames" in texto,
          [l for l in texto.splitlines() if "harmonic" in l][:1])
    check("voice_present se cuenta, no se describe de oido",
          "2 of 3 frames with a measured track" in texto,
          [l for l in texto.splitlines() if "measured track" in l][:1])


def test_un_lote_sin_voice_present_sale_como_cadena_antigua():
    """Sin `voice_present`, y con valores fuera de cotas, el README lo dice con las cifras."""
    texto, _ = _readme_de(VOZ_SPLINE)
    check("sin voice_present el regimen medido es la spline",
          "| voice chain regime | **spline** |" in texto,
          [l for l in texto.splitlines() if "regime" in l][:1])
    check("la seccion antigua nombra la cadena antigua y apunta al arreglo",
          "comes out of the old chain" in texto and "upstream_parole" in texto)
    check("las cotas se cuentan sobre estos datos: pitch fuera en 2, HNR bajo el suelo en 1",
          "outside 60-600 Hz | 2 frames" in texto
          and "below -200 dB | 1 frames" in texto
          and "outside 0-5500 Hz | 1 values" in texto,
          [l for l in texto.splitlines() if "frames |" in l or "values |" in l][:4])


def test_un_lote_con_restos_de_spline_no_se_lla_ma_arreglado():
    """`voice_present` esta pero queda un valor fuera de cotas: regimen «tramos con restos»."""
    voz = dict(VOZ_ARREGLADA, harmonicity=[8.0, -400.0, 9.0])
    texto, _ = _readme_de(voz)
    check("un valor fuera de cotas impide llamarse arreglada",
          "**segments with leftovers**" in texto,
          [l for l in texto.splitlines() if "regime" in l][:1])


def test_voice_present_sale_documentada_con_su_fuente():
    """La columna nueva tiene fila, familia y fuente en el CSV; no solo menciones sueltas."""
    _, csv = _readme_de(VOZ_ARREGLADA)
    d = pd.read_csv(csv).set_index("columna")
    check("voice_present esta documentada con su familia y su fuente",
          d.loc["voice_present", "familia"] == "voice (Praat)"
          and "process_prosody.R" in d.loc["voice_present", "fuente"],
          f'{d.loc["voice_present", "familia"]} | {d.loc["voice_present", "fuente"]}')
    check("pitch cita el alineador por tramos, no la spline",
          "prosody_align.R" in d.loc["pitch", "fuente"], d.loc["pitch", "fuente"])


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[fn]()
        except Exception as e:  # noqa: BLE001
            check(f"{fn} (excepción)", False, repr(e))
    print("\nTOTAL:", "FAIL " + ", ".join(FAIL) if FAIL else "todo OK")
    sys.exit(1 if FAIL else 0)
