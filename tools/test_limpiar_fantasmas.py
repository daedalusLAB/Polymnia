#!/usr/bin/env python3
"""Tests de `limpiar_fantasmas.py` con trayectorias conocidas.

Casos que pinchan el comportamiento, en el mismo orden en que se midió en el corpus:
  1. hablante + blob de 2-3 puntos sin torso  -> se tira el blob, se queda el hablante
  2. dos cuerpos reales (con torso, sin negro)-> no se tira NADA (eso es admisión)
  3. hablante + esqueleto sobre el negro      -> se tira por la señal de negro
  4. frames de un solo candidato              -> copiados byte a byte
  5. frames sin candidatos                    -> copiados, sin petar
  6. --criterio nada                          -> informa, no tira
  7. los JSON originales                      -> intactos siempre
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(AQUI, "limpiar_fantasmas.py")
FAIL = []


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def cuerpo(puntos):
    """Un `pose_keypoints_2d` de BODY_25 con los puntos indicados en {i: (x, y, conf)}."""
    v = [0.0] * 75
    for i, (x, y, c) in puntos.items():
        v[3 * i:3 * i + 3] = [x, y, c]
    return v


def hablante(x=100.0, conf=0.8):
    """12 puntos fiables con los dos hombros: el tipo de cuerpo que da ASD."""
    return cuerpo({0: (x, 40, conf), 1: (x, 60, conf), 2: (x - 20, 62, conf), 5: (x + 20, 62, conf),
                   3: (x - 30, 90, conf), 6: (x + 30, 90, conf), 4: (x - 35, 120, conf),
                   7: (x + 35, 120, conf), 8: (x - 22, 140, conf), 11: (x + 22, 140, conf),
                   9: (x - 25, 170, conf), 12: (x + 25, 170, conf)})


def blob(x=250.0):
    """Candidato de artefacto: nariz y polvo, sin hombros ni muñecas (2-3 puntos)."""
    return cuerpo({0: (x, 60, 0.35), 1: (x, 70, 0.30), 15: (x + 3, 74, 0.25)})


def sobre_negro_cuerpo(x=290.0, conf=0.55):
    """7 puntos con hombros, pero colocados sobre el negro de la máscara (x > 260)."""
    return cuerpo({0: (x, 60, conf), 1: (x, 80, conf), 2: (x - 10, 82, conf), 5: (x + 10, 82, conf),
                   3: (x - 14, 105, conf), 6: (x + 14, 105, conf), 4: (x - 16, 130, conf)})


def frame(personas):
    return {"version": "1.5.1", "people": [{"pose_keypoints_2d": p} for p in personas]}


def monta_lote(frames, nombre="clip_test", con_video=True):
    """Reproduce el layout real: <lote>/dataset/OpenPose/<clip>/JSON_FILES y videos/masked."""
    lote = tempfile.mkdtemp(prefix="limpiar-")
    jd = os.path.join(lote, "dataset", "OpenPose", nombre, "JSON_FILES")
    os.makedirs(jd)
    if con_video:
        vdir = os.path.join(lote, "videos", "masked")
        os.makedirs(vdir)
        w = cv2.VideoWriter(os.path.join(vdir, nombre + ".mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 29.97, (320, 240))
        for _ in range(max(len(frames), 1)):
            img = np.full((240, 320, 3), 120, np.uint8)   # cuadro visible hasta x=260
            img[:, 260:] = 0                              # de ahí en adelante, la máscara
            w.write(img)
        w.release()
    for i, personas in enumerate(frames):
        with open(os.path.join(jd, f"{nombre}_{i:012d}_keypoints.json"), "w") as fh:
            json.dump(frame(personas), fh)
    return lote, jd


def sale(texto, patron):
    """El informe es texto alineado: se comprueba con patrón, no con literales."""
    return re.search(patron, texto) is not None


def corre(lote, *extra):
    r = subprocess.run([sys.executable, SCRIPT, "--lote", lote, *extra],
                       capture_output=True, text=True)
    return r


def lee(jd_dir):
    out = {}
    for n in sorted(os.listdir(jd_dir)):
        out[n] = json.load(open(os.path.join(jd_dir, n)))
    return out


# ------------------------------------------------------------------ 1. blob ----

def test_tira_el_blob_y_conserva_el_hablante():
    lote, jd = monta_lote([[hablante()], [hablante(), blob()], [hablante(), blob()]])
    r = corre(lote)
    limpio = os.path.join(os.path.dirname(jd), "JSON_FILES_limpio")
    data = lee(limpio)
    n_personas = [len(v["people"]) for v in data.values()]
    check("blob sin torso tirado: todos los frames acaban con 1 candidato",
          n_personas == [1, 1, 1], n_personas)
    check("el candidato que queda es el hablante (12 puntos fiables)",
          all(len(p["pose_keypoints_2d"]) == 75 and
              sum(1 for j in range(25) if p["pose_keypoints_2d"][3 * j + 2] >= 0.2) == 12
              for v in data.values() for p in v["people"]))
    check("el informe dice 2 candidatos tirados por falta de torso",
          sale(r.stdout, r"tirados\s+2") and "sin torso" in r.stdout and
          sale(r.stdout, r"amb\s+2/3") and sale(r.stdout, r"podados\s+2"), r.stdout[-260:])
    shutil.rmtree(lote)


# --------------------------------------------------- 2. dos personas reales ----

def test_no_toca_dos_cuerpos_reales():
    """Segunda persona con torso y fuera del negro: eso es admisión, no limpieza."""
    lote, jd = monta_lote([[hablante(), hablante(200.0)], [hablante(), hablante(200.0)]])
    r = corre(lote)
    limpio = os.path.join(os.path.dirname(jd), "JSON_FILES_limpio")
    n_personas = [len(v["people"]) for v in lee(limpio).values()]
    check("dos cuerpos reales se dejan los dos (no decide esta herramienta)",
          n_personas == [2, 2], n_personas)
    check("el informe separa ambigüedad (2) de poda (0)",
          sale(r.stdout, r"amb\s+2/2") and sale(r.stdout, r"podados\s+0")
          and sale(r.stdout, r"tirados\s+0"), r.stdout[-260:])
    shutil.rmtree(lote)


def bulto(x=200.0, conf=0.5):
    """3 puntos con un hombro, sobre contenido visible.

    Así se ve a la vez una segunda presentadora real de lejos (run50: 2-5 kp,
    0 % sobre negro, 99 % de los frames) y el artefacto de la máscara de ASD
    (Tavis 05-20: 3 kp, 0 % sobre negro, 4 % de los frames).
    """
    return cuerpo({0: (x, 60, conf), 2: (x - 12, 66, conf), 6: (x + 14, 88, conf)})


# --------------------------------------- persistencia: persona real vs ruido ----

def test_segunda_persona_persistente_no_se_poda_aunque_escasa():
    """Presente en la mayoría de los frames: es una segunda persona, no ruido."""
    frames = [[hablante(), bulto()] if i < 8 else [hablante()] for i in range(10)]
    lote, jd = monta_lote(frames)
    r = corre(lote)
    limpio = os.path.join(os.path.dirname(jd), "JSON_FILES_limpio")
    n = [len(v["people"]) for v in lee(limpio).values()]
    check("bulto escaso PERSISTENTE (80% de frames) se conserva: ambigüedad sostenida",
          n[:8] == [2] * 8 and n[8:] == [1, 1] and "sostenida" in r.stdout
          and sale(r.stdout, r"tirados\s+0"), (n[:3], r.stdout[-300:]))
    shutil.rmtree(lote)


def test_artefecto_transitorio_escaso_si_se_poda():
    """El mismo bulto en el 10 % de los frames es ruido de la máscara: se poda."""
    frames = [[hablante(), bulto()] if i in (3, 17) else [hablante()] for i in range(20)]
    lote, jd = monta_lote(frames)
    r = corre(lote)
    limpio = os.path.join(os.path.dirname(jd), "JSON_FILES_limpio")
    n = [len(v["people"]) for v in lee(limpio).values()]
    check("bulto escaso TRANSITORIO (10% de frames) podado: todo a un candidato",
          n == [1] * 20 and "transitoria" in r.stdout, (sorted(set(n)), r.stdout[-300:]))
    check("y el motivo dice que fue por persistencia, no por la forma del candidato",
          sale(r.stdout, r"edad transitoria \(10% de frames\) x2"), r.stdout[-300:])
    informe = json.load(open(os.path.join(lote, "informes", "limpieza_fantasmas.json")))
    datos = list(informe.values())[0]
    check("el informe deja constancia del tipo de ambigüedad y su fracción",
          datos["ambiguedad"] == "transitoria" and datos["ambiguedad_frac"] == 0.1, datos)
    shutil.rmtree(lote)



# ------------------------------------------------------ 3. negro de la máscara ----

def test_tira_lo_que_vive_sobre_el_negro():
    lote, jd = monta_lote([[hablante(), sobre_negro_cuerpo()], [hablante(), sobre_negro_cuerpo()]])
    r = corre(lote)
    limpio = os.path.join(os.path.dirname(jd), "JSON_FILES_limpio")
    n_personas = [len(v["people"]) for v in lee(limpio).values()]
    check("esqueleto sobre el negro tirado (aunque tenga hombros y 7 puntos)",
          n_personas == [1, 1], n_personas)
    check("el motivo registrado es el porcentaje sobre negro",
          "sobre negro" in r.stdout, r.stdout[-260:])
    shutil.rmtree(lote)


# ------------------------------------------------- 4/5. frames sin decidir ----

def test_frames_de_uno_y_de_cero_se_copian_iguales():
    lote, jd = monta_lote([[], [hablante()], [hablante()]])
    antes = lee(jd)
    r = corre(lote)
    limpio = os.path.join(os.path.dirname(jd), "JSON_FILES_limpio")
    despues = lee(limpio)
    check("frames de 0 y 1 candidato se copian sin alterar",
          antes == despues and len(despues) == 3, (len(antes), len(despues)))
    check("un clip limpio no tiene ni ambigüedad que reportar",
          sale(r.stdout, r"amb\s+0/3") and sale(r.stdout, r"tirados\s+0"), r.stdout[-260:])
    shutil.rmtree(lote)


# ------------------------------------------------------------ 6. solo informar ----

def test_criterio_nada_informa_sin_tirar():
    lote, jd = monta_lote([[hablante(), blob()], [hablante(), blob(), sobre_negro_cuerpo()]])
    r = corre(lote, "--criterio", "nada")
    limpio = os.path.join(os.path.dirname(jd), "JSON_FILES_limpio")
    n_personas = [len(v["people"]) for v in lee(limpio).values()]
    check("con --criterio nada no se tira ningún candidato",
          n_personas == [2, 3] and sale(r.stdout, r"tirados\s+0"), (n_personas, r.stdout[-260:]))
    check("pero sí reporta la ambigüedad que vio, sin podar nada",
          sale(r.stdout, r"amb\s+2/2") and sale(r.stdout, r"podados\s+0"), r.stdout[-260:])
    shutil.rmtree(lote)


# ------------------------------------------------------------ 7. no destructivo ----

def test_los_originales_quedan_intactos():
    lote, jd = monta_lote([[hablante(), blob()], [hablante(), blob()]])
    antes = {n: open(os.path.join(jd, n)).read() for n in sorted(os.listdir(jd))}
    corre(lote)
    despues = {n: open(os.path.join(jd, n)).read() for n in sorted(os.listdir(jd))}
    check("los JSON originales no se modifican (se escribe en JSON_FILES_limpio)",
          antes == despues and os.path.isdir(os.path.join(os.path.dirname(jd), "JSON_FILES_limpio")))
    check("el informe en JSON queda en informes/limpieza_fantasmas.json",
          os.path.exists(os.path.join(lote, "informes", "limpieza_fantasmas.json")))
    shutil.rmtree(lote)


def test_dry_run_no_escribe():
    lote, jd = monta_lote([[hablante(), blob()]])
    r = corre(lote, "--dry-run")
    limpio = os.path.join(os.path.dirname(jd), "JSON_FILES_limpio")
    check("--dry-run informa sin crear el árbol limpio",
          not os.path.exists(limpio) and "dry-run" in r.stdout, r.stdout[-200:])
    check("--dry-run no escribe el informe",
          not os.path.exists(os.path.join(lote, "informes", "limpieza_fantasmas.json")))
    shutil.rmtree(lote)


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[fn]()
        except Exception as e:  # noqa: BLE001
            check(f"{fn} (excepcion)", False, repr(e))
    print("\nTOTAL:", "FAIL " + ", ".join(FAIL) if FAIL else "todo OK")
    sys.exit(1 if FAIL else 0)
