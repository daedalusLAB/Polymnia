#!/usr/bin/env python3
"""Tests de la lógica pura de `wrist_prescreen.py`, sin YOLO ni vídeos.

El prescreen hace dos cosas que se pueden equivocar en silencio y no se ven hasta
que has gastado GPU: elegir los fotogramas de la muestra, y decidir de QUÉ persona
se leen las muñecas. Aquí se prueban esas dos decisiones con matrices de
confianza fabricadas, sin importar `ultralytics` (el módulo lo importa dentro de
`prescreen_folder`, así que se puede importar sin modelo ni GPU).

Casos que pinchan el comportamiento:
  1. clip corto            -> la muestra no inventa fotogramas que no existen
  2. clip largo            -> `frames` índices distintos, extremos incluidos
  3. sin keypoints         -> nadie, no un fallo
  4. filas de tamaño falso -> se descartan las que no son un cuerpo COCO-17
  5. una persona           -> estricto y hablante coinciden
  6. dos personas          -> estricto NO suma (exige un solo cuerpo)
  7. hablante sin muñecas  -> el hablante manda, aunque el otro las tenga claras
  8. lote vacío            -> ValueError, no un 0 % silencioso
"""
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

from wrist_prescreen import COCO, person_confs, sample_frame_indices, wrist_scores  # noqa: E402

FAIL = []
TOTAL = 0


def check(nombre, ok, detalle=""):
    global TOTAL
    TOTAL += 1
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


class _Keypoints:
    """Lo que lee `person_confs`: una matriz de confianza por persona, con len()."""

    def __init__(self, conf):
        self.conf = np.asarray(conf, dtype="float64")

    def __len__(self):
        return int(self.conf.shape[0])


class _Result:
    """Un `Results` de ultralytics recortado a lo que usa el prescreen."""

    def __init__(self, conf):
        self.keypoints = None if conf is None else _Keypoints(conf)


def cuerpo(nariz, muneca_izq, muneca_der):
    """Una fila COCO-17 con nariz y muñecas; el resto a cero."""
    fila = [0.0] * 17
    fila[COCO["nose"]] = nariz
    fila[COCO["left_wrist"]] = muneca_izq
    fila[COCO["right_wrist"]] = muneca_der
    return fila


def main():
    # 1. clip corto: se pide más muestra de la que hay, y no se inventa nada.
    indices = sample_frame_indices(5, 15)
    check("clip corto: 5 fotogramas, no 15 inventados", len(indices) == 5, f"->{len(indices)}")
    check("clip corto: indices dentro del clip", bool(indices.min() >= 0 and indices.max() <= 4),
          f"->{indices.tolist()}")

    # 2. clip largo: `frames` muestras distintas y extremos cubiertos.
    indices = sample_frame_indices(100, 15)
    check("clip largo: 15 muestras distintas", len(set(indices.tolist())) == 15, f"->{len(set(indices.tolist()))}")
    check("clip largo: cubre el primer y ultimo fotograma",
          indices[0] == 0 and indices[-1] == 99, f"->{indices[[0, -1]].tolist()}")
    check("clip largo: ordenados", bool(np.all(np.diff(indices) > 0)), f"->{indices.tolist()}")

    # 3. y 4. sobre la matriz de confianza.
    check("sin keypoints no es un fallo", person_confs(_Result(None)) == [], "->no vacio")
    check("keypoints presentes pero cero personas", person_confs(_Result([])) == [], "->no vacio")
    filas = person_confs(_Result([cuerpo(0.9, 0.8, 0.7), cuerpo(0.6, 0.4, 0.3)]))
    check("dos personas -> dos filas", len(filas) == 2, f"->{len(filas)}")
    check("matriz que no llega a los indices COCO se descarta",
          person_confs(_Result([[0.5] * 8, [0.4] * 8])) == [], "->conserva algo")

    # 5. una persona con las dos muñecas claras: estricto y hablante coinciden.
    (scored, sr, sl, sb, pr, pl, pb, maxima) = wrist_scores(
        [_Result([cuerpo(0.90, 0.80, 0.70)])], min_conf=0.5)
    check("una persona: estricto cuenta las dos munecas y el par",
          (scored, sr, sl, sb) == (1, 1, 1, 1), f"->{(scored, sr, sl, sb)}")
    check("una persona: hablante igual que estricto", (pr, pl, pb) == (1, 1, 1), f"->{(pr, pl, pb)}")
    check("una persona: nadie cuenta como no visible -> score 0 en frame vacio",
          wrist_scores([_Result(None)], min_conf=0.5)[1:4] == (0, 0, 0), "->cuenta algo")

    # 6., 7. y 8. el lote mezclado: es el caso que decide el ranking.
    #   A una persona con munecas | B dos, hablante SIN munecas (el otro las tiene
    #   clarisimas) | C dos, hablante CON munecas | D nadie.
    lote = [_Result([cuerpo(0.90, 0.80, 0.70)]),
            _Result([cuerpo(0.10, 0.90, 0.90), cuerpo(0.95, 0.30, 0.30)]),
            _Result([cuerpo(0.95, 0.80, 0.80), cuerpo(0.10, 0.95, 0.95)]),
            _Result([])]
    (scored, sr, sl, sb, pr, pl, pb, maxima) = wrist_scores(lote, min_conf=0.5)
    check("lote: se puntan los 4 fotogramas", scored == 4, f"->{scored}")
    check("lote: con dos personas el estricto no suma (exige un solo cuerpo)",
          (sr, sl, sb) == (1, 1, 1), f"->{(sr, sl, sb)}")
    check("lote: el hablante manda, no el primero de la lista",
          (pr, pl, pb) == (2, 2, 2), f"->{(pr, pl, pb)}  (3 seria leer al primero en B)")
    check("lote: maxima gente observada", maxima == 2, f"->{maxima}")

    try:
        wrist_scores([], min_conf=0.5)
        check("lote vacio petar en vez de informar 0 %", False, "->no peto")
    except ValueError:
        check("lote vacio petar en vez de informar 0 %", True)

    if FAIL:
        print(f"\nTOTAL: {TOTAL - len(FAIL)} / {TOTAL}   fallos: {', '.join(FAIL)}")
        return 1
    print(f"\nTOTAL: {TOTAL} / {TOTAL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
