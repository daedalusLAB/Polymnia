#!/usr/bin/env python3
"""Deja en cada frame el candidato que de verdad se puede medir.

Contexto medido en un lote propio de 30 clips de NewsScape (los numeros de abajo
son de esa medida, hecha sobre los JSON de OpenPose de raw y masked; los nombres
de clip son de ese lote):

* La máscara de TalkNet-ASD deja un borde duro contra el negro, y OpenPose emite
  ahí candidatos que no son nadie: 2-3 keypoints, sin hombros. En
  `Kimmel 08-25` y `Tavis 05-20` el RAW tiene **1 candidato en los 155 frames** y
  el MASKED tiene 2 en 12 y 7 frames. El segundo candidato lo fabrica la máscara.
* Cuando sí hay una segunda persona en el plano (`CNN Newsroom`: 3 candidatos en
  151/155 frames del RAW), la máscara la recorta y OpenPose dibuja un esqueleto
  deformado whose puntos caen al 71-86 % sobre el negro.

Por qué importa: `multimolang::dfMaker` hace `people_id = índice en el array` y
`max_people_classification.R` clasifica el clip con `num_people <- max(people_id)`.
Un candidato fantasma en el índice 1 convierte un clip de una persona en
`2_persons`, y `--n_persons 1` se queda sin clips. Medido en la pasada de prueba:
4 de 4 clips a `2_persons`, `1_persons` vacío.

Esto NO toca el contrato de admisión del repo (`body_visibility.py` sigue igual).
Es un paso de limpieza sobre los JSON de trabajo:

* los JSON originales no se tocan: se escriben copias en otro árbol;
* por frame se conserva el candidato con más keypoints fiables (empate a más
  nariz), que es el sujeto que ASD eligió como hablante;
* se registra frame a frame lo que se tiró, para que se pueda auditar.

Uso:
    limpiar_fantasmas.py --lote /tmp/lote_smoke
    limpiar_fantasmas.py --lote ... --criterio negro      # solo el descarte por negro
    limpiar_fantasmas.py --lote ... --dry-run             # no escribe, solo informa
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np

BODY_25_POINTS = 25
BODY_25_VALUES = 75
CONF = 0.2                       # el mismo umbral de keypoint que usa el pipeline
TORSO = (2, 5)                   # BODY_25 hombros
NEGRO_FRAC = 0.5                 # >50 % de las coyunturas sobre negro => no es un cuerpo
NEGRO_PIXEL = 12                 # luma por debajo de esto es el negro de la máscara
PERSIST_FRAC = 0.25              # ambigüedad por debajo de esto es transitoria (fantasma)


def keypoint_body(person):
    """Los 25 puntos (x, y, conf) de un candidato, o None si no es un cuerpo BODY_25."""
    kp = person.get("pose_keypoints_2d")
    if not kp or len(kp) != BODY_25_VALUES:
        return None
    return np.asarray(kp, dtype=float).reshape(BODY_25_POINTS, 3)


def quality(body):
    """Cuánto de cuerpo hay: nº de puntos fiables, y la nariz como desempate."""
    validas = body[:, 2] >= CONF
    return int(validas.sum()), float(body[0, 2])


def sobre_negro(body, gray):
    """Fracción de las coyunturas fiables que caen sobre el negro de la máscara.

    Una persona real está sobre contenido visible; un esqueleto completado sobre el
    recorte negro tiene la mayoría de sus puntos en negro. Medido: hablante 0-20 %,
    esqueleto sobre la máscara 71-86 %.
    """
    if gray is None:
        return None
    validas = np.where(body[:, 2] >= CONF)[0]
    if len(validas) < 2:
        return None
    oscuras = 0
    for i in validas:
        x, y = int(round(body[i, 0])), int(round(body[i, 1]))
        if not (2 <= x < gray.shape[1] - 2 and 2 <= y < gray.shape[0] - 2):
            oscuras += 1                      # fuera de cuadro tampoco es un cuerpo
            continue
        oscuras += int(np.median(gray[y - 2:y + 3, x - 2:x + 3]) < NEGRO_PIXEL)
    return oscuras / len(validas)


def es_descartable(cuerpo, fraction_negro, criterio):
    """Qué candidatos se tiran. Dos señales medidas, independientes."""
    kp, _ = quality(cuerpo)
    tiene_torso = any(cuerpo[i, 2] >= CONF for i in TORSO)
    if criterio in ("todo", "torso") and kp < 6 and not tiene_torso:
        return "sin torso y <6 puntos"
    if criterio in ("todo", "negro") and fraction_negro is not None and fraction_negro > NEGRO_FRAC:
        return f"{fraction_negro * 100:.0f}% de puntos sobre negro"
    return None


def mejor_restante(candidatos, quitados):
    """El índice del candidato más completo de los que quedan."""
    disponibles = [i for i in range(len(candidatos)) if i not in quitados]
    if not disponibles:
        return None
    return max(disponibles, key=lambda i: quality(candidatos[i]))


def limpiar_clip(json_dir, video, criterio, dry_run):
    """Reescribe los JSON de un clip en `json_dir/../JSON_FILES_limpio`.

    La decisión es a nivel de clip, no de frame, y la razón está medida: un segundo
    candidato con 2-5 puntos fiables, un hombro y sin negro fue a la vez la segunda
    presentadora real de un debate (`run50`, presente en 425 de 427 frames) y el
    artefacto de la máscara de ASD (`Tavis 05-20`, presente en 7 de 155). Por frame
    son el mismo bulto; lo que los separa es cuánto dura.

    * ambigüedad transitoria (< PERSIST_FRAC de los frames): se poda hasta dejar el
      candidato más completo en cada frame ambiguo;
    * ambigüedad sostenida: solo se poda lo que no es cuerpo medible (sin torso y con
      pocos puntos, o vivendo sobre el negro). Lo demás es admisión, no limpieza.

    Los frames con un solo candidato se copian tal cual: un clip limpio no sufre
    ninguna alteración, y los JSON originales no se tocan.
    """
    limpio_dir = os.path.join(os.path.dirname(json_dir.rstrip("/")), "JSON_FILES_limpio")
    cap = cv2.VideoCapture(video) if criterio in ("todo", "negro") else None
    rutas = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    informe = {
        "json_originales": len(rutas),
        "json_escritos": 0,
        "frames_con_ambiguedad": 0,   # frames donde OpenPose emitió más de un candidato
        "ambiguedad_frac": 0.0,
        "ambiguedad": "-",
        "frames_podados": 0,          # frames de los que además se tiró algo
        "candidatos_tirados": 0,
        "motivos": {},
        "kp_minimo_del_sujeto": None,
        "dir_limpio": limpio_dir,
    }
    if not rutas:
        return informe
    if not dry_run:
        shutil.rmtree(limpio_dir, ignore_errors=True)
        os.makedirs(limpio_dir, exist_ok=True)

    dados = []
    for ruta in rutas:
        payload = json.load(open(ruta))
        personas = payload.get("people") or []
        cuerpos = [keypoint_body(p) for p in personas]
        valido = [i for i, c in enumerate(cuerpos) if c is not None]
        dados.append((ruta, payload, personas, cuerpos, valido))
        if len(valido) > 1:
            informe["frames_con_ambiguedad"] += 1
    informe["ambiguedad_frac"] = round(informe["frames_con_ambiguedad"] / max(len(rutas), 1), 4)
    transitorio = informe["frames_con_ambiguedad"] / max(len(rutas), 1) < PERSIST_FRAC
    informe["ambiguedad"] = "transitoria" if transitorio else "sostenida"

    for ruta, payload, personas, cuerpos, valido in dados:
        if len(valido) <= 1:
            # nada que decidir: se copia el frame intacto (incluido el 0 personas)
            if not dry_run:
                shutil.copy2(ruta, os.path.join(limpio_dir, os.path.basename(ruta)))
                informe["json_escritos"] += 1
            continue

        m = re.search(r"_(\d{12})_keypoints\.json$", ruta)
        idx = int(m.group(1)) if m else None
        gray = None
        if cap is not None and idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        quitados = set()
        elegido = mejor_restante(cuerpos, quitados)
        if transitorio:
            # la ambigüedad dura unos pocos frames: todo lo que no sea el sujeto sobra
            quitados = {i for i in valido if i != elegido}
            motivo = (f"ambigüedad transitoria ({informe['ambiguedad_frac'] * 100:.0f}% de frames)")
            for i in quitados:
                informe["candidatos_tirados"] += 1
                informe["motivos"][motivo] = informe["motivos"].get(motivo, 0) + 1
        else:
            # ambigüedad sostenida: solo se va lo que no es un cuerpo medible
            while len([i for i in valido if i not in quitados]) > 1:
                elegido = mejor_restante(cuerpos, quitados)
                tiro = False
                for i in [j for j in valido if j not in quitados and j != elegido]:
                    motivo = es_descartable(cuerpos[i], sobre_negro(cuerpos[i], gray), criterio)
                    if motivo:
                        quitados.add(i)
                        informe["candidatos_tirados"] += 1
                        informe["motivos"][motivo] = informe["motivos"].get(motivo, 0) + 1
                        tiro = True
                        break
                if not tiro:
                    # los que sobran son cuerpos con torso y fuera del negro: segunda
                    # persona real. No se toca nada: eso es admisión, no limpieza.
                    quitados = set()
                    break
        if quitados:
            informe["frames_podados"] += 1
            payload["people"] = [personas[i] for i in valido if i not in quitados]
            informe["kp_minimo_del_sujeto"] = min(
                [quality(cuerpos[i])[0] for i in valido if i not in quitados]
                + [informe["kp_minimo_del_sujeto"] or 99]
            )
        if not dry_run:
            with open(os.path.join(limpio_dir, os.path.basename(ruta)), "w") as fh:
                json.dump(payload, fh)
            informe["json_escritos"] += 1
    if cap is not None:
        cap.release()
    return informe


def main(argv=None):
    ap = argparse.ArgumentParser(description="deja en cada frame el candidato medible")
    ap.add_argument("--lote", required=True, help="carpeta del lote (con dataset/OpenPose)")
    ap.add_argument("--criterio", choices=["todo", "torso", "negro", "nada"], default="todo",
                    help="qué señal usa para tirar candidatos (nada = solo informa)")
    ap.add_argument("--dry-run", action="store_true", help="no escribe los JSON limpios")
    args = ap.parse_args(argv)

    op = os.path.join(args.lote, "dataset", "OpenPose")
    clips = sorted(d for d in glob.glob(os.path.join(op, "*")) if os.path.isdir(d))
    if not clips:
        print(f"❌ no hay JSON de OpenPose en {op}", file=sys.stderr)
        return 1

    total_tirados = 0
    filas = []
    for cdir in clips:
        clip = os.path.basename(cdir)
        json_dir = os.path.join(cdir, "JSON_FILES")
        video = os.path.join(args.lote, "videos", "masked", clip + ".mp4")
        if not os.path.isdir(json_dir):
            continue
        inf = limpiar_clip(json_dir, video if os.path.exists(video) else None,
                           args.criterio, args.dry_run)
        total_tirados += inf["candidatos_tirados"]
        filas.append((clip, inf))
        motivos = ", ".join(f"{k} x{v}" for k, v in sorted(inf["motivos"].items())) or "-"
        print(f"   {clip[:52]:52s} amb{inf['frames_con_ambiguedad']:>4}/{inf['json_originales']:<4}"
              f" {inf['ambiguedad']:<11} podados{inf['frames_podados']:>4}"
              f" tirados{inf['candidatos_tirados']:>4}  {motivos}")

    informe_ruta = os.path.join(args.lote, "informes", "limpieza_fantasmas.json")
    if not args.dry_run:
        os.makedirs(os.path.dirname(informe_ruta), exist_ok=True)
        with open(informe_ruta, "w") as fh:
            json.dump({os.path.basename(c): i for c, i in filas}, fh, indent=1)
    print(f"   total de candidatos tirados: {total_tirados}"
          + ("  (dry-run: no se escribió nada)" if args.dry_run else
             f"\n   JSON limpios en dataset/OpenPose/*/JSON_FILES_limpio"
             f"\n   informe en {informe_ruta}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
