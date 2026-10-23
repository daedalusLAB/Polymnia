#!/usr/bin/env python3
"""Pre-screen clips for WRIST visibility with YOLO11-pose, before OpenPose runs.

Why this exists: in a 29-clip NewsScape batch we delivered, only 4 clips have
BODY_25 wrists 4 and 7 visible with confidence >= 0.2 in >= 40 % of frames, and
the yield curve is flat
(4 clips pass at 40 %, the same 4 at 95 %). The corpus is bimodal, so a 15-frame
sample decides it. The exact gate is `argos.py --filter-body-visibility`, but that
runs AFTER OpenPose; this runs before, and its only job is to stop spending GPU on
clips that cannot yield wrist features. It is a ranker, not an admission decision:
what a person count means is the other gate's job, and OpenPose output is the
evidence there.

Differences from `is_there_a_person_in_the_video.py`, which this is based on:

* confidence, not zero: it scores `keypoints.conf >= --min-conf`. A non-zero test
  passes low-confidence keypoints that OpenPose later drops at c < 0.2 (pose rows
  in the delivered corpus have median c = 0.244).
* a score, not a boolean: one CSV row per clip with the sampled-frame fraction for
  each wrist and for both, ranked. A boolean plus `shutil.copy` cannot rank a wide
  batch down to the best K.
* safe sampling: `--frames` indices from np.linspace over the real frame count, so
  a short clip cannot collapse to `int(length/16) == 0` and re-read one frame 15x.
* no side effects: no `images/*.jpg` dumps, no copied videos, no `chmod -R 777`.

Two wrist metrics per clip, because which body is the subject depends on the stage:

* `both` - strict: only frames with exactly one detected person. Sound on the
  post-ASD `videos_masked/`, where the mask keeps the active speaker's body column
  and nothing else.
* `both_speaker` - the person whose FACE is most confident in each frame, i.e. the
  body that contains the speaker's face, which is what ASD's tracking selects. On
  `videos_raw/` a two-shot has several bodies and "the wrist" is ambiguous; the face
  picks the one that means the same thing as ASD's track.

A third column, `sp_any` = max(sp_right, sp_left), is what `pipeline_lote.sh` admits on
(`--min-sp-any`, default 0.6): a clip with one clean wrist does yield wrist features,
because the feature stack treats the two wrists independently and can only fail to
compute what it never saw. Requiring BOTH wrists in the same sampled frame (`sp_both`, the
old criterion) threw away a clip OpenPose later rated 91.9 % / 100 % with `gate=accepted`
- see `tools/auditoria_criba.py`, which exists to keep this claim measurable instead of
received wisdom. `sp_both` is still written, as a ranking column and as the evidence of
what the old rule did.

Keypoint indices: ultralytics returns COCO-17, so wrists are 9 = left, 10 = right
(person's own left/right, same convention as BODY_25, where they are 7 and 4).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# COCO-17 indices (ultralytics order).
COCO = {"nose": 0, "left_shoulder": 5, "right_shoulder": 6, "left_wrist": 9, "right_wrist": 10}
DEFAULT_WEIGHTS = "yolo11x-pose.pt"


def sample_frame_indices(count, frames):
    """`frames` distinct frame indices spread over the clip, safe for short clips."""
    if count <= 0:
        return np.array([], dtype="int64")
    n = min(int(frames), int(count))
    return np.unique(np.linspace(0, count - 1, n).round().astype("int64"))


def extract_frames(video_path, frames):
    """(indices, frames, reported_length). Frames that cannot be decoded are skipped."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(count, frames)
    kept_index, kept_frames = [], []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok and frame is not None:
            kept_index.append(int(index))
            kept_frames.append(frame)
    cap.release()
    return np.array(kept_index, dtype="int64"), kept_frames, count


def person_confs(result):
    """Per-person COCO keypoint confidences as a list of float arrays (host memory)."""
    if result.keypoints is None or len(result.keypoints) == 0:
        return []
    conf = result.keypoints.conf
    conf = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
    conf = np.asarray(conf, dtype="float64")
    return [row for row in conf if row.ndim == 1 and row.size > max(COCO.values())]


def wrist_scores(results, min_conf):
    """Per-frame wrist visibility from one batch of YOLO results.

    Returns (n_frames, strict hits, speaker hits, max_people) for right / left / both.
    Strict counts only frames with exactly one person. Speaker counts, for every
    frame with at least one person, the wrists of the person with the most confident
    face. A frame with no person counts as not visible in both.
    """
    hits = {"strict": [0, 0, 0], "speaker": [0, 0, 0]}
    scored, max_people = 0, 0
    for result in results:
        scored += 1
        rows = person_confs(result)
        max_people = max(max_people, len(rows))
        for kind, row in (("strict", rows[0] if len(rows) == 1 else None),
                          ("speaker", max(rows, key=lambda r: r[COCO["nose"]]) if rows else None)):
            if row is None:
                continue
            r = bool(row[COCO["right_wrist"]] >= min_conf)
            l = bool(row[COCO["left_wrist"]] >= min_conf)
            hits[kind][0] += r
            hits[kind][1] += l
            hits[kind][2] += (r and l)
    if scored == 0:
        raise ValueError("YOLO devolvio cero resultados para un lote de fotogramas")
    return (scored, hits["strict"][0], hits["strict"][1], hits["strict"][2],
            hits["speaker"][0], hits["speaker"][1], hits["speaker"][2], max_people)


def best_wrist(sp_right, sp_left):
    """La mejor de las dos muñecas del hablante: el criterio con el que se ADMITE.

    Por qué una y no las dos, medido: la regla anterior (`sp_both`) exige ver las dos
    muñecas en el MISMO frame muestreado, y en el censo de los 16 clips que descartó
    FREQ025 se dejó uno que OpenPose valoró luego en 91,9 % / 100 % de los frames
    (`gate=accepted`). Ese clip era irrecuperable: la criba corre antes de OpenPose.
    Con `sp_any >= umbral` ese falso negativo desaparece y no entra ninguno nuevo
    (matriz completa de los 20 clips del lote, medida con OpenPose de verdad).

    Y admitir una muñeca no es bajar la calidad de los features: `gesture_features.py`
    calcula derecha e izquierda por separado y tolera que falte una (`*_n_observed`,
    `*_frac_valid`); lo que no puede fabricar son las dos.

    NaN solo si falta también la otra: un NaN no se traga la muñeca que sí se vio.
    """
    visto = [v for v in (sp_right, sp_left) if v == v]  # NaN != NaN
    return max(visto) if visto else float("nan")


def orden_de_fila(row):
    """El orden del CSV: primero la métrica que decide, `sp_both` solo desempata.

    Con `sp_any` los empates son legión (un lote entero de clips a 0,0), y sin desempate el
    orden del CSV dependía del orden de carpeta. Se desempata por `sp_both` para que, a
    igual de «algo se ve», salga antes el clip con las dos muñecas más presentes.

    Un score NaN (clip no decodificable) va al final, no al principio: NaN comparado con
    `<` es falso, y sin este cuidado un vídeo corrupto encabezaría el ranking.
    """
    def negativo(clave):
        v = row.get(clave)
        return -(v if v == v else -1.0)
    return (negativo("sp_any"), negativo("sp_both"), row["video"])


def prescreen_folder(videos_folder, weights=DEFAULT_WEIGHTS, frames=15, min_conf=0.2,
                     device=None, imgsz=None, model=None):
    """One ranked row per .mp4: fraction of sampled frames with each wrist visible."""
    if model is None:
        from ultralytics import YOLO
        model = YOLO(weights)
    paths = sorted(os.path.join(root, name)
                   for root, _, names in os.walk(videos_folder)
                   for name in names if name.endswith(".mp4"))
    rows = []
    for path in paths:
        indices, batch, count = extract_frames(path, frames)
        nan = float("nan")
        if not batch:
            rows.append({"video": os.path.basename(path), "path": os.path.abspath(path),
                         "reported_frames": count, "n_sampled": 0, "right": nan, "left": nan,
                         "both": nan, "sp_right": nan, "sp_left": nan, "sp_both": nan,
                         "sp_any": nan,
                         "max_people": 0, "status": "no decodificable"})
            continue
        kwargs = {"verbose": False, "conf": 0.30}
        if device:
            kwargs["device"] = device
        if imgsz:
            kwargs["imgsz"] = imgsz
        results = model(batch, **kwargs)
        scored, r, l, b, sr, sl, sb, max_people = wrist_scores(results, min_conf)
        rows.append({"video": os.path.basename(path), "path": os.path.abspath(path),
                     "reported_frames": count, "n_sampled": scored,
                     "right": r / scored, "left": l / scored, "both": b / scored,
                     "sp_right": sr / scored, "sp_left": sl / scored, "sp_both": sb / scored,
                     "sp_any": best_wrist(sr / scored, sl / scored),
                     "max_people": max_people,
                     "status": "ok" if scored == len(batch) else "muestreo parcial"})
    # Se rankea por la métrica que decide (sp_any); ver `orden_de_fila`.
    rows.sort(key=orden_de_fila)
    return rows


def write_csv(rows, out_csv):
    # sp_any va justo despues de video porque es la columna con la que pipeline_lote.sh
    # admite; sp_both se queda como ranking y como evidencia de la regla anterior.
    fields = ["video", "sp_any", "sp_both", "both", "sp_right", "sp_left", "right", "left",
              "n_sampled", "max_people", "reported_frames", "status", "path"]
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) and v == v else v)
                             for k, v in row.items()})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos_folder", required=True, help="carpeta con .mp4 (recursiva)")
    parser.add_argument("--out_csv", required=True, help="CSV de salida; se sobrescribe")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="pesos YOLO-pose (COCO-17)")
    parser.add_argument("--frames", type=int, default=15,
                        help="fotogramas muestreados por clip, repartidos en el tiempo")
    parser.add_argument("--min-conf", dest="min_conf", type=float, default=0.2,
                        help="confianza minima del keypoint para contarlo como visible")
    parser.add_argument("--device", default=None, help="device ultralytics (p.ej. 0 o cpu)")
    parser.add_argument("--imgsz", type=int, default=None, help="tamano de inferencia")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.videos_folder):
        print(f"❌ no existe la carpeta: {args.videos_folder}", file=sys.stderr)
        return 1
    if not 0.0 <= args.min_conf <= 1.0:
        print(f"❌ --min-conf fuera de [0,1]: {args.min_conf}", file=sys.stderr)
        return 1
    rows = prescreen_folder(args.videos_folder, args.weights, args.frames, args.min_conf,
                            args.device, args.imgsz)
    write_csv(rows, args.out_csv)
    print(f"✅ {args.out_csv}")
    for row in rows:
        print(f"   sp_any={row['sp_any']:5.2f} sp_both={row['sp_both']:5.2f} "
              f"both={row['both']:5.2f} "
              f"n={row['n_sampled']:3d} people={row['max_people']} {row['video'][:52]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
