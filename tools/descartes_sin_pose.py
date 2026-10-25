#!/usr/bin/env python3
"""Clips que OpenPose dejo sin una sola deteccion: se apartan, no tumban el lote.

`max_people_classification.R` escribe `num_persons.txt` junto a los JSON de cada clip. Cuando
OpenPose devolvio `{"people": []}` en todos los frames, `dfMaker` devuelve NULL y el
`write_parquet(NULL)` de antes reventaba: un solo clip sin pose abortaba el lote entero en la
etapa 6 (paso con ayesha_century_tspan: 2 de 103; ayesha_ago_d_time: 3 de 174).

El R ya no revienta: deja `num_persons.txt = 0` y sale 0. Este helper los cuenta, copia su mp4 a
`videos/descartados/sin_pose/`, los apunta en `informes/descartados_sin_pose.tsv` y aplica el
umbral: si el patron no son unos pocos clips (o no sobrevive ninguno) es OpenPose roto, y ahi si
hay que parar.

Uso:
    descartes_sin_pose.py --lote lotes/X [--max-pct 5]

Salida: `sin_pose=N de M (X.X %)` y una linea por clip. Exit 1 solo cuando se pasa el umbral o
no queda ningun clip utilizable.
"""
import argparse
import glob
import os
import shutil
import sys

DEFAULT_MAX_PCT = 5.0


def carpetas_de_clip(lote):
    """(clip, carpeta con los JSON) por clip, prefiriendo JSON_FILES_limpio."""
    base = os.path.join(lote, "dataset", "OpenPose")
    limpio = {}
    for ruta in glob.glob(os.path.join(base, "*", "JSON_FILES_limpio")):
        limpio[os.path.basename(os.path.dirname(ruta))] = ruta
    for ruta in glob.glob(os.path.join(base, "*", "JSON_FILES")):
        limpio.setdefault(os.path.basename(os.path.dirname(ruta)), ruta)
    return sorted(limpio.items())


def personas(jd):
    """num_persons.txt del clip, o None si la pasada es vieja y no lo dejo."""
    ruta = os.path.join(jd, "num_persons.txt")
    if not os.path.isfile(ruta):
        return None
    try:
        with open(ruta, encoding="utf-8") as fh:
            return int(float(fh.read().strip() or "0"))
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lote", required=True, help="carpeta del lote")
    parser.add_argument("--max-pct", type=float, default=DEFAULT_MAX_PCT,
                        help="%% de descartes que para el lote; 0 lo desactiva (defecto %(default)s)")
    args = parser.parse_args()

    clips = carpetas_de_clip(args.lote)
    vistos = [(clip, jd, personas(jd)) for clip, jd in clips]
    conocidos = [(c, jd, n) for c, jd, n in vistos if n is not None]
    sin_pose = [(c, jd) for c, jd, n in conocidos if n == 0]
    total = len(conocidos)
    pct = (100.0 * len(sin_pose) / total) if total else 0.0

    informe = os.path.join(args.lote, "informes", "descartados_sin_pose.tsv")
    os.makedirs(os.path.dirname(informe), exist_ok=True)
    with open(informe, "w", encoding="utf-8") as fh:
        fh.write("clip\treason\tsource\n")
        for clip, jd in sin_pose:
            fh.write(f"{clip}\tsin_pose\t{os.path.join(jd, 'num_persons.txt')}\n")

    destino = os.path.join(args.lote, "videos", "descartados", "sin_pose")
    os.makedirs(destino, exist_ok=True)
    copiados = 0
    for clip, _ in sin_pose:
        origen = os.path.join(args.lote, "videos", "1_person", f"{clip}.mp4")
        if os.path.isfile(origen):
            shutil.copy2(origen, os.path.join(destino, f"{clip}.mp4"))
            copiados += 1

    print(f"sin_pose={len(sin_pose)} de {total} ({pct:.1f} %)")
    for clip, _ in sin_pose:
        print(f"  - {clip}")
    if sin_pose:
        print(f"  apartados: videos/descartados/sin_pose/ ({copiados} mp4) + "
              f"informes/descartados_sin_pose.tsv")

    if sin_pose and total and len(sin_pose) == total:
        print("[ERROR] no sobrevive ningun clip con pose: OpenPose no detecto a nadie en el lote",
              file=sys.stderr)
        return 1
    if sin_pose and args.max_pct and pct > args.max_pct:
        print(f"[ERROR] {len(sin_pose)} de {total} clips sin pose ({pct:.1f} %) supera el umbral "
              f"{args.max_pct:g} %: no son unos pocos clips, revisa OpenPose", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
