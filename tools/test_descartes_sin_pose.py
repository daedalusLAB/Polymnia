#!/usr/bin/env python3
"""`tools/descartes_sin_pose.py`: el clip sin una sola deteccion se descarta, no tumba el lote.

OpenPose devuelve `{"people": []}` en todos los frames de algunos clips (nadie en camara).
`max_people_classification.R` hacia `write_parquet(NULL)` con ellos y reventaba: un clip sin
pose abortaba el lote entero en la etapa 6 (paso con ayesha_century_tspan: 2 de 103).

Ahora el R deja `num_persons.txt = 0` y sale 0; este helper los cuenta, los aparta y aplica el
umbral. Casos:

  1. un descarte de 10 con umbral alto: exit 0, TSV con el motivo y el mp4 copiado
  2. el mismo caso con el umbral por defecto (5 %): exit 1
  3. todos descartados, aunque el umbral este apagado: exit 1 (no se publica un lote vacio)
  4. lote sin `num_persons.txt` (pasada vieja): 0 de 0, exit 0 y TSV solo con cabecera
  5. el clip descartado no tiene mp4: se registra igual, sin reventar
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.join(REPO, "tools", "descartes_sin_pose.py")
FAIL = []


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def monta(base, personas, con_mp4=True):
    """`personas`: dict clip -> num_persons (None = sin num_persons.txt, como una pasada vieja)."""
    os.makedirs(os.path.join(base, "videos", "1_person"), exist_ok=True)
    for clip, n in personas.items():
        jd = os.path.join(base, "dataset", "OpenPose", clip, "JSON_FILES_limpio")
        os.makedirs(jd, exist_ok=True)
        if n is not None:
            with open(os.path.join(jd, "num_persons.txt"), "w", encoding="utf-8") as fh:
                fh.write(f"{n}\n")
        if con_mp4:
            with open(os.path.join(base, "videos", "1_person", f"{clip}.mp4"), "wb") as fh:
                fh.write(b"x")
    return base


def corre(lote, *extra):
    return subprocess.run([sys.executable, HELPER, "--lote", lote, *extra],
                          capture_output=True, text=True)


def filas(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return [l for l in fh if l.strip()]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # 1. 1 de 10 con umbral 20 %: se descarta y el lote sigue
        lote = monta(os.path.join(tmp, "a"), {f"clip_{i}": (0 if i == 3 else 1) for i in range(10)})
        p = corre(lote, "--max-pct", "20")
        check("un descarte con umbral alto sale 0", p.returncode == 0, p.stderr[-300:])
        check("cuenta 1 de 10", "sin_pose=1 de 10" in p.stdout, p.stdout)
        filas_tsv = filas(os.path.join(lote, "informes", "descartados_sin_pose.tsv")) or []
        check("el TSV lleva cabecera y una fila", len(filas_tsv) == 2, f"{filas_tsv}")
        if len(filas_tsv) == 2:
            check("nombra el clip y el motivo", filas_tsv[1].split("\t")[:2] == ["clip_3", "sin_pose"],
                  filas_tsv[1])
        check("el mp4 queda apartado",
              os.path.isfile(os.path.join(lote, "videos", "descartados", "sin_pose", "clip_3.mp4")))

        # 2. el mismo caso con el umbral por defecto (5 %)
        p = corre(lote)
        check("10 % de descartes para el lote por defecto", p.returncode == 1, f"codigo {p.returncode}")
        check("y lo dice con numeros", "10.0 %" in p.stdout, p.stdout)

        # 3. todos descartados
        lote = monta(os.path.join(tmp, "b"), {"clip_0": 0, "clip_1": 0})
        p = corre(lote, "--max-pct", "0")
        check("sin supervivientes sale 1", p.returncode == 1, f"codigo {p.returncode}")

        # 4. lote de una pasada vieja (sin num_persons.txt)
        lote = monta(os.path.join(tmp, "c"), {"clip_0": None, "clip_1": None})
        p = corre(lote, "--max-pct", "0")
        check("sin num_persons no hay nada que descartar",
              p.returncode == 0 and "sin_pose=0 de 0" in p.stdout, p.stdout)

        # 5. descartado sin mp4
        lote = monta(os.path.join(tmp, "d"), {"clip_0": 0}, con_mp4=False)
        p = corre(lote, "--max-pct", "0")
        check("un descartado sin mp4 no revienta", p.returncode != 1 or "sin_pose=1" in p.stdout,
              f"codigo {p.returncode}: {p.stderr[-200:]}")

    print()
    if FAIL:
        print(f"TOTAL: FAIL {', '.join(FAIL)}")
        return 1
    print("TOTAL: todo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
