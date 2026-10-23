#!/usr/bin/env python3
"""Informe de fallos de descarga de UN lote, sacado del log compartido del repo.

`download_clips.py` escribe `failed_downloads.log` en el cwd del repo (que `uvrun` fija),
asi que todas las descargas appenden al mismo fichero, tambien cuando corren dos lotes en
paralelo. Sacar el delta por offset de bytes mezcla los lotes.

Este filtro se queda con las lineas cuyo nombre de clip lleva `_<searchterm>.mp4` y, si se
pasa `--desde`, solo las posteriores a la hora de inicio de la etapa.

Uso:
    filtrar_fallos_descarga.py --log Polymnia/failed_downloads.log \\
        --searchterm freq013 --desde "2026-10-23 06:14:00" --out lote/informes/....

Salida: `fallos=N (leidas M de LOG)`. Siempre exit 0: un log ausente es "0 fallos", no un
error de la etapa.
"""
import argparse
import os
import re
import sys

TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", required=True, help="failed_downloads.log compartido")
    parser.add_argument("--searchterm", required=True,
                        help="termino del lote; el nombre del clip acaba en _<searchterm>.mp4")
    parser.add_argument("--desde", default=None,
                        help='hora mínima "YYYY-mm-dd HH:MM:SS" (inicio de la descarga)')
    parser.add_argument("--out", required=True, help="fichero de salida (solo este lote)")
    args = parser.parse_args()

    carpeta = os.path.dirname(args.out)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)

    leidas = guardadas = 0
    with open(args.out, "w", encoding="utf-8") as salida:
        if os.path.isfile(args.log):
            with open(args.log, encoding="utf-8", errors="replace") as entrada:
                for linea in entrada:
                    leidas += 1
                    if args.searchterm not in linea:
                        continue
                    sello = TIMESTAMP.match(linea)
                    if args.desde and (sello is None or sello.group(1) < args.desde):
                        continue
                    salida.write(linea if linea.endswith("\n") else linea + "\n")
                    guardadas += 1

    print(f"fallos={guardadas} (leidas {leidas} de {args.log})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
