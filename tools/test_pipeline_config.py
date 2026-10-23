#!/usr/bin/env python3
"""`pipeline_lote.sh --print-config`: una sola orden y los parámetros resueltos.

Objetivo: `--lote lotes/FREQ013` debe bastar. El TSV se busca por el nombre del lote, el
`searchterm` se deduce y, si el lote ya existía, `informes/parametros.json` devuelve el
`--offset`/`--limit` con los que se lanzó - para que un reanudado no salga con otra holgura.

Casos:

  1. lote nuevo con el TSV en ./MULTIFLOW_TSVs/... => tsv, searchterm y offset por defecto
  2. lote nuevo sin TSV en ningún sitio => exit 1 y dice qué falta
  3. --print-config no crea ni toca la carpeta del lote
  4. lote existente con parametros.json => recupera offset, searchterm y limit guardados
  5. lo que se pasa por CLI gana a lo guardado
  6. --tsv explícito gana a la búsqueda por nombre
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO, "final_dataset", "pipeline_lote.sh")
FAIL = []


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def tsv_de(base, lote, campana="tanvir"):
    carpeta = os.path.join(base, "MULTIFLOW_TSVs", campana, "Recurrence_Tabulations")
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"{lote}_Tab.txt")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("clip.txt\tid\t0\t0\t1\t0\t\n")
    return ruta


def corre(base, *args):
    entorno = dict(os.environ)
    entorno.pop("POLYMNIA_TSV_DIR", None)
    return subprocess.run(["bash", PIPELINE, *args, "--print-config"],
                          cwd=base, capture_output=True, text=True, env=entorno)


def campo(salida, nombre):
    for linea in salida.splitlines():
        if linea.startswith(f"{nombre}:"):
            return linea.split(":", 1)[1].strip()
    return ""


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # 1. lote nuevo, TSV encontrado por nombre
        esperado = tsv_de(tmp, "FREQ999")
        p = corre(tmp, "--lote", "lotes/FREQ999")
        check("un lote nuevo se configura solo", p.returncode == 0, p.stderr[-300:])
        check("encuentra el TSV por el nombre del lote", campo(p.stdout, "tsv") == esperado,
              campo(p.stdout, "tsv"))
        check("deduce el searchterm del nombre", campo(p.stdout, "searchterm") == "freq999",
              campo(p.stdout, "searchterm"))
        check("offset por defecto 2.0", campo(p.stdout, "offset") == "2.0", campo(p.stdout, "offset"))
        check("limit vacío si no se pidió", campo(p.stdout, "limit") == "(sin límite)",
              campo(p.stdout, "limit"))

        # 3. no crea la carpeta del lote
        check("--print-config no crea la carpeta del lote",
              not os.path.exists(os.path.join(tmp, "lotes", "FREQ999")))

        # 2. sin TSV en ningún sitio
        p = corre(tmp, "--lote", "lotes/FREQ000")
        check("sin TSV sale distinto de 0", p.returncode != 0, f"codigo {p.returncode}")
        check("y dice que lo pases con --tsv", "--tsv" in p.stderr, p.stderr[-200:])

        # 4. lote existente con parámetros recordados
        lote = os.path.join(tmp, "lotes", "FREQ999")
        os.makedirs(os.path.join(lote, "informes"), exist_ok=True)
        with open(os.path.join(lote, "informes", "parametros.json"), "w", encoding="utf-8") as fh:
            json.dump({"offset": 4.25, "searchterm": "freq999b", "limit": 100,
                       "tsv": "FREQ999_Tab.txt", "tsv_sha256": "x"}, fh)
        p = corre(tmp, "--lote", "lotes/FREQ999")
        check("recupera el offset guardado", campo(p.stdout, "offset") == "4.25",
              campo(p.stdout, "offset"))
        check("recupera el searchterm guardado", campo(p.stdout, "searchterm") == "freq999b",
              campo(p.stdout, "searchterm"))
        check("recupera el limit guardado", campo(p.stdout, "limit") == "100",
              campo(p.stdout, "limit"))

        # 5. lo explícito gana
        p = corre(tmp, "--lote", "lotes/FREQ999", "--offset", "1.5", "--searchterm", "otro")
        check("--offset explícito gana al guardado", campo(p.stdout, "offset") == "1.5",
              campo(p.stdout, "offset"))
        check("--searchterm explícito gana al guardado", campo(p.stdout, "searchterm") == "otro",
              campo(p.stdout, "searchterm"))

        # 6. --tsv explícito gana a la búsqueda por nombre
        otro = tsv_de(tmp, "OTRO")
        p = corre(tmp, "--lote", "lotes/FREQ999", "--tsv", otro)
        check("--tsv explícito se respeta", campo(p.stdout, "tsv") == otro, campo(p.stdout, "tsv"))

        # 7. otra campaña (MULTIFLOW_TSVs/ayesha/...) también vale
        otra = tsv_de(tmp, "FREQ777", campana="ayesha")
        p = corre(tmp, "--lote", "lotes/FREQ777")
        check("encuentra el TSV en cualquier campaña", campo(p.stdout, "tsv") == otra,
              campo(p.stdout, "tsv"))

    print()
    if FAIL:
        print(f"TOTAL: FAIL {', '.join(FAIL)}")
        return 1
    print("TOTAL: todo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
