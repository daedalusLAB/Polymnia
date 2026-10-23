#!/usr/bin/env python3
"""Guarda de la etapa Theia: no se publica un lote sin una sola anotacion.

`theia.py` inserta una fila por clip en `video_annotations`: `error` vacio (o NULL) cuando
va bien, y el motivo cuando falla. Si el endpoint rechaza la clave (401/403) o esta caido
(429/500), la etapa terminaba "con avisos" y el pipeline publicaba igual - paso con
FREQ028_4 (114/114 con error) y FREQ029 (161/161) en la tanda de octubre.

Uso:
    check_annotations.py --db lotes/X/X.duckdb [--table video_annotations]

Salida: `anotados=N error=M total=T pct_ok=X.X`
Exit: 0 si hay al menos una anotacion buena; 1 si no hay ninguna o no existe la tabla.
"""
import argparse
import os
import sys

try:
    import duckdb
except ImportError:  # pragma: no cover - el lote siempre corre en el venv del repo
    print("no duckdb")
    sys.exit(2)

SIN_ERROR = "error IS NULL OR trim(error) = ''"
CON_ERROR = "error IS NOT NULL AND trim(error) <> ''"


def cuentas(db, tabla):
    con = duckdb.connect(db, read_only=True)
    try:
        existe = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [tabla]
        ).fetchone()[0]
        if not existe:
            return None
        fila = con.execute(
            f"SELECT count(*) FILTER (WHERE {SIN_ERROR}), "
            f"count(*) FILTER (WHERE {CON_ERROR}), count(*) FROM {tabla}"
        ).fetchone()
        return tuple(int(x) for x in fila)
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, help="DuckDB del lote")
    parser.add_argument("--table", default="video_annotations", help="tabla de anotaciones")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print(f"anotados=0 error=0 total=0 pct_ok=0.0  (no existe {args.db})")
        return 1
    cuenta = cuentas(args.db, args.table)
    if cuenta is None:
        print(f"anotados=0 error=0 total=0 pct_ok=0.0  (no existe la tabla {args.table})")
        return 1
    anotados, con_error, total = cuenta
    pct = (100.0 * anotados / total) if total else 0.0
    print(f"anotados={anotados} error={con_error} total={total} pct_ok={pct:.1f}")
    return 0 if anotados > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
