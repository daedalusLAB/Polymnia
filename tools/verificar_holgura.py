#!/usr/bin/env python3
"""Comprueba con qué holgura están cortados los clips que ya están descargados.

Contexto medido en un lote propio de 29 clips de NewsScape:

* La holgura (`--offset`) es lo que se añade a cada lado de la ventana marcada en el
  TSV. En este corpus las ventanas marcadas son cortisimas --de 0,002 a 0,160 s-- así
  que el clip ES practicamente la holgura: con 2.0 s cada lado el clip dura 4 s y con
  2.5 s dura 5 s. Un cambio de holgura es un 20 % del material.
* La holgura NO se puede comprobar desde el nombre del fichero: el nombre lo escribe
  `download_clips.py` con la misma holgura que uso al pedir el corte, asi que comparar
  nombre contra TSV cancela el offset y da igual con 2.0 que con 2.5. Es lo que le pasa
  a `check_tab.py`: `esperado = nombre - offset` contra `calculado = raw - offset`, la
  diferencia no depende del offset. Nunca puede ver un desfase.
* Si se puede comprobar midiendo la duracion del video: `ffprobe` contra la ventana raw
  del TSV. Diferencia medida entre duracion real y lo que anuncia el nombre: de +0,006 a
  +0,025 s en 6/6 clips (una fraccion de frame a ~30 fps). Con tolerancia de 0,15 s un
  error de holgura de 0,5 s por lado (1,0 s de duracion) salta sin falsos positivos.

Por eso el emparejamiento TSV<->fichero usa el **centro de la ventana**, que es
invariante a la holgura: anadir `pad` a un lado y restarlo al otro no lo mueve. Asi el
centro del fichero localiza la fila del TSV sin adivinar, y la holgura usada sale de:

    holgura_usada = (duracion_real - ventana_raw_del_TSV) / 2

que es una medida independiente del nombre y del offset que uno crea haber pedido.

Salida: una linea por clip muestreado y un resumen. Codigos de salida:

  0  todas las holguras medidas dentro de tolerancia
  1  alguna holgura fuera de tolerancia, o no se ha podido medir ni un clip

Uso:
    verificar_holgura.py --videos DIR --tab TSV [--offset 2.0] [--searchterm ozbek1]
                         [--sample 12] [--tol 0.15] [--show N]
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys

# El divisor del cuarto campo cambia segun la epoca del programa: a partir de 2017 el
# campo se llama centisegundos por compatibilidad pero lleva milisegundos.
def divisor_de(nombre_programa: str) -> int:
    return 1000 if os.path.basename(nombre_programa).startswith("2017") else 100


def filas_del_tab(ruta: str) -> tuple[dict[tuple[str, float], list[float]], int]:
    """(programa, centro de la ventana) -> duraciones raw marcadas en el TSV, y nº de invertidas.

    Una lista con mas de un elemento significa que dos filas comparten programa y
    centro: ese clip no se puede atribuir a una sola fila y se deja fuera.

    Las filas con ventana invertida (fin <= inicio) NO son un error aqui: en un TSV real
    de 300 filas hay 4 asi, de -0.306 a -0.001 s, y el pipeline las acepta porque la
    holgura se las traga (inicio - pad sigue siendo menor que fin + pad). La formula
    holgura = (duracion - ventana_raw) / 2 funciona igual con ventana negativa, asi que
    se miden y solo se cuentan en el resumen.
    """
    por_centro: dict[tuple[str, float], list[float]] = {}
    invertidas = 0
    with open(ruta, encoding="utf-8") as handle:
        for numero, linea in enumerate(handle, 1):
            if not linea.strip():
                continue
            campos = linea.rstrip("\n").split("\t")
            if len(campos) < 6:
                raise ValueError(f"{ruta}:{numero}: hacen falta >=6 campos separados por tab")
            try:
                d = divisor_de(campos[0])
                inicio = int(campos[2]) + int(campos[3]) / d
                fin = int(campos[4]) + int(campos[5]) / d
            except ValueError:
                raise ValueError(f"{ruta}:{numero}: los campos 3-6 tienen que ser enteros") from None
            if fin <= inicio:
                invertidas += 1
            programa = os.path.basename(campos[0]).replace(".txt", "")
            centro = round((inicio + fin) / 2, 6)
            por_centro.setdefault((programa, centro), []).append(round(fin - inicio, 6))
    return por_centro, invertidas


def trocea_nombre(archivo: str, searchterm: str) -> tuple[str, float, float] | None:
    """`programa_inicio_fin_<searchterm>.mp4` -> (programa, inicio, fin) o None.

    `inicio` y `fin` son los que escribio el descargador, holgura incluida, por eso no
    se usan nunca como medida de la holgura: solo sirven para el centro.
    """
    nombre = os.path.basename(archivo)
    if nombre.endswith(".mp4"):
        nombre = nombre[:-4]
    sufijo = f"_{searchterm}"
    if not nombre.endswith(sufijo):
        return None
    campo = nombre[: -len(sufijo)]
    partes = campo.rsplit("_", 2)
    if len(partes) != 3:
        return None
    programa, inicio, fin = partes
    try:
        return programa, float(inicio), float(fin)
    except ValueError:
        return None


def duracion(real: str, ffprobe: str) -> float:
    """Segundos del fichero segun ffprobe. Lanza si el fichero no se puede leer."""
    proceso = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", real],
        capture_output=True, text=True, check=True,
    )
    texto = proceso.stdout.strip()
    if not texto:
        raise ValueError(f"{real}: ffprobe no devuelve duracion")
    return float(texto)


def muestrea(nombres: list[str], muestra: int) -> list[str]:
    """Seleccion determinista: orden alfabetico y indices equiespaciados.

    Determinista para que dos pasadas del mismo lote comprueben los mismos clips; si
    fallan, la lista es reproducible.
    """
    ordenados = sorted(nombres)
    if muestra <= 0 or muestra >= len(ordenados):
        return ordenados
    paso = (len(ordenados) - 1) / (muestra - 1) if muestra > 1 else 0
    indices = sorted({round(i * paso) for i in range(muestra)})
    return [ordenados[i] for i in indices]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", required=True, help="directorio con los .mp4 descargados")
    ap.add_argument("--tab", required=True, help="el TSV efectivo con el que se descargo")
    ap.add_argument("--offset", type=float, default=2.0,
                    help="holgura por lado que crees que se uso (por defecto 2.0)")
    ap.add_argument("--searchterm", default="beginning_to_end",
                    help="sufijo del nombre del clip, el mismo de la descarga")
    ap.add_argument("--sample", type=int, default=12,
                    help="cuantos clips se miden, 0 = todos (por defecto 12)")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="desvio maximo admitido por lado en segundos (por defecto 0.15)")
    ap.add_argument("--show", type=int, default=0, help="imprime detalle de N clips medidos")
    args = ap.parse_args()

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        print("[FALLO] no hay ffprobe en el PATH: no se puede medir la duracion de los clips")
        return 1

    try:
        por_centro, invertidas = filas_del_tab(args.tab)
    except (OSError, ValueError) as exc:
        print(f"[FALLO] {exc}")
        return 1

    if not os.path.isdir(args.videos):
        print(f"[FALLO] no existe el directorio de videos: {args.videos}")
        return 1
    mp4s = [os.path.join(raiz, f) for raiz, _, fs in os.walk(args.videos) for f in fs if f.endswith(".mp4")]
    if not mp4s:
        print(f"[FALLO] no hay ningun .mp4 en {args.videos}")
        return 1

    sin_sufijo = ambiguo = sin_tab = rotos = 0
    medidas: list[tuple[float, str]] = []
    for archivo in muestrea(mp4s, args.sample):
        trozo = trocea_nombre(archivo, args.searchterm)
        if trozo is None:
            sin_sufijo += 1
            continue
        programa, inicio, fin = trozo
        candidatas = por_centro.get((programa, round((inicio + fin) / 2, 6)), [])
        if len(candidatas) == 0:
            sin_tab += 1
            continue
        if len(candidatas) > 1:
            ambiguo += 1
            continue
        try:
            dur = duracion(archivo, ffprobe)
        except (subprocess.CalledProcessError, ValueError, OSError):
            rotos += 1
            continue
        holgura = (dur - candidatas[0]) / 2
        medidas.append((holgura, archivo))
        if args.show and len(medidas) <= args.show:
            desvio = holgura - args.offset
            marca = "OK  " if abs(desvio) <= args.tol else "MAL "
            print(f"[{marca}] {os.path.basename(archivo)[:70]}")
            print(f"        ventana raw {candidatas[0]:.3f}s  duracion {dur:.3f}s  "
                  f"holgura medida {holgura:.3f}s  pedida {args.offset:.3f}s  desvio {desvio:+.3f}s")
    if args.show:
        print()

    fuera = [(h, a) for h, a in medidas if abs(h - args.offset) > args.tol]
    print(f"clips en {args.videos}: {len(mp4s)}  muestreados: {len(medidas) + sin_sufijo + ambiguo + sin_tab + rotos}")
    if medidas:
        holguras = sorted(h for h, _ in medidas)
        print(f"holgura medida por lado:  min {holguras[0]:.3f}  mediana {statistics.median(holguras):.3f}  "
              f"max {holguras[-1]:.3f}   (pedida {args.offset:.3f}, tolerancia {args.tol:.3f})")
    if sin_sufijo:
        print(f"  nombres sin el sufijo _{args.searchterm}: {sin_sufijo}  "
              f"(el searchterm de la descarga no es este?)")
    if sin_tab:
        print(f"  clips sin fila atribuible en el tab: {sin_tab}")
    if invertidas:
        print(f"  filas del tab con ventana invertida (las traga la holgura): {invertidas}")
    if ambiguo:
        print(f"  clips con dos filas del tab en el mismo centro: {ambiguo}")
    if rotos:
        print(f"  clips que ffprobe no ha podido leer: {rotos}")

    if not medidas:
        print("[FALLO] no se ha podido medir ningun clip: no se puede confirmar la holgura")
        return 1
    if fuera:
        holguras = sorted({round(h, 3) for h, _ in fuera})
        print(f"[FALLO] {len(fuera)}/{len(medidas)} clips estan cortados con otra holgura "
              f"(holguras medidas: {holguras})")
        peor = max(fuera, key=lambda x: abs(x[0] - args.offset))[0]
        # la medida lleva el sesgo de una fraccion de frame (hasta ~0.02 s), asi que el
        # valor que se sugiere se redondea al cuarto de segundo mas cercano
        sugerida = round(peor * 4) / 4
        print(f"        ej. holgura medida {peor:.3f}s por lado, sesgo de ~0.02 s incluido")
        print(f"        pasa --offset {sugerida:g} (o regenera los clips con {args.offset:g} s por lado) "
              f"y repite")
        return 1
    print(f"[OK] {len(medidas)} clips medidos, todos dentro de {args.tol:.3f}s de {args.offset:.3f}s por lado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
