#!/usr/bin/env python3
"""Tests de tools/verificar_holgura.py con clips sinteticos de duracion conocida.

No tocan red ni GPU: se generan mp4 de color puro con ffmpeg a duracion exacta, se escribe
su TSV, y se comprueba que la herramienta mide la holgura con la que estan cortados.

Cubren lo medido en el lote real de 29 clips:
  * la holgura NO se deduce del nombre (lo escribe el descargador), asi que el emparejamiento
    va por centro de ventana y la medida sale de la duracion;
  * la duracion real se pasa de lo que anuncia el nombre en +0.006..+0.025 s (fraccion de
    frame), por eso la tolerancia por defecto es 0.15 s y no 0;
  * 4 de 300 filas del TSV real tienen ventana invertida y el pipeline las acepta.

Salida: linea por comprobacion, exit 0 si todas pasan, 1 si alguna falla.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zlib

AQUI = os.path.dirname(os.path.abspath(__file__))
HERRAMIENTA = os.path.join(AQUI, "verificar_holgura.py")
PYTHON = sys.executable

fallan = 0
hechas = 0


def chequea(nombre, condicion, detalle=""):
    global fallan, hechas
    hechas += 1
    if condicion:
        print(f"[ok]    {nombre}")
    else:
        fallan += 1
        print(f"[FALLO] {nombre}" + (f"  -- {detalle}" if detalle else ""))


def graba_mp4(ruta, segundos, fps=25):
    """mp4 de color puro, con ffmpeg, de duracion ~segundos (cuantizado a 1/fps)."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=c=black:s=32x32:r={fps}:d={segundos:.3f}",
         "-pix_fmt", "yuv420p", ruta],
        check=True, capture_output=True,
    )


def linea_tab(programa, inicio, fin, searchterm, offset, carpeta):
    """Escribe el fichero .txt que firmaria NewsScape y el mp4 con ese nombre."""
    d = 1000 if os.path.basename(programa).startswith("2017") else 100
    s_sec, s_ms = int(inicio), int(round((inicio - int(inicio)) * d))
    e_sec, e_ms = int(fin), int(round((fin - int(fin)) * d))
    nombre = os.path.basename(programa).replace(".txt", "")
    clip = f"{nombre}_{round(inicio - offset, 3)}_{round(fin + offset, 3)}_{searchterm}"
    return (f"{programa}\tt__{zlib.crc32(nombre.encode()) & 0xffffff:06x}\t"
            f"{s_sec}\t{s_ms}\t{e_sec}\t{e_ms}\tcontexto"), f"{clip}.mp4"


def monta_lote(dirs, casos, offset_grabado):
    """casos = [(programa, inicio_raw, fin_raw)]. Devuelve (tab, videos, duraciones)."""
    tsv = os.path.join(dirs, "efectivo.tsv")
    videos = os.path.join(dirs, "videos")
    os.makedirs(videos, exist_ok=True)
    filas, nombres = [], []
    for programa, inicio, fin in casos:
        fila, nombre = linea_tab(programa, inicio, fin, "testhol", offset_grabado, dirs)
        filas.append(fila)
        nombres.append(nombre)
        graba_mp4(os.path.join(videos, nombre), (fin + offset_grabado) - (inicio - offset_grabado))
    with open(tsv, "w", encoding="utf-8") as fh:
        fh.write("\n".join(filas) + "\n")
    return tsv, videos


def corre(videos, tab, offset, extra=()):
    proceso = subprocess.run(
        [PYTHON, HERRAMIENTA, "--videos", videos, "--tab", tab,
         "--offset", str(offset), "--searchterm", "testhol", *extra],
        capture_output=True, text=True,
    )
    return proceso.returncode, proceso.stdout + proceso.stderr


def main():
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not (ffmpeg and ffprobe):
        print("[skip] verificar_holgura necesita ffmpeg y ffprobe en el PATH")
        print("       (sin ellos no se puede medir la duracion de un clip: es el fondo de la")
        print("        comprobacion, no se sustituye por una mock)")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        # --- lote cortado a 2.0 s por lado -------------------------------------
        dirs = os.path.join(tmp, "a20")
        os.makedirs(dirs)
        tsv, videos = monta_lote(dirs, [
            ("2017-01-02_0600_US_CNN_The_Eighties.txt", 900.000, 900.320),
            ("2017-03-03_0700_US_KABC_News.txt",        400.004, 400.010),
        ], offset_grabado=2.0)

        code, salida = corre(videos, tsv, 2.0)
        chequea("clip cortado a 2.0, se pregunta 2.0 -> OK", code == 0, salida[-400:])
        chequea("la medida reportada ronda 2.0, no el valor del nombre",
                "2.0" in salida and "mediana" in salida, salida[-400:])

        code, salida = corre(videos, tsv, 2.5)
        chequea("clip cortado a 2.0, se pregunta 2.5 -> FALLO", code == 1, salida[-400:])
        chequea("el fallo dice la holgura medida y propone la salida",
                "otra holgura" in salida and "--offset 2" in salida, salida[-400:])

        # --- ventana invertida, como 4 de las 300 filas reales -----------------
        dirs = os.path.join(tmp, "inv")
        os.makedirs(dirs)
        tsv, videos = monta_lote(dirs, [("2017-11-14_0400_US_WKYC_News.txt", 1160.028, 1160.006)],
                                 offset_grabado=2.0)
        code, salida = corre(videos, tsv, 2.0)
        chequea("ventana raw invertida no rompe y se sigue midiendo",
                code == 0 and "invertida" in salida, salida[-400:])

        # --- divisor: un programa pre-2017 lleva centisegundos (/100) ----------
        dirs = os.path.join(tmp, "pre2017")
        os.makedirs(dirs)
        tsv, videos = monta_lote(dirs, [("2016-06-06_0600_US_MSNBC_Live.txt", 500.00, 500.40)],
                                 offset_grabado=2.0)
        code, salida = corre(videos, tsv, 2.0)
        chequea("programa pre-2017 usa el divisor 100 y mide bien", code == 0, salida[-400:])
        # mal interpretation: si alguien usara /1000, la ventana raw saldria 10 veces mayor
        # y la holgura medida sube ~4 s -> debe fallar al preguntar por 2.0
        fila = open(tsv, encoding="utf-8").read()
        con_msec = os.path.join(tmp, "pre2017", "msec.tsv")
        with open(con_msec, "w", encoding="utf-8") as fh:
            fh.write(fila.replace("\t500\t0\t500\t40\t", "\t500\t0000\t500\t4000\t"))
        code, salida = corre(videos, con_msec, 2.0)
        chequea("divisor equivocado en el tab se detecta, no se traga", code == 1, salida[-400:])

        # --- searchterm equivocado: nada medible -> no dice OK por defecto -----
        code, salida = corre(videos, tsv, 2.0, extra=("--searchterm", "otro"))
        chequea("searchterm que no casa con los nombres -> FALLO, no un OK vacio",
                code == 1 and "sin el sufijo" in salida, salida[-400:])

        # --- carpeta sin clips --------------------------------------------------
        vacio = os.path.join(tmp, "vacio")
        os.makedirs(os.path.join(vacio, "videos"))
        code, salida = corre(os.path.join(vacio, "videos"), tsv, 2.0)
        chequea("directorio sin mp4 -> FALLO claro", code == 1 and "ningun .mp4" in salida, salida[-300:])

        # --- dos filas del tab en el mismo centro: no se puede atribuir --------
        dirs = os.path.join(tmp, "amb")
        os.makedirs(dirs)
        tsv_amb, videos_amb = monta_lote(dirs, [("2017-01-01_0000_US_Twin_Show.txt", 10.0, 10.4)],
                                         offset_grabado=2.0)
        filas = open(tsv_amb, encoding="utf-8").read().strip()
        gemela = filas.split("\n")[0].replace("\tt__", "\tt__z").replace("contexto", "contexto2")
        with open(tsv_amb, "w", encoding="utf-8") as fh:
            fh.write(filas + "\n" + gemela + "\n")
        code, salida = corre(videos_amb, tsv_amb, 2.0)
        chequea("centro compartido por dos filas del tab -> FALLO en vez de adivinar",
                code == 1 and "mismo centro" in salida, salida[-400:])

    # --- determinismo del muestreo (sin ffmpeg) --------------------------------
    sys.path.insert(0, AQUI)
    import verificar_holgura as vh
    nombres = [f"c{i:03d}.mp4" for i in range(50)]
    una = vh.muestrea(nombres, 12)
    otra = vh.muestrea(list(reversed(nombres)), 12)
    chequea("la muestra es determinista e independiente del orden de entrada",
            len(una) == 12 and una == otra, f"{una[:3]} vs {otra[:3]}")
    chequea("trocea nombres con searchterm de varias palabras",
            vh.trocea_nombre("2017_A_B_3.0_8.0_beginning_to_end.mp4", "beginning_to_end")
            == ("2017_A_B", 3.0, 8.0))
    chequea("no trocea nombres de otro searchterm",
            vh.trocea_nombre("2017_A_B_3.0_8.0_far.mp4", "beginning_to_end") is None)

    print(f"\n{hechas - fallan}/{hechas} comprobaciones correctas")
    if fallan:
        print(f"{fallan} FALLAN")
    return 1 if fallan else 0


if __name__ == "__main__":
    sys.exit(main())
