#! /usr/bin/env python3

import os
import sys
from typing import Dict, List, Any
import pandas as pd
import pyarrow.parquet as pq
import json
import duckdb
import argparse
from datetime import datetime, timezone

# Discrepancias por causa. Solo las tres primeras significan "no hay voz que alinear"
# y se descartan (el clip no entra, el lote sigue): un fallo de insercion es un bug y
# sigue abortando. Ver INVENTARIO_FALLOS_LOTES.md §3.
DISCARD_REASONS = ("prosody_missing", "prosody_empty", "combined_empty")
DEFAULT_MAX_DISCARD_PCT = 5.0

discarded = {motivo: [] for motivo in DISCARD_REASONS}   # motivo -> [(clip, origen)]
fatal_errors = []                                        # rutas con fallo de insercion


def clip_name(path: str) -> str:
    """Nombre del clip a partir de cualquiera de sus parquets."""
    return os.path.basename(path).replace(".parquet", "")


def record_discard(motivo: str, path: str) -> None:
    """Anota un descarte con su motivo; nunca en silencio."""
    discarded[motivo].append((clip_name(path), path))

def prepare_prosody_data(prosody_path: str) -> pd.DataFrame:
    """
    Load and prepare prosody data from a parquet file.
    """
    try:
        prosody_df = pq.read_table(prosody_path).to_pandas()
    except Exception as e:
        print(f"[ERROR] Reading prosody file {prosody_path}: {e}")
        return pd.DataFrame()
    
    # One prosody row per video frame. Positional sampling (`iloc[::6]`, the
    # previous behaviour) assumes the producer emits exactly 6 rows per frame in
    # frame order; parole already interpolates onto the frame timestamps (1 row per
    # frame), so that slice silently dropped 5/6 of valid measurements. Selecting by
    # frame value is correct for either density and for any row order.
    prosody_reduced_df = prosody_df.rename(columns={"Frame": "frame"})
    prosody_reduced_df['frame'] = prosody_reduced_df['frame'].astype(int)
    prosody_reduced_df = prosody_reduced_df.drop_duplicates(subset='frame', keep='first')
    prosody_reduced_df = prosody_reduced_df.sort_values('frame')
    
    # Convert datetime to UTC, then remove timezone information
    if 'datetime' in prosody_reduced_df.columns:
        if prosody_reduced_df['datetime'].dt.tz:
            prosody_reduced_df['datetime'] = prosody_reduced_df['datetime'].dt.tz_convert(timezone.utc)
        prosody_reduced_df['datetime'] = prosody_reduced_df['datetime'].dt.tz_localize(None)
        
        # Ensure microsecond precision
        prosody_reduced_df['datetime'] = pd.to_datetime(prosody_reduced_df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    
    return prosody_reduced_df

def prepare_words_data(words_path: str) -> pd.DataFrame:
    """
    Load and prepare words data from a JSON file.
    """
    if not os.path.exists(words_path):
        return pd.DataFrame({
            "frame": [0],
            "words": [""],
            "scores_w": [""]
        })
    
    try:
        with open(words_path, 'r') as f:
            words = json.load(f)
        
        processed_words = []
        for frame in words:
            processed_words.append({
                "frame": frame["frame_number"],
                "words": ", ".join(frame["words"]),
                "scores_w": ", ".join(map(str, frame["scores"]))
            })
        
        return pd.DataFrame(processed_words)
    except Exception as e:
        print(f"[ERROR] Processing words file {words_path}: {e}")
        return pd.DataFrame({
            "frame": [0],
            "words": [""],
            "scores_w": [""]
        })

def prepare_pose_data(pose_path: str) -> pd.DataFrame:
    """
    Load and prepare pose data from a parquet file.
    """
    try:
        pose_df = pq.read_table(pose_path).to_pandas()
        pose_df["frame"] += 1
        return pose_df
    except Exception as e:
        print(f"[ERROR] Reading pose file {pose_path}: {e}")
        return pd.DataFrame()

def combine_all_data(prosody_data: pd.DataFrame, words_data: pd.DataFrame, pose_data: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Combine prosody, words, and pose data into a single DataFrame.
    """
    if len(prosody_data) != len(words_data):
       print(f"[WARNING] Mismatch in row numbers for file: {filename} in speech analysis and words data")
    
    combined_prosody_words = prosody_data.merge(words_data, on='frame', how='outer')
    final_combined_data = pd.merge(combined_prosody_words, pose_data, on="frame", how="outer")
    
    # Ensure datetime is in correct format
    if 'datetime' in final_combined_data.columns:
        final_combined_data['datetime'] = pd.to_datetime(final_combined_data['datetime'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    
    return final_combined_data

def process_single_file(parquet_file_path: str, db_file: str) -> None:
    """
    Process a single pose file and related files.
    """
    print(f"Processing file: {parquet_file_path}")

    # Derivar rutas de archivos relacionados
    prosody_file_path = parquet_file_path.replace("parquet_files", "parquet_speech_analysis")
    words_file_path = parquet_file_path.replace("parquet_files", "whisperx_framer_output").replace(".parquet", ".json")
    
    # Verificar existencia del archivo de prosodia
    if not os.path.exists(prosody_file_path):
        print(f"[WARNING] Prosody file not found: {prosody_file_path}")
        record_discard("prosody_missing", prosody_file_path)
        return
    
    prosody_data = prepare_prosody_data(prosody_file_path)
    if prosody_data.empty:
        print(f"[WARNING] Prosody data is empty for file: {prosody_file_path}")
        record_discard("prosody_empty", prosody_file_path)
        return

    words_data = prepare_words_data(words_file_path)
    pose_data = prepare_pose_data(parquet_file_path)
    combined_data = combine_all_data(prosody_data, words_data, pose_data, parquet_file_path)
    
    if combined_data.empty:
        print(f"[WARNING] Combined data is empty for file: {parquet_file_path}")
        record_discard("combined_empty", parquet_file_path)
        return
    
    try:
        con = duckdb.connect(db_file)
        con.register("combined_data", combined_data)
        con.execute("CREATE TABLE IF NOT EXISTS multi_data AS SELECT * FROM combined_data LIMIT 0")
        con.execute("INSERT INTO multi_data SELECT * FROM combined_data")
        # Eliminar filas donde id es null
        con.execute("DELETE FROM multi_data WHERE id is null")
        con.close()
        print(f"File processed and inserted into DB: {parquet_file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to insert data from {parquet_file_path} into DuckDB: {e}")
        fatal_errors.append(parquet_file_path)


def write_discard_report(path: str) -> None:
    """TSV de descartes: una fila por clip, con motivo y origen. Sirve de evidencia."""
    carpeta = os.path.dirname(path)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("clip\treason\tsource\n")
        for motivo in DISCARD_REASONS:
            for clip, origen in discarded[motivo]:
                fh.write(f"{clip}\t{motivo}\t{origen}\n")
    print(f"Discard report: {path}")


def main() -> None:
    """
    Main function to parse command line arguments and process files.
    """
    parser = argparse.ArgumentParser(description="Process multimodal data files.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to the input folder")
    parser.add_argument("-n", "--n_persons", type=str, required=True, help="Number of persons")
    parser.add_argument("-d", "--db_name", type=str, default="minos.duckdb", help="Path to DuckDB database file")
    parser.add_argument("--discard-report", type=str, default=None,
                        help="Escribe aqui el TSV de clips descartados (clip, motivo, origen)")
    parser.add_argument("--max-discard-pct", type=float, default=DEFAULT_MAX_DISCARD_PCT,
                        help="Umbral de descartes (%% de los clips) que para el lote; 0 lo desactiva "
                             "(defecto %(default)s)")
    parser.add_argument("--strict-discard", action="store_true",
                        help="Cualquier descarte aborta el lote: conducta vieja, para depurar")
    args = parser.parse_args()
    
    n_persons = f"{args.n_persons}_persons"
    parquet_dir = os.path.join(args.input_folder, "dataset", n_persons, "parquet_files")

    if not os.path.exists(parquet_dir):
        print(f"[ERROR] Parquet folder does not exist: {parquet_dir}")
        return

    print(f"Input folder: {args.input_folder}")
    print(f"Number of persons: {args.n_persons}")
    print(f"DuckDB database file: {args.db_name}")
    parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    print(f"Number of Parquet files found: {len(parquet_files)}")

    for parquet_file in parquet_files:
        file_path = os.path.join(parquet_dir, parquet_file)
        process_single_file(file_path, args.db_name)
    
    # ---------------------------------------------------------------- resumen
    total_discards = sum(len(v) for v in discarded.values())
    total_fatal = len(fatal_errors)
    total_clips = len(parquet_files)
    pct = (100.0 * total_discards / total_clips) if total_clips else 0.0

    # La linea historica se mantiene: la leen scripts y logs viejos.
    print(f"\nTotal discrepancies: {total_discards + total_fatal}")
    print("Descartados: " + " ".join(f"{m}={len(discarded[m])}" for m in DISCARD_REASONS))
    if total_fatal:
        print(f"Errores fatales: {total_fatal}")
    if total_discards:
        print(f"Descartes: {total_discards}/{total_clips} ({pct:.1f} %)")
    issues = [origen for motivo in DISCARD_REASONS for _, origen in discarded[motivo]] + fatal_errors
    if issues:
        print("Files with issues:")
        for f in issues:
            print(f)

    if args.discard_report:
        write_discard_report(args.discard_report)

    # Un fallo de insercion no es un clip mudo: sigue abortando.
    if total_fatal:
        print(f"[ERROR] {total_fatal} clip(s) no se pudieron insertar en multi_data "
              f"(bug de datos/esquema, no un clip sin voz)")
        sys.exit(1)

    # Politica: descartar el clip, no el lote. Pero un lote que descarta todo, o que pasa
    # del umbral, no es un corpus mudo: es una extraccion de voz rota.
    if total_discards:
        if args.strict_discard:
            print("[ERROR] --strict-discard: cualquier descarte aborta")
            sys.exit(1)
        if total_clips and total_discards == total_clips:
            print("[ERROR] no sobrevive ningun clip: el lote se quedaria sin datos")
            sys.exit(1)
        if args.max_discard_pct and pct > args.max_discard_pct:
            print(f"[ERROR] {total_discards} de {total_clips} clips descartados ({pct:.1f} %) "
                  f"supera el umbral {args.max_discard_pct:g} %: revisa la extraccion de voz "
                  f"(no es un corpus mudo)")
            sys.exit(1)

    # Una carpeta de parquets que no existe NO entra aqui: un bucket <n>_persons vacio
    # es un resultado normal al recorrer las clasificaciones.
    sys.exit(0)

if __name__ == "__main__":
    main()

