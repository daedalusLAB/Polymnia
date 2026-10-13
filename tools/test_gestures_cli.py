"""Tests for gestures.py: DuckDB (read-only) -> Parquet gesture features.

The command is exercised the way a user runs it - as a subprocess against a
synthetic DuckDB in the shape `hefesto.py` leaves `multi_data` (long format:
one row per keypoint, `points` 1/4/7 = neck / right wrist / left wrist) - and
the Parquet outputs are read back and checked against hand-computed geometry.

Pinned behaviour: the source database is never touched, an existing output
folder is refused, a failure publishes nothing, and reruns into a new folder
reproduce the same numbers.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile

import duckdb
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GESTURES = os.path.join(REPO, "gestures.py")
sys.path.insert(0, REPO)
import gesture_features as gf  # noqa: E402

FAIL = []
POSE = "pose_keypoints"
NECK = (0.0, 0.0, 0.9)          # dfMaker puts the neck at the origin
ABSENT = (None, None, 0.0)

DDL = ("CREATE TABLE {table} (id VARCHAR, people_id DOUBLE, frame BIGINT, "
       "frame_timestamp DOUBLE, type_points VARCHAR, points DOUBLE, x DOUBLE, "
       "y DOUBLE, c DOUBLE, nx DOUBLE, ny DOUBLE)")
def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------- fixtures ----

def pose_rows(clip, spec, people=1.0, neck=NECK, table_points=POSE):
    """Long pose rows: spec = [(frame, t, (nx4,ny4,c4), (nx7,ny7,c7)), ...]."""
    rows = []
    for frame, stamp, right, left in spec:
        for point, (nx, ny, c) in ((1, neck), (4, right), (7, left), (0, ABSENT)):
            rows.append([clip, people, frame, stamp, table_points, float(point),
                         None, None, c, nx, ny])
    # a face row must never be mistaken for a wrist row
    rows.append([clip, people, spec[0][0], spec[0][1], "face_keypoints", 4.0,
                 None, None, 0.9, 7.7, 7.7])
    return rows


def ramp(clip, count=10, slope=0.03, start=1, fps=30.0, gap_at=(), people=1.0):
    """Right wrist at a constant `slope` units/frame; `gap_at` frames are absent."""
    spec = []
    for i in range(count):
        frame, stamp = start + i, (start + i - 1) / fps
        right = (slope * i, 0.0, 0.9) if frame not in gap_at else ABSENT
        spec.append((frame, stamp, right, ABSENT))
    return pose_rows(clip, spec, people=people)


def build_db(path, rows, table="multi_data", columns=None):
    rows = [list(r) for r in rows]
    con = duckdb.connect(path)
    con.execute(DDL.format(table=table) if columns is None else columns.format(table=table))
    con.executemany(f"INSERT INTO {table} VALUES ({', '.join(['?'] * len(rows[0]))})", rows)
    con.close()
    return path


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def run_cli(*args):
    return subprocess.run([sys.executable, GESTURES, *[str(a) for a in args]],
                          capture_output=True, text=True, cwd=REPO)


def expect_fail(name, proc, needle=""):
    check(f"{name} falla claro", proc.returncode != 0 and needle.lower() in (proc.stderr + proc.stdout).lower(),
          f"rc={proc.returncode} stderr={proc.stderr[-400:]}")


def good_db(root, clips=("clipA",)):
    rows = []
    for clip in clips:
        rows += ramp(clip)
    return build_db(os.path.join(root, "source.duckdb"), rows)


def published(out):
    return sorted(os.listdir(out)) if os.path.isdir(out) else None


# ------------------------------------------------------------- happy path ----

def test_cli_writes_parquet_and_config():
    with tempfile.TemporaryDirectory() as root:
        db = good_db(root, clips=("clipA", "clipB"))
        out = os.path.join(root, "features")
        proc = run_cli("--db_name", db, "--output_folder", out,
                       "--min-keypoint-confidence", "0.2", "--max-gap-frames", "3")
        check("cmd termina 0", proc.returncode == 0, proc.stderr[-500:])
        check("los tres artefactos publicados",
              published(out) == ["gesture_config.json", "gesture_frames.parquet",
                                 "gesture_sequences.parquet"], str(published(out)))
        frames = pd.read_parquet(os.path.join(out, "gesture_frames.parquet"))
        seqs = pd.read_parquet(os.path.join(out, "gesture_sequences.parquet"))
        cfg = json.load(open(os.path.join(out, "gesture_config.json")))
        check("esquema de frames = FRAME_COLUMNS", list(frames.columns) == list(gf.FRAME_COLUMNS))
        check("esquema de secuencias = SEQUENCE_COLUMNS", list(seqs.columns) == list(gf.SEQUENCE_COLUMNS))
        check("una fila de secuencia por clip", len(seqs) == 2 and set(seqs["id"]) == {"clipA", "clipB"})
        check("10 frames por clip", len(frames) == 20)
        check("columnas de cuello fuera de la salida",
              not [c for c in frames.columns if c.endswith(("_1", "_0"))])
        check("velocidad 0.9 u/s leida del Parquet",
              all(abs(v - 0.9) < 1e-7 for v in frames["right_velocity_x"]), str(frames["right_velocity_x"][:3]))
        check("config guarda parametros pedidos, referencia, unidades y span",
              set(cfg["parameters"]) == {"min_keypoint_confidence", "max_gap_frames",
                                         "window_size", "poly_order", "requested"}
              and cfg["parameters"]["requested"] == {"min_keypoint_confidence": 0.2,
                                                     "max_gap_frames": 3,
                                                     "window_size": 7, "poly_order": 2}
              and "neck" in json.dumps(cfg["reference"]).lower()
              and "units" in cfg and "source_frame_span" in cfg["reference"], json.dumps(cfg)[:300])
        check("config declara versiones", {"python", "pandas", "numpy", "scipy", "duckdb"} <= set(cfg["versions"]))
        check("sin restos de staging en el padre",
              [d for d in os.listdir(root) if d.startswith(".gesture-staging")] == [])


def test_cli_source_database_is_untouched():
    with tempfile.TemporaryDirectory() as root:
        db = good_db(root)
        before, digest = sha256(db), sha256(db)
        run_cli("--db_name", db, "--output_folder", os.path.join(root, "features"))
        con = duckdb.connect(db, read_only=True)
        rows = con.execute("SELECT count(*) FROM multi_data").fetchone()[0]
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        con.close()
        check("bytes de la fuente identicos", sha256(db) == digest, "la fuente cambio de contenido")
        check("tabla y filas sin tocar", rows == 41 and tables == ["multi_data"], f"{rows} {tables}")
        check("sin -wal/.wal lateral", not os.path.exists(db + ".wal"))
        check("el checksum inicial tenia sentido", before == digest)


def test_cli_rerun_into_new_folder_is_deterministic():
    with tempfile.TemporaryDirectory() as root:
        db = good_db(root)
        first, second = os.path.join(root, "a"), os.path.join(root, "b")
        p1 = run_cli("--db_name", db, "--output_folder", first)
        p2 = run_cli("--db_name", db, "--output_folder", second)
        check("ambas pasadas terminan 0", p1.returncode == 0 and p2.returncode == 0, p2.stderr[-300:])
        fa, fb = pd.read_parquet(f"{first}/gesture_frames.parquet"), pd.read_parquet(f"{second}/gesture_frames.parquet")
        sa, sb = pd.read_parquet(f"{first}/gesture_sequences.parquet"), pd.read_parquet(f"{second}/gesture_sequences.parquet")
        check("frames identicos entre pasadas", fa.equals(fb))
        check("secuencias identicas entre pasadas", sa.equals(sb))
        check("los dos directorios conviven", published(first) == published(second))


# ---------------------------------------------------------------- refusals ----

def test_cli_refuses_existing_output_folder():
    with tempfile.TemporaryDirectory() as root:
        db = good_db(root)
        out = os.path.join(root, "features")
        os.makedirs(out)
        with open(os.path.join(out, "sentinel.txt"), "w") as fh:
            fh.write("previo")
        proc = run_cli("--db_name", db, "--output_folder", out)
        expect_fail("carpeta de salida existente", proc, "exist")
        check("el contenido previo sigue intacto",
              published(out) == ["sentinel.txt"] and open(f"{out}/sentinel.txt").read() == "previo",
              str(published(out)))
        # an empty existing folder is still refused: never overwrite, never reuse
        os.remove(os.path.join(out, "sentinel.txt"))
        expect_fail("carpeta vacia existente tambien se rechaza",
                    run_cli("--db_name", db, "--output_folder", out), "exist")


def test_cli_rejects_missing_or_unusable_database():
    with tempfile.TemporaryDirectory() as root:
        missing = os.path.join(root, "nope.duckdb")
        expect_fail("base de datos inexistente",
                    run_cli("--db_name", missing, "--output_folder", f"{root}/o1"), "not")
        empty = os.path.join(root, "empty.duckdb")
        duckdb.connect(empty).close()
        expect_fail("sin tabla multi_data",
                    run_cli("--db_name", empty, "--output_folder", f"{root}/o2"), "multi_data")
        rows = ramp("clipA")
        narrow = ("CREATE TABLE {table} (id VARCHAR, people_id DOUBLE, frame BIGINT, "
                  "type_points VARCHAR, points DOUBLE, x DOUBLE, y DOUBLE, c DOUBLE, "
                  "nx DOUBLE, ny DOUBLE)")
        bad = build_db(os.path.join(root, "narrow.duckdb"), [r[:3] + r[4:] for r in rows], columns=narrow)
        expect_fail("faltan columnas de pose",
                    run_cli("--db_name", bad, "--output_folder", f"{root}/o3"), "frame_timestamp")
        check("ningun intento publico salida",
              [d for d in os.listdir(root) if d.startswith("o")] == [], str(os.listdir(root)))


def test_cli_rejects_ambiguous_and_impossible_rows():
    with tempfile.TemporaryDirectory() as root:
        base = ramp("clipA")
        conflict = [list(r) for r in base if r[2] == 3 and r[5] == 4.0]
        conflict[0][9] = 0.75                      # same key, different coordinate
        db = build_db(os.path.join(root, "conflict.duckdb"), base + conflict)
        expect_fail("filas duplicadas contradictorias",
                    run_cli("--db_name", db, "--output_folder", f"{root}/o1"), "conflict")

        stampy = [list(r) for r in base]
        for row in stampy:
            if row[2] == 3 and row[5] == 4.0:
                row[3] = 99.0                      # same frame, two timestamps
        db = build_db(os.path.join(root, "stamp.duckdb"), stampy)
        expect_fail("dos timestamps para el mismo frame",
                    run_cli("--db_name", db, "--output_folder", f"{root}/o2"), "timestamp")

        backwards = [list(r) for r in base]
        for row in backwards:
            if row[2] == 4:
                row[3] = -1.0                      # timestamp goes backwards
        db = build_db(os.path.join(root, "back.duckdb"), backwards)
        expect_fail("timestamps no monotonicos",
                    run_cli("--db_name", db, "--output_folder", f"{root}/o3"), "increase")


def test_cli_exact_duplicate_rows_are_dropped_not_double_counted():
    """`merge` appends, so an identical row twice must not fake a second sample."""
    with tempfile.TemporaryDirectory() as root:
        base = ramp("clipA")
        db = build_db(os.path.join(root, "dup.duckdb"), base + [list(r) for r in base])
        clean = build_db(os.path.join(root, "clean.duckdb"), base)
        out1, out2 = os.path.join(root, "dup"), os.path.join(root, "clean")
        p1 = run_cli("--db_name", db, "--output_folder", out1)
        p2 = run_cli("--db_name", clean, "--output_folder", out2)
        check("duplicados exactos aceptados", p1.returncode == 0, p1.stderr[-300:])
        check("mismos valores que sin duplicados",
              pd.read_parquet(f"{out1}/gesture_frames.parquet").equals(
                  pd.read_parquet(f"{out2}/gesture_frames.parquet")))


def test_cli_requires_a_verifiable_neck_origin():
    """Nose-origin (or any other) data must fail, not silently mean something else."""
    with tempfile.TemporaryDirectory() as root:
        nose = [list(r) for r in ramp("clipA")]
        for row in nose:
            if row[5] == 1.0:
                row[9], row[10] = 0.42, -0.13      # neck not at the origin
        db = build_db(os.path.join(root, "nose.duckdb"), nose)
        expect_fail("origen que no es el cuello",
                    run_cli("--db_name", db, "--output_folder", f"{root}/o1"), "neck")

        no_neck = [r for r in ramp("clipA") if r[5] != 1.0]
        db = build_db(os.path.join(root, "noneck.duckdb"), no_neck)
        expect_fail("origen no verificable sin point 1",
                    run_cli("--db_name", db, "--output_folder", f"{root}/o2"), "neck")
        check("nada publicado", [d for d in os.listdir(root) if d.startswith("o")] == [])


def test_cli_failure_publishes_nothing_and_cleans_its_staging():
    with tempfile.TemporaryDirectory() as root:
        good = ramp("clipA")
        broken = [list(r) for r in ramp("clipB")]
        for row in broken:
            if row[2] == 5:
                row[3] = -7.0                      # clipB is impossible, clipA is fine
        rows = good + broken
        db = build_db(os.path.join(root, "mixed.duckdb"), rows)
        out = os.path.join(root, "features")
        proc = run_cli("--db_name", db, "--output_folder", out)
        expect_fail("un clip imposible tumba la ejecucion", proc, "increase")
        check("no se publica nada parcial", not os.path.exists(out))
        check("el staging propio se limpia",
              [d for d in os.listdir(root) if d.startswith(".gesture-staging")] == [], str(os.listdir(root)))


# ------------------------------------------------------------------- flags ----

def test_cli_gap_flags_aliases_and_conflicts():
    with tempfile.TemporaryDirectory() as root:
        rows = ramp("clipA", count=8, gap_at=(4, 5))
        db = build_db(os.path.join(root, "gap.duckdb"), rows)
        off, on = os.path.join(root, "off"), os.path.join(root, "on")
        p0 = run_cli("--db_name", db, "--output_folder", off, "--max_gap_frames", 0)
        p1 = run_cli("--db_name", db, "--output_folder", on, "--max-gap-frames", "3")
        check("guion bajo y guion medio valen igual", p0.returncode == 0 and p1.returncode == 0,
              p1.stderr[-300:])
        s_off = pd.read_parquet(f"{off}/gesture_sequences.parquet")
        s_on = pd.read_parquet(f"{on}/gesture_sequences.parquet")
        check("gap 0 no intercala nada", s_off["right_n_interpolated"].iloc[0] == 0.0, str(s_off["right_n_interpolated"]))
        check("gap 3 rellena los 2 huecos", s_on["right_n_interpolated"].iloc[0] == 2.0, str(s_on["right_n_interpolated"]))
        f_on = pd.read_parquet(f"{on}/gesture_frames.parquet")
        gap = f_on[f_on["frame"].isin([4, 5])]
        check("los frames interpolados salen marcados",
              bool(gap["right_interpolated"].all()) and not bool(gap["right_observed"].any()))

        bad = run_cli("--db_name", db, "--output_folder", os.path.join(root, "x1"),
                      "--max_gap_frames", 1, "--max-gap-frames", 4)
        check("alias en conflicto fallan (exit 2)", bad.returncode == 2 and "conflict" in bad.stderr.lower(),
              f"rc={bad.returncode} {bad.stderr[-300:]}")
        bad_conf = run_cli("--db_name", db, "--output_folder", os.path.join(root, "x2"),
                           "--min-keypoint-confidence", "1.5")
        check("confidence fuera de [0,1] falla (exit 2)",
              bad_conf.returncode == 2 and "min_keypoint_confidence" in bad_conf.stderr,
              f"rc={bad_conf.returncode} {bad_conf.stderr[-300:]}")
        bad_win = run_cli("--db_name", db, "--output_folder", os.path.join(root, "x3"),
                          "--window-size", "6")
        check("window par falla (exit 2)", bad_win.returncode == 2 and "odd" in bad_win.stderr,
              f"rc={bad_win.returncode} {bad_win.stderr[-300:]}")
        check("los intentos fallidos no publican", [d for d in os.listdir(root) if d == "x1" or d == "x2" or d == "x3"] == [])


def test_cli_help_and_optional_repo_wrapper():
    wrapper = os.path.join(REPO, "bin", "polymnia")
    has_wrapper = os.path.isfile(wrapper)
    if has_wrapper:
        proc = subprocess.run([wrapper, "python", "gestures.py", "--help"],
                              capture_output=True, text=True, cwd=REPO)
    else:
        proc = run_cli("--help")
    check("help del comando de gestos funciona", proc.returncode == 0, proc.stderr[-300:])
    check("help explica db_name y output_folder",
          "--db_name" in proc.stdout and "--output_folder" in proc.stdout
          and "read-only" in proc.stdout.lower(), proc.stdout[:400])
    check("help menciona los flags de hueco y suavizado",
          "--max-gap-frames" in proc.stdout and "--window-size" in proc.stdout
          and "--poly-order" in proc.stdout, proc.stdout[:600])
    if not has_wrapper:
        print("[skip] wrapper opcional ausente; CLI standalone verificada")
        return
    helper = subprocess.run([wrapper, "help"], capture_output=True, text=True, cwd=REPO)
    check("bin/polymnia sigue igual (no se inventa un alias gestures)",
          helper.returncode == 0 and "polymnia test" in helper.stdout
          and "polymnia gestures" not in helper.stdout, helper.stdout[:200])
    untouched = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", "bin/polymnia"],
                               capture_output=True, text=True)
    check("bin/polymnia no se modifico", untouched.returncode == 0 and untouched.stdout.strip() == "",
          untouched.stdout)


def test_cli_keeps_untimed_frames_and_reports_them_truthfully():
    """The shipped DB has no frame_timestamp on the last frame of each clip.

    Those frames carry real geometry, so they stay in the output with a NULL
    clock and explicit flags; nothing is claimed about "dropping" them.
    """
    with tempfile.TemporaryDirectory() as root:
        rows = [list(r) for r in ramp("clipA", count=10)]
        for row in rows:
            if row[2] == 10:
                row[3] = None                      # neck + both wrists of the final frame
        db = build_db(os.path.join(root, "lastframe.duckdb"), rows)
        out = os.path.join(root, "features")
        proc = run_cli("--db_name", db, "--output_folder", out)
        check("filas sin timestamp no tumban la ejecucion", proc.returncode == 0, proc.stderr[-300:])
        check("el aviso se imprime", "frame_timestamp" in proc.stdout, proc.stdout[-300:])
        frames = pd.read_parquet(f"{out}/gesture_frames.parquet")
        check("el frame sin reloj se conserva", len(frames) == 10 and frames["frame"].max() == 10)
        last = frames[frames["frame"] == 10].iloc[0]
        check("su geometria real se mide (lo que el descarte perdia)",
              abs(last["right_neck_distance"] - math.hypot(0.03 * 9, 0.0)) < 1e-12
              and not bool(last["clock_known"]),
              f'{last["right_neck_distance"]} {last["clock_known"]}')
        check("sin reloj no hay velocidad", math.isnan(last["right_speed"]))
        check("solo ese frame carece de reloj",
              int(frames["clock_known"].sum()) == 9 and int(frames["timestamp_inferred"].sum()) == 0)
        seq = pd.read_parquet(f"{out}/gesture_sequences.parquet").iloc[0]
        check("span de reloj declarado hasta el ultimo frame con marca, hueco contado",
              seq["frames_grid"] == 10.0 and seq["n_timed_frames"] == 9.0
              and seq["n_missing_timestamps"] == 1.0
              and abs(seq["t_start"] - 0.0) < 1e-12 and abs(seq["t_end"] - 8 / 30.0) < 1e-12,
              str(seq[["frames_grid", "n_timed_frames", "n_missing_timestamps", "t_start",
                       "t_end"]].to_dict()))
        cfg = json.load(open(f"{out}/gesture_config.json"))
        check("el JSON declara el reloj faltante sin hablar de descartes",
              cfg["source_clock"]["frames_without_timestamp"] == 1
              and cfg["source_clock"]["pose_rows_without_timestamp"] == 3
              and cfg["source_clock"]["usable_wrists_without_timestamp"] == {"right": 1, "left": 0}
              and "dropped" not in json.dumps(cfg).lower(), json.dumps(cfg.get("source_clock")))

        lost = [list(r) for r in ramp("clipZ")]
        for row in lost:
            row[3] = None
        db2 = build_db(os.path.join(root, "notimed.duckdb"), lost)
        out2 = os.path.join(root, "z")
        check("un clip sin ningun reloj conserva su geometria",
              run_cli("--db_name", db2, "--output_folder", out2).returncode == 0)
        zf = pd.read_parquet(f"{out2}/gesture_frames.parquet")
        zs = pd.read_parquet(f"{out2}/gesture_sequences.parquet").iloc[0]
        check("todos sus frames presentes, sin reloj alguno",
              len(zf) == 10 and not bool(zf["clock_known"].any()) and zs["n_timed_frames"] == 0.0)
        check("temporales NULL y geometria presente",
              math.isnan(zs["duration_s"]) and abs(zf["right_neck_distance"].iloc[0] - 0.0) < 1e-12,
              str(zf["right_neck_distance"].iloc[0]))


def test_cli_origin_requires_finite_neck_coordinates():
    """A neck row whose coordinates are NULL proves nothing about the origin."""
    with tempfile.TemporaryDirectory() as root:
        nullxy = [list(r) for r in ramp("clipA")]
        for row in nullxy:
            if row[5] == 1.0:
                row[9], row[10] = None, None
        db = build_db(os.path.join(root, "nullxy.duckdb"), nullxy)
        expect_fail("cuello con nx/ny NULL no verifica el origen",
                    run_cli("--db_name", db, "--output_folder", f"{root}/o1"), "origin")
        check("nada publicado", not os.path.exists(os.path.join(root, "o1")))


def test_cli_origin_confidence_must_be_a_probability():
    with tempfile.TemporaryDirectory() as root:
        rows = [list(r) for r in ramp("clipA")]
        for row in rows:
            if row[5] == 1.0:
                row[8] = 1.5                       # impossible neck confidence
        db = build_db(os.path.join(root, "conf15.duckdb"), rows)
        proc = run_cli("--db_name", db, "--output_folder", f"{root}/o")
        check("confidence 1.5 se rechaza (exit 1, nada publicado)",
              proc.returncode == 1 and "confidence" in proc.stderr.lower()
              and not os.path.exists(os.path.join(root, "o")),
              f"rc={proc.returncode} {proc.stderr[-300:]}")


def test_cli_origin_is_per_sequence_and_neck_columns_are_controlled():
    with tempfile.TemporaryDirectory() as root:
        off = [list(r) for r in ramp("clipB")]
        for row in off:
            if row[5] == 1.0:
                row[9], row[10] = 0.42, -0.13
        db = build_db(os.path.join(root, "two.duckdb"), ramp("clipA") + off)
        proc = run_cli("--db_name", db, "--output_folder", f"{root}/o1")
        check("una secuencia con cuello desplazado se rechaza", proc.returncode == 1,
              f"rc={proc.returncode} {proc.stderr[-200:]}")
        check("el fallo nombra solo la secuencia afectada",
              "clipB" in proc.stderr and "clipA" not in proc.stderr, proc.stderr[-300:])

        only_wrists = [r for r in ramp("clipA") if r[5] != 1.0]
        db2 = build_db(os.path.join(root, "noneck.duckdb"), only_wrists)
        expect_fail("munecas sin cuello: origen no verificable",
                    run_cli("--db_name", db2, "--output_folder", f"{root}/o2"), "origin")

        only_neck = [r for r in ramp("clipA") if r[5] == 1.0]
        db3 = build_db(os.path.join(root, "neckonly.duckdb"), only_neck)
        out3 = os.path.join(root, "o3")
        proc3 = run_cli("--db_name", db3, "--output_folder", out3)
        check("sin munecas medidas no se inventan features", proc3.returncode == 0, proc3.stderr[-200:])
        check("las columnas de muneca existen y estan NULL",
              pd.read_parquet(f"{out3}/gesture_frames.parquet")["right_neck_distance"].isna().all())

        import gestures
        try:
            gestures.verify_neck_origin(pd.DataFrame({"id": ["a"], "nx_4": [0.1]}))
            check("columnas de cuello ausentes: GestureRunError", False, "no fallo")
        except gestures.GestureRunError:
            check("columnas de cuello ausentes: GestureRunError", True)
        except Exception as exc:  # noqa: BLE001
            check("columnas de cuello ausentes: GestureRunError", False, repr(exc))


def test_cli_never_overwrites_a_concurrently_created_folder():
    """os.replace() would silently publish over a folder made after the first guard."""
    import gestures
    frames, seqs = pd.DataFrame({"id": ["a"], "frame": [1]}), pd.DataFrame({"id": ["a"]})
    with tempfile.TemporaryDirectory() as root:
        target = os.path.join(root, "features")

        def concurrent_actor(_staging):
            os.mkdir(target)                       # another publisher wins, for real

        try:
            gestures.publish(target, frames, seqs, {"tool": "t"}, on_staged=concurrent_actor)
            check("no se publica encima de una carpeta concurrente", False, "no fallo")
        except gestures.GestureRunError as exc:
            check("no se publica encima de una carpeta concurrente",
                  "exist" in str(exc).lower(), str(exc))
        check("la carpeta del otro actor queda intacta y vacia",
              published(target) == [], str(published(target)))
        check("el staging propio se limpia",
              [d for d in os.listdir(root) if d.startswith(".gesture-staging")] == [], str(os.listdir(root)))


def test_cli_two_publishers_exactly_one_wins():
    """Two real publishers, one target: the loser must not half-overwrite the winner."""
    import gestures
    import threading
    with tempfile.TemporaryDirectory() as root:
        db = good_db(root, clips=("clipA", "clipB"))
        target = os.path.join(root, "features")
        results = []

        def publisher(tag):
            try:
                results.append((tag, gestures.run(db, target, gf.GestureConfig())))
            except gestures.GestureRunError as exc:
                results.append((tag, exc))

        threads = [threading.Thread(target=publisher, args=(tag,)) for tag in ("x", "y")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        won = [r for r in results if isinstance(r[1], dict)]
        lost = [r for r in results if isinstance(r[1], gestures.GestureRunError)]
        check("un solo publicador gana la carrera", len(won) == 1 and len(lost) == 1,
              str([(t, type(r).__name__) for t, r in results]))
        check("el ganador publico los tres artefactos completos",
              published(target) == ["gesture_config.json", "gesture_frames.parquet",
                                    "gesture_sequences.parquet"], str(published(target)))
        check("sin restos de staging",
              [d for d in os.listdir(root) if d.startswith(".gesture-staging")] == [])


def test_cli_never_follows_a_symlinked_output_path():
    import gestures
    frames, seqs = pd.DataFrame({"id": ["a"], "frame": [1]}), pd.DataFrame({"id": ["a"]})
    with tempfile.TemporaryDirectory() as root:
        elsewhere = os.path.join(root, "other")
        os.makedirs(elsewhere)
        link = os.path.join(root, "link")
        os.symlink(elsewhere, link)
        try:
            gestures.publish(link, frames, seqs, {})
            check("una ruta symlink no se sigue", False, "no fallo")
        except gestures.GestureRunError as exc:
            check("una ruta symlink no se sigue", "exist" in str(exc).lower(), str(exc))
        check("el symlink y su destino quedan intactos",
              os.path.islink(link) and published(elsewhere) == [], str(published(elsewhere)))


def test_cli_identity_flags_reject_conflicting_aliases():
    with tempfile.TemporaryDirectory() as root:
        db = good_db(root)
        before = sha256(db)
        cases = {
            "--db-name": ["--db_name", db, "--db-name", os.path.join(root, "otro.duckdb"),
                          "--output_folder", os.path.join(root, "o1")],
            "--output-folder": ["--db_name", db, "--output_folder", os.path.join(root, "o2"),
                                "--output-folder", os.path.join(root, "o2b")],
            "--table-name": ["--db_name", db, "--output_folder", os.path.join(root, "o3"),
                             "--table_name", "multi_data", "--table-name", "otra"],
        }
        for flag, args in cases.items():
            proc = run_cli(*args)
            check(f"alias en conflicto de {flag} fallan (exit 2)",
                  proc.returncode == 2 and "conflict" in proc.stderr.lower(),
                  f"rc={proc.returncode} {proc.stderr[-200:]}")
        check("los conflictos no crean ninguna salida",
              sorted(d for d in os.listdir(root) if d.startswith("o")) == [], str(os.listdir(root)))
        check("la fuente sigue intacta", sha256(db) == before)
        check("un valor unico para table_name sigue valiendo",
              run_cli("--db_name", db, "--output_folder", os.path.join(root, "ok"),
                      "--table_name", "multi_data").returncode == 0)


def test_cli_absent_left_wrist_column_keeps_right_features():
    """A source that never reports BODY_25 point 7 must not lose the right wrist."""
    with tempfile.TemporaryDirectory() as root:
        rows = [r for r in ramp("clipA") if r[5] != 7.0]
        db = build_db(os.path.join(root, "noleft.duckdb"), rows)
        out = os.path.join(root, "features")
        proc = run_cli("--db_name", db, "--output_folder", out)
        check("la ejecucion pasa", proc.returncode == 0, proc.stderr[-300:])
        frames = pd.read_parquet(f"{out}/gesture_frames.parquet")
        check("la mano presente conserva features",
              abs(frames["right_velocity_x"].iloc[1] - 0.9) < 1e-7,
              str(frames["right_velocity_x"].iloc[:3]))
        check("la mano ausente sale NULL con su esquema",
              frames["left_neck_distance"].isna().all()
              and list(frames.columns) == list(gf.FRAME_COLUMNS))
        check("el par no se afirma",
              frames["both_hands_valid"].notna().all()
              and not bool(frames["both_hands_valid"].any())
              and frames["pair_separation"].isna().all())
        check("la secuencia existe con su conteo",
              pd.read_parquet(f"{out}/gesture_sequences.parquet")["right_n_valid"].iloc[0] == 10.0)


def test_cli_reads_the_shipped_database_read_only():
    """If the real entrega_1p database is present, the adapter must accept it."""
    db = os.path.join(REPO, "..", "entrega_1p", "1p.duckdb")
    if not os.path.exists(db):
        check("entrega_1p presente (opcional)", True, "ausente: prueba omitida")
        return
    digest = sha256(db)
    with tempfile.TemporaryDirectory() as root:
        out = os.path.join(root, "features")
        proc = run_cli("--db_name", db, "--output_folder", out)
        check("base real leida y procesada", proc.returncode == 0, proc.stderr[-400:])
        frames = pd.read_parquet(f"{out}/gesture_frames.parquet")
        check("frames reales con origen en el cuello",
              len(frames) > 0 and frames["frame"].min() >= 1)
        check("la base real no se toco", sha256(db) == digest)
        check("los clips reales tienen secuencia",
              pd.read_parquet(f"{out}/gesture_sequences.parquet")["id"].nunique() > 1)
        cfg = json.load(open(f"{out}/gesture_config.json"))
        check("la base real declara 84 filas / 28 frames sin reloj",
              cfg["source_clock"]["pose_rows_without_timestamp"] == 84
              and cfg["source_clock"]["frames_without_timestamp"] == 28,
              str(cfg["source_clock"]))
        check("y las 5 derechas / 6 izquierdas utilizables de esos frames",
              cfg["source_clock"]["usable_wrists_without_timestamp"] == {"right": 5, "left": 6},
              str(cfg["source_clock"]))
        untimed = frames[~frames["clock_known"].astype("bool")]
        check("los 28 frames reales sin reloj salen con geometria, sin reloj inventado",
              len(untimed) == 28 and int(frames["timestamp_inferred"].sum()) == 0
              and untimed["right_neck_distance"].notna().sum() + untimed["left_neck_distance"].notna().sum() > 0,
              f"{len(untimed)} {int(frames['timestamp_inferred'].sum())}")
        check("nada de 'dropped' en el metadata", "dropped" not in json.dumps(cfg).lower())


def test_final_untimed_metadata_matches_requested_usability():
    cases = ((.3, .8, 1., 0), (.15, .1, 1., 1),
             (.9, .2, float("inf"), 0), (0., 0., 1., 0))
    with tempfile.TemporaryDirectory() as root:
        for i, (confidence, threshold, x, expected) in enumerate(cases):
            db = build_db(f"{root}/s{i}.duckdb", pose_rows(
                "a", [(1, None, (x, 0., confidence), ABSENT)]))
            out = f"{root}/o{i}"
            before = sha256(db)
            p = run_cli("--db_name", db, "--output_folder", out,
                        "--min-keypoint-confidence", threshold)
            check(f"metadata run {i}", p.returncode == 0, p.stderr)
            cfg = json.load(open(f"{out}/gesture_config.json"))
            check(f"requested usability {i}",
                  cfg["source_clock"]["usable_wrists_without_timestamp"]
                  == {"right": expected, "left": 0})
            check(f"source unchanged {i}", sha256(db) == before)


def test_final_empty_table_name_is_not_silently_defaulted():
    with tempfile.TemporaryDirectory() as root:
        db = good_db(root)
        before = sha256(db)
        out = f"{root}/out"
        p = run_cli("--db_name", db, "--output_folder", out,
                    "--table_name", "")
        check("empty explicit table rejected",
              p.returncode != 0 and not os.path.lexists(out))
        check("source unchanged", sha256(db) == before)


def test_final_inferred_clock_is_not_reported_as_geometry_only():
    with tempfile.TemporaryDirectory() as root:
        db = build_db(f"{root}/s.duckdb", pose_rows("a", [
            (1, 0., (1., 0., .9), ABSENT),
            (2, None, (2., 0., .9), ABSENT),
            (3, 2., (3., 0., .9), ABSENT)]))
        out = f"{root}/out"
        p = run_cli("--db_name", db, "--output_folder", out)
        check("clock inference run", p.returncode == 0, p.stderr)
        f = pd.read_parquet(f"{out}/gesture_frames.parquet")
        check("middle clock inferred",
              bool(f.iloc[1]["timestamp_inferred"])
              and abs(float(f.iloc[1]["right_speed"]) - 1.0) < 1e-8)
        check("no false geometry-only notice",
              "keep geometry only, with clock_known = false" not in p.stdout)
        cfg = json.load(open(f"{out}/gesture_config.json"))
        check("no unconditional NULL promise",
              "its timed features stay NULL" not in cfg["source_clock"]["note"])


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[fn]()
        except Exception as e:  # noqa: BLE001
            check(f"{fn} (excepcion)", False, repr(e))
    print("\nTOTAL:", "FAIL " + ", ".join(FAIL) if FAIL else "todo OK")
    sys.exit(1 if FAIL else 0)
