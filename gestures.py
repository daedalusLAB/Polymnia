#!/usr/bin/env python3
"""Neck-relative wrist gesture features: DuckDB (read-only) -> Parquet.

Reads the long `multi_data` table that `hefesto.py` writes, keeps the BODY_25
neck (point 1) and wrist (points 4 / 7) rows, pivots them into one row per
clip / person / frame, and hands that table to `gesture_features.compute_gesture_features`
- the same public function a notebook would call. Nothing here recomputes math.

    python3 gestures.py --db_name out/climate.duckdb \
        --output_folder out/gestures --max-gap-frames 3 --min-keypoint-confidence 0.2

Guarantees
----------
* the source database is opened `read_only=True` and is never written, altered
  or re-imported; no original media or delivered table is touched;
* the output folder must not exist yet: results are staged in a temporary
  directory next to it and published with a single **no-replace** rename (Linux
  `renameat2(..., RENAME_NOREPLACE)`), so a concurrent folder - even an empty one
  created after the first check - is never overwritten. On a platform without
  that guarantee the run fails closed and cleans only its own staging directory
  instead of publishing unsafely;
* the three artefacts are `gesture_frames.parquet`, `gesture_sequences.parquet`
  and `gesture_config.json` (parameters, versions, reference frame, units and
  the meaning of the frame span). No CSV side files, no resume log.

The adapter also refuses data it cannot interpret: a missing table or column, two
identities for one option alias (`--db_name` and `--db-name` disagreeing), two
different values for the same clip / person / frame / keypoint, one frame that is
both timed and untimed, or `nx` / `ny` whose reference origin cannot be verified
per clip and person as neck-centred (a nose origin, or raw pixels) - because
every feature below is defined around the neck.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import platform
import shutil
import sys
import tempfile

import duckdb
import numpy
import pandas
import pyarrow
import scipy

import gesture_features as gf

FRAMES_FILE = "gesture_frames.parquet"
SEQUENCES_FILE = "gesture_sequences.parquet"
CONFIG_FILE = "gesture_config.json"
STAGING_PREFIX = ".gesture-staging-"
DEFAULT_TABLE = "multi_data"

POSE_TYPE_POINTS = "pose_keypoints"          # 25 BODY_25 points per person/frame
NECK_ORIGIN_TOLERANCE = 1e-6                 # dfMaker puts the neck at exactly (0, 0)
NECK_COLUMNS = tuple(f"{field}_{gf.BODY_25_NECK}"
                     for field in ("nx", "ny", "c"))   # used for validation only
WRIST_COLUMNS = tuple(f"{field}_{point}" for field in ("nx", "ny", "c")
                      for point in (gf.BODY_25_RIGHT_WRIST, gf.BODY_25_LEFT_WRIST))
LONG_REQUIRED = ("id", "people_id", "frame", "frame_timestamp",
                 "type_points", "points", "nx", "ny", "c")
# Defaults live here so an unset flag is distinguishable from an explicit value.
DEFAULT_PARAMETERS = {"min_keypoint_confidence": 0.2, "max_gap_frames": 3,
                      "window_size": 7, "poly_order": 2}
# renameat2(2) numbers by machine; publishing needs RENAME_NOREPLACE (Linux).
RENAMEAT2_SYSCALL = {"x86_64": 316, "aarch64": 276, "i686": 353, "armv7l": 382}
AT_FDCWD, RENAME_NOREPLACE = -100, 1


class GestureRunError(ValueError):
    """The run cannot start or cannot be trusted (input or output problem)."""


# --------------------------------------------------------------------- read ----

def _quote(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def load_pose_keypoints(db_path, table=DEFAULT_TABLE):
    """Neck and wrist rows of `table`, read from a strictly read-only connection."""
    if not os.path.isfile(db_path):
        raise GestureRunError(f"database not found: {db_path}")
    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception as exc:  # duckdb raises its own error types
        raise GestureRunError(f"cannot open {db_path} read-only: {exc}") from None
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if table not in tables:
            raise GestureRunError(f"table {table!r} not found in {db_path} (found: {', '.join(tables) or 'none'})")
        columns = {r[0] for r in con.execute(f"DESCRIBE {_quote(table)}").fetchall()}
        missing = [c for c in LONG_REQUIRED if c not in columns]
        if missing:
            raise GestureRunError(f"table {table!r} is missing pose columns: {', '.join(missing)}")
        query = (f"SELECT id, people_id, frame, frame_timestamp, points, nx, ny, c "
                 f"FROM {_quote(table)} WHERE type_points = ? AND points IN (?, ?, ?)")
        long_df = con.execute(query, [POSE_TYPE_POINTS, gf.BODY_25_NECK,
                                      gf.BODY_25_RIGHT_WRIST, gf.BODY_25_LEFT_WRIST]).df()
    finally:
        con.close()
    if long_df.empty:
        raise GestureRunError(f"no BODY_25 pose rows (type_points={POSE_TYPE_POINTS!r}, points "
                              f"{gf.BODY_25_NECK}/{gf.BODY_25_RIGHT_WRIST}/{gf.BODY_25_LEFT_WRIST}) "
                              f"in table {table!r}: nothing to measure")
    return long_df


def pivot_wrists(long_df, min_keypoint_confidence=0.2):
    """Long keypoint rows -> one wide row per id / people_id / frame.

    Returns (wide, stats). A keypoint row whose `frame_timestamp` is NULL is kept:
    it is a real frame that carries geometry but no clock, and the features say so
    (`clock_known = false`).
    """
    stats = clock_stats(long_df, min_keypoint_confidence)
    deduped = long_df.drop_duplicates()
    stats["identical_rows_collapsed"] = int(len(long_df) - len(deduped))
    if deduped.duplicated(subset=["id", "people_id", "frame", "points"]).any():
        bad = deduped[deduped.duplicated(subset=["id", "people_id", "frame", "points"], keep=False)]
        sample = ", ".join(f"({r.id}, {r.people_id:g}, {r.frame}, point {r.points:g})"
                           for r in bad.head(3).itertuples())
        raise GestureRunError(f"conflicting pose rows for the same id/people_id/frame/points: {sample}")
    # One frame cannot be both timed and untimed: that source is ambiguous.
    stamps = deduped.groupby(["id", "people_id", "frame"])["frame_timestamp"].nunique(dropna=False)
    if (stamps > 1).any():
        bad = stamps[stamps > 1].head(3)
        raise GestureRunError("frame_timestamp differs (or is missing for only some keypoints) of one "
                              f"frame (e.g. {', '.join(f'({i[0]}, {i[1]:g}, {i[2]})' for i in bad.index)})")

    wide = deduped.pivot(index=["id", "people_id", "frame", "frame_timestamp"],
                         columns="points", values=["nx", "ny", "c"])
    wide.columns = [f"{field}_{int(point)}" for field, point in wide.columns]
    wide = wide.reset_index()
    # A keypoint this source never reports becomes a NULL column: the other wrist
    # keeps all of its features and the between-hand metrics stay NULL.
    for column in [c for c in gf.REQUIRED_WIDE_COLUMNS if c not in wide.columns]:
        wide[column] = numpy.nan
    return wide, stats


def clock_stats(long_df, min_keypoint_confidence):
    """How many rows / frames / usable wrists have no frame_timestamp mark."""
    untimed = long_df["frame_timestamp"].isna()
    keys = ["id", "people_id", "frame"]
    stats = {"pose_rows_without_timestamp": int(untimed.sum()),
             "frames_without_timestamp": int(long_df.loc[untimed, keys].drop_duplicates().shape[0])}
    usable = {}
    for hand, point in gf.HAND_POINTS.items():
        trusted = ((long_df["points"] == point) & untimed
                   & _finite(long_df["nx"]) & _finite(long_df["ny"])
                   & (long_df["c"] > 0.0) & (long_df["c"] <= 1.0)
                   & (long_df["c"] >= min_keypoint_confidence))
        usable[hand] = int(long_df.loc[trusted, keys].drop_duplicates().shape[0])
    stats["usable_wrists_without_timestamp"] = usable
    return stats


def verify_neck_origin(wide):
    """Per clip/person: dfMaker centers `nx`/`ny` on the neck; refuse any other frame.

    A neck row only counts as evidence when its coordinates are finite and its
    confidence is a valid probability. Each id / people_id sequence is assessed on
    its own, and a sequence that measures wrists but cannot locate its neck is
    refused instead of having its origin guessed.
    """
    absent = [c for c in NECK_COLUMNS if c not in wide.columns]
    if absent:
        raise GestureRunError(
            f"cannot verify the reference origin: the table has no BODY_25 point {gf.BODY_25_NECK} "
            f"(neck) columns to check: {', '.join(absent)}")
    verified, unverifiable, labels = 0, 0, sorted({str(i) for i in wide["id"].unique()})
    for (clip_id, people), grp in wide.groupby(["id", "people_id"], sort=True):
        label = f"{clip_id} / person {people:g}"
        conf = grp["c_1"].to_numpy("float64")
        nx, ny = grp["nx_1"].to_numpy("float64"), grp["ny_1"].to_numpy("float64")
        impossible = _finite(conf) & ((conf < 0.0) | (conf > 1.0))
        if impossible.any():
            raise GestureRunError(f"{label}: neck (BODY_25 point {gf.BODY_25_NECK}) confidence must be "
                                  f"within [0, 1], got {float(conf[impossible][0]):.6g}")
        evidence = _finite(nx) & _finite(ny) & _finite(conf) & (conf > 0.0) & (conf <= 1.0)
        off = evidence & ((numpy.abs(nx) > NECK_ORIGIN_TOLERANCE)
                           | (numpy.abs(ny) > NECK_ORIGIN_TOLERANCE))
        if off.any():
            first = grp[off].iloc[0]
            raise GestureRunError(
                f"{label}: nx/ny are not neck-centered - the neck sits at "
                f"({first['nx_1']:.6g}, {first['ny_1']:.6g}) in {int(off.sum())} of "
                f"{int(evidence.sum())} reported frames. Every feature here is defined around the "
                "neck, so a nose origin or raw pixels are refused; re-run dfMaker (fast_scaling=TRUE)")
        wrists = _measures_wrists(grp)
        if evidence.any():
            verified += 1
        elif wrists:
            unverifiable += 1
            raise GestureRunError(
                f"{label}: wrists are reported but BODY_25 point {gf.BODY_25_NECK} (neck) never comes "
                "with finite coordinates and a confidence in [0, 1], so the reference origin of these "
                "nx/ny values cannot be verified")
    return {"sequences_checked": verified + unverifiable, "origin_verified": verified,
            "origin_unverifiable": unverifiable, "clips_seen": len(labels)}


def _finite(values):
    return numpy.isfinite(numpy.asarray(values, "float64"))


def _measures_wrists(grp):
    """Does this sequence carry any wrist coordinate that could produce geometry?"""
    points = set(gf.HAND_POINTS.values())
    if any(f"{field}_{point}" not in grp.columns
           for point in points for field in ("nx", "ny", "c")):
        return False
    return any((_finite(grp[f"nx_{point}"]) & _finite(grp[f"ny_{point}"])
                & (grp[f"c_{point}"].to_numpy("float64") > 0.0)).any() for point in points)


# -------------------------------------------------------------------- write ----

def build_payload(db_path, table, config, requested, frames, sequences, origin, stats):
    """What a reader needs to interpret the Parquet files years from now."""
    return {
        "tool": "gestures.py",
        "stage": "first-stage neck-relative BODY_25 wrist features",
        "source_database": os.path.abspath(db_path),
        "source_table": table,
        "source_opened": "read_only (the database was not modified)",
        "parameters": dict(config.as_dict(), requested=requested),
        "source_clock": dict({k: v for k, v in stats.items() if k != "identical_rows_collapsed"},
                             note="frames without source timestamps are retained; bounded internal "
                                  "clock gaps may be interpolated and flagged timestamp_inferred. "
                                  "Frames still without a clock keep geometry but no temporal features"),
        "duplicate_source_rows_collapsed": stats["identical_rows_collapsed"],
        "origin_check": origin,
        "reference": dict(gf.SEMANTICS,
                          origin=f"BODY_25 point {gf.BODY_25_NECK} (neck), dfMaker nx/ny with "
                                 f"Y positive up; verified at (0, 0) for "
                                 f"{origin['origin_verified']} of {origin['sequences_checked']} "
                                 "sequence(s)",
                          publish_requirement="Linux renameat2(RENAME_NOREPLACE): publishing never "
                                              "replaces an existing path, other platforms fail closed",
                          keypoints={"right_wrist": gf.BODY_25_RIGHT_WRIST,
                                     "left_wrist": gf.BODY_25_LEFT_WRIST}),
        "units": gf.FEATURE_UNITS,
        "schema": {"gesture_frames": list(gf.FRAME_COLUMNS),
                   "gesture_sequences": list(gf.SEQUENCE_COLUMNS),
                   "types": "float64 (NaN/NULL = not measured), boolean (nullable), Int64 frame"},
        "rows": {"gesture_frames": int(len(frames)), "gesture_sequences": int(len(sequences))},
        "versions": {"python": platform.python_version(), "python_package": sys.version.split()[0],
                     "gesture_features": gf.FEATURE_VERSION, "pandas": pandas.__version__,
                     "numpy": numpy.__version__, "scipy": scipy.__version__,
                     "duckdb": duckdb.__version__, "pyarrow": pyarrow.__version__},
    }


def _atomic_rename(source, target):
    """Publish by a rename that never replaces an existing path.

    os.replace() would overwrite an empty directory another process created after
    the initial existence check, so this calls Linux
    renameat2(AT_FDCWD, source, AT_FDCWD, target, RENAME_NOREPLACE) through the C
    library. Returns False when the target is taken; raises when the platform
    cannot make that promise, so nothing is ever published unsafely.
    """
    syscall = RENAMEAT2_SYSCALL.get(platform.machine()) if sys.platform.startswith("linux") else None
    if syscall is None:
        raise GestureRunError(
            "cannot publish safely: the atomic no-replace rename needs Linux "
            f"renameat2(RENAME_NOREPLACE) on x86_64/aarch64, this is {sys.platform}/"
            f"{platform.machine()}; nothing was written to {target}")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_char_p,
                             ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if libc.syscall(syscall, AT_FDCWD, os.fsencode(source), AT_FDCWD,
                    os.fsencode(target), RENAME_NOREPLACE) != 0:
        code = ctypes.get_errno() or errno.EIO
        if code in (errno.EEXIST, errno.ENOTEMPTY):
            return False                       # someone else owns that path: keep it
        raise GestureRunError(f"cannot publish to {target}: {os.strerror(code)}")
    return True


def publish(output_folder, frames, sequences, payload, on_staged=None):
    """Write into a sibling temporary directory, then rename it into place.

    `on_staged` is a test seam called while the staging directory is still private,
    so a test can make a concurrent publisher appear at the exact risky moment.
    """
    target = os.path.abspath(output_folder)
    if os.path.exists(target) or os.path.islink(target):
        raise GestureRunError(f"output folder already exists: {target} "
                              "(results are never overwritten; pick a new --output_folder)")
    parent = os.path.dirname(target)
    if not os.path.isdir(parent):
        raise GestureRunError(f"parent folder does not exist: {parent}")
    staging = tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=parent)
    try:
        frames.to_parquet(os.path.join(staging, FRAMES_FILE), index=False)
        sequences.to_parquet(os.path.join(staging, SEQUENCES_FILE), index=False)
        with open(os.path.join(staging, CONFIG_FILE), "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        if on_staged is not None:
            on_staged(staging)
        if not _atomic_rename(staging, target):
            raise GestureRunError(f"output folder already exists: {target} "
                                  "(another publisher got there first; nothing was overwritten)")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)   # only this run's own staging directory
        raise
    return target


# ---------------------------------------------------------------------- run ----

def run(db_path, output_folder, config, table=DEFAULT_TABLE, requested=None, on_staged=None):
    """Features for every clip/person sequence in `db_path`."""
    long_df = load_pose_keypoints(db_path, table)
    wide, stats = pivot_wrists(long_df, config.min_keypoint_confidence)
    origin = verify_neck_origin(wide)
    frames, sequences = gf.compute_gesture_features(wide.drop(columns=list(NECK_COLUMNS)), config)
    payload = build_payload(db_path, table, config, dict(DEFAULT_PARAMETERS, **(requested or {})),
                            frames, sequences, origin, stats)
    target = publish(output_folder, frames, sequences, payload, on_staged=on_staged)
    return dict(stats, output_folder=target, sequences=int(len(sequences)),
                frames=int(len(frames)), source_rows=int(len(long_df)),
                frames_clock_inferred=int(frames["timestamp_inferred"].sum()),
                frames_still_without_clock=int((~frames["clock_known"]).sum()), **origin)


class _SingleValue(argparse.Action):
    """Reject an option (or its alias) that appears twice with different values."""

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if current is not None and current != values:
            parser.error(f"conflicting values for {self.dest}: {current!r} and "
                         f"{values!r} ({option_string}); give one value")
        setattr(namespace, self.dest, values)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="gestures.py",
        description="Neck-relative BODY_25 wrist gesture features from a POLYMNIA DuckDB. "
                    "The database is opened read-only; nothing in it is modified.",
        epilog="Writes gesture_frames.parquet, gesture_sequences.parquet and gesture_config.json "
               "into a NEW --output_folder (an existing folder is refused, and a failure publishes "
               "nothing). Coordinates are dfMaker neck-space units, not metres; timestamps are real "
               "elapsed seconds, so velocities are units/s and accelerations units/s^2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db_name", "--db-name", dest="db_name", required=True,
                        action=_SingleValue,
                        help="existing DuckDB written by hefesto.py (opened read-only)")
    parser.add_argument("--output_folder", "--output-folder", dest="output_folder", required=True,
                        action=_SingleValue,
                        help="folder to create for the three artefacts")
    parser.add_argument("--table_name", "--table-name", dest="table_name", default=None,
                        action=_SingleValue,
                        help=f"table holding the long pose rows [default: {DEFAULT_TABLE}]")
    parser.add_argument("--min-keypoint-confidence", "--min_keypoint_confidence",
                        dest="min_keypoint_confidence", type=float, default=None, action=_SingleValue,
                        help="wrist confidence needed to trust a frame (in [0, 1], inclusive; "
                             "0 keeps 0-confidence points out too) [default: 0.2]")
    parser.add_argument("--max-gap-frames", "--max_gap_frames", dest="max_gap_frames",
                        type=int, default=None, action=_SingleValue,
                        help="interpolate interior gaps of at most this many missing frames "
                             "(0 disables interpolation) [default: 3]")
    parser.add_argument("--window-size", "--window_size", dest="window_size",
                        type=int, default=None, action=_SingleValue,
                        help="odd Savitzky-Golay window, shrunk to fit short runs "
                             "(1 disables smoothing) [default: 7]")
    parser.add_argument("--poly-order", "--poly_order", "--polyorder", dest="poly_order",
                        type=int, default=None, action=_SingleValue,
                        help="polynomial order of that window [default: 2]")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.table_name == "":
        parser.error("table_name must not be empty")
    requested = {name: getattr(args, name) for name in gf.PARAM_NAMES if getattr(args, name) is not None}
    try:
        config = gf.GestureConfig.from_input(requested)
    except gf.GestureConfigError as exc:
        parser.error(str(exc))
    try:
        summary = run(args.db_name, args.output_folder, config,
                      table=DEFAULT_TABLE if args.table_name is None else args.table_name,
                      requested=requested)
    except (GestureRunError, gf.GestureDataError, gf.GestureConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {summary['frames']} frame rows and {summary['sequences']} sequence rows "
          f"to {summary['output_folder']}")
    if summary["frames_without_timestamp"]:
        print(f"note: {summary['pose_rows_without_timestamp']} source row(s) in "
              f"{summary['frames_without_timestamp']} frame(s) lack a source frame_timestamp; retained. "
              f"Output: {summary['frames_clock_inferred']} inferred clock(s), "
              f"{summary['frames_still_without_clock']} frame(s) still without a clock")
    if summary["identical_rows_collapsed"]:
        print(f"note: collapsed {summary['identical_rows_collapsed']} exactly duplicated source row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
