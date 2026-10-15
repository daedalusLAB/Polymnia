#!/usr/bin/env python3
"""Neck-relative BODY_25 wrist features - the pure calculation core.

This module turns dfMaker's person-relative wrist coordinates into bounded,
auditable geometry and timing features. It reads no files and no database: the
public entry point is

    compute_gesture_features(wide_dataframe, config) -> (frame_features, sequence_features)

`gestures.py` (the CLI) is only an adapter that produces the wide frame table
below from the long `multi_data` table and writes the two Parquet files.

Coordinates (see README "Neck-relative gesture features")
---------------------------------------------------------
BODY_25 point 4 is the right wrist and point 7 the left wrist. The values come
from dfMaker's ``nx`` / ``ny`` columns (``fast_scaling = TRUE``, default
``transformation_coords = c(1, 1, 5, 5)``): the origin is the NECK (point 1, not
the nose), X is divided by the neck-to-point-5 X component (BODY_25 point 5 sits
at exactly ``nx = 1.0`` in delivered data; point 5 is ONE shoulder, so this is
not a bilateral shoulder span) and Y is positive UP. So ``ny > 0`` means "above
the neck" and every distance is in neck-space normalized units, not metres.

Input schema (one row per id / people_id / frame)
-------------------------------------------------
``id, people_id, frame, frame_timestamp, nx_4, ny_4, c_4, nx_7, ny_7, c_7``
Extra columns are ignored, and a missing wrist column is filled with NULL (the
other hand still gets its features). ``frame_timestamp`` is elapsed seconds:
known marks must be finite and increase with the frame, while a NULL mark is a
real frame with no clock. It keeps its geometry and its coordinate features and
loses every timed feature; derivatives never cross it, and a clock is only
borrowed across a short internal gap (``timestamp_inferred`` says when).

Pipeline, per (id, people_id) sequence, in this exact order
-----------------------------------------------------------
1. keep a coordinate only when its confidence is a finite probability, > 0 and
   >= threshold; keep a clock only where the source reported a finite one;
2. expand the missing frame indices inside the observed frame span (those rows
   are marked ``frame_inferred``; the span is the *observed* frames, never a
   claim about the whole raw video);
3. inside the grid, linearly interpolate a clock across at most
   ``max_gap_frames`` consecutive untimed internal frames bounded by timed ones
   (marked ``timestamp_inferred``; edges are never extrapolated), and linearly
   interpolate a coordinate across at most ``max_gap_frames`` missing frames
   that have trusted endpoints on both sides and a finite clock span
   (``max_gap_frames`` 0 disables both, long gaps stay NULL);
4. smooth each contiguous run of usable samples with a known clock separately -
   neither an unfilled coordinate gap nor a clock hole is bridged - with an
   adaptive Savitzky-Golay window that fits in the run; sampled uniformly the
   filter runs on the real time step, sampled irregularly the same polynomial
   is fitted against the actual timestamps;
5. velocities and accelerations come from those smoothed coordinates against
   the clock, and nothing timed is invented where the clock is unknown; neck
   distance/angle, above-neck, adjacent-frame step distances and between-hand
   metrics only need trusted coordinates, so they survive an unknown clock.

Missing data is always NULL, never 0 and never inf: geometry needs one valid
sample, velocity two connected samples, acceleration three, a between-hand
metric needs both hands at that frame, and every "time to" / duration statistic
needs a clock on the frame it describes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ----------------------------------------------------------------- vocabulary ----

BODY_25_NECK = 1
BODY_25_RIGHT_WRIST = 4
BODY_25_LEFT_WRIST = 7
HAND_POINTS = {"right": BODY_25_RIGHT_WRIST, "left": BODY_25_LEFT_WRIST}

FEATURE_VERSION = "2"          # bump when the feature meaning changes
PARAM_NAMES = ("min_keypoint_confidence", "max_gap_frames", "window_size", "poly_order")

REQUIRED_WIDE_COLUMNS = ("id", "people_id", "frame", "frame_timestamp",
                         "nx_4", "ny_4", "c_4", "nx_7", "ny_7", "c_7")

# Per-hand columns, in output order. "(b)" columns are nullable booleans.
HAND_MEASURES = ("confidence", "nx", "ny", "nx_smooth", "ny_smooth",
                 "valid(b)", "observed(b)", "interpolated(b)",
                 "neck_distance", "neck_angle", "above_neck(b)",
                 "velocity_x", "velocity_y", "speed",
                 "accel_x", "accel_y", "accel_magnitude", "step_distance")

KEY_COLUMNS = ("id", "people_id")

FRAME_COLUMNS = (KEY_COLUMNS + ("frame", "frame_timestamp", "source_observed",
                                "frame_inferred", "clock_known", "timestamp_inferred")
                 + tuple(f"{hand}_{m.split('(')[0]}" for hand in ("right", "left") for m in HAND_MEASURES)
                 + ("both_hands_valid", "pair_separation", "pair_angle",
                    "centroid_x", "centroid_y", "mirror_error_x", "mirror_error_y"))

FRAME_BOOL_COLUMNS = tuple(c for c in FRAME_COLUMNS
                           if c not in ("id", "people_id", "frame", "frame_timestamp")
                           and ((c in ("source_observed", "frame_inferred", "clock_known",
                                       "timestamp_inferred", "both_hands_valid"))
                                or c.split("_", 1)[1] in {m.split("(")[0] for m in HAND_MEASURES
                                                          if m.endswith("(b)")}))

SEQUENCE_MEASURES = ("n_observed", "n_interpolated", "n_valid",
                     "frac_observed", "frac_interpolated", "frac_valid",
                     "mean_speed", "max_speed", "mean_accel_magnitude", "max_accel_magnitude",
                     "path_length", "nx_range", "ny_range", "mean_neck_distance",
                     "angle_mean", "angle_std", "time_to_peak_speed",
                     "early_speed_mean", "late_speed_mean")

SEQUENCE_COLUMNS = (KEY_COLUMNS + ("frames_source", "frames_grid", "frames_missing_grid",
                                   "n_timed_frames", "n_timestamp_inferred", "n_missing_timestamps",
                                   "t_start", "t_end", "duration_s")
                    + tuple(f"{hand}_{m}" for hand in ("right", "left") for m in SEQUENCE_MEASURES)
                    + ("n_both_hands", "frac_both_hands", "mean_pair_separation",
                       "max_pair_separation", "mean_mirror_error_x", "mean_mirror_error_y",
                       "time_to_peak_pair_separation"))

# Written next to the outputs so a Parquet file can be read years later.
FEATURE_UNITS = {
    "coordinates (nx, ny, nx_smooth, ny_smooth, centroid_*)": "dfMaker neck-space normalized units, origin BODY_25 neck (point 1), Y positive up",
    "distances (neck_distance, step_distance, pair_separation, path_length, nx_range, ny_range, mirror_error_*)": "normalized neck-space units (NOT metres)",
    "velocity (velocity_x, velocity_y, speed, early/late_speed_mean)": "normalized neck-space units per second",
    "acceleration (accel_x, accel_y, accel_magnitude)": "normalized neck-space units per second squared (NOT joules, no body-mass model)",
    "angles (neck_angle, pair_angle, angle_mean, angle_std)": "radians, atan2(ny, nx); NULL when the vector has zero length",
    "time (t_start, t_end, duration_s, time_to_peak_*, early/late windows)": "elapsed seconds from the source frame_timestamp; NULL where that frame has no clock mark (clock_known = false)",
    "counts and fractions": "float64; fractions divide by frames_grid, every frame of the OBSERVED frame span including frames with no clock mark",
    "NULL": "not measured: absent keypoint, low confidence, or too few connected samples",
}

SEMANTICS = {
    "reference_origin": ("BODY_25 point 1 (neck) as transformed by dfMaker; never the nose. X is "
                         "normalized by the neck-to-point-5 X component (point 5 is one shoulder, "
                         "not a bilateral shoulder span)"),
    "source_frame_span": ("features cover the observed frame span min(frame)..max(frame) of each "
                          "id/people_id sequence. Frames absent from the source inside that span are "
                          "marked frame_inferred; nothing is claimed about frames outside it, and the "
                          "span is not a claim about the length of the raw video."),
    "clock": ("frame_timestamp NULL means the source gave no clock for that frame: geometry and "
              "coordinate features are kept, timed features are NULL. A missing clock is interpolated "
              "only across at most max_gap_frames consecutive untimed INTERNAL frames bounded by timed "
              "ones (timestamp_inferred); edges are never extrapolated, and known marks must increase "
              "with the frame"),
    "interpolation": ("coordinate gaps of at most max_gap_frames missing frames between valid endpoints "
                      "with a finite clock span are linearly interpolated in elapsed time and flagged "
                      "interpolated; edges are never extrapolated and longer gaps are never bridged"),
    "smoothing": ("adaptive Savitzky-Golay per contiguous run of usable coordinates with a known clock "
                  "(window shrinks to fit the run and its polynomial order with it, window_size=1 "
                  "disables smoothing); unfilled gaps and clock holes are never crossed"),
    "derivatives": "computed from smoothed coordinates against real frame_timestamp seconds, never an assumed frame rate",
    "adjacent_step_distance": ("displacement to the previous frame of a contiguous coordinate run; it "
                               "needs no clock, so path_length stays meaningful where the clock has holes"),
    "angles": ("angle_mean/angle_std are the circular mean and the sqrt(-2 ln R) spread of R = "
               "|mean(exp(i*angle))|; that spread assumes a single concentrated mode (it is a "
               "concentration measure, not a full von Mises fit) and is NULL, together with the mean, "
               "when R is numerically zero and no direction is meaningful"),
    "aggregates": "mean/max over finite values only; NULL when no value exists (a NULL is not a measured 0)",
}


class GestureConfigError(ValueError):
    """The requested features cannot be computed (bad parameter)."""


class GestureDataError(ValueError):
    """The input table cannot be trusted (missing keys, conflicting or impossible rows)."""


# --------------------------------------------------------------------- config ----

_PARAMS = PARAM_NAMES


def _clean_key(name):
    return str(name).strip().replace("-", "_")


def _number(value, name, low, high, integer=False):
    """One finite number inside [low, high] (integral when `integer`)."""
    if isinstance(value, bool) or value is None:
        raise GestureConfigError(f"{name} must be a number, got {value!r}")
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError):
        raise GestureConfigError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(num) or not low <= num <= high:
        raise GestureConfigError(f"{name} must be a finite number in [{low}, {high}], got {value!r}")
    if integer and num != int(num):
        raise GestureConfigError(f"{name} must be a whole number, got {value!r}")
    if integer:
        try:
            return int(num)
        except (TypeError, ValueError, OverflowError):
            raise GestureConfigError(f"{name} is too large to use, got {value!r}") from None
    return num


class GestureConfig:
    """Validated parameters for one feature run."""

    def __init__(self, min_keypoint_confidence=0.2, max_gap_frames=3,
                 window_size=7, poly_order=2):
        self.min_keypoint_confidence = _number(min_keypoint_confidence, "min_keypoint_confidence", 0.0, 1.0)
        self.max_gap_frames = _number(max_gap_frames, "max_gap_frames", 0, 10_000, integer=True)
        window = _number(window_size, "window_size", 1, 10_001, integer=True)
        if window % 2 == 0:
            raise GestureConfigError(f"window_size must be odd (1 disables smoothing), got {window_size!r}")
        self.window_size = window
        self.poly_order = _number(poly_order, "poly_order", 1, 10_000, integer=True)
        if window > 1 and self.poly_order >= window:
            raise GestureConfigError(
                f"poly_order {self.poly_order} must be smaller than window_size {window}; "
                "short runs still shrink the window and its order, and window_size 1 disables smoothing")

    @classmethod
    def from_input(cls, params=None):
        """Build from a mapping, accepting hyphens, rejecting conflicts and typos."""
        if params is None:
            return cls()
        if isinstance(params, GestureConfig):
            return params
        if not hasattr(params, "items"):
            raise GestureConfigError(f"config must be a mapping or GestureConfig, got {params!r}")
        chosen, seen = {}, {}
        for key, value in params.items():
            name = _clean_key(key)
            if name not in _PARAMS:
                raise GestureConfigError(
                    f"unknown gesture parameter {key!r}; supported: {', '.join(_PARAMS)}")
            if name in seen and seen[name] != value:
                raise GestureConfigError(
                    f"conflicting values for {name}: {seen[name]!r} and {value!r}")
            seen[name], chosen[name] = value, value
        return cls(**chosen)

    def as_dict(self):
        return {name: getattr(self, name) for name in _PARAMS}

    def smoothing_params(self, run_length):
        """(window, poly_order) that fit a run; window 1 means "do not smooth".

        ``poly_order`` is already checked against ``window_size``, so an order
        that only has to shrink for a short run can be adapted here.
        """
        window = min(self.window_size, run_length if run_length % 2 else run_length - 1)
        order = min(self.poly_order, window - 1)
        if window < 3 or order < 1:
            return 1, 0
        return window, order


# ------------------------------------------------------------------- plumbing ----

def _runs(mask):
    """Contiguous [start, stop) stretches where `mask` is True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def _is_uniform(times):
    if len(times) < 2:
        return True
    steps = np.diff(times)
    step = float(np.median(steps))
    return step > 0 and float(np.max(np.abs(steps - step))) <= 1e-6 * step


def _local_polyfit(values, times, window, order):
    """Savitzky-Golay for samples that are NOT evenly spaced in time."""
    out = np.empty_like(values)
    half, last = window // 2, len(values) - window
    for i in range(len(values)):
        lo = min(max(i - half, 0), last)
        design = np.vander(times[lo:lo + window] - times[i], order + 1, increasing=True)
        coef = np.linalg.lstsq(design, values[lo:lo + window], rcond=None)[0]
        out[i] = coef[0]
    return out


def _smooth_run(values, times, config):
    """Smooth one contiguous run; never looks outside it."""
    window, order = config.smoothing_params(len(values))
    if window == 1:
        return values.copy()
    if _is_uniform(times):
        return np.asarray(savgol_filter(values, window, order,
                                       delta=float(times[1] - times[0]), mode="interp"), dtype="float64")
    return _local_polyfit(values, times, window, order)


def _derivative(values, times, degree):
    """d/dt (velocity degree 1, acceleration degree 2) or None when too short."""
    if len(values) < degree + 1:
        return None
    out = np.asarray(values, dtype="float64")
    for _ in range(degree):
        out = np.gradient(out, times, edge_order=2 if len(values) >= 3 else 1)
    return out


def _interpolate_gaps(values, times, valid, max_gap):
    """Fill bounded interior gaps in `values` linearly in `times`.

    Returns (filled, interpolated). Gaps touching an edge, longer than `max_gap`,
    or without a finite clock at both endpoints and a positive span stay NULL.
    """
    filled, interpolated = values.copy(), np.zeros(len(values), dtype=bool)
    if max_gap <= 0:
        return filled, interpolated
    known = np.flatnonzero(valid)
    for left, right in zip(known[:-1], known[1:]):
        missing = right - left - 1
        span = times[right] - times[left]
        if not 1 <= missing <= max_gap:
            continue
        if not (np.isfinite(times[left]) and np.isfinite(times[right])) or not span > 0:
            continue                     # no clock, no elapsed time to interpolate on
        between = np.arange(left + 1, right)
        frac = (times[between] - times[left]) / span
        filled[between] = values[left] + frac * (values[right] - values[left])
        interpolated[between] = True
    return filled, interpolated


# ------------------------------------------------------------------- sequences ----

def _validate_input(wide_df):
    """Normalize and trust-check the wide table; returns a sorted copy."""
    if not hasattr(wide_df, "columns"):
        wide_df = pd.DataFrame(wide_df, columns=list(REQUIRED_WIDE_COLUMNS))
    missing_keys = [c for c in ("id", "people_id", "frame") if c not in wide_df.columns]
    if missing_keys:
        raise GestureDataError(f"input is missing required columns: {', '.join(missing_keys)}")
    df = wide_df.copy()
    # A wrist this source never reports is NULL: the other hand keeps its features.
    for column in REQUIRED_WIDE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    df = df.loc[:, list(REQUIRED_WIDE_COLUMNS)]
    if df.empty:
        return df.assign(id=pd.Series(dtype="object"), people_id=pd.Series(dtype="float64"),
                        frame=pd.Series(dtype="int64"), frame_timestamp=pd.Series(dtype="float64"))

    for name in ("id", "people_id", "frame"):
        if df[name].isna().any():
            raise GestureDataError(f"{name} is NULL in {int(df[name].isna().sum())} row(s); "
                                   f"every pose row needs a clip id, person and frame")
    limits = np.iinfo(np.int64)
    # Reject exact Python/NumPy integers before conversion can round them into
    # range (e.g. int64.min - 1 becomes int64.min as a float).
    for value in df["frame"]:
        if isinstance(value, (int, np.integer)) and not limits.min <= int(value) <= limits.max:
            raise GestureDataError("frame must fit int64 frame indices")
    try:
        people = pd.to_numeric(df["people_id"], errors="raise").astype("float64")
        frames = pd.to_numeric(df["frame"], errors="raise")
        stamps = pd.to_numeric(df["frame_timestamp"], errors="raise").astype("float64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise GestureDataError(f"frame / people_id / frame_timestamp must be numeric: {exc}") from None
    if not np.all(np.isfinite(people)):
        raise GestureDataError(f"people_id must be a finite number in every row "
                               f"({int(np.sum(~np.isfinite(people)))} row(s) are not)")
    if not np.all(np.isfinite(frames)):
        raise GestureDataError(f"frame must be a finite integer index "
                               f"({int(np.sum(~np.isfinite(frames)))} row(s) are not)")
    if not np.all(frames == np.floor(frames)):
        raise GestureDataError("frame must hold whole frame indices")
    if pd.api.types.is_integer_dtype(frames.dtype):
        in_range = frames.map(lambda value: limits.min <= int(value) <= limits.max)
    else:
        # float(int64.max) is already 2**63: this upper bound is EXCLUSIVE.
        in_range = (frames >= float(limits.min)) & (frames < float(2**63))
    if not np.all(in_range):
        raise GestureDataError(f"frame must fit int64 frame indices (got values outside "
                               f"[{limits.min}, {limits.max}])")
    # A NULL timestamp is a frame with no clock; a reported one must be a real time.
    reported = ~np.isnan(stamps)
    if not np.all(np.isfinite(stamps[reported])):
        raise GestureDataError(f"frame_timestamp must be finite where reported "
                               f"({int(np.sum(~np.isfinite(stamps[reported])))} row(s) are not)")
    for hand, point in HAND_POINTS.items():
        _check_confidence(df[f"c_{point}"], hand)

    df = df.assign(people_id=people, frame=frames.astype("int64"), frame_timestamp=stamps)
    # A full duplicate row is harmless (re-runs of `merge` append); a duplicate
    # key carrying different values means the frame is ambiguous - fail closed.
    deduped = df.drop_duplicates(keep="first")
    if deduped.duplicated(subset=["id", "people_id", "frame"]).any():
        bad = deduped[deduped.duplicated(subset=["id", "people_id", "frame"], keep=False)]
        keys = ", ".join(f"({r.id}, {r.people_id:g}, {r.frame:d})" for r in bad.head(5).itertuples())
        raise GestureDataError(f"conflicting rows for the same id/people_id/frame: {keys}")
    df = deduped.sort_values(["id", "people_id", "frame"], kind="mergesort").reset_index(drop=True)
    return df


def _check_confidence(values, hand):
    """OpenPose-style confidence: NULL is absent, anything else must be a probability."""
    numbers = values.to_numpy("float64")
    reported = ~np.isnan(numbers)
    if not np.all((numbers[reported] >= 0.0) & (numbers[reported] <= 1.0)):
        bad = numbers[reported][(numbers[reported] < 0.0) | (numbers[reported] > 1.0)][:5]
        raise GestureDataError(f"{hand} wrist confidence must be within [0, 1] where reported "
                               f"(got {', '.join(repr(float(v)) for v in bad)}); "
                               "NULL is treated as absent, an impossible value is not")


def _sequence_grid(grp, label, config):
    """Frame grid, clock, clock-known / inferred flags and source rows over the span.

    A clock is only borrowed across at most `max_gap_frames` consecutive untimed
    INTERNAL frames bounded by timed ones; edges are never extrapolated.
    """
    frames = grp["frame"].to_numpy("int64")
    marks = grp["frame_timestamp"].to_numpy("float64")
    if np.any(np.diff(frames) <= 0):
        raise GestureDataError(f"{label}: repeated frame index inside the sequence")
    reported = np.isfinite(marks)
    if reported.any() and np.any(np.diff(marks[reported]) <= 0):
        raise GestureDataError(f"{label}: frame_timestamp must increase with the frame wherever "
                               "a timestamp is reported, so elapsed time is well defined")

    grid = np.arange(int(frames[0]), int(frames[-1]) + 1, dtype="int64")
    times = np.full(len(grid), np.nan, dtype="float64")
    times[frames - frames[0]] = marks
    inferred = np.zeros(len(grid), dtype=bool)
    timed = np.flatnonzero(np.isfinite(times))
    if config.max_gap_frames > 0:
        for left, right in zip(timed[:-1], timed[1:]):
            if not 1 <= right - left - 1 <= config.max_gap_frames:
                continue
            between = np.arange(left + 1, right)
            frac = (grid[between] - grid[left]) / (grid[right] - grid[left])
            times[between] = times[left] + frac * (times[right] - times[left])
            inferred[between] = True

    observed = np.zeros(len(grid), dtype=bool)
    observed[frames - frames[0]] = True
    return grid, times, np.isfinite(times), inferred, observed


def _hand_arrays(grp, hand, size, first_frame):
    """Raw nx / ny / confidence placed on the frame grid (NaN where no row)."""
    point = HAND_POINTS[hand]
    out = []
    for field in ("nx", "ny", "c"):
        arr = np.full(size, np.nan, dtype="float64")
        arr[grp["frame"].to_numpy("int64") - first_frame] = grp[f"{field}_{point}"].to_numpy("float64")
        out.append(arr)
    return out


def _hand_rows(nx_raw, ny_raw, conf, times, clock, observed, config):
    """All per-hand frame features for one sequence, as a dict of grids."""
    usable = (np.isfinite(nx_raw) & np.isfinite(ny_raw) & np.isfinite(conf)
              & (conf > 0.0) & (conf >= config.min_keypoint_confidence))
    nx_fill, interp_x = _interpolate_gaps(nx_raw, times, usable, config.max_gap_frames)
    ny_fill, _ = _interpolate_gaps(ny_raw, times, usable, config.max_gap_frames)
    valid = usable | interp_x
    # A coordinate that arrived with low confidence is still not usable: report
    # NULL, never the value nobody trusted.
    nx_usable = np.where(valid, nx_fill, np.nan)
    ny_usable = np.where(valid, ny_fill, np.nan)

    out = {"confidence": conf, "valid": valid, "observed": usable & observed,
           "interpolated": interp_x, "nx": nx_usable, "ny": ny_usable}
    # Geometry follows trusted coordinates; smoothing and derivatives only run
    # where that coordinate also has a clock, so a clock hole stops the physics
    # without losing the position.
    sx, sy = nx_usable.copy(), ny_usable.copy()
    velocity = {k: np.full(len(times), np.nan) for k in ("velocity_x", "velocity_y", "speed")}
    accel = {k: np.full(len(times), np.nan) for k in ("accel_x", "accel_y", "accel_magnitude")}
    step = np.full(len(times), np.nan)

    for start, stop in _runs(valid & clock):
        run_t, run_x, run_y = times[start:stop], sx[start:stop], sy[start:stop]
        smooth_x, smooth_y = _smooth_run(run_x, run_t, config), _smooth_run(run_y, run_t, config)
        sx[start:stop], sy[start:stop] = smooth_x, smooth_y
        velocity_x, velocity_y = _derivative(smooth_x, run_t, 1), _derivative(smooth_y, run_t, 1)
        if velocity_x is not None and velocity_y is not None:
            velocity["velocity_x"][start:stop] = velocity_x
            velocity["velocity_y"][start:stop] = velocity_y
            velocity["speed"][start:stop] = np.hypot(velocity_x, velocity_y)
        ax, ay = _derivative(smooth_x, run_t, 2), _derivative(smooth_y, run_t, 2)
        if ax is not None and ay is not None:
            accel["accel_x"][start:stop], accel["accel_y"][start:stop] = ax, ay
            accel["accel_magnitude"][start:stop] = np.hypot(ax, ay)

    # Adjacent-frame displacement needs two neighbouring trusted coordinates, not
    # a clock, so path_length stays meaningful across a hole in the timestamps.
    for start, stop in _runs(valid):
        if stop - start >= 2:
            step[start + 1:stop] = np.hypot(np.diff(sx[start:stop]), np.diff(sy[start:stop]))

    distance = np.hypot(sx, sy)
    angle = np.where(distance > 0, np.arctan2(sy, sx), np.nan)
    above_neck = np.full(len(times), None, dtype=object)
    known = np.isfinite(sy)
    above_neck[known] = sy[known] > 0
    out.update({"nx_smooth": sx, "ny_smooth": sy, "neck_distance": distance,
                "neck_angle": angle, "above_neck": above_neck,
                "velocity_x": velocity["velocity_x"], "velocity_y": velocity["velocity_y"],
                "speed": velocity["speed"], "accel_x": accel["accel_x"], "accel_y": accel["accel_y"],
                "accel_magnitude": accel["accel_magnitude"], "step_distance": step})
    return out


def _sequence_frame_rows(clip_id, people, grp, config):
    """Every grid row of one (id, people_id) sequence as a list of dicts."""
    grid, times, clock, clock_inferred, observed = _sequence_grid(
        grp, f"sequence {clip_id} / person {people:g}", config)
    hands = {hand: _hand_rows(*_hand_arrays(grp, hand, len(grid), grid[0]),
                              times, clock, observed, config)
             for hand in ("right", "left")}

    both = hands["right"]["valid"] & hands["left"]["valid"]
    rx, ry = hands["right"]["nx_smooth"], hands["right"]["ny_smooth"]
    lx, ly = hands["left"]["nx_smooth"], hands["left"]["ny_smooth"]
    separation = np.where(both, np.hypot(rx - lx, ry - ly), np.nan)
    pair = {
        "both_hands_valid": both,
        "pair_separation": separation,
        "pair_angle": np.where(both & (separation > 0), np.arctan2(ry - ly, rx - lx), np.nan),
        "centroid_x": np.where(both, (rx + lx) / 2.0, np.nan),
        "centroid_y": np.where(both, (ry + ly) / 2.0, np.nan),
        "mirror_error_x": np.where(both, np.abs(rx + lx), np.nan),
        "mirror_error_y": np.where(both, np.abs(ry - ly), np.nan),
    }

    rows = []
    for i, frame in enumerate(grid):
        row = {"id": clip_id, "people_id": people, "frame": int(frame),
               "frame_timestamp": times[i], "source_observed": bool(observed[i]),
               "frame_inferred": not bool(observed[i]), "clock_known": bool(clock[i]),
               "timestamp_inferred": bool(clock_inferred[i])}
        for hand in ("right", "left"):
            for measure in HAND_MEASURES:
                name = measure.split("(")[0]
                row[f"{hand}_{name}"] = hands[hand][name][i]
        row.update({name: values[i] for name, values in pair.items()})
        rows.append(row)
    return rows


# ------------------------------------------------------------------ aggregates ----

def _finite(values):
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


def _stat(values, reducer, default=np.nan):
    values = _finite(values)
    return float(reducer(values)) if values.size else default


def _clock_span(times, clock):
    """(first, last) known clock of a sequence, or (NULL, NULL) when there is none."""
    known = np.asarray(times, "float64")[np.asarray(clock, dtype=bool)]
    if not known.size:
        return np.nan, np.nan
    return float(known.min()), float(known.max())


def _time_to_peak(values, times, clock, t_start):
    """Elapsed time from the start of the known clock to the highest value.

    When the winning frame has no clock the answer is NULL: silently reporting
    the second-highest timed value would describe a different peak.
    """
    values, times = np.asarray(values, "float64"), np.asarray(times, "float64")
    trusted = np.asarray(clock, dtype=bool) & np.isfinite(values) & np.isfinite(times)
    peak = np.flatnonzero(np.isfinite(values))
    if not peak.size:
        return np.nan
    winner = peak[np.argmax(values[peak])]
    if not trusted[winner] or not np.isfinite(t_start):
        return np.nan
    return float(times[winner] - t_start)


def _edge_window_mean(values, times, clock, t_start, t_end, first):
    """Mean of the finite values inside the first/last 10% of the known clock span."""
    values, times = np.asarray(values, "float64"), np.asarray(times, "float64")
    trusted = np.asarray(clock, dtype=bool) & np.isfinite(times)
    if not np.isfinite(t_start) or not np.isfinite(t_end):
        return np.nan
    edge = 0.1 * (t_end - t_start)
    inside = trusted & ((times <= t_start + edge) if first else (times >= t_end - edge))
    return _stat(values[inside], np.mean)


def _circular_stats(angles, zero_resultant=1e-12):
    """Circular mean and the sqrt(-2 ln R) spread of the resultant R.

    That spread is a concentration measure for a single mode (not a full von
    Mises fit). When R is numerically zero no direction is meaningful, so both
    the mean and the spread are NULL rather than an arbitrary angle.
    """
    angles = _finite(angles)
    if not angles.size:
        return np.nan, np.nan
    mean_vector = complex(float(np.cos(angles).mean()), float(np.sin(angles).mean()))
    resultant = min(abs(mean_vector), 1.0)
    if resultant <= zero_resultant:
        return np.nan, np.nan
    mean = math.atan2(mean_vector.imag, mean_vector.real)
    return mean, math.sqrt(max(-2.0 * math.log(resultant), 0.0))


def _hand_summary(frames, hand, frames_grid):
    def column(name):
        return frames[f"{hand}_{name}"].to_numpy("float64")

    def count(name):
        return float(frames[f"{hand}_{name}"].to_numpy("bool").sum())

    n_observed, n_interp, n_valid = count("observed"), count("interpolated"), count("valid")
    angle_mean, angle_std = _circular_stats(column("neck_angle"))
    times = frames["frame_timestamp"].to_numpy("float64")
    clock = frames["clock_known"].to_numpy("bool")
    t_start, t_end = _clock_span(times, clock)
    speed = column("speed")
    return {
        f"{hand}_n_observed": n_observed, f"{hand}_n_interpolated": n_interp,
        f"{hand}_n_valid": n_valid,
        f"{hand}_frac_observed": n_observed / frames_grid,
        f"{hand}_frac_interpolated": n_interp / frames_grid,
        f"{hand}_frac_valid": n_valid / frames_grid,
        f"{hand}_mean_speed": _stat(speed, np.mean),
        f"{hand}_max_speed": _stat(speed, np.max),
        f"{hand}_mean_accel_magnitude": _stat(column("accel_magnitude"), np.mean),
        f"{hand}_max_accel_magnitude": _stat(column("accel_magnitude"), np.max),
        f"{hand}_path_length": _stat(column("step_distance"), np.sum),
        f"{hand}_nx_range": _stat(column("nx_smooth"), lambda a: a.max() - a.min()),
        f"{hand}_ny_range": _stat(column("ny_smooth"), lambda a: a.max() - a.min()),
        f"{hand}_mean_neck_distance": _stat(column("neck_distance"), np.mean),
        f"{hand}_angle_mean": angle_mean, f"{hand}_angle_std": angle_std,
        f"{hand}_time_to_peak_speed": _time_to_peak(speed, times, clock, t_start),
        f"{hand}_early_speed_mean": _edge_window_mean(speed, times, clock, t_start, t_end, True),
        f"{hand}_late_speed_mean": _edge_window_mean(speed, times, clock, t_start, t_end, False),
    }


def _sequence_summary(clip_id, people, frames):
    times = frames["frame_timestamp"].to_numpy("float64")
    clock = frames["clock_known"].to_numpy("bool")
    both = frames["both_hands_valid"].to_numpy("bool")
    separation = frames["pair_separation"].to_numpy("float64")
    n_both = float(both.sum())
    frames_grid = float(len(frames))
    n_timed = float(clock.sum())
    t_start, t_end = _clock_span(times, clock)
    summary = {
        "id": clip_id, "people_id": people,
        "frames_source": float(frames["source_observed"].to_numpy("bool").sum()),
        "frames_grid": frames_grid,
        "frames_missing_grid": float(frames["frame_inferred"].to_numpy("bool").sum()),
        "n_timed_frames": n_timed,
        "n_timestamp_inferred": float(frames["timestamp_inferred"].to_numpy("bool").sum()),
        "n_missing_timestamps": frames_grid - n_timed,
        "t_start": t_start, "t_end": t_end,
        "duration_s": t_end - t_start if np.isfinite(t_start) and np.isfinite(t_end) else np.nan,
        "n_both_hands": n_both, "frac_both_hands": n_both / frames_grid,
        "mean_pair_separation": _stat(separation, np.mean),
        "max_pair_separation": _stat(separation, np.max),
        "mean_mirror_error_x": _stat(frames["mirror_error_x"].to_numpy("float64"), np.mean),
        "mean_mirror_error_y": _stat(frames["mirror_error_y"].to_numpy("float64"), np.mean),
        "time_to_peak_pair_separation": _time_to_peak(separation, times, clock, t_start),
    }
    for hand in ("right", "left"):
        summary.update(_hand_summary(frames, hand, frames_grid))
    return summary


# ----------------------------------------------------------------- public API ----

def compute_gesture_features(wide_df, config=None):
    """Neck-relative wrist features for a wide per-frame table.

    Args:
        wide_df: rows of REQUIRED_WIDE_COLUMNS (never modified).
        config: GestureConfig, a mapping of GestureConfig parameters, or None.

    Returns:
        (frame_features, sequence_features) DataFrames with FRAME_COLUMNS and
        SEQUENCE_COLUMNS - the same schema even when nothing was measurable.
    """
    cfg = config if isinstance(config, GestureConfig) else GestureConfig.from_input(config)
    df = _validate_input(wide_df)

    records = []
    for (clip_id, people), grp in df.groupby(["id", "people_id"], sort=True):
        records.extend(_sequence_frame_rows(clip_id, people, grp, cfg))

    frames = _typed_frame(pd.DataFrame(records, columns=list(FRAME_COLUMNS)))
    if frames.empty:
        return frames, _typed_frame(pd.DataFrame(columns=list(SEQUENCE_COLUMNS)), sequence=True)

    summaries = [_sequence_summary(clip_id, people, grp)
                 for (clip_id, people), grp in frames.groupby(["id", "people_id"], sort=True)]
    return frames, _typed_frame(pd.DataFrame(summaries, columns=list(SEQUENCE_COLUMNS)), sequence=True)


def _typed_frame(df, sequence=False):
    """Fixed dtypes: float64 features (NULL = NaN), nullable bool, int64 frame."""
    out = pd.DataFrame(index=range(len(df)), columns=list(SEQUENCE_COLUMNS if sequence else FRAME_COLUMNS))
    for name in out.columns:
        values = df[name].to_numpy(dtype=object) if len(df) else np.empty(0, dtype=object)
        if name == "id":
            out[name] = pd.Series(values, dtype="object")
        elif name == "frame":
            out[name] = pd.Series(pd.array(values, dtype="Int64"), dtype="Int64")
        elif name in FRAME_BOOL_COLUMNS and not sequence:
            out[name] = pd.Series(pd.array([None if v is None or (isinstance(v, float) and math.isnan(v))
                                            else bool(v) for v in values], dtype="boolean"),
                                  dtype="boolean")
        else:
            numbers = np.full(len(values), np.nan, dtype="float64")
            for i, value in enumerate(values):
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    numbers[i] = float(value)
            out[name] = np.where(np.isfinite(numbers), numbers, np.nan) if len(values) else numbers
    return out
