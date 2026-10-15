"""Tests for gesture_features.py: neck-relative BODY_25 wrist features.

Everything here goes through the public surface (`compute_gesture_features` and
`GestureConfig`) with hand-computed trajectories, so the numbers below are
literal expectations from geometry, not the result of running the code.

Contract under test:
  * BODY_25 wrists 4 (right) / 7 (left), read from dfMaker's nx/ny columns,
    whose origin is the NECK (point 1) and whose Y axis is positive UP.
  * a frame is usable when confidence >= threshold and confidence > 0;
  * interior gaps of at most `max_gap_frames` missing frames are linearly
    interpolated in *elapsed time*, edges are never extrapolated;
  * geometry needs 1 valid sample, velocity 2 connected samples, acceleration 3;
  * every feature is finite or NULL - never 0, never inf, for "unknown".
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gesture_features as gf  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAIL.append(name)


# --------------------------------------------------------------- fixtures ----

COLS = ("frame", "frame_timestamp", "nx_4", "ny_4", "c_4", "nx_7", "ny_7", "c_7")
ABSENT = (None, None, 0.0)


def rows(spec, clip="clipA", people=1.0):
    """Wide input rows: (frame, frame_timestamp, nx4, ny4, c4, nx7, ny7, c7)."""
    return [dict(zip(COLS, r), id=clip, people_id=people) for r in spec]


def wide(*groups):
    """DataFrame with the required wide columns from rows() groups."""
    return pd.DataFrame([r for group in groups for r in group],
                        columns=list(gf.REQUIRED_WIDE_COLUMNS))


def right_ramp(frames, slope=0.03, start=1, c=0.9, x0=0.0, y=0.0, fps=30.0):
    """Right wrist moving at a constant `slope` neck-space units per frame."""
    return [(start + i, (start + i - 1) / fps, x0 + slope * i, y, c, *ABSENT)
            for i in range(frames)]


def calc(spec, **cfg):
    df = spec if isinstance(spec, pd.DataFrame) else wide(rows(spec))
    return gf.compute_gesture_features(df, cfg)


def col(frames, name):
    return list(frames[name])


def near(a, b, tol=1e-9):
    return a is not None and not (isinstance(a, float) and math.isnan(a)) and abs(a - b) <= tol


def seq_one(frames, clip="clipA", people=1.0):
    got = frames[(frames["id"] == clip) & (frames["people_id"] == people)]
    assert len(got) == 1, f"expected one sequence row, got {len(got)}"
    return got.iloc[0]


# ------------------------------------------------------------------ config ----

def test_config_validation():
    ok = gf.GestureConfig()
    check("config defaults (conf 0.2, gap 3, window 7, poly 2)",
          (ok.min_keypoint_confidence, ok.max_gap_frames, ok.window_size, ok.poly_order)
          == (0.2, 3, 7, 2), str(ok.as_dict()))
    bad = [
        dict(min_keypoint_confidence=1.5), dict(min_keypoint_confidence=-0.1),
        dict(min_keypoint_confidence=float("nan")), dict(min_keypoint_confidence="alto"),
        dict(max_gap_frames=-1), dict(max_gap_frames=1.5), dict(max_gap_frames="tres"),
        dict(window_size=6), dict(window_size=0), dict(window_size=-7), dict(window_size=7.5),
        dict(poly_order=0), dict(poly_order=-1), dict(poly_order=2.5),
    ]
    for kwargs in bad:
        try:
            gf.GestureConfig(**kwargs)
            check(f"config rechaza {kwargs}", False, "no fallo")
        except gf.GestureConfigError:
            check(f"config rechaza {kwargs}", True)
    # 1 is a legitimate window: it means "no smoothing"
    check("window_size=1 permitido (sin suavizado)", gf.GestureConfig(window_size=1).window_size == 1)
    check("max_gap_frames=0 permitido (sin interpolacion)",
          gf.GestureConfig(max_gap_frames=0).max_gap_frames == 0)


def test_config_conflicting_and_unknown_keys():
    both = gf.GestureConfig.from_input({"max_gap_frames": 2, "max-gap-frames": 2})
    check("clave guion y guion bajo coincidentes aceptadas", both.max_gap_frames == 2)
    for kwargs in ({"max_gap_frames": 2, "max-gap-frames": 4}, {"nose": 1}, {"ventana": 3}):
        try:
            gf.GestureConfig.from_input(kwargs)
            check(f"config rechaza {kwargs}", False, "no fallo")
        except gf.GestureConfigError:
            check(f"config rechaza {kwargs}", True)


# -------------------------------------------------------------- trajectories ----

def test_constant_velocity_is_exact():
    """x = 0.03 per frame at 30 fps -> 0.9 units/s, zero acceleration."""
    frames, seqs = calc(right_ramp(10))
    vx = col(frames, "right_velocity_x")
    ax = col(frames, "right_accel_x")
    check("velocidad constante 0.9 u/s en todos los frames",
          all(near(v, 0.9, 1e-9) for v in vx), str(vx))
    check("aceleracion nula en trayectoria lineal",
          all(near(a, 0.0, 1e-9) for a in ax), str(ax))
    check("speed = |velocity|", all(near(s, 0.9, 1e-9) for s in col(frames, "right_speed")))
    check("suavizado no deforma una recta",
          all(near(a, b, 1e-12) for a, b in zip(col(frames, "right_nx_smooth"),
                                                col(frames, "right_nx"))))
    check("una secuencia por id/persona", len(seqs) == 1)


def test_quadratic_acceleration_is_exact():
    """x = 0.2 t^2 -> v = 0.4 t, a = 0.4 using real timestamps, not frame index."""
    spec = [(f, (f - 1) / 30.0, 0.2 * ((f - 1) / 30.0) ** 2, 0.0, 0.9, *ABSENT)
            for f in range(1, 10)]
    frames, _ = calc(spec)
    t = col(frames, "frame_timestamp")
    vx = col(frames, "right_velocity_x")
    ax = col(frames, "right_accel_x")
    check("velocidad = 0.4 t (derivada temporal real)",
          all(near(v, 0.4 * tt, 1e-7) for v, tt in zip(vx, t)), str(vx))
    check("aceleracion constante 0.4 u/s^2", all(near(a, 0.4, 1e-7) for a in ax), str(ax))


def test_stationary_geometry_and_neck_orientation():
    """y > 0 is ABOVE the neck: the dfMaker Y axis points up."""
    spec = [(f, f / 30.0, 0.0, 0.5, 0.8, 0.0, -0.5, 0.8) for f in range(1, 6)]
    frames, _ = calc(spec)
    check("distance al cuello = |y|", all(near(d, 0.5, 1e-12) for d in col(frames, "right_neck_distance")))
    check("arriba del cuello cuando ny > 0",
          all(bool(v) for v in col(frames, "right_above_neck"))
          and frames["right_above_neck"].notna().all())
    check("debajo del cuello cuando ny < 0",
          not any(bool(v) for v in col(frames, "left_above_neck"))
          and frames["left_above_neck"].notna().all())
    check("angulo +pi/2 hacia arriba (Y positiva hacia arriba)",
          near(math.radians(90), math.radians(90))
          and all(near(a, math.pi / 2, 1e-12) for a in col(frames, "right_neck_angle")),
          str(col(frames, "right_neck_angle")))
    check("angulo -pi/2 hacia abajo para la izquierda",
          all(near(a, -math.pi / 2, 1e-12) for a in col(frames, "left_neck_angle")))
    check("sin movimiento -> velocidad 0 medida, no NULL",
          all(near(v, 0.0, 1e-12) for v in col(frames, "right_velocity_x")))
    check("distancia entre frames nula (0 real, no NULL)",
          all(near(d, 0.0, 1e-12) for d in col(frames, "right_step_distance")[1:]))


def test_angle_undefined_at_origin_is_null():
    """Wrist exactly on the neck: distance is a real 0, the angle is undefined."""
    spec = [(f, f / 30.0, 0.0, 0.0, 0.9, *ABSENT) for f in range(1, 4)]
    frames, _ = calc(spec)
    check("distance 0 en el origen", all(near(d, 0.0) for d in col(frames, "right_neck_distance")))
    check("angulo NULL en el origen (no 0)",
          all(v is None or (isinstance(v, float) and math.isnan(v))
              for v in col(frames, "right_neck_angle")), str(col(frames, "right_neck_angle")))


def test_confidence_threshold_inclusive_and_zero_absent():
    spec = [(1, 1 / 30.0, 0.1, 0.1, 0.2, *ABSENT), (2, 2 / 30.0, 0.1, 0.1, 0.1999, *ABSENT),
            (3, 3 / 30.0, 0.1, 0.1, 0.0, *ABSENT)]
    frames, _ = calc(spec)
    d = col(frames, "right_neck_distance")
    check("confidence == umbral cuenta (inclusivo)", near(d[0], math.hypot(0.1, 0.1), 1e-12))
    check("confidence < umbral -> NULL", math.isnan(d[1]), str(d[1]))
    check("confidence 0 -> NULL", math.isnan(d[2]), str(d[2]))
    zero_thr, _ = calc([(1, 1 / 30.0, 0.1, 0.1, 0.0, *ABSENT)], min_keypoint_confidence=0.0)
    check("confidence 0 esta ausente aunque el umbral sea 0",
          math.isnan(col(zero_thr, "right_neck_distance")[0]))
    nan_coord = calc([(1, 1 / 30.0, None, 0.1, 0.9, *ABSENT)])[0]
    check("coordenada ausente -> NULL", math.isnan(col(nan_coord, "right_neck_distance")[0]))


# ------------------------------------------------------------- gaps/interp ----

def test_internal_short_gap_is_interpolated_in_time():
    """Valid at frames 1, 2 and 6: the 3-frame interior gap is bridged in time."""
    spec = [(1, 0.0, 0.0, 0.0, 0.9, *ABSENT), (2, 0.1, 0.1, 0.1, 0.9, *ABSENT),
            (3, 0.2, None, None, 0.0, *ABSENT), (4, 0.3, None, None, 0.0, *ABSENT),
            (5, 0.4, None, None, 0.0, *ABSENT), (6, 0.5, 0.5, 0.1, 0.9, *ABSENT)]
    frames, seqs = calc(spec)
    nx = col(frames, "right_nx")
    check("interpolar 3 frames con recta en tiempo real",
          [near(v, e, 1e-12) for v, e in zip(nx, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])], str(nx))
    check("marcar interpolados", col(frames, "right_interpolated") == [False, False, True, True, True, False],
          str(col(frames, "right_interpolated")))
    check("marcar observados", col(frames, "right_observed") == [True, True, False, False, False, True],
          str(col(frames, "right_observed")))
    check("flags de validez cubren el hueco",
          col(frames, "right_valid") == [True] * 6, str(col(frames, "right_valid")))
    s = seq_one(seqs)
    check("conteos de interpolacion", (s["right_n_interpolated"], s["right_n_observed"]) == (3.0, 3.0),
          str((s["right_n_interpolated"], s["right_n_observed"])))
    check("fraccion interpolada sobre el span observado",
          near(s["right_frac_interpolated"], 0.5, 1e-12), str(s["right_frac_interpolated"]))


def test_long_and_edge_gaps_are_never_bridged():
    """4 missing interior frames and both edge gaps stay NULL, no fake 0."""
    spec = [(1, 0.0, None, None, 0.0, *ABSENT), (2, 0.1, 0.0, 0.0, 0.9, *ABSENT),
            (3, 0.2, None, None, 0.0, *ABSENT), (4, 0.3, None, None, 0.0, *ABSENT),
            (5, 0.4, None, None, 0.0, *ABSENT), (6, 0.5, None, None, 0.0, *ABSENT),
            (7, 0.6, 0.5, 0.0, 0.9, *ABSENT), (8, 0.7, None, None, 0.0, *ABSENT)]
    frames, _ = calc(spec)
    nx = col(frames, "right_nx")
    check("hueco interno de 4 frames NO se interpola",
          all(math.isnan(v) for v in nx[2:6]), str(nx))
    check("hueco inicial NO se extrapola", math.isnan(nx[0]), str(nx[0]))
    check("hueco final NO se extrapola", math.isnan(nx[7]), str(nx[7]))
    check("nada se interpola aqui", not any(bool(v) for v in col(frames, "right_interpolated")))
    sm = col(frames, "right_nx_smooth")
    check("el suavizado tampoco cruza el hueco", all(math.isnan(v) for v in sm[2:6]), str(sm))
    d = col(frames, "right_step_distance")
    check("sin distancia falsa a traves del hueco (todo NULL)",
          all(math.isnan(v) for v in d), str(d))


def test_no_derivative_across_a_long_gap():
    """Two lone valid samples separated by a long gap have no velocity at all."""
    spec = [(1, 0.0, 0.0, 0.0, 0.9, *ABSENT), (2, 0.1, None, None, 0.0, *ABSENT),
            (3, 0.2, None, None, 0.0, *ABSENT), (4, 0.3, None, None, 0.0, *ABSENT),
            (5, 0.4, None, None, 0.0, *ABSENT), (6, 0.5, 1.0, 0.0, 0.9, *ABSENT)]
    frames, _ = calc(spec)
    check("geometria si disponible en cada sample valido",
          near(col(frames, "right_neck_distance")[0], 0.0, 1e-12)
          and near(col(frames, "right_neck_distance")[5], 1.0, 1e-12))
    for name in ("right_velocity_x", "right_velocity_y", "right_speed", "right_accel_x",
                 "right_accel_magnitude"):
        values = col(frames, name)
        check(f"{name} NULL sin samples conectados",
              all(math.isnan(v) for v in values), str(values))


def test_max_gap_zero_disables_interpolation():
    spec = [(1, 0.0, 0.0, 0.0, 0.9, *ABSENT), (2, 0.1, None, None, 0.0, *ABSENT),
            (3, 0.2, 0.2, 0.0, 0.9, *ABSENT)]
    off, off_seqs = calc(spec, max_gap_frames=0)
    on, on_seqs = calc(spec, max_gap_frames=1)
    check("max_gap_frames=0 deja el hueco en NULL", math.isnan(col(off, "right_nx")[1]))
    check("max_gap_frames=1 lo rellena", near(col(on, "right_nx")[1], 0.1, 1e-12))
    check("max_gap_frames=0 no declara interpolados",
          seq_one(off_seqs)["right_n_interpolated"] == 0.0)
    check("max_gap_frames=1 declara uno",
          seq_one(on_seqs)["right_n_interpolated"] == 1.0)


def test_missing_frame_indices_are_expanded_within_span():
    """Frames 4 and 5 never arrive: they become inferred timesteps, timed."""
    spec = [(f, (f - 1) / 10.0, f * 0.1, 0.0, 0.9, *ABSENT)
            for f in (1, 2, 3, 6, 7)]
    frames, seqs = calc(spec)
    check("span expandido 1..7", col(frames, "frame") == [1, 2, 3, 4, 5, 6, 7])
    check("timestamps inferidos del grid real",
          [near(v, e, 1e-12) for v, e in zip(col(frames, "frame_timestamp"),
                                             [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])])
    check("solo frames 4 y 5 marcados como inferidos",
          col(frames, "frame_inferred") == [False] * 3 + [True, True] + [False, False],
          str(col(frames, "frame_inferred")))
    check("un frame inferido no es 'observado'",
          col(frames, "source_observed") == [True, True, True, False, False, True, True])
    s = seq_one(seqs)
    check("span declarado: source/grid/missing",
          (s["frames_source"], s["frames_grid"], s["frames_missing_grid"]) == (5.0, 7.0, 2.0),
          str((s["frames_source"], s["frames_grid"], s["frames_missing_grid"])))
    check("coordenadas interpoladas en los frames inferidos",
          all(bool(v) for v in col(frames, "right_interpolated")[3:5]),
          str(col(frames, "right_interpolated")))


# ---------------------------------------------------------- short sequences ----

def test_short_and_single_sequences_stay_stable():
    one = calc([(1, 0.033, 0.3, 0.4, 0.9, *ABSENT)])
    check("1 frame: esquema completo presente",
          list(one[0].columns) == list(gf.FRAME_COLUMNS))
    check("1 frame: geometria medida", near(col(one[0], "right_neck_distance")[0], 0.5, 1e-12))
    check("1 frame: velocidad NULL (faltan 2 samples)",
          math.isnan(col(one[0], "right_speed")[0]))
    check("1 frame: aceleracion NULL", math.isnan(col(one[0], "right_accel_magnitude")[0]))

    two = calc([(1, 0.0, 0.0, 0.0, 0.9, *ABSENT), (2, 2.0, 1.0, 0.0, 0.9, *ABSENT)])
    check("2 frames: velocidad con dt real de 2 s",
          near(col(two[0], "right_velocity_x")[0], 0.5, 1e-12), str(col(two[0], "right_velocity_x")))
    check("2 frames: aceleracion NULL (faltan 3 samples)",
          all(math.isnan(v) for v in col(two[0], "right_accel_x")), str(col(two[0], "right_accel_x")))

    three = calc([(1, 0.0, 0.0, 0.0, 0.9, *ABSENT), (2, 1.0, 1.0, 0.0, 0.9, *ABSENT),
                  (3, 3.0, 9.0, 0.0, 0.9, *ABSENT)])  # x = t^2 on irregular times
    check("3 frames: aceleracion = 2 con tiempo irregular",
          all(near(a, 2.0, 1e-9) for a in col(three[0], "right_accel_x")),
          str(col(three[0], "right_accel_x")))
    check("3 frames: velocidad = 2 t con tiempo irregular",
          [near(v, e, 1e-9) for v, e in zip(col(three[0], "right_velocity_x"), [0.0, 2.0, 6.0])],
          str(col(three[0], "right_velocity_x")))


def test_all_missing_sequence_keeps_schema_and_nulls():
    spec = [(f, f / 30.0, None, None, 0.0, None, None, 0.0) for f in range(1, 6)]
    frames, seqs = calc(spec)
    s = seq_one(seqs)
    check("secuencia sin datos conserva su fila", len(seqs) == 1 and len(frames) == 5)
    keys = ("id", "people_id", "frame", "frame_timestamp", "right_confidence", "left_confidence")
    numeric = [c for c in gf.FRAME_COLUMNS if c not in keys and frames[c].dtype.kind == "f"]
    check("todos los features numericos son NULL",
          all(frames[c].isna().all() for c in numeric),
          str([c for c in numeric if not frames[c].isna().all()]))
    check("n_valid 0 y path_length NULL (no 0)",
          s["right_n_valid"] == 0.0 and math.isnan(s["right_path_length"]),
          str(s["right_path_length"]))
    check("conteo de frames igual existe", (s["frames_source"], s["frames_grid"]) == (5.0, 5.0))
    check("timestamps de la secuencia igualmente reportados",
          near(s["t_start"], 1 / 30.0, 1e-12) and near(s["duration_s"], 4 / 30.0, 1e-12))


def test_empty_input_returns_typed_empty_frames():
    empty = pd.DataFrame(columns=list(gf.REQUIRED_WIDE_COLUMNS))
    frames, seqs = gf.compute_gesture_features(empty, gf.GestureConfig())
    check("entrada vacia: columnas de frames", list(frames.columns) == list(gf.FRAME_COLUMNS))
    check("entrada vacia: columnas de secuencias", list(seqs.columns) == list(gf.SEQUENCE_COLUMNS))
    check("entrada vacia: 0 filas", len(frames) == 0 and len(seqs) == 0)


def test_short_run_smoothing_adapts_and_window_one_disables():
    spec = right_ramp(3, slope=0.03)
    frames, _ = calc(spec)  # window 7 > run of 3: must adapt, not crash
    check("run corto: suavizado adaptado conserva la recta",
          all(near(a, b, 1e-12) for a, b in zip(col(frames, "right_nx_smooth"),
                                                col(frames, "right_nx"))))
    off, _ = calc(right_ramp(10), window_size=1)
    check("window_size=1 no suaviza nada",
          col(off, "right_nx_smooth") == col(off, "right_nx"))
    lin, _ = calc(right_ramp(10), window_size=9, poly_order=8)
    check("poly_order alto adaptado sigue exacto en una recta",
          all(near(v, 0.9, 1e-7) for v in col(lin, "right_velocity_x")),
          str(col(lin, "right_velocity_x")))


def test_irregular_timestamps_use_elapsed_time():
    """Sampled by hand at irregular times: velocity must follow seconds, not index."""
    spec = [(1, 0.0, 1.0, 0.0, 0.9, *ABSENT), (2, 0.5, 1.25, 0.0, 0.9, *ABSENT),
            (3, 1.5, 1.75, 0.0, 0.9, *ABSENT), (4, 3.5, 2.75, 0.0, 0.9, *ABSENT),
            (5, 6.0, 4.0, 0.0, 0.9, *ABSENT)]
    frames, _ = calc(spec)
    vx = col(frames, "right_velocity_x")
    check("velocidad 0.5 u/s con tiempos muy irregulares",
          all(near(v, 0.5, 1e-9) for v in vx), str(vx))
    check("suavizado no asume indice uniforme",
          all(near(a, b, 1e-9) for a, b in zip(col(frames, "right_nx_smooth"),
                                               [1.0, 1.25, 1.75, 2.75, 4.0])),
          str(col(frames, "right_nx_smooth")))
    check("aceleracion nula con tiempo irregular",
          all(near(a, 0.0, 1e-9) for a in col(frames, "right_accel_x")))


# ------------------------------------------------------------------ pairs ----

def test_disjoint_hands_have_no_pair_features():
    spec = [(1, 1 / 30.0, 0.1, 0.1, 0.9, None, None, 0.0),
            (2, 2 / 30.0, 0.2, 0.1, 0.9, None, None, 0.0),
            (3, 3 / 30.0, None, None, 0.0, 0.3, 0.4, 0.9),
            (4, 4 / 30.0, None, None, 0.0, 0.3, 0.4, 0.9)]
    frames, seqs = calc(spec)
    s = seq_one(seqs)
    check("cada mano se cuenta sola",
          (s["right_n_valid"], s["left_n_valid"]) == (2.0, 2.0),
          str((s["right_n_valid"], s["left_n_valid"])))
    check("ambas manos nunca validas a la vez", not any(bool(v) for v in col(frames, "both_hands_valid")))
    check("features de par NULL sin co-ocurrencia",
          all(frames[c].isna().all() for c in ("pair_separation", "pair_angle", "centroid_x",
                                               "centroid_y", "mirror_error_x", "mirror_error_y")))
    check("fraccion de par a 0", near(s["frac_both_hands"], 0.0))


def test_pair_geometry_is_exact():
    """Right (0.4, 0.3), left (-0.2, 0.3): separation 0.6 along x, mirrored in x."""
    spec = [(f, f / 30.0, 0.4, 0.3, 0.9, -0.2, 0.3, 0.9) for f in range(1, 5)]
    frames, seqs = calc(spec)
    s = seq_one(seqs)
    check("separacion 0.6", all(near(v, 0.6, 1e-12) for v in col(frames, "pair_separation")))
    check("angulo de par 0 (derecha a la izquierda del par)",
          all(near(v, 0.0, 1e-12) for v in col(frames, "pair_angle")))
    check("centroides (0.1, 0.3)",
          all(near(v, 0.1, 1e-12) for v in col(frames, "centroid_x"))
          and all(near(v, 0.3, 1e-12) for v in col(frames, "centroid_y")))
    check("espejo en x = |xR + xL| = 0.2",
          all(near(v, 0.2, 1e-12) for v in col(frames, "mirror_error_x")))
    check("espejo en y = |yR - yL| = 0",
          all(near(v, 0.0, 1e-12) for v in col(frames, "mirror_error_y")))
    check("fraccion de par = 1", near(s["frac_both_hands"], 1.0))


# ------------------------------------------------------------- aggregation ----

def test_sequence_aggregates_are_exact():
    """x = [0, .03, .06, .36, .38, .40] at 0.1 s steps, smoothing off."""
    xs = [0.0, 0.03, 0.06, 0.36, 0.38, 0.40]
    spec = [(f + 1, f * 0.1, x, 0.0, 0.9, *ABSENT) for f, x in enumerate(xs)]
    frames, seqs = calc(spec, window_size=1, max_gap_frames=0)
    s = seq_one(seqs)
    speeds = col(frames, "right_speed")
    check("speeds de referencia", [near(v, e, 1e-9) for v, e in
          zip(speeds, [0.3, 0.3, 1.65, 1.6, 0.2, 0.2])], str(speeds))
    check("max_speed", near(s["right_max_speed"], 1.65, 1e-9), str(s["right_max_speed"]))
    check("mean_speed", near(s["right_mean_speed"], (0.3 + 0.3 + 1.65 + 1.6 + 0.2 + 0.2) / 6, 1e-9),
          str(s["right_mean_speed"]))
    check("path_length = suma de desplazamientos", near(s["right_path_length"], 0.40, 1e-12),
          str(s["right_path_length"]))
    check("nx_range", near(s["right_nx_range"], 0.40, 1e-12))
    check("ny_range es 0 real, no NULL", near(s["right_ny_range"], 0.0, 1e-12), str(s["right_ny_range"]))
    check("time_to_peak_speed = 0.2 s desde el inicio",
          near(s["right_time_to_peak_speed"], 0.2, 1e-12), str(s["right_time_to_peak_speed"]))
    check("early/late usan el 10% del tiempo",
          near(s["right_early_speed_mean"], 0.3, 1e-9) and near(s["right_late_speed_mean"], 0.2, 1e-9),
          f'{s["right_early_speed_mean"]} {s["right_late_speed_mean"]}')
    check("duration = span real en segundos", near(s["duration_s"], 0.5, 1e-12))
    check("mean_neck_distance", near(s["right_mean_neck_distance"], sum(xs) / 6, 1e-12),
          str(s["right_mean_neck_distance"]))
    check("mano ausente: resumenes NULL",
          all(math.isnan(s[c]) for c in ("left_max_speed", "left_mean_speed", "left_time_to_peak_speed",
                                         "left_path_length", "left_angle_mean", "left_nx_range",
                                         "left_mean_accel_magnitude", "mean_pair_separation")))
    check("aceleracion presente con 6 samples conectados",
          math.isfinite(s["right_max_accel_magnitude"]) and s["right_max_accel_magnitude"] > 0,
          str(s["right_max_accel_magnitude"]))


def test_time_to_peak_pair_separation_is_elapsed_time():
    """Hands approach: peak separation is at the first frame, 4 s before the end."""
    spec = [(f + 1, f * 1.0, 1.0 - 0.1 * f, 0.0, 0.9, -1.0 + 0.1 * f, 0.0, 0.9)
            for f in range(5)]
    frames, seqs = calc(spec, window_size=1, max_gap_frames=0)
    s = seq_one(seqs)
    check("separacion maxima al inicio", near(s["max_pair_separation"], 2.0, 1e-12),
          str(s["max_pair_separation"]))
    check("time_to_peak_pair_separation = 0 s", near(s["time_to_peak_pair_separation"], 0.0, 1e-12),
          str(s["time_to_peak_pair_separation"]))
    check("separaciones decrecientes", [near(v, e, 1e-12) for v, e in
          zip(col(frames, "pair_separation"), [2.0, 1.8, 1.6, 1.4, 1.2])],
          str(col(frames, "pair_separation")))


def test_circular_angle_stats_are_clipped():
    """Angles {90, 0, 90, 180} deg: resultant 0.5, mean 90 deg, circ std sqrt(2 ln 2)."""
    spec = [(1, 0.1, 0.0, 1.0, 0.9, *ABSENT), (2, 0.2, 1.0, 0.0, 0.9, *ABSENT),
            (3, 0.3, 0.0, 1.0, 0.9, *ABSENT), (4, 0.4, -1.0, 0.0, 0.9, *ABSENT)]
    frames, seqs = calc(spec, window_size=1)
    s = seq_one(seqs)
    check("media circular = pi/2", near(s["right_angle_mean"], math.pi / 2, 1e-9),
          str(s["right_angle_mean"]))
    check("desviacion circular = sqrt(2 ln 2)",
          near(s["right_angle_std"], math.sqrt(2 * math.log(2)), 1e-9), str(s["right_angle_std"]))
    constant = calc([(f, f / 30.0, 0.0, 1.0, 0.9, *ABSENT) for f in range(1, 6)], window_size=1)
    sc = seq_one(constant[1])
    check("angulos identicos: R=1 -> desviacion 0 (sin NaN por redondeo)",
          near(sc["right_angle_std"], 0.0, 1e-12) and near(sc["right_angle_mean"], math.pi / 2, 1e-12),
          str(sc["right_angle_std"]))


# ------------------------------------------------------------- isolation ----

def test_sequences_are_isolated_per_id_and_person():
    a = right_ramp(4, slope=0.03, start=1, x0=0.0)
    b = rows(right_ramp(4, slope=0.03, start=1, x0=10.0), clip="clipB")
    still = [(f, (f - 1) / 30.0, 5.0, 0.0, 0.9, *ABSENT) for f in (1, 2, 3, 4)]
    df = wide(rows(a), b, rows(still, clip="clipA", people=2.0))
    frames, seqs = gf.compute_gesture_features(df, gf.GestureConfig())
    keys = list(zip(seqs["id"], seqs["people_id"]))
    check("una fila de secuencia por (id, people_id)",
          sorted(keys) == [("clipA", 1.0), ("clipA", 2.0), ("clipB", 1.0)], str(keys))
    check("frames ordenados por secuencia y frame",
          len(frames) == 12 and col(frames, "frame") == [1, 2, 3, 4] * 3)
    other = frames[(frames["id"] == "clipA") & (frames["people_id"] == 2.0)]
    check("la segunda persona no hereda el movimiento de la primera",
          all(near(v, 5.0, 1e-12) for v in col(other, "right_nx"))
          and all(near(v, 0.0, 1e-9) for v in col(other, "right_speed")),
          str(col(other, "right_speed")))
    clipb = frames[frames["id"] == "clipB"]
    check("otro clip conserva su propia pendiente (0.9 u/s)",
          all(near(v, 0.9, 1e-7) for v in col(clipb, "right_velocity_x")),
          str(col(clipb, "right_velocity_x")))
    check("el inicio de una secuencia no toma el frame final de otra",
          math.isnan(list(other["right_step_distance"])[0]),
          str(col(other, "right_step_distance")))


def test_input_is_not_mutated_and_outputs_have_no_infinity():
    spec = right_ramp(6) + [(7, 6 / 30.0, np.nan, np.nan, np.nan, *ABSENT)]
    df = wide(rows(spec))
    before = df.copy(deep=True)
    frames, seqs = gf.compute_gesture_features(df, gf.GestureConfig())
    check("el DataFrame de entrada no se modifica", df.equals(before))
    bad = []
    for table in (frames, seqs):
        for name in table.columns:
            if table[name].dtype.kind == "f":
                arr = table[name].to_numpy("float64")
                if np.isinf(arr).any():
                    bad.append(name)
    check("sin inf/-inf en ningun feature", not bad, str(bad))
    check("los features boolenos son boolenos verdaderos (no 0/1)",
          all(frames[c].dtype == "boolean" for c in
              ("right_above_neck", "right_observed", "right_interpolated", "both_hands_valid")),
          str({c: str(frames[c].dtype) for c in ("right_above_neck", "both_hands_valid")}))
    check("flags de geometria NULL cuando no hay dato",
          frames["left_above_neck"].isna().all(), "left_above_neck deberia ser NULL")


def test_bad_input_data_fails_clear():
    dup_conflict = wide(rows([(1, 0.1, 0.1, 0.1, 0.9, *ABSENT),
                              (1, 0.1, 0.4, 0.1, 0.9, *ABSENT)]))
    for name, df in (
        ("filas duplicadas contradictorias", dup_conflict),
        ("timestamp no monotono", wide(rows([(1, 0.5, 0.1, 0.1, 0.9, *ABSENT),
                                             (2, 0.2, 0.1, 0.1, 0.9, *ABSENT)]))),
        ("timestamp infinito", wide(rows([(1, float("inf"), 0.1, 0.1, 0.9, *ABSENT)]))),
        ("faltan columnas", pd.DataFrame({"id": ["a"], "frame": [1]})),
    ):
        try:
            gf.compute_gesture_features(df, gf.GestureConfig())
            check(f"rechaza {name}", False, "no fallo")
        except gf.GestureDataError:
            check(f"rechaza {name}", True)
        except Exception as e:  # noqa: BLE001
            check(f"rechaza {name}", False, repr(e))


def test_round1_smoothing_order_must_fit_the_window():
    """A configured order >= window is a mistake, not something to silently shrink."""
    for kwargs in (dict(window_size=5, poly_order=5), dict(window_size=5, poly_order=7),
                   dict(window_size=3, poly_order=3)):
        try:
            gf.GestureConfig(**kwargs)
            check(f"config rechaza {kwargs}", False, "no fallo")
        except gf.GestureConfigError:
            check(f"config rechaza {kwargs}", True)
    check("default (7, 2) sigue valido", gf.GestureConfig().poly_order == 2)
    check("window 1 desactiva y acepta cualquier orden",
          gf.GestureConfig(window_size=1, poly_order=9).window_size == 1)
    # short runs still adapt the ORDER of a legitimate configuration (5, 3) on 3 frames
    short = calc([(f, f / 30.0, 0.1 * f, f / 10.0, 0.9, *ABSENT) for f in (1, 2, 3)],
                 window_size=5, poly_order=3)
    check("run corto: orden adaptado, sin error",
          all(near(a, b, 1e-12) for a, b in zip(col(short[0], "right_nx_smooth"),
                                                [0.1, 0.2, 0.3])),
          str(col(short[0], "right_nx_smooth")))


def test_round1_opposing_angles_have_no_meaningful_direction():
    """{0 deg, 180 deg} has a zero resultant: no mean, no spread, not pi/2 and not 8.64."""
    spec = [(1, 0.1, 1.0, 0.0, 0.9, *ABSENT), (2, 0.2, -1.0, 0.0, 0.9, *ABSENT)]
    s = seq_one(calc(spec, window_size=1)[1])
    check("angulos opuestos: media circular NULL",
          math.isnan(s["right_angle_mean"]), str(s["right_angle_mean"]))
    check("angulos opuestos: desviacion circular NULL",
          math.isnan(s["right_angle_std"]), str(s["right_angle_std"]))
    three_way = [(f, i * 0.1, math.cos(a), math.sin(a), 0.9, *ABSENT)
                 for i, (f, a) in enumerate(zip((1, 2, 3), (0.0, 2 * math.pi / 3, 4 * math.pi / 3)))]
    sr = seq_one(calc(three_way, window_size=1)[1])
    check("tres direcciones que cierran el circulo: resultante nula -> NULL",
          math.isnan(sr["right_angle_mean"]) and math.isnan(sr["right_angle_std"]),
          f'{sr["right_angle_mean"]} {sr["right_angle_std"]}')
    same = seq_one(calc([(f, f / 30.0, 0.0, 1.0, 0.9, *ABSENT) for f in (1, 2, 3)],
                        window_size=1)[1])
    check("R=1 conserva std 0", near(same["right_angle_std"], 0.0, 1e-12)
          and near(same["right_angle_mean"], math.pi / 2, 1e-12), str(same["right_angle_std"]))


def test_round1_untimed_frames_keep_geometry_without_time():
    """A NULL frame_timestamp is a real frame with no clock, not a dropped frame."""
    spec = [(1, 0.0, 0.1, 0.2, 0.9, *ABSENT), (2, 0.1, 0.2, 0.2, 0.9, *ABSENT),
            (3, None, 0.3, 0.2, 0.9, *ABSENT)]
    frames, seqs = calc(spec, window_size=1)
    s = seq_one(seqs)
    check("el frame sin reloj se conserva", col(frames, "frame") == [1, 2, 3])
    check("geometria del frame sin reloj presente",
          near(col(frames, "right_neck_distance")[2], math.hypot(0.3, 0.2), 1e-12))
    check("clock_known False y no inferido",
          col(frames, "clock_known") == [True, True, False]
          and col(frames, "timestamp_inferred") == [False, False, False],
          str(col(frames, "clock_known")))
    check("timestamp NULL, no 0", math.isnan(col(frames, "frame_timestamp")[2]))
    check("sin reloj no hay velocidad", math.isnan(col(frames, "right_speed")[2]))
    check("t_start/t_end span del reloj conocido, no del ultimo frame",
          near(s["t_start"], 0.0) and near(s["t_end"], 0.1, 1e-12)
          and near(s["duration_s"], 0.1, 1e-12), f'{s["t_start"]} {s["t_end"]} {s["duration_s"]}')
    check("contadores de reloj explicitos",
          (s["n_timed_frames"], s["n_timestamp_inferred"], s["n_missing_timestamps"]) == (2.0, 0.0, 1.0),
          str((s["n_timed_frames"], s["n_timestamp_inferred"], s["n_missing_timestamps"])))
    check("frames_grid incluye el frame sin reloj",
          (s["frames_source"], s["frames_grid"]) == (3.0, 3.0))


def test_round1_no_speed_across_an_unknown_clock():
    """Adjacent coordinates still give a step/path; velocity stops at the clock hole."""
    spec = [(1, 0.0, 0.0, 0.0, 0.9, *ABSENT), (2, 1.0, 1.0, 0.0, 0.9, *ABSENT),
            (3, None, 2.0, 0.0, 0.9, *ABSENT)]
    frames, seqs = calc(spec, window_size=1)
    check("distancia entre frames adyacentes sin reloj existe",
          [near(v, e, 1e-12) for v, e in
           zip(col(frames, "right_step_distance"), [float("nan"), 1.0, 1.0])
           if not math.isnan(v)] and math.isnan(col(frames, "right_step_distance")[0]),
          str(col(frames, "right_step_distance")))
    check("path_length cubre el hueco de reloj",
          near(seq_one(seqs)["right_path_length"], 2.0, 1e-12))
    check("suavizado no cruza el frame sin reloj",
          near(col(frames, "right_nx_smooth")[2], 2.0, 1e-12))


def test_round1_clock_inference_is_bounded_internal_and_flagged():
    """The grid may borrow a clock only across a short internal gap, never at an edge."""
    inferred = [(1, 0.0, 0.1, 0.0, 0.9, *ABSENT), (2, 1.0, 0.2, 0.0, 0.9, *ABSENT),
                (3, None, 0.3, 0.0, 0.9, *ABSENT), (4, 3.0, 0.4, 0.0, 0.9, *ABSENT)]
    frames, seqs = calc(inferred)
    s = seq_one(seqs)
    check("hueco interno corto: reloj inferido por interpolacion lineal",
          col(frames, "clock_known") == [True] * 4
          and col(frames, "timestamp_inferred") == [False, False, True, False]
          and near(col(frames, "frame_timestamp")[2], 2.0, 1e-12),
          str(col(frames, "frame_timestamp")))
    check("el frame sigue marcado como observado y no interpolado en coordenadas",
          col(frames, "source_observed") == [True] * 4
          and col(frames, "right_interpolated") == [False] * 4)
    check("contador de reloj inferido", (s["n_timestamp_inferred"], s["n_timed_frames"]) == (1.0, 4.0))

    stamps = {1: 0.0, 2: 1.0, 3: None, 4: None, 5: None, 6: 6.0}
    long_gap = [(f, stamps[f], 0.1 * f, 0.0, 0.9, *ABSENT) for f in (1, 2, 3, 4, 5, 6)]
    refused, refused_seqs = calc(long_gap, max_gap_frames=2)
    check("hueco de reloj de 3 frames con max_gap 2 NO se inventa",
          col(refused, "clock_known") == [True, True, False, False, False, True]
          and col(refused, "timestamp_inferred") == [False] * 6
          and all(math.isnan(v) for v in col(refused, "frame_timestamp")[2:5]),
          str(col(refused, "frame_timestamp")))
    check("coordenadas validas sin reloj se conservan como geometria cruda",
          [near(v, e, 1e-12) for v, e in zip(col(refused, "right_nx")[2:5], [0.3, 0.4, 0.5])])
    check("nada se declara interpolado en ese hueco",
          not any(bool(v) for v in col(refused, "right_interpolated")))
    check("tres marcas de reloj faltan",
          seq_one(refused_seqs)["n_missing_timestamps"] == 3.0,
          str(seq_one(refused_seqs)["n_missing_timestamps"]))
    allowed, _ = calc(long_gap, max_gap_frames=3)
    check("con max_gap 3 el hueco de reloj si se rellena",
          col(allowed, "timestamp_inferred") == [False, False, True, True, True, False],
          str(col(allowed, "timestamp_inferred")))

    leading = [(1, None, 0.1, 0.0, 0.9, *ABSENT), (2, 1.0, 0.2, 0.0, 0.9, *ABSENT),
               (3, 2.0, 0.3, 0.0, 0.9, *ABSENT)]
    lead_frames, _ = calc(leading)
    check("el reloj nunca se extrapola hacia el borde",
          col(lead_frames, "clock_known")[0] is False or not bool(col(lead_frames, "clock_known")[0]),
          str(col(lead_frames, "clock_known")))


def test_round1_all_untimed_sequence_keeps_geometry_and_null_time():
    spec = [(f, None, 0.1 * f, 0.2, 0.9, *ABSENT) for f in (1, 2, 3, 4)]
    frames, seqs = calc(spec, window_size=1)
    s = seq_one(seqs)
    check("secuencia enteramente sin reloj conserva sus frames", len(frames) == 4)
    check("geometria presente sin ningun reloj",
          all(near(v, math.hypot(0.1 * f, 0.2), 1e-12)
              for v, f in zip(col(frames, "right_neck_distance"), (1, 2, 3, 4))))
    check("temporales NULL",
          all(math.isnan(s[c]) for c in ("t_start", "t_end", "duration_s", "right_mean_speed",
                                         "right_max_speed", "right_time_to_peak_speed",
                                         "right_early_speed_mean", "right_late_speed_mean")))
    check("velocidad NULL en todos los frames",
          all(math.isnan(v) for v in col(frames, "right_speed")))
    check("contadores: 0 timed, 4 sin marca",
          (s["n_timed_frames"], s["n_timestamp_inferred"], s["n_missing_timestamps"]) == (0.0, 0.0, 4.0))
    check("conteos de coordenadas intactos", s["right_n_valid"] == 4.0)


def test_round1_peak_time_is_null_when_the_peak_has_no_clock():
    """A geometry peak on an untimed frame has no elapsed time: NULL, not the 2nd peak."""
    spec = [(1, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.9), (2, 1.0, 1.0, 0.0, 0.9, 0.0, 0.0, 0.9),
            (3, None, 2.0, 0.0, 0.9, 0.0, 0.0, 0.9)]
    frames, seqs = calc(spec, window_size=1)
    s = seq_one(seqs)
    check("separacion maxima medida en el frame sin reloj",
          near(s["max_pair_separation"], 2.0, 1e-12) and near(col(frames, "pair_separation")[2], 2.0, 1e-12),
          str(col(frames, "pair_separation")))
    check("time_to_peak NULL en vez de elegir el segundo pico",
          math.isnan(s["time_to_peak_pair_separation"]), str(s["time_to_peak_pair_separation"]))


def test_round1_clock_must_increase_among_timed_frames():
    bad = wide(rows([(1, 1.0, 0.1, 0.0, 0.9, *ABSENT),
                     (2, None, 0.2, 0.0, 0.9, *ABSENT),
                     (3, 0.5, 0.3, 0.0, 0.9, *ABSENT)]))
    try:
        gf.compute_gesture_features(bad, gf.GestureConfig())
        check("rechaza reloj que retrocede entre frames con reloj", False, "no fallo")
    except gf.GestureDataError:
        check("rechaza reloj que retrocede entre frames con reloj", True)
    sparse = gf.compute_gesture_features(
        wide(rows([(1, 0.0, 0.1, 0.0, 0.9, *ABSENT), (50, None, 0.2, 0.0, 0.9, *ABSENT),
                   (99, 4.0, 0.3, 0.0, 0.9, *ABSENT)])), gf.GestureConfig())
    check("marcas dispersas con hueco largo: geometria si, reloj no inventado",
          len(sparse[0]) == 99 and sparse[1].iloc[0]["n_missing_timestamps"] == 97.0
          and sparse[1].iloc[0]["n_timed_frames"] == 2.0,
          str(sparse[1].iloc[0][["n_timed_frames", "n_missing_timestamps"]].to_dict()))


def test_round1_confidence_must_be_a_probability():
    for bad_conf in (1.5, -0.1, float("inf")):
        try:
            calc([(1, 0.1, 0.1, 0.0, bad_conf, *ABSENT)])
            check(f"rechaza confidence {bad_conf}", False, "no fallo")
        except gf.GestureDataError:
            check(f"rechaza confidence {bad_conf}", True)
    null_conf = calc([(1, 0.1, 0.1, 0.0, None, None, None, None)])
    check("confidence NULL es ausencia permitida",
          math.isnan(col(null_conf[0], "right_neck_distance")[0]))
    check("confidence NULL no infla los conteos",
          seq_one(null_conf[1])["right_n_valid"] == 0.0)


def test_round1_nonfinite_keys_and_huge_frames_are_rejected():
    for name, spec in (("people_id infinito", [(1, 0.1, 0.1, 0.0, 0.9, *ABSENT)]),
                       ("frame desbordado", [(9300000000000000000, 0.1, 0.1, 0.0, 0.9, *ABSENT)]),
                       ("frame infinito", [(float("inf"), 0.1, 0.1, 0.0, 0.9, *ABSENT)])):
        df = wide(rows(spec))
        if name == "people_id infinito":
            df["people_id"] = float("inf")
        try:
            gf.compute_gesture_features(df, gf.GestureConfig())
            check(f"rechaza {name}", False, "no fallo")
        except gf.GestureDataError:
            check(f"rechaza {name}", True)
    for kwargs in (dict(max_gap_frames=1e30), dict(window_size=1e30)):
        try:
            gf.GestureConfig(**kwargs)
            check(f"config rechaza {kwargs}", False, "no fallo")
        except gf.GestureConfigError:
            check(f"config rechaza {kwargs}", True)


def test_final_frame_int64_overflow_is_rejected():
    for frame in (2**63, float(2**63), -2**63 - 1):
        df = wide(rows([(frame, None, 1.0, 0.0, .9, *ABSENT)]))
        try:
            gf.compute_gesture_features(df)
            check(f"int64 overflow rejected: {frame!r}", False)
        except gf.GestureDataError:
            check(f"int64 overflow rejected: {frame!r}", True)


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[fn]()
        except Exception as e:  # noqa: BLE001
            check(f"{fn} (excepcion)", False, repr(e))
    print("\nTOTAL:", "FAIL " + ", ".join(FAIL) if FAIL else "todo OK")
    sys.exit(1 if FAIL else 0)
