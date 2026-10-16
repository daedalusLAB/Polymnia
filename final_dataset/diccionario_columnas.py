#!/usr/bin/env python3
"""Column-by-column dictionary of the two final tables, generated from the code.

Nothing this script writes is invented here: every formula and every unit comes from the
constants that live next to the maths (`gesture_features.py`: FEATURE_UNITS,
EXTENDED_FEATURE_UNITS, SEMANTICS) or from the function that computes the column
(`_sequence_summary`, `load_frame_metadata`). The only hand-written line is the "what it
measures" one, and every column carries a `fuente` saying where the rest came from. If a
column shows up unresolved, the script fails instead of staying quiet: that is how a
README is kept from lying once the columns change.

Besides the description, it measures in the delivered parquet how many clips carry a real
value in each column: a statistic with 3 of 29 clips is not the same as one with 29, and
that is the first thing a student has to be able to see.

With `--readme` it also writes the README of the delivery: the same document with a cover
of measured numbers (rows, columns, features version, padding, date) and not a single
figure inherited from another batch. That file is the one `final_dataset.py` publishes as
`DATASET.md` inside `final/`, and that is why it is generated **before** publishing:
`pipeline_lote.sh` takes a measurement pass into `.medicion`, draws the dictionary and the
README from there, publishes `final/` with the README inside, and compares the parquet of
both passes so the document cannot describe anything but what was delivered.

    python diccionario_columnas.py --final lote3/final \
        --csv lote3/informes/DICCIONARIO_COLUMNAS.csv \
        --md lote3/DICCIONARIO_COLUMNAS.md \
        --readme lote3/informes/README_entrega.md
"""

import argparse
import json
import math
import os
import re
import sys

import pandas as pd

TABLA_MANO = {"right": "right hand", "left": "left hand"}

# What each measure measures, on one line. The formula and the unit go underneath, taken
# from the code; all that is needed here is the meaning, which the code does not tell.
MIDE_V1 = {
    "n_observed": "frames of the span where the source did give that keypoint, even if it was later rejected on confidence",
    "n_interpolated": "frames filled by linear interpolation between two valid endpoints (gaps of at most max_gap_frames)",
    "n_valid": "frames with a usable measurement: the real denominator for everything else on this hand",
    "frac_observed": "n_observed over frames_grid",
    "frac_interpolated": "n_interpolated over frames_grid; if it is high, half of the trajectory is reconstructed",
    "frac_valid": "n_valid over frames_grid",
    "mean_speed": "mean wrist speed over the valid frames",
    "max_speed": "the highest instantaneous wrist speed",
    "mean_accel_magnitude": "mean acceleration magnitude",
    "max_accel_magnitude": "maximum acceleration magnitude",
    "path_length": "path length: sum of the displacements between consecutive frames of the same contiguous run",
    "nx_range": "how horizontal the sweeps are: max - min of the smoothed x coordinate",
    "ny_range": "how vertical the sweeps are: max - min of the smoothed y coordinate",
    "mean_neck_distance": "how far from the neck the wrist moves, on average",
    "angle_mean": "mean direction of the neck->wrist vector (circular mean)",
    "angle_std": "spread of that direction (a concentration measure, not a fitted von Mises)",
    "time_to_peak_speed": "seconds from the first frame with a clock to the fastest frame",
    "early_speed_mean": "mean speed in the first 10% of the clock span",
    "late_speed_mean": "mean speed in the last 10% of the clock span",
}

MIDE_BILATERAL_V1 = {
    "n_both_hands": "frames with BOTH wrists valid: without this, a bilateral statistic says nothing",
    "frac_both_hands": "n_both_hands over frames_grid",
    "mean_pair_separation": "mean separation between the two wrists",
    "max_pair_separation": "the largest separation between the two wrists in a single frame",
    "mean_mirror_error_x": "mean of |x_R + x_L|: how far the pair drifts from the vertical axis of the body",
    "mean_mirror_error_y": "mean of |y_R - y_L|: how much the two hands sit at different heights",
    "time_to_peak_pair_separation": "seconds to the frame of maximum separation between the hands",
}

# Families of the extended block: one line per family, the formula comes from the code.
# The keys are the EXACT family names that come out of the keys of
# EXTENDED_FEATURE_UNITS; if the code renames a family, the column falls into the
# "undocumented" guard instead of keeping an old description.
MIDE_FAMILIA = {
    "distribution aggregates": "the full shape of that measure's distribution in the clip, not just its mean",
    "extent": "how far the wrist reaches on each axis: extremes and spread of the smoothed coordinate",
    "reversals": "how many times the movement changes direction on that axis (waving, not displacement)",
    "reversals_per_s": "direction changes per second, to compare clips of different length",
    "hull_area": "area of the convex hull of the trajectory: the area the hand sweeps",
    "ellipse_area": "area of the 2 standard deviation ellipse of the point cloud: the 'usual' area",
    "bbox_area": "area of the axis-aligned rectangle that encloses the whole trajectory",
    "scale-free intra-clip ratios": "ratios between two statistics of the SAME clip: shape without depending on other clips",
    "shape of the speed distribution": "skew and peakedness of the speed distribution",
    "frac_speed_gt_0p5 / 1p0 / 2p0": "fraction of the time with the hand above a given absolute speed",
    "speed_sq_mean, speed_sq_sum": "mean and sum of speed squared (what the reference code called kinetic energy)",
    "n_speed": "frames with finite speed: the denominator behind every speed statistic on this hand",
    "longest_moving_s": "the longest continuous run moving that hand above the threshold",
    "cos_phase_*": "which way the two hands go at once: cos of the angle between their velocities (+1 same direction, -1 opposite)",
    "frac_cos_gt_0p8, frac_cos_lt_m0p2": "how much of the time is genuinely in-phase or anti-phase",
    "n_cos_phase": "frames behind the phase figures: if it is small, the phase of that clip is not a result",
    "coactivity_frac_0p5, coactivity_frac_1p0": "fraction of the time with BOTH hands above the speed threshold",
    "coactivity_bout_*": "how many bouts there are and how long they last: the alternation rhythm between hands",
    "mirror_error_x_std, mirror_error_y_std": "how variable the symmetry is inside the clip, not just its mean",
    "abs_y_pair_sum_mean": "magnitude of the joint height of the pair of hands (this is NOT symmetry)",
    "pair_elevation_mean": "mean height of the pair of hands relative to the neck, signed (positive = above)",
    "bbox_area_wrists": "area of the rectangle that has the two wrists as opposite corners",
    "jerk": "how abruptly the acceleration changes: the third derivative of the smoothed coordinate",
}

MIDE_PROCEDENCIA = {
    "id": "clip identifier: the processed name that lines up with the window of the TSV",
    "people_id": "index of the OpenPose candidate inside the clip; in this delivery always 1",
    "frames_source": "rows the source gave for this clip",
    "frames_grid": "frame span between the first and the last observed frame (min..max), inclusive",
    "frames_missing_grid": "gaps inside that range the source did not give, marked as inferred",
    "n_timed_frames": "frames with a real clock stamp",
    "n_timestamp_inferred": "frames whose clock was borrowed by short interpolation between two timed frames",
    "n_missing_timestamps": "frames_grid - n_timed_frames: frames without a clock, where anything temporal is NULL",
    "t_start": "first known clock of the sequence (seconds of the source frame_timestamp)",
    "t_end": "last known clock of the sequence",
    "duration_s": "t_end - t_start; this is the MEASURED duration of the interval, not of the raw video",
}

MIDE_ANNOTATION = {
    "indoor_outdoor": "visual model label: indoor or outdoor",
    "show_type": "visual model label: genre of the programme (news anchor, weather report, interview, monologue)",
    "hands_free": "visual model label: whether the person has their hands free to gesture",
    "sitting_standing": "visual model label: sitting or standing",
    "screen_interaction": "visual model label: whether they interact with a screen",
    "sex": "visual model label: sex assigned to the person by the model",
    "age": "age estimated by the visual model, in years",
    "status": "status of the visual model call for this clip",
    "error": "annotation error text; empty when the annotation came out fine",
}

MIDE_META = {
    "newscape_clip_id": "the clip name the pipeline produced (raw -> ASD -> OpenPose), verbatim",
    "tab_context": "field 7 of the input TSV: the context asked for by whoever made the selection",
    "tab_start_raw": "start of the window as it came in the TSV, undecoded",
    "tab_end_raw": "end of the window as it came in the TSV, undecoded",
    "tab_padding_s": "padding added to the TSV window to match it with the processed clip",
    "processed_padding_s": "padding the processed clip name already carried",
    "id_source": "how this row was paired with the TSV: exact id or window centre",
    "datetime_utc": "date and time the programme aired, in UTC",
    "country_code": "country code of the programme",
    "network_code": "network code of the programme",
    "exp_search": "searchterm from the TSV: the search this clip came out of",
}

MIDE_FRAME = {
    "formant.1_frequency": "first formant of the voice in that frame (Hz, Praat)",
    "formant.2_frequency": "second formant of the voice in that frame (Hz, Praat)",
    "formant.3_frequency": "third formant of the voice in that frame (Hz, Praat)",
    "formant.4_frequency": "fourth formant of the voice in that frame (Hz, Praat)",
    "formant.5_frequency": "fifth formant of the voice in that frame (Hz, Praat)",
    "frame": "frame number inside the clip",
    "frame_timestamp": "seconds of the source frame_timestamp; NULL where there is no clock",
    "source_observed": "the source gave this frame (false = filled gap)",
    "frame_inferred": "this frame was not in the source and was reconstructed",
    "clock_known": "this frame has a real clock",
    "timestamp_inferred": "this frame's clock was borrowed by short interpolation",
    "both_hands_valid": "both wrists are valid in this frame",
    "pair_separation": "distance between the two wrists",
    "pair_angle": "angle of the left wrist -> right wrist vector",
    "centroid_x": "x coordinate of the midpoint between the two wrists",
    "centroid_y": "y coordinate of the midpoint between the two wrists",
    "mirror_error_x": "|x_R + x_L| about the vertical axis of the body, frame by frame",
    "mirror_error_y": "|y_R - y_L|: height difference between the two hands",
    "pitch": "fundamental frequency of the voice in that frame (Hz, Praat)",
    "voice_present": "whether Praat was following a pitch track in that frame: True measured, False no track, "
                     "NULL if the clip has no pitch track. It is the door to open before any voice "
                     "statistic (what right_n_speed is for the wrists)",
    "intensity": "voice intensity in that frame (dB, Praat)",
    "harmonicity": "harmonics-to-noise ratio of the voice in that frame (Praat)",
    "vad": "voice activity: there is speech in that frame",
    "words": "word active in that frame according to WhisperX",
    "openpose_json": "name of the OpenPose JSON this row came out of (traceability)",
    "datetime_utc": "date and time the programme aired, in UTC",
    "country_code": "country code of the programme",
    "network_code": "network code of the programme",
    "exp_search": "searchterm from the TSV: the search this clip came out of",
    "id": "clip identifier, the same as in final_videos",
    "people_id": "index of the OpenPose candidate; in this delivery always 1",
}

MIDE_FRAME_MANO = {
    "nx": "raw x coordinate of the keypoint, normalised by the neck",
    "ny": "raw y coordinate of the keypoint, normalised by the neck",
    "nx_smooth": "smoothed x coordinate (Savitzky-Golay per contiguous run)",
    "ny_smooth": "smoothed y coordinate",
    "confidence": "the confidence OpenPose assigns to that keypoint in this frame",
    "observed": "the source gave this keypoint in this frame",
    "interpolated": "the coordinate comes from interpolating a short gap",
    "valid": "the measurement is usable in this frame: keypoint present and confidence >= minimum, or interpolated",
    "above_neck": "the wrist is above the height of the neck (smoothed ny > 0)",
    "neck_distance": "distance from the wrist to the neck",
    "neck_angle": "angle of the neck -> wrist vector",
    "velocity_x": "wrist velocity in x",
    "velocity_y": "wrist velocity in y",
    "speed": "magnitude of the wrist velocity",
    "accel_x": "wrist acceleration in x",
    "accel_y": "wrist acceleration in y",
    "accel_magnitude": "magnitude of the acceleration",
    "step_distance": "displacement to the previous frame of the same contiguous run (needs no clock)",
    "jerk_x": "change of acceleration in x",
    "jerk_y": "change of acceleration in y",
    "jerk_magnitude": "magnitude of the jerk: how abrupt the change of acceleration is",
}

STATS = ("min", "p05", "p25", "p50", "p75", "p95", "max", "std")

# Counters whose definition lives in the code INSIDE the note of another family instead of
# having a key of their own. Without this they would come out "undocumented" even though the
# code does explain them; they are linked to that note so that nobody writes a second version
# that can drift out of sync.
NOTA_DE = {"n_cos_phase": "cos_phase_"}
MIDE_CONTADORES = {"n_cos_phase": "frames behind the phase figures: if it is small, the phase of that clip is not a result"}
MIDE_STAT = {"min": "minimum", "max": "maximum", "std": "sample standard deviation (ddof=1)",
             "p05": "5th percentile", "p25": "25th percentile", "p50": "median",
             "p75": "75th percentile", "p95": "95th percentile"}
NOMBRE_MEDIDA_LARGA = {"accel_magnitude": "acceleration", "jerk_magnitude": "jerk",
                       "neck_distance": "distance to the neck", "speed": "speed"}

# Formulas and units that do NOT live in the code's unit constants. They are transcribed from
# the function that computes them (every row names which one), so their truth is checked by
# reading that function. Without this table, ~95 columns would come out of the dictionary with
# an empty formula: the student would see "what it measures" but not "how".
FORMULA_PROPIA = {
    # --- per frame and hand: gesture_features.py: _hand_rows
    "confidence": ("keypoint confidence",
                   "confidence score OpenPose gives for the keypoint of that wrist, in [0,1]; it is not a calibrated probability",
                   "gesture_features.py: _hand_rows"),
    "valid": ("keypoint confidence",
              "usable = finite coordinates and conf > 0 and conf >= min_keypoint_confidence (0.2 by default); valid = usable | interpolated. If it is false, everything else for that hand in this frame is NULL",
              "gesture_features.py: _hand_rows"),
    "observed": ("keypoint confidence",
                 "usable and the source gave this frame (not filled). Differs from valid exactly on the interpolated frames",
                 "gesture_features.py: _hand_rows"),
    "interpolated": ("keypoint confidence",
                     "the coordinate comes from _interpolate_gaps: a short gap between two valid endpoints, never at the edges",
                     "gesture_features.py: _hand_rows"),
    "above_neck": ("per-frame geometry",
                   "ny_smooth > 0: the wrist is above the neck. NULL where there is no valid coordinate",
                   "gesture_features.py: _hand_rows"),
    # --- clock and span: gesture_features.py: _sequence_grid
    "frame": ("identification and provenance", "frame number inside the clip, an integer from the OpenPose JSON",
              "gesture_features.py: _sequence_grid"),
    "frame_timestamp": ("time",
                        "seconds of the source frame_timestamp; NULL where there is no clock. Only interpolated in short internal gaps (timestamp_inferred); the edges are never extrapolated",
                        "gesture_features.py: _sequence_grid"),
    "source_observed": ("identification and provenance", "the source gave this frame inside the observed span",
                        "gesture_features.py: _sequence_grid"),
    "frame_inferred": ("identification and provenance", "it was not in the source: internal gap inside the observed range",
                       "gesture_features.py: _sequence_grid"),
    "clock_known": ("time", "this frame has a real clock (finite frame_timestamp)",
                    "gesture_features.py: _sequence_grid"),
    "timestamp_inferred": ("time", "the clock is borrowed for an internal gap of at most max_gap_frames frames",
                           "gesture_features.py: _sequence_grid"),
    "both_hands_valid": ("counts and fractions", "right_valid and left_valid in the same frame: the door of everything bilateral",
                         "gesture_features.py: _sequence_frame_rows"),
    # --- v1 aggregates: gesture_features.py: _hand_summary
    "n_observed": ("counts and fractions", "sum of the observed flag of that hand over the observed span (unit: frames)",
                   "gesture_features.py: _hand_summary"),
    "n_interpolated": ("counts and fractions", "sum of the interpolated flag of that hand (frames)",
                       "gesture_features.py: _hand_summary"),
    "n_valid": ("counts and fractions", "sum of the valid flag of that hand: the real denominator of its measurements",
                "gesture_features.py: _hand_summary"),
    "frac_observed": ("counts and fractions", "n_observed / frames_grid, dimensionless",
                      "gesture_features.py: _hand_summary"),
    "frac_interpolated": ("counts and fractions", "n_interpolated / frames_grid, dimensionless",
                          "gesture_features.py: _hand_summary"),
    "frac_valid": ("counts and fractions", "n_valid / frames_grid, dimensionless",
                   "gesture_features.py: _hand_summary"),
    "mean_speed": ("velocity", "np.mean over the finite speed values of the clip, in neck units per second",
                   "gesture_features.py: _hand_summary"),
    "max_speed": ("velocity", "np.max of the finite speed values: a single frame defines it, it is not robust to shot cuts",
                  "gesture_features.py: _hand_summary"),
    "mean_accel_magnitude": ("acceleration", "np.mean of finite accel_magnitude, in neck units per second squared",
                             "gesture_features.py: _hand_summary"),
    "max_accel_magnitude": ("acceleration", "np.max of finite accel_magnitude", "gesture_features.py: _hand_summary"),
    "mean_neck_distance": ("distances", "np.mean of finite neck_distance, in neck units",
                           "gesture_features.py: _hand_summary"),
    "early_speed_mean": ("velocity", "mean of speed in the first 10% of the [t_start, t_end] span of known clock",
                         "gesture_features.py: _edge_window_mean"),
    "late_speed_mean": ("velocity", "mean of speed in the last 10% of the [t_start, t_end] span of known clock",
                        "gesture_features.py: _edge_window_mean"),
    # --- bilateral v1: gesture_features.py: _sequence_summary
    "n_both_hands": ("counts and fractions", "sum of both_hands_valid (frames)",
                     "gesture_features.py: _sequence_summary"),
    "frac_both_hands": ("counts and fractions", "n_both_hands / frames_grid, dimensionless",
                        "gesture_features.py: _sequence_summary"),
    "mean_pair_separation": ("distances", "np.mean of pair_separation = hipot(nx_R - nx_L, ny_R - ny_L) of the smoothed coordinates, in neck units",
                             "gesture_features.py: _sequence_summary"),
    "max_pair_separation": ("distances", "np.max of pair_separation", "gesture_features.py: _sequence_summary"),
    "mean_mirror_error_x": ("distances",
                            "np.mean of |nx_R + nx_L|: with X centred on the neck, a symmetric pair gives ~0; it grows when both hands go to the same side",
                            "gesture_features.py: _sequence_summary"),
    "mean_mirror_error_y": ("distances", "np.mean of |ny_R - ny_L|: height difference between the two hands",
                            "gesture_features.py: _sequence_summary"),
    # --- provenance of the sequence: gesture_features.py: _sequence_summary
    "frames_source": ("identification and provenance", "how many frames of the span brought a row from the source (sum of source_observed)",
                      "gesture_features.py: _sequence_summary"),
    "frames_grid": ("identification and provenance",
                    "size of the span min(frame)..max(frame) inclusive: the denominator of every frac_*",
                    "gesture_features.py: _sequence_grid"),
    "frames_missing_grid": ("identification and provenance", "sum of frame_inferred: internal gaps with no row from the source",
                            "gesture_features.py: _sequence_summary"),
    "n_timed_frames": ("time", "sum of clock_known: frames with a real clock", "gesture_features.py: _sequence_summary"),
    "n_timestamp_inferred": ("time", "sum of timestamp_inferred: frames with a borrowed clock",
                             "gesture_features.py: _sequence_summary"),
    "n_missing_timestamps": ("time", "frames_grid - n_timed_frames: frames without a clock",
                             "gesture_features.py: _sequence_summary"),
    # --- voice: produced by the voice stage (parole), NOT by the wrist maths. The bounds of
    # the analysis are fixed and live in extract_prosody.praat: pitch 60-600 Hz, formants up
    # to 5500 Hz, floor of Harmonicity (cc) at -200 dB. Any value OUTSIDE those bounds was not
    # produced by Praat: it betrays the old chain (one spline over the whole track plus the
    # pitch<10 -> 0 sentinel). The regime of each delivery is MEASURED, not assumed.
    "pitch": ("voice (Praat)",
              "fundamental frequency in Hz. Analysis bound: 60-600 Hz (Praat To Pitch, step 0.001 s). Praat marks silence with the 'mute' candidate (frequency = 0), which in the current chain comes out as NULL and not as 0 Hz. A value outside 60-600 could not have been produced by Praat",
              "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "intensity": ("voice (Praat)",
                  "intensity in dB (Praat To Intensity: min pitch 75 Hz, step 0.001 s). Praat keeps measuring it during silence, so this column has almost no gaps: do not use it as a voice indicator, that is what voice_present is for",
                  "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "harmonicity": ("voice (Praat)",
                    "harmonics-to-noise ratio in dB (Praat To Harmonicity (cc): step 0.01 s, from 75 Hz). Negative values are normal (more noise than harmonics) and the floor of the method is -200 dB: in silence it is -200. What does not exist is a value BELOW -200",
                    "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "formant.1_frequency": ("voice (Praat)",
                            "first formant in Hz (Praat To Formant (burg): step 0.001 s, 5 formants with a 5500 Hz ceiling). Any frequency outside 0-5500 Hz is impossible: that is not a rare value, it is a different processing chain",
                            "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "formant.2_frequency": ("voice (Praat)", "second formant in Hz (same extraction and same bounds as formant.1_frequency)",
                            "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "formant.3_frequency": ("voice (Praat)", "third formant in Hz (same extraction and same bounds as formant.1_frequency)",
                            "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "formant.4_frequency": ("voice (Praat)", "fourth formant in Hz (same extraction and same bounds as formant.1_frequency)",
                            "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "formant.5_frequency": ("voice (Praat)", "fifth formant in Hz (same extraction and same bounds as formant.1_frequency)",
                            "create_datasets/parole/scripts/prosody_align.R: align_track"),
    "voice_present": ("voice (Praat)",
                      "TRUE only where Praat was following a real pitch track in that frame, FALSE where there was none, NULL if the clip has no pitch track. It is the door to open before any voice statistic, and the equivalent of right_n_speed for the wrists: without it, NULL, silence and an interpolated value all dress the same",
                      "create_datasets/parole/scripts/process_prosody.R: voice_present = !is.na(pitch) aligned per run"),
    "vad": ("voice (Praat)",
            "voice activity per frame, from the Silero VAD segments: it is the only voice column that is a detection and not a sampling of a track. Published as a boolean (measured: BOOLEAN in the three bases of the corpus, with a value in ~99% of the frames). If some lot ever arrived whole, the constructor casts it to boolean before publishing",
            "create_datasets/parole/scripts/silero_vad_runner.py via parole.sh"),
    "words": ("voice (Praat)",
              "word active in that frame according to WhisperX with the timings aligned to the video. NULL where there is no transcribed word: here NULL really does mean silence",
              "apate.py (whisperx + framer)"),
    # --- visual annotations: theia.py writes these, they are not measured over the video
    "indoor_outdoor": ("visual annotation",
                       "label from the visual model over frames of the RAW video, in {indoor, outdoor}. These are two calls per video: this is the scene one. NULL if the call failed",
                       "theia.py: SCENE_FIELDS + ENUMS"),
    "show_type": ("visual annotation",
                  "programme genre label in {news anchor, monologue, weather report, interview}; the vocabulary is closed, the model cannot invent another class",
                  "theia.py: SCENE_FIELDS + ENUMS"),
    "hands_free": ("visual annotation",
                   "BOOLEAN: the model understands that the person has their hands free to gesture. It comes from the speaker call (MASKED + RAW frames) and is normalised from true/yes/si/1",
                   "theia.py: SPEAKER_FIELDS + TRUEISH/FALSEISH"),
    "sitting_standing": ("visual annotation",
                         "label in {sitting, standing} from the speaker call. Useful for stratifying: a seated presenter gestures differently from a standing one",
                         "theia.py: SPEAKER_FIELDS + ENUMS"),
    "screen_interaction": ("visual annotation",
                           "BOOLEAN: interacts with a screen, the same normalisation as hands_free. In this lot only 3 of 29 clips say yes",
                           "theia.py: SPEAKER_FIELDS + TRUEISH/FALSEISH"),
    "sex": ("visual annotation",
            "label in {male, female} from the speaker call: an assignment by the model, not a registered variable",
            "theia.py: SPEAKER_FIELDS + ENUMS"),
    "age": ("visual annotation",
            "estimated age in years, an integer validated in [0, 120]. Careful: the model gives round numbers; useful for stratifying, not as a real age",
            "theia.py: SPEAKER_FIELDS + AGE_RANGE"),
    # --- identification, window and metadata: final_dataset.py
    "id": ("identification and provenance",
           "clip key, rebuilt from the processed name and matched with the TSV window; it is the only join between the two tables",
           "final_dataset.py: parse_clip_id / match_tab_to_clips"),
    "people_id": ("identification and provenance",
                  "index of the OpenPose candidate inside the clip (person_index of the JSON, no tracking across frames)",
                  "final_dataset.py: load_frame_metadata"),
    "newscape_clip_id": ("identification and provenance", "the exact clip name the pipeline produced",
                         "final_dataset.py: parse_clip_id"),
    "tab_context": ("identification and provenance", "field 7 of the input TSV, literal",
                    "final_dataset.py: read_tab"),
    "tab_start_raw": ("identification and provenance",
                      "window start as it came in the TSV. Remember the divisor: msec is divided by 1000 if the programme starts with 2017, by 100 otherwise",
                      "final_dataset.py: fractional_divisor / read_tab"),
    "tab_end_raw": ("identification and provenance", "window end as it came in the TSV, with the same divisor",
                    "final_dataset.py: fractional_divisor / read_tab"),
    "tab_padding_s": ("identification and provenance",
                      "padding added to the TSV window to match it. Not a universal constant: it depends on the lot (deliveries with already processed ids use 2.5 s)",
                      "final_dataset.py: clip_window"),
    "processed_padding_s": ("identification and provenance", "padding the processed clip name already carried",
                            "final_dataset.py: parse_clip_id"),
    "id_source": ("identification and provenance",
                  "how the row was paired: exact id or window centre. Best not to mix the two in one analysis",
                  "final_dataset.py: match_tab_to_clips"),
    "status": ("visual annotation", "status of the visual model call for this clip",
               "final_dataset.py: load_annotations"),
    "error": ("visual annotation",
              "error text; EMPTY (not NULL) when the annotation came out fine. Counting the OKs means looking at the empty string",
              "final_dataset.py: load_annotations"),
    "openpose_json": ("identification and provenance",
                      "name of the OpenPose JSON the row came out of. program_name and time_range are that same name, not programme metadata",
                      "final_dataset.py: load_frame_metadata"),
    "datetime_utc": ("identification and provenance", "date and time the programme aired, in UTC",
                     "final_dataset.py: load_frame_metadata"),
    "country_code": ("identification and provenance", "country code of the programme, as NewsScape gives it",
                     "final_dataset.py: load_frame_metadata"),
    "network_code": ("identification and provenance", "network code of the programme", "final_dataset.py: load_frame_metadata"),
    "exp_search": ("identification and provenance", "searchterm from the TSV: the search this clip came out of",
                   "final_dataset.py: read_tab"),
}


def tabla_desde_codigo(gf):
    """The definitions read from gf's constants, at three levels of precision.

    The keys of FEATURE_UNITS and EXTENDED_FEATURE_UNITS carry the list of measures they
    apply to in parentheses; that is where the real formula is. They are split into
    `exactos` (the full name of a measure), `prefijos` (`cos_phase_*`) and `sufijos`
    (`*_p95`). Resolution tries exact first, then the longest prefix, then a suffix: that
    way the specific definition of `mirror_error_x_std` is not covered by the general
    family `mirror_error_*`.
    """
    exactos, prefijos, sufijos = {}, [], []

    def apunta(nombre, familia, texto):
        nombre = nombre.strip()
        if not nombre:
            return
        if nombre.endswith("_*"):
            prefijos.append((nombre[:-1], familia, texto))
        elif nombre.startswith("*_"):
            sufijos.append((nombre[1:], familia, texto))
        elif "/" in nombre:                          # "frac_speed_gt_0p5 / 1p0 / 2p0"
            base = nombre.rsplit("_", 1)[0]
            for parte in nombre.split("/"):
                exactos.setdefault(f"{base}_{parte.strip().rsplit('_', 1)[-1]}", (familia, texto))
        else:
            exactos.setdefault(nombre, (familia, texto))

    for dicc in (gf.EXTENDED_FEATURE_UNITS, gf.FEATURE_UNITS):
        for clave, texto in dicc.items():
            encontrado = re.match(r"^(.*?)\s*\((.*)\)$", clave.strip(), re.S)
            contenido = encontrado.group(2) if encontrado else clave
            familia = (encontrado.group(1) if encontrado else clave).strip()
            for trozo in contenido.split(","):
                apunta(trozo, familia, texto)
    prefijos.sort(key=lambda t: len(t[0]), reverse=True)
    for nombre, prefijo in NOTA_DE.items():
        definicion = next(((f, t) for p, f, t in prefijos if p == prefijo), None)
        if definicion is not None:
            exactos.setdefault(nombre, definicion)
    return exactos, prefijos, sufijos


def _limpiar(texto):
    return re.sub(r"\s+", " ", texto).strip()


def resolver(tabla, columna, gf, codigo):
    """One row of the dictionary, or None if there is no honest way to document it."""
    exactos, prefijos, sufijos = codigo
    ambito, base = "clip" if tabla == "final_videos" else "frame", columna
    for prefijo, nombre in TABLA_MANO.items():
        if columna.startswith(prefijo + "_"):
            ambito, base = nombre, columna[len(prefijo) + 1:]
            break
    familia = formula = mides = None

    definicion = exactos.get(base)
    if definicion is None:
        definicion = next(((f, t) for p, f, t in prefijos if base.startswith(p)), None)
    if definicion is None:
        sufijo = next(((f, t) for s, f, t in sufijos if base.endswith(s)), None)
        if sufijo is not None:
            medida, estadistico = base.rsplit("_", 1)
            definicion = sufijo
            mides = (f"{MIDE_STAT[estadistico]} of {NOMBRE_MEDIDA_LARGA.get(medida, medida)} "
                     "in the clip, over the finite values of that measure")
    if definicion is not None:
        familia, formula = _limpiar(definicion[0]), _limpiar(definicion[1])
    elif base in FORMULA_PROPIA:
        familia, formula = _limpiar(FORMULA_PROPIA[base][0]), _limpiar(FORMULA_PROPIA[base][1])

    if mides is None:
        mides = (MIDE_V1.get(base) or MIDE_BILATERAL_V1.get(base) or MIDE_PROCEDENCIA.get(base)
                 or MIDE_ANNOTATION.get(base) or MIDE_META.get(base) or MIDE_FRAME.get(base)
                 or MIDE_FRAME_MANO.get(base) or MIDE_CONTADORES.get(base))
    if mides is None and familia:
        mides = MIDE_FAMILIA.get(_limpiar(familia))
    if mides is None:
        return None

    fuente = ("gesture_features.py: FEATURE_UNITS/EXTENDED_FEATURE_UNITS" if formula else
              "gesture_features.py: _hand_summary/_sequence_summary")
    # Source precedence: the specific citation wins over the generic one. Without this
    # order, `frames_grid` lost _sequence_grid and ended up in _sequence_summary.
    if base in FORMULA_PROPIA:
        fuente = FORMULA_PROPIA[base][2]
    elif base in MIDE_META or base in MIDE_ANNOTATION:
        fuente = "final_dataset.py: load_annotations / CLIP_META"
    elif base in MIDE_PROCEDENCIA:
        fuente = "gesture_features.py: _sequence_summary"
    elif tabla == "final_frames" and (base in MIDE_FRAME_MANO or base in MIDE_FRAME):
        fuente = "gesture_features.py: FRAME_COLUMNS"
    return {"tabla": tabla, "columna": columna, "ambito": ambito,
            "familia": familia or "identification and provenance",
            "que_mide": mides, "formula_unidades": formula or "", "fuente": fuente}


def cubrir(df, fila):
    """How much real data that column carries, in THIS lot."""
    serie = df[fila["columna"]]
    fila["no_nulos"] = int(serie.notna().sum())
    fila["pct_no_nulos"] = round(100.0 * fila["no_nulos"] / len(df), 1)
    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
        limpios = serie.dropna()
        if len(limpios):
            fila["minimo"] = round(float(limpios.min()), 4)
            fila["mediana"] = round(float(limpios.median()), 4)
            fila["maximo"] = round(float(limpios.max()), 4)
    fila["tipo"] = str(serie.dtype)
    return fila


def generar(ruta_final, polymnia_root):
    sys.path.insert(0, polymnia_root)
    try:
        import gesture_features as gf
    except ImportError as exc:
        raise SystemExit(f"cannot import gesture_features from {polymnia_root}: {exc}")
    codigo = tabla_desde_codigo(gf)

    manifiesto = {}
    ruta_manifest = os.path.join(ruta_final, "manifest.json")
    if os.path.isfile(ruta_manifest):
        with open(ruta_manifest, encoding="utf-8") as lector:
            manifiesto = json.load(lector)

    filas, pendientes = [], []
    for tabla in ("final_videos", "final_frames"):
        ruta = os.path.join(ruta_final, f"{tabla}.parquet")
        if not os.path.isfile(ruta):
            continue
        df = pd.read_parquet(ruta)
        for columna in df.columns:
            fila = resolver(tabla, columna, gf, codigo)
            if fila is None:
                pendientes.append(f"{tabla}.{columna}")
                continue
            filas.append(cubrir(df, {**fila, "filas_tabla": len(df)}))
    if pendientes:
        raise SystemExit("undocumented columns (add them to the MIDE tables, do not "
                         "invent them here): " + ", ".join(pendientes))

    informe = {
        "feature_version": manifiesto.get("features", {}).get("version"),
        "extendidas": manifiesto.get("features", {}).get("extended"),
        # Data for the README cover. From the manifest of the measurement pass, which is the
        # same manifest that gets published: the pipeline compares the parquets before trusting.
        "fecha": manifiesto.get("generated_at"),
        "padding_s": (manifiesto.get("clip_id_mapping") or {}).get("requested_padding_s"),
        "searchterm": (manifiesto.get("clip_id_mapping") or {}).get("searchterm"),
        "umbral_movimiento": manifiesto.get("movement_speed_units_per_s", gf.MOVEMENT_SPEED),
        "suelo_muestras": gf.MIN_STAT_SAMPLES,
        "reglas_admision": manifiesto.get("admission_rules", {}),
        "semantica_diseno": {**gf.SEMANTICS, **gf.EXTENDED_SEMANTICS},
    }
    return pd.DataFrame(filas), informe


ORDEN_FAMILIAS = ("identification and provenance", "coordinates", "distances", "velocity",
                  "acceleration", "angles", "time", "counts and fractions", "NULL",
                  "distribution aggregates", "extent", "reversals", "hull_area", "ellipse_area",
                  "bbox_area", "scale-free intra-clip ratios", "shape of the speed distribution",
                  "speed_sq_mean, speed_sq_sum", "cos_phase_*", "coactivity_bout_*",
                  "jerk")


def _familia_de_orden(familia):
    for i, nombre in enumerate(ORDEN_FAMILIAS):
        if familia.startswith(nombre):
            return i
    return len(ORDEN_FAMILIAS)


def _portada(diccionario, informe):
    """The header that turns the document into the README of one concrete delivery.

    Only the measured numbers: rows and columns come from the parquet, version and padding from
    the manifest. There is not a single figure from another batch, which is how this whole
    problem was born.
    """
    lineas = [
        "> **README of the delivery.** It ships next to `final_videos.parquet` and `final_frames.parquet`: "
        "it is the documentation of the two tables in front of you, column by column.",
        "",
        "| data about this delivery | value |",
        "|---|---|",
    ]
    for tabla in ("final_videos", "final_frames"):
        sub = diccionario[diccionario["tabla"] == tabla]
        if sub.empty:
            continue
        granos = "clips" if int(sub["filas_tabla"].iloc[0]) < 1000 else "rows"
        if tabla == "final_frames":
            granos = "frames"
        lineas.append(f"| `{tabla}` | {int(sub['filas_tabla'].iloc[0])} {granos} x "
                      f"{len(sub)} columns |")
    extendidas = "included" if informe["extendidas"] else "not included"
    lineas.append(f"| features version | {informe['feature_version']} "
                  f"(extended block: {extendidas}) |")
    if informe.get("padding_s") is not None:
        extra = f", searchterm `{informe['searchterm']}`" if informe.get("searchterm") else ""
        lineas.append(f"| window padding | {informe['padding_s']} s{extra} |")
    if informe.get("fecha"):
        lineas.append(f"| generated | {informe['fecha']} |")
    lineas += [
        "",
        "This document **is generated**, not written: it comes out of the manifest, of the "
        "constants of `gesture_features.py` and of the parquet itself. Every coverage figure here "
        "belongs to THIS delivery. If you want to change the text, you change the generator and "
        "regenerate. The date above is that of the manifest this coverage was computed with.",
        "",
    ]
    return lineas


def muestra_de_ejemplos(ruta_final):
    """What the document claims about the lot, measured over the delivered parquet."""
    clips = frames = 0
    sustento, clip_max, max_speed = None, None, None
    ruta = os.path.join(ruta_final, "final_videos.parquet")
    if os.path.isfile(ruta):
        videos = pd.read_parquet(ruta)
        clips = len(videos)
        # The extended block is optional: a FEATURE_VERSION 2 delivery has no n_speed and no
        # per-hand max, and the README cannot crash or write "0 of 29" for reading columns that
        # do not exist. Whatever is there gets measured, and if it is not there, that is what it
        # says instead of inventing a number.
        if {"right_n_speed", "left_n_speed"} <= set(videos.columns):
            sustento = int(((videos.right_n_speed >= 30) & (videos.left_n_speed >= 30)).sum())
        if {"id", "right_max_speed"} <= set(videos.columns) and videos.right_max_speed.notna().any():
            rapido = videos.right_max_speed.idxmax()
            clip_max = str(videos.loc[rapido, "id"])
            max_speed = float(videos.loc[rapido, "right_max_speed"])
    ruta = os.path.join(ruta_final, "final_frames.parquet")
    if os.path.isfile(ruta):
        frames = int(pd.read_parquet(ruta, columns=["frame"]).shape[0])
    return {"clips": clips, "frames": frames, "sustento": sustento,
            "clip_max_speed": clip_max, "max_speed": max_speed,
            "voz": sanidad_de_voz(ruta_final)}


def escribir_markdown(diccionario, informe, ruta, muestra_ejemplos, portada=False):
    """The document that gets handed to the class."""
    lineas = ["# What every column of this dataset is", ""]
    if portada:
        lineas += _portada(diccionario, informe)
    clips = muestra_ejemplos["clips"]
    frames = muestra_ejemplos["frames"]
    lineas += [
        "Two tables, one row each. `final_videos` has **one row per clip** "
        f"({clips} clips here): each clip is one person (`people_id`) inside a television clip. "
        "`final_frames` has **one row per frame** of each clip "
        f"({frames} rows), and that is where the time series and the voice live.",
        "",
        "## The two units of the dataset",
        "",
        "- **Time**: seconds of the source `frame_timestamp`. Nothing assumes 30 fps.",
        "- **Space**: *neck units*. The origin is the neck (BODY_25 point 1) and the coordinates "
        "are normalised by the shoulder. **These are not metres and they cannot be converted**: "
        "there is no metric scale in the data. That is why `mean_speed = 3` means 'three times the "
        "neck-to-shoulder distance per second', and it is only comparable with other numbers of the "
        "same dataset.",
        "- Angles in radians. Areas in neck units squared.",
        "",
        "## NULL is not zero",
        "",
        "A NULL means **not measured**: the keypoint was absent, its confidence was too low, or "
        "there were too few samples behind it. A 0 means measured and zero. And a statistic with few "
        "samples is not a small result, it is noise with decimals.",
        "",
        f"The code applies two constants, declared in the manifest: a movement threshold of "
        f"**{informe['umbral_movimiento']} neck units/s** (below that a hand counts as still and its "
        f"direction changes are not counted) and a floor of **{informe['suelo_muestras']} "
        "samples** to publish a percentile or an aggregated mean.",
        "",
        "**Before using a statistic, look at its counter.** They live in the same row:",
        "",
        "| if you are about to use | look first at |",
        "|---|---|",
        "| any `*_speed`, `*_accel`, `jerk_*` | `right_n_speed` / `left_n_speed` |",
        "| any `cos_phase_*` | `n_cos_phase` |",
        "| any bilateral column | `n_both_hands`, `frac_both_hands` |",
        "| `*_p05`, `*_p95`, `*_std` | the `n` of that hand: percentiles over 5 frames are not percentiles |",
        "| any `frac_*` | `frames_grid` |",
        "",
        "## The 7 annotations are model labels, not measured reality",
        "",
        "`indoor_outdoor`, `show_type`, `hands_free`, `sitting_standing`, `screen_interaction`, "
        "`sex`, `age` were written by a visual model over frames of the clip. They are good for "
        "**stratifying** ('let us compare seated anchors with standing presenters'), not as truth. "
        "In particular `age` is round numbers: a model does not measure 44 years, it estimates them.",
        "",
        *(_seccion_voz(muestra_ejemplos["voz"]) if muestra_ejemplos else []),
        "## Columns",
        "",
        "`data` is how many of the clips (or frames) of this lot carry a value in that column. "
        "If it is low, the column exists but has no support here: do not use it to compare clips.",
        "",
    ]

    for tabla in ("final_videos", "final_frames"):
        sub = diccionario[diccionario["tabla"] == tabla].copy()
        if sub.empty:
            continue
        sub["orden"] = sub["familia"].map(_familia_de_orden)
        sub = sub.sort_values(["orden", "columna"], kind="stable")
        lineas += [f"### `{tabla}.parquet` — {int(sub['filas_tabla'].iloc[0])} rows x {len(sub)} columns", ""]
        actual = None
        for _, fila in sub.iterrows():
            if fila["familia"] != actual:
                actual = fila["familia"]
                lineas += ["", f"#### {actual}", "",
                           "| column | scope | what it measures | data | min | median | max |",
                           "|---|---|---|---|---|---|---|"]
            def numero(clave):
                valor = fila.get(clave)
                return "" if valor is None or (isinstance(valor, float) and math.isnan(valor)) else valor
            lineas.append(f"| `{fila['columna']}` | {fila['ambito']} | {fila['que_mide']} | "
                          f"{fila['pct_no_nulos']}% | {numero('minimo')} | {numero('mediana')} | "
                          f"{numero('maximo')} |")
        lineas += [""]

    familias = (diccionario[diccionario["formula_unidades"] != ""]
                .drop_duplicates("familia")[["familia", "formula_unidades", "fuente"]]
                .sort_values("familia", key=lambda s: s.map(_familia_de_orden), kind="stable"))
    lineas += ["## Definitions, exactly as they are in the code", "",
               "This is not a summary: it is the text that lives next to the maths "
               "(`gesture_features.py`). If this and the code disagree, the code wins.", ""]
    for _, fila in familias.iterrows():
        lineas += [f"**{fila['familia']}**  ", f"{fila['formula_unidades']}  ",
                   f"*{fila['fuente']}*", ""]

    lineas += ["## The design decisions you need to know about", ""]
    for titulo, texto in informe["semantica_diseno"].items():
        lineas += [f"- **{titulo}** — {texto}", ""]

    lineas += [
        "## Five things to do with this",
        "",
        "```python",
        "import pandas as pd",
        "videos = pd.read_parquet('final_videos.parquet')",
        "frames = pd.read_parquet('final_frames.parquet')",
        "```",
        "",
        "1. **Filter by support before filtering by value.** Keep the clips with "
        "`right_n_speed >= 30` and `left_n_speed >= 30` and watch how the distribution of "
        "`right_mean_speed` changes: that is the difference between gesticulation and detection noise. "
        + (f"In this lot that threshold leaves **{muestra_ejemplos['sustento']} of "
           f"{muestra_ejemplos['clips']} clips**: it barely filters anything, and knowing that is part "
           "of the exercise too (the old 29-clip delivery left 6). Raise the threshold and watch "
           "when the mean starts to move: that is the point where you stop averaging "
           "noise." if muestra_ejemplos["sustento"] is not None else
           "These counters come from the extended block: if this delivery does not carry it, support "
           "is checked with `right_n_valid` and `left_n_valid`."),
        "2. **Look for gesticulators without using the mean.** Sort by `right_frac_speed_gt_1p0` and "
        "compare it with `right_mean_speed`: an editing cut can inflate the mean, the frac no.",
        "3. **Hands taking turns vs hands moving together.** High `coactivity_frac_0p5` with low "
        "`cos_phase_mean` = alternation; both high = both hands in the same gesture. "
        "Only in clips with `n_cos_phase >= 10`.",
        "4. **What the voice says about the gesture (open the door first).** Cross `intensity` and `pitch`",
        "of `final_frames` with `right_speed` of the same row, **only in the frames with voice**",
        "(`pitch` between 60 and 600, or `vad` if it comes filled in): without filtering you are also",
        "averaging the silence frames, which carry a number because of the spline. Measured here: the",
        f"correlation is {muestra_ejemplos['voz']['corr_todo']} unfiltered and",
        f"{muestra_ejemplos['voz']['corr_voz']} when filtering by voice. 'Almost nothing in both",
        "cases' is a valid answer, and it is the one this lot gives. Voice **only exists",
        "in `final_frames`**: `final_videos` has not a single voice column.",
        "5. **Check that the number is real.** Take the fastest clip by `right_max_speed` "
        + (f"(here **{muestra_ejemplos['clip_max_speed']}: "
           f"{muestra_ejemplos['max_speed']:.1f} neck units per second, and a single frame signs that**)"
           if muestra_ejemplos["max_speed"] is not None else "")
        + ", extract four frames of the video and look at them. A two-digit `max_speed` almost "
        "always is a shot cut or a zoom, not a swipe: speed is a quotient between two positions, and "
        "if the framing changes, the position changes on its own.",
        "",
        "---",
        "",
        f"Generated with `final_dataset/diccionario_columnas.py` (FEATURE_VERSION "
        f"{informe['feature_version']}, extended {informe['extendidas']}). Every formula comes from the "
        "constants of `gesture_features.py`; the data coverage is measured over this same parquet. Do "
        "not edit this file by hand: regenerate it.",
    ]
    with open(ruta, "w", encoding="utf-8") as escritor:
        escritor.write("\n".join(lineas) + "\n")


COTA_PITCH = (60.0, 600.0)      # Praat To Pitch: min 60, max 600 (extract_prosody.praat)
COTA_FORMANTE = 5500.0           # Praat To Formant (burg): 5 formants up to 5500 Hz
SUELO_HNR = -200.0               # floor To Harmonicity (cc) reports in silence
VOZ_COLUMNAS = ("pitch", "intensity", "harmonicity", "voice_present",
                "formant.1_frequency", "formant.2_frequency", "formant.3_frequency",
                "formant.4_frequency", "formant.5_frequency")


def _seccion_voz(voz):
    """The voice section of the README, in two versions chosen by MEASUREMENT.

    The key is not 'which one I like': these are the bounds of the Praat analysis, which are
    fixed in extract_prosody.praat (To Pitch 60-600 Hz; To Formant (burg) with a 5500 Hz
    ceiling; To Harmonicity (cc) with a -200 dB floor). No value outside those bounds could
    have been produced by Praat, so it betrays the old chain. And `voice_present` only exists
    since the fix. A lot regenerated with `pipeline_lote.sh --desde voz` changes section by itself.
    """
    if voz["regimen"] == "no voice":
        # Staying quiet is not honest: if there are no voice columns, it gets said. And no numbers
        # are shown, because a '0 of 0' in a table looks like a measurement of zero frames.
        return [
            "## The voice: the part that comes from another chain, and how to check it",
            "",
            "**This delivery does not carry voice columns**: there is no `pitch`, `intensity`, "
            "`harmonicity`, no formants and no `voice_present` in `final_frames` (the voice stage did "
            "not run, or left nothing publishable). No voice number is shown because there is nowhere "
            "to take it from: if a 0 shows up in a voice analysis of this lot, that 0 is yours.",
            "",
        ]
    cabecera = [
        "## The voice: the part that comes from another chain, and how to check it",
        "",
        "The voice columns are not made by the wrist maths: they are made by "
        "`create_datasets/parole` (Praat + WhisperX + Silero VAD). And for years they did not honour "
        "the NULL contract, for three reasons that are checked with Praat's own analysis bounds "
        f"(`To Pitch: 60, 600`; formants up to `{COTA_FORMANTE:.0f}` Hz; the floor of "
        f"Harmonicity `(cc)` at `{SUELO_HNR:.0f}` dB):",
        "",
        "- One single cubic spline over the WHOLE track, evaluated at every frame: with 4 valid "
        "points in the entire clip, even the silence received a number.",
        "- `splinefun` cannot switch extrapolation off (it does not accept the `extrapolation` "
        "argument), so outside the start and the end of the track it invented values: that is where "
        "an F5 of 29,144,321 Hz and an HNR of -246.7 dB in corpus deliveries came from.",
        "- The `pitch < 10 -> 0` sentinel painted the extrapolation 0. In Praat `frequency = 0` is "
        "the `mute` candidate ('there is no pitch'), so the 0 really DID mean silence; the problem is "
        "that published as a number it sneaks into the means: measured, the mean pitch dropped from "
        "182.5 Hz to 125.3 Hz because of those zeros.",
        "",
        "The current chain (`prosody_align.R`) splits each track into runs of consecutive samples and "
        "only interpolates **inside** a run; outside a run, NULL. And it publishes "
        "`voice_present`. This is what this document measures over the Parquet in front of you:",
        "",
        "| check in this lot | value |",
        "|---|---|",
        f"| voice chain regime | **{voz['regimen']}** |",
        f"| `voice_present` | {voz['voice_present']} |",
        f"| `pitch == 0` | {voz['pitch_ceros']} frames |",
        f"| `pitch` outside {COTA_PITCH[0]:.0f}-{COTA_PITCH[1]:.0f} Hz | {voz['pitch_fuera']} frames |",
        f"| `harmonicity` below {SUELO_HNR:.0f} dB | {voz['hnr_bajo_suelo']} frames |",
        f"| formants outside 0-{COTA_FORMANTE:.0f} Hz | {voz['formantes_fuera']} values |",
        f"| `intensity` range | {voz['int_rango']} |",
        f"| `intensity` vs `right_speed` correlation: unfiltered / filtered by voice ({voz['n_voz']} frames) | {voz['corr_todo']} / {voz['corr_voz']} |",
        "",
    ]
    if voz["regimen"] == "segments":
        resto = [
            "This lot comes out of the chain with `prosody_align.R`: there is no voice where Praat was "
            "not following a track, there are no values outside the bounds, and `voice_present` "
            "exists. Even so:",
            "",
            "1. **Open the door with `voice_present` before any voice statistic.** It is the exact "
            "equivalent of demanding `right_n_speed` for the wrists. The correlation in the table "
            "above uses this door.",
            "2. **`intensity` is not a voice detector**: Praat keeps measuring it during silence, so it "
            "has almost no gaps. To know whether there is voice there is `voice_present`.",
            "3. **`words` is transcriber text**, with its errors; here NULL really does mean silence.",
            "4. **Voice is not in `final_videos`.** If you want a voice statistic per clip, compute it "
            "yourself from `final_frames` and say how many frames support it, the way `right_n_speed` "
            "and `n_cos_phase` do.",
            "",
            "If a value outside the bounds still shows up, do not impute it: set those frames aside and "
            "record that you did. And if you compare this lot with an old one, remember that in the old "
            "one a third of the voice frames are sentinel zeros.",
            "",
        ]
    else:
        resto = [
            "**This lot comes out of the old chain** (one spline over the whole track + sentinel), and "
            "that shows in the numbers above: there are values the Praat analysis cannot produce, and "
            "there is no `voice_present` column, so the only door available is the pitch "
            f"band ({voz['puerta']}).",
            "",
            "1. **Do not treat a voice value as a measurement of that frame.** It is the curve of the "
            "clip evaluated there: it is good for the global shape of a segment, not for 'what "
            "happened in frame 12'.",
            f"2. **Filter before computing anything** (pitch between {COTA_PITCH[0]:.0f} and "
            f"{COTA_PITCH[1]:.0f} Hz). Unfiltered, some of the frames are zeros that do not mean "
            "'no voice': measured, the mean pitch goes from 125.3 Hz to 182.5 Hz once you filter.",
            "3. **`intensity` is the most usable part** of this side, but only with the door open.",
            "4. The negatives of `harmonicity` are normal (a negative HNR exists: it is noise); what "
            f"does not exist is anything below {SUELO_HNR:.0f} dB. And a negative formant frequency or "
            f"one greater than {COTA_FORMANTE:.0f} Hz is plain impossibility, not an outlier.",
            "",
            "The fix lives in `final_dataset/upstream_parole/` (with a patch to send upstream); when a "
            "lot is regenerated with `pipeline_lote.sh --desde voz`, the voice comes out with "
            "`voice_present` and inside the bounds, and this section changes on its own.",
            "",
        ]
    return cabecera + resto


def sanidad_de_voz(ruta_final):
    """Voice numbers measured over the delivered Parquet, not from memory.

    What decides the regime is NOT an opinion: these are the bounds of the analysis. With
    `To Pitch: 60, 600`, `To Formant (burg): ..., 5500` and `To Harmonicity (cc)` you cannot
    get a pitch of 0 Hz, an HNR of -1193 dB or a formant of 29 MHz. If they are there, this
    delivery comes from the old chain (one cubic spline over the whole track, evaluated where
    there was no measurement, plus the pitch<10 -> 0 sentinel). Measured in the old corpus:
    1408 frames with pitch == 0, an F5 of 29,144,321 Hz and an HNR of -246.7 dB.
    """
    vacio = {"regimen": "no voice", "voice_present": "(no voice column)", "pitch_ceros": 0,
             "pitch_fuera": 0, "hnr_bajo_suelo": 0, "formantes_fuera": 0, "int_rango": "(no column)",
             "corr_todo": "(no column)", "corr_voz": "(no column)", "n_voz": 0, "puerta": "n/a"}
    ruta = os.path.join(ruta_final, "final_frames.parquet")
    if not os.path.isfile(ruta):
        return vacio
    columnas = set(pd.read_parquet(ruta).columns)
    if "pitch" not in columnas:
        return vacio
    datos = pd.read_parquet(ruta, columns=[c for c in list(VOZ_COLUMNAS) + ["right_speed"] if c in columnas])
    total = len(datos)
    bajo_suelo = int((datos["harmonicity"] < SUELO_HNR).sum()) if "harmonicity" in datos else 0
    forma_cols = [c for c in VOZ_COLUMNAS if c.startswith("formant.") and c in datos.columns]
    formantes_fuera = int(sum(((datos[c] < 0) | (datos[c] > COTA_FORMANTE)).sum() for c in forma_cols)) if forma_cols else 0
    pitch_fuera = int(((datos["pitch"] < COTA_PITCH[0]) | (datos["pitch"] > COTA_PITCH[1])).sum())
    pitch_ceros = int((datos["pitch"] == 0).sum())
    # regime: the band 0 < pitch < 60 is impossible by construction (the analysis does not look
    # below 60 Hz), and the same goes for under the HNR floor or above the formant ceiling.
    # `voice_present` only exists since the fix, so it works as a witness too.
    imposible = pitch_fuera + bajo_suelo + formantes_fuera
    if "voice_present" in columnas:
        regimen = "segments"
        vp = datos["voice_present"]
        con_voz = int(vp.fillna(False).sum())
        voice_present = (f"{con_voz} of {total} frames with a measured track "
                         f"({100 * con_voz / max(total, 1):.1f} %); {int(vp.isna().sum())} with no pitch track")
        puerta = "voice_present"
        voz = datos[vp.fillna(False)]
    else:
        regimen = "spline"
        voice_present = "(no column: this lot predates the prosody fix)"
        puerta = f"{COTA_PITCH[0]:.0f}-{COTA_PITCH[1]:.0f} Hz"
        voz = datos[(datos["pitch"] >= COTA_PITCH[0]) & (datos["pitch"] <= COTA_PITCH[1])]
    if regimen == "segments" and imposible:
        regimen = "segments with leftovers"   # no trust: it gets said

    def correlacion(tabla):
        if {"intensity", "right_speed"} <= set(tabla.columns):
            pares = tabla[["intensity", "right_speed"]].dropna()
            if len(pares) > 2 and pares.std().min() > 0:
                return f"{pares.corr().iloc[0, 1]:+.3f} (n={len(pares)})"
        return "not computable"

    return {
        "regimen": regimen, "voice_present": voice_present, "puerta": puerta,
        "pitch_ceros": pitch_ceros, "pitch_fuera": pitch_fuera,
        "hnr_bajo_suelo": bajo_suelo, "formantes_fuera": formantes_fuera,
        "int_rango": (f"{datos['intensity'].min():.1f} .. {datos['intensity'].max():.1f} dB"
                      if "intensity" in datos else "(no column)"),
        "corr_todo": correlacion(datos), "corr_voz": correlacion(voz), "n_voz": len(voz),
    }


def _polymnia_por_defecto():
    """Where Polymnia is: it can be the parent (final_dataset/ lives inside the repo)
    or a sibling (a loose copy next to a Polymnia checkout)."""
    arriba = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return arriba if os.path.exists(os.path.join(arriba, "gesture_features.py")) \
        else os.path.join(arriba, "Polymnia")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", required=True, help="final/ folder with the parquets and the manifest")
    parser.add_argument("--csv", help="where to write the dictionary as CSV")
    parser.add_argument("--md", help="where to write the dictionary as Markdown")
    parser.add_argument("--readme", help="where to write the README with the delivery cover "
                        "(it is the file published as DATASET.md inside final/)")
    parser.add_argument("--polymnia_root", default=_polymnia_por_defecto())
    args = parser.parse_args(argv)

    diccionario, informe = generar(args.final, args.polymnia_root)
    # Schema names, not prose: in ASCII on purpose, and the guard below fails loudly if one of
    # them does not exist (a rename here empties CSV columns in silence, which is exactly the
    # kind of bug this script cannot have).
    columnas = ["tabla", "columna", "ambito", "familia", "que_mide", "formula_unidades",
                "tipo", "no_nulos", "pct_no_nulos", "minimo", "mediana", "maximo", "fuente"]
    faltantes = [c for c in columnas if c not in diccionario.columns]
    if faltantes:
        raise SystemExit(f"the dictionary does not produce these schema columns: {faltantes}. "
                         "Field names are not class-facing prose: they stay plain ASCII.")
    vista = diccionario.reindex(columns=columnas)

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        vista.to_csv(args.csv, index=False)
    muestra = muestra_de_ejemplos(args.final) if (args.md or args.readme) else None
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
        escribir_markdown(diccionario, informe, args.md, muestra)
    if args.readme:
        destino = os.path.abspath(args.readme)
        dentro = os.path.abspath(args.final) + os.sep
        if destino.startswith(dentro):
            raise SystemExit(f"--readme cannot write inside the delivery it describes ({args.final}): "
                             "final/ is an atomic package with a sha256 in the manifest. Write it "
                             "outside and let final_dataset.py bring it in with its own --readme.")
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        escribir_markdown(diccionario, informe, destino, muestra, portada=True)

    print(f"documented columns: {len(vista)} "
          f"({(vista['tabla'] == 'final_videos').sum()} in final_videos, "
          f"{(vista['tabla'] == 'final_frames').sum()} in final_frames), unresolved: 0")
    if args.csv:
        print(f"csv: {args.csv}")
    if args.md:
        print(f"markdown: {args.md}")
    if args.readme:
        print(f"readme: {os.path.abspath(args.readme)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
