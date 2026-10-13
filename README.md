<p align="center">
  <img src="assets/logo/polymnia_logo.png" alt="POLYMNIA logo" width="50%">
</p>

<p align="center">
  <strong style="font-size: 2em;">POLYMNIA</strong><br>
  <span style="font-size: 1.2em;">Modular pipeline for multimodal language analysis</span>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.XXXXXXXX"><img src="https://zenodo.org/badge/DOI_REPO_ID.svg" alt="DOI"></a>
  <img src="https://github.com/daedalusLAB/POLYMNIA/actions/workflows/main.yml/badge.svg" alt="Test POLYMNIA pipeline">
</p>

> **Check** [CHANGELOG.md](./CHANGELOG.md) **for recent updates.**

---

## 🧬 Overview

**POLYMNIA** is a modular pipeline for the automated processing of multimodal linguistic data. It integrates modules for body pose estimation (OpenPose), prosodic analysis (Parole), gesture normalization (dfMaker), and alignment of text and audio (WhisperX), combining them into a unified data structure optimized for research and visualization.


## 🔧 Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

> `parole.sh` (used for prosody analysis) is part of the [parole system](https://github.com/daedalusLAB/parole), which includes its own installer (`install_parole.sh`) for Praat, R scripts, and dependencies. See `parole/` and install it before running `ecos.py`.

## 📦 Requirements

See `requirements.txt`:

```txt
argparse
opencv-python
ultralytics
praat-parselmouth
moviepy
pandas
duckdb
pyarrow
git+https://github.com/m-bain/whisperx.git
```

External dependencies:

* `curl` and `ffmpeg` (for video/audio processing)
* R + packages (`arrow`, `logger`, `multimolang`)
* OpenPose (compiled binary)
* Praat (called by `parole.sh`)



## 1️⃣ `download_clips.py` — Download video clips from NewsScape

Downloads `.mp4` video clips from `gallo.case.edu` based on a tabulated `.txt` file. Adds a configurable offset to clip timestamps. Retries failed downloads and logs errors.

**Inputs**:

* `--txt_file`: Tab-separated file with clip metadata
* `--searchterm`: Label for clip naming
* `--offset`: Seconds before/after clip (default: 2.5)
* `--output_dir`: Target folder for videos

**Example**:

```bash
python3 download_clips.py \
  --txt_file clips.txt \
  --searchterm climate_change \
  --output_dir ./output/videos/raw
```



## 2️⃣ `is_there_a_person_in_the_video.py` — Filter usable clips

Uses YOLOv8-Pose to accept only videos with exactly **one visible person** (head, shoulders, and optionally hands). Extracts 15 representative frames per clip.

**Inputs**:

* `--videos_folder`: Folder with `.mp4` files
* `--matched_videos`: Output folder for accepted clips
* `--discarded_videos`: Folder to store discarded clips
* `--check_hands`: Require hand keypoints (default: False)

**Example**:

```bash
python3 is_there_a_person_in_the_video.py \
  --videos_folder ./output/videos/masked \
  --matched_videos ./output/videos/1_person \
  --discarded_videos ./output/videos/discarded \
  --check_hands True
```



## 3️⃣ `argos.py` — Run OpenPose and classify by number of people

Runs OpenPose on each video and saves body keypoints in `JSON_FILES/`. Then calls an R script to standardize coordinates and classify videos into `1_persons/`, `2_persons/`, etc.

**Inputs**:

* `--videos_folder`: Folder with filtered `.mp4` videos
* `--output_folder`: Root output directory
* `--openpose_path`: Path to OpenPose build
* `--face_hands`: Enable face/hands detection
* `--skeletons`: Save skeleton render videos

**Example**:

```bash
python3 argos.py \
  --videos_folder ./output/videos/1_person \
  --output_folder ./output \
  --openpose_path /opt/openpose \
  --face_hands True \
  --skeletons True
```



## 4️⃣ `max_people_classification.R` — Coordinate normalization and person counting

Called automatically by `argos.py`. Uses the `multimolang::dfMaker()` function to:

* Normalize OpenPose JSONs into structured Parquet
* Detect number of people per video
* Write `num_persons.txt` and save `.parquet` to:
  `dataset/{n_persons}_persons/parquet_files/`

Dependencies are installed locally in `R_libs/` if not available.



## 5️⃣ `ecos.py` — Run prosody analysis via Parole

Calls `parole.sh` for each video using the `.mp4` and corresponding `.parquet` coordinates. Generates prosodic curves (pitch, intensity, etc.) aligned to frames.

**Inputs**:

* `--input_folder`: Root folder with dataset
* `--n_persons`: Number of people to select (e.g., 1)

**Output**:

* Parquet files stored in `parquet_speech_analysis/`

**Example**:

```bash
python3 ecos.py \
  --input_folder ./output \
  --n_persons 1
```



## 6️⃣ `apate.py` — Run WhisperX and align words to frames

* Extracts audio from `.mp4` using `ffmpeg`
* Runs `whisperx` for transcription + word timing
* Aligns words with video frames
* Saves frame-level lexical info as `.json`

**Output**:

* `whisperx/` and `whisperx_framer_output/` directories with aligned data

**Example**:

```bash
python3 apate.py \
  --input_folder ./output \
  --n_persons 1
```



## 7️⃣ `hefesto.py` — Merge all modalities into DuckDB

Merges pose data, prosody, and lexical info into a unified DuckDB database. Performs outer joins by `frame`. Skips missing or corrupted files and logs discrepancies.

**Inputs**:

* Pose `.parquet`: `parquet_files/`
* Prosody `.parquet`: `parquet_speech_analysis/`
* Words `.json`: `whisperx_framer_output/`

**Output**:

* `dataset.duckdb` with a `multi_data` table

**Example**:

```bash
python3 hefesto.py \
  --input_folder ./output \
  --n_persons 1 \
  --db_name ./output/dataset.duckdb
```



## Optional neck-relative gesture features

After merging with `hefesto.py`, run `gestures.py` to derive wrist geometry and
kinematics. This is movement measurement, **not recognition of named gestures**.
The source DuckDB is opened **read-only**. Results go into a **new** folder:
`gesture_frames.parquet`, `gesture_sequences.parquet`, and `gesture_config.json`.

```bash
python3 gestures.py --db_name ./output/dataset.duckdb \
  --output_folder ./output/gestures \
  --max-gap-frames 3 --min-keypoint-confidence 0.2
```

| Option | Default | Meaning |
|---|---|---|
| `--db_name`, `--table_name` | required, `multi_data` | existing database and long-format pose table |
| `--output_folder` | required | directory to create; existing paths are never replaced |
| `--min-keypoint-confidence` | `0.2` | inclusive wrist threshold; confidence zero is always absent |
| `--max-gap-frames` | `3` | maximum internal coordinate/clock gap to interpolate; `0` disables interpolation |
| `--window-size`, `--poly-order` | `7`, `2` | odd smoothing window and smaller polynomial order; short runs adapt; window `1` disables smoothing |

Hyphen/underscore aliases are accepted, but conflicting values are rejected.
The input must use **BODY_25** and dfMaker's default fast normalization:
origin **neck (point 1)**, wrists **4/7**, **Y positive up**. Scale comes from the
neck-to-point-5 shoulder X component, not bilateral shoulder span or a rotated
basis. The adapter verifies the origin per clip/person. Distances are normalized
units, speeds units/s, accelerations units/s² — not metres or kinetic energy.

Calculation is isolated by `id`/`people_id`, ordered by frame. IDs are used as
provided: this stage does not track people or repair changing detection order.
Geometry needs one usable sample; velocity two connected timed samples;
acceleration three. Between-hand features require both wrists. NULL means missing
evidence, not zero movement. Outputs include positions, neck distance/angle,
above-neck flags, velocity/acceleration, connected path lengths, separation,
centroids, mirror residuals, and per-sequence summaries. There are no labels,
class-aware scores, batch-dependent scaling, q-cut rings, or coordination proxies.

Coverage is the **observed frame span**, not the whole video. Internal gaps up to
the configured limit may be estimated and flagged; edges are never extrapolated
and long gaps stay NULL. Untimed source frames are retained with geometry.
`timestamp_inferred` identifies estimated clocks; `clock_known` indicates available
clocks. Without a clock, smoothing and temporal features stop. Sequence timing
uses the known-clock span, with explicit timed/inferred/missing counts; a peak on
an untimed frame has a NULL time-to-peak. Irregular timed samples use a local
polynomial fit rather than assuming uniform FPS. Configuration JSON records
parameters, schema, units, versions and source-clock/origin checks.

Publication uses Linux `renameat2(RENAME_NOREPLACE)` and fails closed on unsupported
platforms. Even a concurrent empty destination is preserved. For CPU-only tests,
run `python3 tools/test_gesture_features.py` and `python3 tools/test_gestures_cli.py`.
These use known trajectories and temporary synthetic databases, not live models.
For notebook usage, see `compute_gesture_features(wide_dataframe, config)` in
[`gesture_features.py`](gesture_features.py).

## 📁 Output Structure

```
output/
├── videos/
│   ├── raw/               # Downloaded videos
│   ├── masked/            # ASD-processed videos
│   ├── discarded/         # Rejected videos
│   └── 1_person/          # Accepted videos (1 visible speaker)
├── dataset/
│   ├── OpenPose/          # OpenPose JSONs
│   ├── 1_persons/
│   │   ├── parquet_files/
│   │   ├── parquet_speech_analysis/
│   │   ├── whisperx/
│   │   └── whisperx_framer_output/
│   └── dataset.duckdb     # Final database
```



## 📌 Notes

* All modules are idempotent and modular — rerun safely with `--resume` or `--start_at_argos`.
* Processing can be parallelized using `concurrent.futures` where applicable.
* All logs and errors are saved for reproducibility.
* This pipeline assumes a well-formed tabulated `txt` and structured `.mp4` filenames.



## 👥 Authors

Developed by **DaedalusLAB**
For multimodal linguistic research, gesture-prosody alignment, and AI-enhanced corpus construction.


## 🧩 Theoretical Framework

**POLYMNIA** is based on the principle that language is a *physical phenomenon* observable in space and time.
Every linguistic act —acoustic, gestural, or digital— involves energy, duration, and spatial organization.
Therefore, language can be modeled as a **dynamic system**, where its components interact non-linearly and evolve together over time.

The pipeline measures and models these physical traces —movement, prosody, and lexical timing— to infer reproducible knowledge (*episteme*) about the **multimodal flow of language** understood as a real, measurable process.
This approach shifts the focus from isolated linguistic units to the continuous interaction among modalities, aligning linguistic research with the empirical standards of the natural sciences.

