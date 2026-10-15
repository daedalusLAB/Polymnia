<p align="center">
  <img src="assets/logo/polymnia_logo.png" alt="POLYMNIA logo" width="50%">
</p>

# POLYMNIA

**A modular pipeline for multimodal language research.** POLYMNIA combines body,
face and hand keypoints, person-relative coordinates, prosodic features and
frame-aligned words in a queryable DuckDB dataset. An optional vision-language
model (VLM) stage adds seven video-level annotations.

Start with the workflow below. For installation details and compatibility notes,
see [SETUP.md](SETUP.md) (Spanish, with commands and local deployment examples).
Use the repository's commit history to track changes.

## Pipeline at a glance

```text
NewsScape tab file → download → TalkNet-ASD → person filter → OpenPose / dfMaker
                                                         ↓
raw audio ──────────────────────────────────────→ Parole + WhisperX
                                                         ↓
                                                    DuckDB merge
                                                         ↓
                                              optional VLM annotations
```

| Stage | CLI command | Purpose |
|---|---|---|
| Validate / download | `check-tab`, `download` | Check timing fields; retrieve NewsScape clips |
| Identify the speaker | `asd` | Produce speaker-masked videos with TalkNet-ASD |
| Filter clips | `filter` | Select clips meeting the single-visible-person criteria |
| Extract movement | `argos` | Run OpenPose, transform coordinates with dfMaker, classify by person count |
| Extract speech | `voice` | Run Parole (`ecos`) and WhisperX (`apate`) in parallel |
| Assemble the dataset | `merge` | Align pose, prosody and words by frame in DuckDB |
| Annotate videos | `theia` | Query a vision model for scene and speaker attributes |

The filter and OpenPose are different detectors: a clip accepted by `filter` can
still be classified into a multiple-person category by `argos`.

## Installation

The current environment targets **Linux with an NVIDIA GPU**. Python dependencies
are managed with [uv](https://docs.astral.sh/uv/); the project selects Python 3.12
and PyTorch 2.8-compatible CUDA 12.8 builds on non-macOS x86_64/AMD64.
Other platforms are not established as supported end-to-end configurations.

### Bootstrap on an account that has nothing yet

Three things finish the setup: this block, OpenPose compiled by whoever administers
the machine (step 8 reports it), and the two TalkNet weights, which are not in any
Git repository (step 4 says why). Run it from the account that will do the work: the
virtual environment, `uv` and the cache directories are all per-user.

```bash
# One work root holds the checkout. The pipeline creates its lots in the directory
# you launch it from, and looks for TalkNet-ASD and modelos/ next to the repo.
export POLYMNIA_WORK="$HOME/polymnia-work"
mkdir -p "$POLYMNIA_WORK" && cd "$POLYMNIA_WORK"

# 1. uv, installed per account (it lives in your own home directory)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"        # or just open a new shell
cd "$POLYMNIA_WORK"

# 2. the repository and its Python environment
git clone https://github.com/daedalusLAB/Polymnia.git
cd Polymnia
uv sync --frozen --extra asd

# 3. Parole, vendored under create_datasets/, plus the prosody fix this repo keeps
#    a canonical copy of (create_datasets/ is in .gitignore, so it is not in the clone)
git clone https://github.com/daedalusLAB/parole create_datasets/parole
( cd create_datasets/parole && bash install_parole.sh )  # Praat and the R packages; its README lists options
bash final_dataset/install_parole_fix.sh               # install the canonical scripts
bash final_dataset/install_parole_fix.sh --comprobar   # must agree with the canonical copies

# 4. TalkNet-ASD as a sibling of the repo. Clone this fork rather than plain
#    upstream: it carries the fixes this pipeline runs on (speaker selection,
#    missing track ids, fps as a float). State in use: e802c54d33bdf4fbdb6d3bcb24ffec9883d8ff02.
git clone https://github.com/raul-agent/TalkNet-ASD.git ../TalkNet-ASD
#    Its two weights are ignored by that repo (*.pth, *.model), so no clone has them;
#    copy them from a machine that already ran a pipeline, and check them:
#      89844381  model/faceDetector/s3fd/sfd_face.pth
#                d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491
#      63227827  pretrain_TalkSet.model
#                d985ddd07dca08864a28337726f5b7d91b6426bebc7327f0e9b21cf9ff1cc937
#    Copy with `cp -r`, never `cp -a`: the source files are mode 600 and -a keeps
#    that mode, which locks the next user out of the ASD stage.

# 5. DuckDB CLI. uv installs the Python binding, not the command line tool.
mkdir -p tools/bin
curl -L -o /tmp/duckdb.zip \
  https://github.com/duckdb/duckdb/releases/download/v1.5.6/duckdb_cli-linux-amd64.zip
unzip -o -d tools/bin /tmp/duckdb.zip && chmod +x tools/bin/duckdb
tools/bin/duckdb --version                   # v1.5.6 (Variegata) 069cc9f9b5

# 6. YOLO pose weights, needed by the wrist prescreen (stage 4 of
#    final_dataset/pipeline_lote.sh). A sibling modelos/ is found automatically;
#    POLYMNIA_YOLO_WEIGHTS=/absolute/path overrides the location.
mkdir -p ../modelos
curl -L -o ../modelos/yolo11x-pose.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt

# 7. check the downloads against values measured on a machine that works
sha256sum -c - <<'SUMS'
61238cfbe9dfeaad4bcfcd8a48f6f7123dae2e9603aaadb1f77a4217aeb8980c  tools/bin/duckdb
013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8  ../modelos/yolo11x-pose.pt
SUMS

# 8. what is still missing, with the path it looked for
./bin/polymnia doctor
```

Step 7 is not decoration: the two hashes above are the SHA-256 of what those two URLs
serve today, and `doctor` compares presence, not content. If a hash differs you have a
different artifact, and `informes/procedencia.json` would later record a weight nobody
can name.

### What `uv sync` does not install

`uv sync` installs Python dependencies, **not the external tools or model weights**:

| Dependency | Required setup |
|---|---|
| OpenPose | Compiled binary; default root `/opt/openpose`, or `--openpose_path` for `argos` |
| TalkNet-ASD | Checkout at `../TalkNet-ASD` (override with `POLYMNIA_ASD_PATH`). Use [the fork this pipeline runs on](https://github.com/raul-agent/TalkNet-ASD): [upstream](https://github.com/RaulKite/TalkNet-ASD) does not carry the fixes for speaker selection, missing track ids or float `fps`. Its two weights are ignored by Git and must be copied by hand (step 4) |
| R | `multimolang`, `arrow`, `logger`; local packages in `R_libs/`, configuration in `config_dfMaker.json` |
| Parole / Praat | [Parole](https://github.com/daedalusLAB/parole) under `create_datasets/parole/`, installed with its own `install_parole.sh`, then `bash final_dataset/install_parole_fix.sh` for the prosody scripts in this repo |
| YOLO pose weights | `modelos/yolo11x-pose.pt`, 118 481 010 bytes, SHA-256 `013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8`, from [ultralytics assets](https://github.com/ultralytics/assets/releases/tag/v8.3.0). Without it the wrist prescreen cannot run and `doctor` exits 1; `POLYMNIA_YOLO_WEIGHTS` overrides the path |
| System tools | Bash, `curl`, GNU `timeout`, `ffmpeg`, `ffprobe` and `Rscript` |
| DuckDB CLI | Executable at `tools/bin/duckdb` for `polymnia duckdb`: the [v1.5.6 CLI release](https://github.com/duckdb/duckdb/releases/tag/v1.5.6), SHA-256 `61238cfbe9dfeaad4bcfcd8a48f6f7123dae2e9603aaadb1f77a4217aeb8980c`. Python DuckDB is installed by uv |
| Vision endpoint | Optional OpenAI-compatible chat-completions API with image support |

The wrapper sets the Python and R paths and uses the existing uv environment;
no manual virtual-environment activation is needed. Run `doctor` before processing.

## Input format

The downloader reads a **headerless, tab-separated file** with at least six fields:

| Position | Meaning |
|---|---|
| 1 | NewsScape filename/path; basename becomes part of the output clip ID |
| 2 | NewsScape clip/program identifier used by the download service |
| 3 / 4 | Start time: whole seconds / fractional integer |
| 5 / 6 | End time: whole seconds / fractional integer |

Extra fields are ignored by the downloader. The fractional divisor is **1000 when
the normalized filename starts with `2017`, otherwise 100**. This is the literal
current rule, not a general guarantee for later years. Padding (`--offset`) defaults
to **2 seconds on each side** in both the standalone downloader and the batch
pipeline, and one value has to be used for the whole batch: it cannot be recovered
later from the clip filename, because the downloader writes the filename with the
padding it used. `verificar-holgura` measures it instead, from the media duration.

Run `check-tab` first: it checks parsing, reversed intervals and agreement with
ranges embedded in filenames. The download wrapper also checks, but warnings do
not abort downloads. `verify_videos.py` checks the resulting media independently.
`verificar-holgura` is the one that *does* stop: it compares each clip's measured
duration with the tab window plus the requested padding, and exits 1 on a mismatch.

Capture provenance at the start, while the environment is still the one doing the work:
`tools/tool_provenance.py` writes which Praat, R, OpenPose build, YOLO weights and repo
commits are in play (`code`, `binaries`, `weights`, `python`, `services`). It records `null`
with a reason for anything it cannot identify and never reads the Theia API key.
`final_dataset.py --provenance` embeds that file in `manifest.json`.

## Run the workflow

Run from the repository root with **absolute paths**. Paths and clip filenames
must also be free of whitespace and shell metacharacters: downstream scripts build
unquoted shell commands. Use trusted inputs; replace the path and search label below.

```bash
set -euo pipefail
TAB="/absolute/path/clips.txt"
OUT="$PWD/output"
mkdir -p "$OUT"
./bin/polymnia python tools/tool_provenance.py --out "$OUT/informes/procedencia.json" --gpu
./bin/polymnia check-tab "$TAB" --offset 2.0
./bin/polymnia download --txt_file "$TAB" --searchterm climate \
  --offset 2.0 --output_dir "$OUT/videos/raw"
./bin/polymnia verificar-holgura --videos "$OUT/videos/raw" --tab "$TAB" \
  --offset 2.0 --searchterm climate
./bin/polymnia python tools/verify_videos.py "$OUT/videos/raw"
./bin/polymnia asd --input_dir "$OUT/videos/raw" --output_dir "$OUT/videos/masked"
./bin/polymnia filter --videos_folder "$OUT/videos/masked" \
  --matched_videos "$OUT/videos/1_person" --discarded_videos "$OUT/videos/discarded"
./bin/polymnia argos --videos_folder "$OUT/videos/1_person" --output_folder "$OUT"

# Process every category actually produced, not a fixed list of person counts.
for category in "$OUT"/dataset/*_persons; do
  [ -d "$category/parquet_files" ] || continue
  n="${category##*/}"; n="${n%_persons}"
  [[ "$n" =~ ^[0-9]+$ ]] || continue
  ./bin/polymnia voice --input_folder "$OUT" --n_persons "$n"
  ./bin/polymnia merge --input_folder "$OUT" --n_persons "$n" \
    --db_name "$OUT/multi_${n}p.duckdb"
done
```

Pose extraction uses the **filtered masked videos** in this example; speech stages
use their matching **raw videos**. `voice` writes `ecos.log` and `apate.log` at the
output root, overwriting those logs for each category. Check outputs as well as
exit codes: earlier stages can discard clips or skip missing files.

Outputs include `videos/{raw,masked,1_person,discarded}/`, OpenPose JSONs in
`dataset/OpenPose/{id}/JSON_FILES/`, and `dataset/{n}_persons/` containing
`parquet_files/`, `parquet_speech_analysis/`, `whisperx/` and
`whisperx_framer_output/`. The example keeps a separate DuckDB file per category.

## Optional visual annotations

After merging, export `THEIA_API_BASE`, `THEIA_MODEL` and `THEIA_API_KEY` into the
process environment. Do not put credentials in Git, shared commands or datasets.
A repository `.env` is ignored by Git, but this version's API-key fallback does
not reliably load a key supplied **only** there; use exported variables.

```bash
./bin/polymnia theia --input_folder "$OUT" --n_persons 1 \
  --db_name "$OUT/multi_1p.duckdb"
```

Theia selects IDs from `multi_data`, samples up to five proportional frame
positions, and makes separate scene (raw) and speaker (masked + raw context)
requests. It validates responses and retries failures; **schema validity is not
annotation accuracy**. Unknown values may be NULL.

| Field | Accepted values (also NULL) |
|---|---|
| `indoor_outdoor` | `indoor`, `outdoor` |
| `show_type` | `news anchor`, `monologue`, `weather report`, `interview` |
| `hands_free`, `screen_interaction` | Boolean |
| `sitting_standing` | `sitting`, `standing` |
| `sex` | `male`, `female` (model estimate, not verified identity) |
| `age` | Integer 0–120 (model estimate) |

Results go to `dataset/{n}_persons/video_annotations.csv`, the `video_annotations`
table, and seven columns in `multi_data`. Inspect `status` / `error` before use.
`--no-db` writes CSV only. Resume skips CSV rows marked `ok` and retries errors;
if all rows are already `ok`, it returns without synchronizing a new database.
For endpoint/model overrides or `--max-tokens`, use
`./bin/polymnia python theia.py --help`: these flags are not forwarded by the
`theia` wrapper. Repeat annotation for each category you want to include.

## Querying and interpreting the data

`multi_data` is **long format**, not one row per frame: pose rows carry clip,
frame, person and keypoint identifiers (`id`, `frame`, `people_id`, `type_points`,
`points`). Speech values repeat across keypoint rows; annotations repeat across
all rows of a clip. Count distinct IDs for clip totals and collapse frame-level
features before statistical analysis. Frame indices are 1-based in the merged pose data.

```sql
SELECT count(*) AS rows, count(DISTINCT id) AS clips FROM multi_data;
SELECT show_type, count(*) AS clips
FROM video_annotations WHERE status = 'ok' GROUP BY show_type;
WITH per_frame AS (
  SELECT id, frame, max(pitch) AS pitch, bool_or(vad) AS vad
  FROM multi_data GROUP BY id, frame
)
SELECT avg(pitch) FILTER (WHERE vad) AS mean_voiced_pitch_hz FROM per_frame;
```

`x` / `y` are pixel coordinates; `nx` / `ny` are dfMaker person-relative
transformed coordinates, **not [0,1] screen coordinates**. Missing keypoints or
anchors can produce NULLs. Parole interpolates prosody onto frame timestamps:
non-null pitch does not prove phonation. VAD is useful for selecting speech, but
is not a guarantee that every pitch estimate is valid.

## Verification and limitations

Run `./bin/polymnia test` for the CPU-focused prosody-alignment and Theia tests
(Theia uses a local fake HTTP server, not a paid endpoint). These do not replace
end-to-end checks of external models, downloads and generated datasets.

- `merge` adapts Parole's nested output into derived flat files expected by
  Hefesto, preserving originals. Missing prosody can skip an entire clip.
- Reruns are stage-specific: **merge appends rows**, so use a fresh database for
  a clean rebuild. Do not assume every stage is idempotent or supports resume.
- Speaker detection and person counting are model decisions, not ground truth.
  Keep raw media and inspect discarded clips before interpreting coverage.
- For new naming conventions, review `config_dfMaker.json` metadata extraction.

## Research context, authors and license

Developed by **DaedalusLAB** for multimodal linguistic research, gesture–prosody
alignment and corpus construction. POLYMNIA treats language as a physical,
dynamic process: movement, acoustic features and lexical timing provide measurable
traces of the interaction among modalities, rather than isolated linguistic units.

The repository is distributed under [GNU GPL v3](LICENSE). External software,
model weights and source media have their own terms; this license does not grant
redistribution rights for NewsScape videos. No release DOI is currently documented
here; cite the repository URL and the exact commit used for reproducibility.
