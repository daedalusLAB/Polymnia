# `final_dataset/` — from a NewsScape TSV to the two tables

One script, ten stages, one folder per batch. Input is a NewsScape clip TSV (the format
`download_clips.py` eats); output is the dataset: `final_frames` (one row per frame) and
`final_videos` (one row per clip), in Parquet and CSV, plus a data dictionary and a README
generated from the numbers actually written.

The docs in this directory are in **Spanish** because they were written for the people who
deliver the data (`RUNBOOK_nuevo_tsv.md` is the one to read first). This README is the
English way in.

## The one command

```bash
final_dataset/pipeline_lote.sh --tsv clips_nuevos.txt --lote lote3
```

It runs `tsv → descarga → asd → preseleccion → openpose → limpieza → voz → merge → theia →
final` and leaves everything in `./lote3/`: the TSV with its SHA-256, the videos, what got
dropped and *why* (`descartados/<motivo>/`), the OpenPose JSON, one Parquet per clip,
`munecas.csv`, all ten logs, and `final/` with the tables. It is resumable:

```bash
final_dataset/pipeline_lote.sh --tsv clips_nuevos.txt --lote lote3 --desde asd
```

`--desde asd` (not `--desde tsv`) to resume after a download: **`download_clips.py` does not
skip clips it already downloaded**, so re-running that stage re-downloads the lot. The
downloaded count is compared against the TSV before anything else is trusted; if it does not
match, the run was resumed, not finished.

## Layout

| file | what it does |
| --- | --- |
| `pipeline_lote.sh` | the whole batch, in order, with logs, counters and a stop-on-failure rule per stage |
| `final_dataset.py` | the last stage: merges the per-clip Parquets + prosody + gestures into the two tables, with the column contract |
| `diccionario_columnas.py` | generates the data dictionary from the tables it just wrote — no hand-written README survives here |
| `pasted_features.py` | the reference 373-column feature stack, kept as the published baseline |
| `install_parole_fix.sh` | installs / **checks** the PAROLE prosody fix in a local PAROLE clone (`--comprobar`) |
| `upstream_parole/` | the canonical copy of that fix, its patch, and the tests that keep it honest |
| `wrist_prescreen.py`, `limpiar_fantasmas.py` | now live in [`../tools/`](../tools) — the pipeline picks them up there |
| `RUNBOOK_nuevo_tsv.md` | step-by-step for a new TSV, including what to check when a stage is suspicious |
| `HOWTO_lote.md` | the operational detail: offsets, resumes, GPU, known traps |

Two files here are **not** generated: `clips_delivered_29.txt` (the 29 clips of the first
published delivery, so the byte-identical regression can be re-run) and
`prescreen_validation_29.csv` (the measured evidence for the "yield curve is flat" claim in
`tools/wrist_prescreen.py`). Everything else that appears in this directory is an artifact.

## Before spending GPU

```bash
Polymnia/bin/polymnia doctor                       # GPU, OpenPose, TalkNet-ASD, WhisperX, Praat, R
final_dataset/install_parole_fix.sh --comprobar    # the prosody fix is still in your PAROLE clone
ls -l modelos/yolo11x-pose.pt                      # prescreen weights (--peso for another path)
```

Needs: the Polymnia env (through `bin/polymnia`, which puts the venv, R, Praat and
`scenedetect` on `PATH`), a PAROLE clone at `create_datasets/parole` (`install_parole.sh`
fetches Praat from GitHub releases), YOLO11-pose weights, and a GPU. If `bin/polymnia` is not
next to this directory it assumes Polymnia is a sibling checkout — nothing to configure.

## Two things that will bite you

**Keep the same `--offset` for the whole batch.** The delivered clips were cut with
`--offset 2.5`; `final_dataset.py` defaults to `2.0`. Mixing them silently changes which
frames exist.

**The wrapper is not optional for GPU stages.** Called bare, TalkNet-ASD cannot find
`scenedetect` or `python`, produces nothing, and **exits 0**.

## Tests

```bash
final_dataset/test_todo.sh
```

which runs `test_diccionario_columnas.py` (dictionary columns match the tables, formulas and
units present, no set-dependent score) and `test_final_dataset_voz.py` (voice columns:
contiguous spans, `NULL` outside a track, no `0` standing in for silence). `polymnia test`
globs `tools/test_*.py` only, so these two are not in that loop.

`test_diccionario_columnas.py` resolves the `EXTENDED_*` families out of
`gesture_features.py`, so on a checkout that predates the version-3 block it prints `[skip]`
and passes instead of failing: the dictionary genuinely cannot resolve those columns there,
and that is a missing piece, not a broken delivery.

## Provenance

Every delivery records the TSV's SHA-256, the offsets used, per-column counters for how many
values were `NULL` and why, and the hashes of the inputs it read. `NULL` means "not
measurable", never `0`, and nothing is extrapolated off the end of a track — including voice.

Voice in the data delivered so far was produced with **Praat 6.4.30** (the version
`install_parole.sh` had pinned before it moved to releases); the `To Pitch` settings are
documented in `RUNBOOK_nuevo_tsv.md`. Pin it with `PRAAT_URL=…/praat6430_linux-intel64-barren.tar.gz`
if you are re-cutting an existing delivery and want the numbers to land identically.
