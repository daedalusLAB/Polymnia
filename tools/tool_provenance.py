#!/usr/bin/env python3
"""Write down which tools actually produced this dataset (Polymnia #14).

Why this exists: before this, the only environment record in a delivery was
`manifest.json`'s `platform` line -- the Python version and the OS. Nothing said which
Praat produced the `.Pitch` files, which OpenPose build produced the keypoints, which
weights the wrist prescreen ranked with. `bin/polymnia doctor` prints most of it, but
printing is not recording: a delivery two years from now cannot ask today's `doctor`.

Three honest constraints this tool obeys:

1. **Some tools have no version.** A `.pt` weight file is not versioned by anybody, so its
   identity is size + SHA-256. Same for an `openpose.bin` whose `--version` may or may not
   be built in. Every entry says `how` it got its identity (`--version`, `git`, `sha256`).
2. **A tool that is not there is recorded as missing, with a reason.** The probe never
   substitutes a plausible number, and never fails the run for a missing optional tool --
   that is the doctor's and the pipeline's business. It writes `null` + `reason`.
3. **Environment at run start, not at report time.** `pipeline_lote.sh` calls this in stage
   0 and `final_dataset.py` embeds *that file*, so the manifest describes the environment
   that did the work rather than whatever got upgraded while a 5000-clip lot was running.

Secrets: the Theia API key is never read, never resolved, never written. Only the base URL
and the model name are recorded, because those are configuration that changes results.

Usage:
    tool_provenance.py --out FILE [--weights PATH] [--gpu] [--root DIR]
    tool_provenance.py                    # prints JSON to stdout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# How many seconds we give a tool to tell us who it is. A tool that hangs is recorded as
# unknown with `reason: timeout`; it must not hang the pipeline.
TIMEOUT_S = 45

# Python modules whose version changes numbers: the pose math lives in the repo, but the
# linear algebra and IO underneath it do not.
PY_MODULES = ("numpy", "pandas", "scipy", "pyarrow", "duckdb", "cv2", "ultralytics", "parselmouth")
PY_GPU_MODULES = ("torch", "torchvision")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(cmd: list[str]) -> dict:
    """Capture a tool's own words. Returns {stdout, reason} with exactly one non-null."""
    exe = shutil.which(cmd[0])
    if not exe:
        return {"path": None, "stdout": None, "reason": f"{cmd[0]} not on PATH"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"path": exe, "stdout": None, "reason": f"timeout after {TIMEOUT_S}s"}
    except OSError as exc:
        return {"path": exe, "stdout": None, "reason": f"could not run: {exc}"}
    text = (proc.stdout or proc.stderr).strip()
    if not text:
        return {"path": exe, "stdout": None, "reason": f"exited {proc.returncode} with no output"}
    return {"path": exe, "stdout": text.splitlines()[0].strip(), "reason": None}


def binary(cmd: list[str]) -> dict:
    """A version line quoted from the tool itself, or null + reason.

    A line without a single digit is not a version. This is not paranoia: this OpenPose
    build answers `-version` with the literal string `openpose.bin`, and recording that as
    the version would be worse than admitting it says nothing -- the identity of that tool
    comes from the SHA-256 of the binary instead.
    """
    got = run(cmd)
    text = got["stdout"]
    reason = got["reason"]
    if text is not None and not any(ch.isdigit() for ch in text):
        reason = (f"{cmd[0]} answered {text!r}, which is not a version; "
                  "identity comes from the sha256 of the binary")
        text = None
    return {"version": text, "path": got["path"], "how": " ".join(cmd), "reason": reason}


def sha256_of(path: str) -> dict:
    """Content identity for things nobody versioned (weights, a compiled binary)."""
    if not os.path.isfile(path):
        return {"sha256": None, "bytes": None, "reason": f"not a file: {path}"}
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            total += len(block)
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": total, "reason": None}


def git_identity(path: str) -> dict:
    """Commit + dirty flag. `dirty` matters: a patched tree is not the commit it points at.

    `dirty` counts changes to **tracked** files only; untracked files are reported separately.
    That is deliberate. The question is "is this tree different from the commit?", and loading
    this very file writes a `__pycache__` into the tree being probed, so counting untracked
    files would let the probe declare a clean repo dirty just by looking at it.
    """
    if not os.path.isdir(path):
        return {"commit": None, "dirty": None, "untracked": None,
                "reason": f"not a directory: {path}"}
    try:
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"commit": None, "dirty": None, "untracked": None, "reason": f"git failed: {exc}"}
    if head.returncode != 0:
        return {"commit": None, "dirty": None, "untracked": None,
                "reason": "no git HEAD here (vendored copy, not a clone)"}
    status = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                            capture_output=True, text=True, timeout=60)
    lineas = [l for l in status.stdout.splitlines() if l.strip()]
    untracked = sum(1 for l in lineas if l.startswith("??"))
    modified = len(lineas) - untracked
    return {"commit": head.stdout.strip(), "dirty": modified > 0, "modified_paths": modified,
            "untracked": untracked, "reason": None}


def module_versions(names: tuple[str, ...]) -> dict:
    """`__version__` when the module has one; `ok` when it imports but does not; null + reason."""
    out = {}
    for name in names:
        try:
            module = __import__(name)
        except Exception as exc:  # noqa: BLE001 - any import failure is a missing tool
            out[name] = {"version": None, "reason": f"{type(exc).__name__}: {exc}"}
            continue
        out[name] = {"version": getattr(module, "__version__", "ok"), "reason": None}
    return out


def env_or_dotenv(root: str, key: str) -> str | None:
    """Configuration from the environment, or from the repo's .env.

    Only ever called for keys that change results and are safe to store. The credential
    (THEIA_API_KEY) is never looked up, so it cannot end up in a manifest by accident.
    """
    value = os.environ.get(key)
    if value:
        return value.strip()
    path = os.path.join(root, ".env")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    except OSError:
        return None
    return None


def theia_config(root: str) -> dict:
    """Theia is an external service: record what changes the answers, never the credential."""
    model = env_or_dotenv(root, "THEIA_MODEL")
    return {
        "api_base": env_or_dotenv(root, "THEIA_API_BASE"),
        "model": model,
        "credentials": "not recorded on purpose (THEIA_API_KEY)",
        "reason": None if model else "THEIA_MODEL not set: the theia stage is skipped and its 7 columns are NULL",
    }


def collect(root: str, weights: str | None, with_gpu: bool) -> dict:
    root = os.path.abspath(root)
    parole = os.path.join(root, "create_datasets", "parole")
    openpose_bin = os.environ.get("OPENPOSE_PATH", "/opt/openpose")
    openpose_bin = os.path.join(openpose_bin, "build", "examples", "openpose", "openpose.bin")
    asd = os.environ.get("POLYMNIA_ASD_PATH", os.path.join(os.path.dirname(root), "TalkNet-ASD"))
    weight_path = weights or os.environ.get("POLYMNIA_YOLO_WEIGHTS") \
        or os.path.join(os.path.dirname(root), "modelos", "yolo11x-pose.pt")
    duckdb_cli = os.path.join(root, "tools", "bin", "duckdb")

    provenance = {
        "recorded_at": now_iso(),
        "note": ("captured at run start by pipeline_lote.sh stage 0; if a tool was upgraded "
                 "mid-run this block does not know about it"),
        "code": {
            "polymnia": git_identity(root),
            "parole_vendored": git_identity(parole),
            "talknet_asd": git_identity(asd),
        },
        "binaries": {
            "praat": binary(["praat", "--version"]),
            "Rscript": binary(["Rscript", "--version"]),
            "ffmpeg": binary(["ffmpeg", "-version"]),
            "scenedetect": binary(["scenedetect", "version"]),
            "duckdb_cli": binary([duckdb_cli, "--version"]) if os.path.isfile(duckdb_cli)
            else {"version": None, "path": None, "how": f"{duckdb_cli} --version", "reason": "not installed in tools/bin"},
            "openpose": {
                **binary([openpose_bin, "-version"]),
                "sha256": sha256_of(openpose_bin)["sha256"],
                "path": openpose_bin,
            } if os.path.isfile(openpose_bin) else
            {"version": None, "path": None, "sha256": None, "how": "openpose.bin -version",
             "reason": "openpose.bin not found (set OPENPOSE_PATH)"},
        },
        "weights": {
            "yolo_pose": {**sha256_of(weight_path), "path": weight_path,
                          "how": "size + sha256: nobody versions a .pt"},
        },
        "python": {
            "interpreter": sys.executable,
            "version": sys.version.split()[0],
            "os": platform.platform(),
            "modules": module_versions(PY_MODULES),
        },
        "services": {"theia": theia_config(root)},
    }
    if with_gpu:
        # Importing torch costs seconds, so the pipeline asks for it and the delivery
        # builder does not.
        provenance["python"]["gpu"] = module_versions(PY_GPU_MODULES)
        try:
            import torch  # noqa: WPS433

            provenance["python"]["gpu"]["cuda_available"] = {
                "version": str(torch.cuda.is_available()), "reason": None}
            if torch.cuda.is_available():
                provenance["python"]["gpu"]["device"] = {
                    "version": torch.cuda.get_device_name(0), "reason": None}
        except Exception:  # noqa: BLE001
            pass
    return provenance


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="where to write the JSON (default: stdout only)")
    ap.add_argument("--weights", help="YOLO weights used by the prescreen; the pipeline passes $PESO")
    ap.add_argument("--gpu", action="store_true", help="also probe torch/CUDA (costs a few seconds)")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="Polymnia repo root (default: the repo this tool lives in)")
    args = ap.parse_args()

    data = collect(args.root, args.weights, args.gpu)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        # write-then-move: a half-written provenance file is worse than none
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        os.replace(tmp, args.out)
        missing = [f"{group}/{name}"
                   for group in ("code", "binaries", "weights")
                   for name, entry in data[group].items()
                   # an entry is identified by a version line, a commit or a content hash:
                   # code is versioned by git, weights and an unversioned binary by sha256
                   if not (entry.get("version") or entry.get("commit") or entry.get("sha256"))]
        print(f"provenance -> {args.out}")
        print(f"   no version recorded for: {len(missing)}"
              + (f"  ({', '.join(missing)})" if missing else ""))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
