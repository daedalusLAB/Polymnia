#!/usr/bin/env python3
"""Tests for the provenance block final_dataset.py writes into manifest.json (#14).

A delivery used to record only its Python version and OS. These checks cover the three
things that make the new block trustworthy:

* it never invents: a missing, unreadable or non-object file is recorded as a reason;
* it embeds the file the pipeline wrote at run start, not a fresh probe of today's box;
* it still reports the commit of the code that computed the features, measured at build
  time, because a locally patched repo is not the commit it points at.

Output: one line per check; exit 0 if all pass, 1 otherwise.
"""

import json
import os
import subprocess
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import final_dataset as fd  # noqa: E402

failed = 0
done = 0


def check(name, ok, detail=""):
    global failed, done
    done += 1
    if ok:
        print(f"[ok]    {name}")
    else:
        failed += 1
        print(f"[FAIL]  {name}" + (f"  -- {detail}" if detail else ""))


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # --- nothing was captured -------------------------------------------------
        ausente = fd.build_provenance(None, tmp)
        check("no --provenance: the manifest says no versions were captured",
              ausente["tools"] is None and "--provenance" in ausente["reason"], ausente["reason"])
        check("no --provenance: captured_from is null rather than a made-up path",
              ausente["captured_from"] is None, ausente)

        # --- a real run-start file, embedded verbatim -----------------------------
        archivo = os.path.join(tmp, "procedencia.json")
        contenido = {"recorded_at": "2026-10-15T09:00:00Z",
                     "binaries": {"praat": {"version": "Praat 6.4.30 (April 29 2025)"},
                                  "openpose": {"version": None, "sha256": "abc123"}},
                     "weights": {"yolo_pose": {"sha256": "013c", "bytes": 118481010}}}
        with open(archivo, "w", encoding="utf-8") as handle:
            json.dump(contenido, handle)
        bloque = fd.build_provenance(archivo, tmp)
        check("the file written at run start is embedded unchanged",
              bloque["tools"] == contenido, bloque["tools"])
        check("its timestamp becomes the block's timestamp (provenance of the provenance)",
              bloque["captured_at"] == "2026-10-15T09:00:00Z" and bloque["reason"] is None, bloque)

        # --- the ways a file can fail, none of which may stop a delivery ----------
        roto = os.path.join(tmp, "roto.json")
        with open(roto, "w", encoding="utf-8") as handle:
            handle.write('{"recorded_at": "2026-10-15')
        sin_leer = fd.build_provenance(roto, tmp)
        check("a truncated JSON file is recorded as unreadable, not as empty provenance",
              sin_leer["tools"] is None and "could not be read" in sin_leer["reason"], sin_leer["reason"])

        lista = os.path.join(tmp, "lista.json")
        with open(lista, "w", encoding="utf-8") as handle:
            json.dump([1, 2, 3], handle)
        raro = fd.build_provenance(lista, tmp)
        check("a JSON file that is not an object is rejected with a reason",
              raro["tools"] is None and "not a JSON object" in raro["reason"], raro["reason"])

        desaparecido = fd.build_provenance(os.path.join(tmp, "nunca.json"), tmp)
        check("a --provenance path that does not exist names the path in the reason",
              "nunca.json" in desaparecido["reason"], desaparecido["reason"])

        # --- the code that produced the numbers -----------------------------------
        repo = os.path.join(tmp, "Polymnia")
        os.makedirs(os.path.join(repo, "tools"))
        with open(os.path.join(repo, "gesture_features.py"), "w", encoding="utf-8") as handle:
            handle.write("MOVEMENT_SPEED = 0.5\n")
        # delivery_code() loads tools/tool_provenance.py from the repo it is given, so the
        # test repo needs a probe of its own -- and it has to be committed, or the "clean"
        # check below would be measuring a tree that is dirty before anything happens.
        with open(os.path.join(repo, "tools", "tool_provenance.py"), "w", encoding="utf-8") as handle:
            handle.write(open(os.path.join(AQUI, "..", "tools", "tool_provenance.py"),
                              encoding="utf-8").read())
        git(["init", "-q"], repo)
        git(["config", "user.email", "t@t"], repo)
        git(["config", "user.name", "t"], repo)
        git(["add", "-A"], repo)
        git(["commit", "-qm", "base"], repo)
        identidad = fd.delivery_code(repo)
        check("the manifest records the commit of the repo holding the wrist maths",
              len(identidad["polymnia"]["commit"] or "") == 40
              and identidad["polymnia"]["dirty"] is False, identidad)
        with open(os.path.join(repo, "gesture_features.py"), "w", encoding="utf-8") as handle:
            handle.write("MOVEMENT_SPEED = 0.9   # un parche local cambia las medias\n")
        sucio = fd.delivery_code(repo)
        check("a locally patched tree is flagged dirty, not reported as clean",
              sucio["polymnia"]["dirty"] is True, sucio)

        sin_sonda = fd.delivery_code(os.path.join(tmp, "otro"))
        check("if the probe cannot be imported the reason is written down, no crash",
              sin_sonda["polymnia"]["commit"] is None and "unavailable" in sin_sonda["polymnia"]["reason"],
              sin_sonda)

    parser = fd.build_parser()
    check("the CLI exposes --provenance", "--provenance" in parser.format_help(),
          parser.format_help()[-200:])
    args = parser.parse_args(["--db_name", "d", "--tab", "t", "--output_folder", "o"])
    check("--provenance defaults to nothing rather than to a guessed file",
          args.provenance is None, args.provenance)

    ayuda = subprocess.run([sys.executable, os.path.join(AQUI, "final_dataset.py"), "--help"],
                           capture_output=True, text=True)
    check("--help documents what the flag is for",
          "Praat" in ayuda.stdout and "procedencia.json" in ayuda.stdout,
          ayuda.stdout[-200:])

    print(f"\n{done - failed}/{done} checks passed")
    if failed:
        print(f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
