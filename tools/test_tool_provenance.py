#!/usr/bin/env python3
"""Tests for tools/tool_provenance.py (Polymnia #14).

The point of the tool is that it never invents an identity, so most of these tests are
about the *missing* case: what a probe must do when a tool is not there, says something
that is not a version, or is a file nobody ever versioned.

Nothing here needs GPU, network or the real corpus. Real tools are probed only through
temp files and temp scripts, so the tests pass the same on a laptop as on the GPU box.

Output: one line per check; exit 0 if all pass, 1 otherwise.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import tool_provenance as tp  # noqa: E402

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


def fake_tool(dirs, name, output, exit_code=0):
    """A script that answers like a tool would, so version parsing is tested for real."""
    path = os.path.join(dirs, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"#!/bin/sh\nprintf '%s\\n' '{output}'\nexit {exit_code}\n")
    os.chmod(path, 0o755)
    return path


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # --- a tool that is simply not there ------------------------------------
        missing = tp.binary(["no-such-tool-anything-xyz", "--version"])
        check("a tool that is not on PATH is recorded as missing, not as empty",
              missing["version"] is None and "not on PATH" in missing["reason"], missing)

        # --- a tool whose answer is not a version -------------------------------
        old_path = os.environ["PATH"]
        os.environ["PATH"] = tmp + os.pathsep + old_path
        try:
            fake_tool(tmp, "saysaname", "openpose.bin")
            nodigits = tp.binary(["saysaname", "-version"])
            check("an answer without a single digit is not recorded as a version",
                  nodigits["version"] is None and "sha256" in nodigits["reason"], nodigits)

            fake_tool(tmp, "saysversion", "Praat 6.4.30 (April 29 2025)")
            digits = tp.binary(["saysversion", "--version"])
            check("a real version line is quoted from the tool, not summarised",
                  digits["version"] == "Praat 6.4.30 (April 29 2025)", digits)

            fake_tool(tmp, "saysnothing", "", exit_code=3)
            quiet = tp.binary(["saysnothing", "--version"])
            check("a tool that exits silently is missing with the exit code as reason",
                  quiet["version"] is None and "exited 3" in quiet["reason"], quiet)
        finally:
            os.environ["PATH"] = old_path

        # --- things nobody versioned --------------------------------------------
        pesos = os.path.join(tmp, "yolo-fake.pt")
        with open(pesos, "wb") as handle:
            handle.write(b"pesos de prueba" * 1000)
        esperado = hashlib.sha256(open(pesos, "rb").read()).hexdigest()
        identidad = tp.sha256_of(pesos)
        check("weights are identified by size + sha256 (no version exists to quote)",
              identidad["sha256"] == esperado and identidad["bytes"] == 15000, identidad)
        ausente = tp.sha256_of(os.path.join(tmp, "no-existe.pt"))
        check("missing weights say so instead of hashing nothing",
              ausente["sha256"] is None and "not a file" in ausente["reason"], ausente)

        # --- code identity, including the part people forget: dirty trees --------
        repo = os.path.join(tmp, "repo")
        os.makedirs(repo)
        git(["init", "-q"], repo)
        git(["config", "user.email", "t@t"], repo)
        git(["config", "user.name", "t"], repo)
        open(os.path.join(repo, "a.py"), "w").write("x = 1\n")
        git(["add", "a.py"], repo)
        git(["commit", "-qm", "uno"], repo)
        limpio = tp.git_identity(repo)
        check("a clean tree records the commit and dirty=False",
              limpio["commit"] == git(["rev-parse", "HEAD"], repo) and limpio["dirty"] is False, limpio)
        # A scratch file must NOT flip the flag: importing the probe itself drops a
        # __pycache__ next to it, and a tool that reports a clean repo dirty is worse than one
        # that says nothing. Untracked files are counted, never trusted.
        open(os.path.join(repo, "notas-scratch.py"), "w").write("y = 2\n")
        rayas = tp.git_identity(repo)
        check("an untracked scratch file is counted but does not make the tree dirty",
              rayas["dirty"] is False and rayas["untracked"] == 1, rayas)
        with open(os.path.join(repo, "a.py"), "w") as handle:
            handle.write("x = 99   # local patch to tracked code\n")
        sucio = tp.git_identity(repo)
        check("a local edit to tracked code flips dirty and counts the paths",
              sucio["dirty"] is True and sucio["modified_paths"] == 1, sucio)
        fuera = tp.git_identity(os.path.join(tmp, "nada"))
        check("a directory that is not a clone says so, without crashing",
              fuera["commit"] is None and "not a directory" in fuera["reason"], fuera)

        # --- configuration, and the secret that must never appear ---------------
        entorno = os.path.join(tmp, "conf")
        os.makedirs(entorno)
        with open(os.path.join(entorno, ".env"), "w", encoding="utf-8") as handle:
            handle.write('# credentials\nTHEIA_API_BASE="https://api.example.test/v1"\n'
                         'THEIA_MODEL=theia-vlm-7b\nTHEIA_API_KEY=SECREDO-NUNCA-GRABADO\n')
        check(".env is read for configuration values",
              tp.env_or_dotenv(entorno, "THEIA_MODEL") == "theia-vlm-7b",
              tp.env_or_dotenv(entorno, "THEIA_MODEL"))
        check("quotes around a .env value are stripped",
              tp.env_or_dotenv(entorno, "THEIA_API_BASE") == "https://api.example.test/v1")
        os.environ["THEIA_MODEL"] = "from-environment"
        check("the real environment wins over .env (that is what the stage actually sees)",
              tp.env_or_dotenv(entorno, "THEIA_MODEL") == "from-environment")
        del os.environ["THEIA_MODEL"]
        config = tp.theia_config(entorno)
        check("the model and base URL are recorded",
              config["model"] == "theia-vlm-7b" and config["api_base"].startswith("https://"), config)
        check("the API key never reaches the JSON",
              "SECREDO-NUNCA-GRABADO" not in json.dumps(config), config)

        # --- the whole block -----------------------------------------------------
        bloque = tp.collect(AQUI + "/..", pesos, with_gpu=False)
        check("the block is JSON serialisable and has the groups the docs promise",
              set(["code", "binaries", "python", "weights", "services"]) <= set(bloque),
              sorted(bloque))
        check("python and its numeric stack are recorded from the running interpreter",
              bloque["python"]["version"] == sys.version.split()[0]
              and "pandas" in bloque["python"]["modules"], bloque["python"]["version"])
        check("the weights block carries the hash of the file the pipeline will use",
              bloque["weights"]["yolo_pose"]["sha256"] == esperado, bloque["weights"]["yolo_pose"])
        sin_pesos = tp.collect(AQUI + "/..", os.path.join(tmp, "iao.pt"), with_gpu=False)
        check("a missing weights file is recorded as missing inside the block too",
              sin_pesos["weights"]["yolo_pose"]["sha256"] is None
              and "not a file" in sin_pesos["weights"]["yolo_pose"]["reason"])
        otra = tp.collect(AQUI + "/..", pesos, with_gpu=False)
        b1, b2 = dict(bloque), dict(otra)
        b1.pop("recorded_at"), b2.pop("recorded_at")
        check("two probes of the same machine agree (only the timestamp moves)",
              json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True))
        check("torch is not imported unless asked for, so the delivery builder stays quick",
              "gpu" not in bloque["python"])

        # --- writing it out ------------------------------------------------------
        destino = os.path.join(tmp, "informes", "procedencia.json")
        proceso = subprocess.run([sys.executable, os.path.join(AQUI, "tool_provenance.py"),
                                  "--out", destino, "--weights", pesos],
                                 capture_output=True, text=True)
        escrito = json.load(open(destino, encoding="utf-8"))
        check("--out creates parent directories and valid JSON",
              proceso.returncode == 0 and escrito["weights"]["yolo_pose"]["sha256"] == esperado,
              proceso.stdout + proceso.stderr)
        check("the file is written atomically (no .tmp left behind)",
              not os.path.exists(destino + ".tmp"))
        check("the summary names what could not be identified",
              "no version recorded for" in proceso.stdout, proceso.stdout)

    print(f"\n{done - failed}/{done} checks passed")
    if failed:
        print(f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
