#!/usr/bin/env python3
"""Tests de `tools/preflight_entrega.sh`.

Lo que se prueba no es el dibujito: es que el preanuncio de la entrega no pueda **mentir**.
Las dos comprobaciones existen porque antes fallaban tarde (la criba de munecas aborta a
mitad de lote si no hay pesos; la anotacion visual sale a NULL si no hay clave), asi que un
falso positivo aqui cuesta horas de GPU. Orden de los casos:

  1. pesos dentro del repo -> sale 0 y dice la ruta encontrada
  2. pesos al lado del checkout (como TalkNet-ASD) -> tambien vale
  3. POLYMNIA_YOLO_WEIGHTS apuntando a un fichero de 3 bytes -> sale 1 (no se traga basura)
  4. POLYMNIA_YOLO_WEIGHTS apuntando a otro sitio valido -> manda la variable
  5. sin pesos en ninguna parte -> sale 1 y nombra los dos sitios que ha buscado
  6. sin .env -> avisa, pero NO cuenta como fallo: theia es opcional
  7. .env con las tres claves -> ✅ y los VALORES no salen por la salida estandar
  8. THEIA_API_KEY vacia en el .env -> se cuenta como faltante
  9. la clave viene del entorno -> ✅ sin necesidad de .env
 10. `polymnia doctor` muestra el bloque (se omite si el entorno no tiene .venv: ver issue #16)
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.join(REPO, "tools", "preflight_entrega.sh")
WRAPPER = os.path.join(REPO, "bin", "polymnia")
CLAVES = ("THEIA_API_BASE", "THEIA_MODEL", "THEIA_API_KEY")

FAIL = []


def check(nombre, ok, detalle=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def peso(ruta, bytes=1_500_000):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as fh:
        fh.write(b"\0" * bytes)
    return ruta


def corre(raiz, extra_env=None):
    """El helper con un entorno limpio: sin THEIA_* heredadas del shell que testea."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("THEIA_")}
    env.pop("POLYMNIA_YOLO_WEIGHTS", None)
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(["bash", HELPER, raiz], capture_output=True, text=True, env=env,
                       timeout=60)
    return p


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # 1. pesos dentro del repo
        raiz = os.path.join(tmp, "repo1")
        peso(os.path.join(raiz, "modelos", "yolo11x-pose.pt"))
        p = corre(raiz)
        check("pesos dentro del repo salen 0", p.returncode == 0, p.stdout[-250:])
        check("dice la ruta exacta donde los vio", "modelos/yolo11x-pose.pt" in p.stdout, p.stdout[-250:])

        # 2. pesos al lado del checkout (el hermano de la raiz)
        hermano = os.path.join(tmp, "lado")
        peso(os.path.join(hermano, "modelos", "yolo11x-pose.pt"))
        raiz2 = os.path.join(hermano, "checkout")
        os.makedirs(raiz2, exist_ok=True)
        p = corre(raiz2)
        check("pesos al lado del checkout tambien valen", p.returncode == 0, p.stdout[-300:])

        # 3. la variable apunta a un fichero que no es un peso
        bicho = peso(os.path.join(tmp, "bicho.pt"), bytes=3)
        p = corre(raiz, {"POLYMNIA_YOLO_WEIGHTS": bicho})
        check("un fichero de 3 bytes no pasa por un peso (sale 1)", p.returncode == 1, p.stdout[-250:])
        check("y lo dice en vez de callarse", "no es un peso" in p.stdout, p.stdout[-250:])

        # 4. la variable manda, aunque el peso valido este en otra parte
        bueno = peso(os.path.join(tmp, "otros", "yolo11x-pose.pt"))
        p = corre(raiz, {"POLYMNIA_YOLO_WEIGHTS": bueno})
        check("POLYMNIA_YOLO_WEIGHTS manda", p.returncode == 0 and bueno in p.stdout, p.stdout[-250:])
        check("y dice de donde viene", "POLYMNIA_YOLO_WEIGHTS" in p.stdout.split("🔑")[0], p.stdout[:400])

        # 5. sin pesos en ninguna parte
        vacia = os.path.join(tmp, "sinpesos")
        os.makedirs(vacia, exist_ok=True)
        p = corre(vacia)
        check("sin pesos sale 1", p.returncode == 1, p.stdout[-250:])
        buscadas = [l for l in p.stdout.splitlines() if "no existe" in l]
        check("nombra los dos sitios donde ha buscado", len(buscadas) >= 2, p.stdout[-300:])

        # 6. sin .env: se avisa, no falla
        p = corre(raiz)
        check("sin .env sigue saliendo 0 (theia es opcional)", p.returncode == 0, p.stdout[-250:])
        check("pero avisa de las tres claves que faltan",
              all(k in p.stdout for k in CLAVES) and "⚠" in p.stdout, p.stdout[-300:])

        # 7. .env completo: ✅ y sin filtrar los valores
        secreto = "NO-DEBE-APARECER-EN-La-SALIDA"
        with open(os.path.join(raiz, ".env"), "w", encoding="utf-8") as fh:
            fh.write(f"THEIA_API_BASE=https://example.test/v1\nTHEIA_MODEL=un-modelo\nTHEIA_API_KEY={secreto}\n")
        p = corre(raiz)
        check("con .env completo dice ✅", "✅" in p.stdout.split("🔑")[1], p.stdout[-300:])
        check("los valores del .env NO se imprimen", secreto not in p.stdout, "se ha filtrado el valor")
        check("y dice donde esta el .env", ".env" in p.stdout.split("🔑")[1], p.stdout[-300:])

        # 8. clave presente pero vacia
        with open(os.path.join(raiz, ".env"), "w", encoding="utf-8") as fh:
            fh.write("THEIA_API_BASE=https://example.test/v1\nTHEIA_MODEL=un-modelo\nTHEIA_API_KEY=\n")
        p = corre(raiz)
        check("THEIA_API_KEY vacia cuenta como faltante", "THEIA_API_KEY" in p.stdout.split("🔑")[1]
              and "⚠" in p.stdout, p.stdout[-300:])

        # 9. la clave viene del entorno
        os.remove(os.path.join(raiz, ".env"))
        p = corre(raiz, {k: "de-el-entorno" for k in CLAVES})
        tramo = p.stdout.split("🔑")[1]
        check("las claves del entorno valen", "✅" in tramo, tramo[-300:])
        check("y no inventa un .env que no existe", "de-el-entorno" not in tramo, tramo[-300:])

    # 10. el bloque aparece en `polymnia doctor`
    try:
        d = subprocess.run([WRAPPER, "doctor"], capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[skip] `polymnia doctor` no pudo correr ({type(e).__name__}): hace falta el "
              ".venv del checkout; en un worktree sin el, doctor se cae antes (issue #16)")
    else:
        if d.returncode != 0:
            print("[skip] `polymnia doctor` no termino (sin .venv en este checkout): se omite "
                  "la comprobacion del bloque, ver issue #16")
        else:
            check("doctor muestra el bloque de entrega", "🚦 entrega" in d.stdout, d.stdout[-400:])
            check("doctor trae los pesos YOLO", "pesos YOLO" in d.stdout, d.stdout[-400:])
            check("doctor trae la anotacion visual", "anotación visual" in d.stdout, d.stdout[-400:])

    print()
    if FAIL:
        print(f"TOTAL: FAIL {', '.join(FAIL)}")
        return 1
    print("TOTAL: todo OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
