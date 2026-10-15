#!/usr/bin/env bash
# Las pruebas de este directorio.
#
# `polymnia test` recorre tools/test_*.py; estas dos no van ahí porque hablan del contrato
# de la ENTREGA (qué columnas salen, qué significa un NULL), no de un tool suelto.
set -uo pipefail

AQUI="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)"
RAIZ="$(cd "$AQUI/.." && pwd -P)"
if [ -n "${POLYMNIA_VENV:-}" ]; then PY="$POLYMNIA_VENV/bin/python"        # explícito
elif [ -x "$RAIZ/.venv/bin/python" ]; then PY="$RAIZ/.venv/bin/python"      # dentro del repo
elif [ -x "$RAIZ/Polymnia/.venv/bin/python" ]; then PY="$RAIZ/Polymnia/.venv/bin/python"
else PY="$(command -v python3 || true)"; fi                                   # último recurso
[ -n "$PY" ] || { echo "❌ no encuentro ningún python (pasa POLYMNIA_VENV=…)" >&2; exit 1; }

echo "python: $PY"

fallos=0
corre() { echo; echo "── ${*/$AQUI\//}"; "$@" || { echo "   ❌ falla"; fallos=$((fallos + 1)); }; }

corre bash -n "$AQUI/pipeline_lote.sh"
corre "$PY" "$AQUI/test_final_dataset_voz.py"

# El diccionario lee el bloque extendido de gesture_features.py. Si tu Polymnia no lo tiene
# todavia, la prueba no puede correr: se anuncia y se omite, no se cuela como fallo (una
# copia limpia de main no esta rota, le falta una pieza).
if "$PY" - "$RAIZ" <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, sys.argv[1])
import gesture_features as gf
sys.exit(0 if hasattr(gf, "EXTENDED_FEATURE_UNITS") else 1)
PYEOF
then
  corre "$PY" "$AQUI/test_diccionario_columnas.py"
else
  echo
  echo "[skip] test_diccionario_columnas.py: el diccionario resuelve las familias EXTENDED_*"
  echo "       de gesture_features.py y tu copia no las tiene (necesita FEATURE_VERSION 3)."
  echo "       Se omite; no es un fallo de estas pruebas."
fi

echo
if [ "$fallos" -eq 0 ]; then echo "TOTAL: todo OK"; else echo "TOTAL: $fallos prueba(s) fallida(s)"; fi
exit $((fallos > 0))
