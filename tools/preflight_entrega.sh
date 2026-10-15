#!/usr/bin/env bash
# Comprobaciones que `doctor` no haciba y que solo necesita la entrega.
#
# `bin/polymnia doctor` ya mira python/uv, OpenPose, TalkNet-ASD (y sus pesos), WhisperX,
# DuckDB, Praat, R y ffmpeg. Cuando lo que vas a correr es `final_dataset/pipeline_lote.sh`
# quedan dos cosas mas, y las dos fallan tarde y cara: los pesos de la criba de munecas
# (etapa 4/10, aborta a mitad de lote) y las credenciales de la anotacion visual (etapa
# 9/10, y sin clave las 7 columnas de anotacion salen NULL con la etapa aparentando haber
# corrido).
#
# Uso: tools/preflight_entrega.sh [RAIZ]     # RAIZ = raiz de Polymnia por defecto
#
# Sale 1 solo si falta lo imprescindible (los pesos). La anotacion visual es opcional: se
# avisa, no se cuenta. Los VALORES de las claves nunca se imprimen, ni siquiera aqui.
set -uo pipefail

RAIZ="${1:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd -P)}"
faltan=0

# ------------------------------------------------- pesos de la criba de munecas
# Misma resolucion que final_dataset/pipeline_lote.sh: la variable manda; si no hay
# variable se aceptan los dos sitios donde han estado siempre los pesos (dentro del repo
# y al lado del checkout, como TalkNet-ASD). Si cambies esto, cambia aquello.
PESO="${POLYMNIA_YOLO_WEIGHTS:-}"
if [ -n "$PESO" ]; then
  CANDIDATOS=("$PESO")
else
  CANDIDATOS=("${RAIZ}/modelos/yolo11x-pose.pt")
  HERMANO="$(cd "${RAIZ}/.." 2>/dev/null && pwd -P)"
  [ -n "$HERMANO" ] && CANDIDATOS+=("${HERMANO}/modelos/yolo11x-pose.pt")
fi

encontrado=""
for c in "${CANDIDATOS[@]}"; do
  [ -f "$c" ] || continue
  # 1 MB es el suelo para no tragar ni un fichero vacio ni una descarga cortada: el peso
  # real de YOLO11x-pose pesa cientos de MB.
  [ "$(wc -c <"$c")" -gt 1000000 ] && encontrado="$c" && break
done

echo "    🥊 pesos YOLO (criba de muñecas, etapa 4/10)"
if [ -n "$encontrado" ]; then
  echo "    ✅ ${encontrado} ($(( $(wc -c <"$encontrado") / 1000000 )) MB)"
  [ -n "${POLYMNIA_YOLO_WEIGHTS:-}" ] && echo "       (viene de POLYMNIA_YOLO_WEIGHTS)"
else
  for c in "${CANDIDATOS[@]}"; do
    if [ -f "$c" ]; then
      echo "    ❌ ${c} existe pero pesa $(wc -c <"$c") bytes: no es un peso de YOLO11x-pose"
    else
      echo "    ❌ ${c} no existe"
    fi
  done
  echo "       → exporta POLYMNIA_YOLO_WEIGHTS=/ruta/yolo11x-pose.pt o pasa --peso al pipeline"
  faltan=1
fi

# ------------------------------------------------ credenciales de anotacion visual
# theia.py lee THEIA_API_BASE / THEIA_MODEL / THEIA_API_KEY del entorno o de un .env en
# la raiz de Polymnia. Es la etapa opcional del pipeline, asi que la ausencia se avisa.
ENVF="${RAIZ}/.env"
sin_clave=""
for k in THEIA_API_BASE THEIA_MODEL THEIA_API_KEY; do
  if [ -n "${!k:-}" ]; then continue; fi                     # ya viene del entorno
  # -E y [^[:space:]] a proposito: en BRE un "+" es literal (la primera version de esta linea
  # no encontraba nunca una clave), y un valor de un solo espacio no es un valor.
  if [ -f "$ENVF" ] && grep -qsE "^${k}=[^[:space:]]" "$ENVF"; then continue; fi
  sin_clave="${sin_clave} ${k}"
done

echo "    🔑 anotación visual (theia, etapa 9/10 — opcional)"
if [ -z "$sin_clave" ]; then
  if [ -f "$ENVF" ]; then
    echo "    ✅ THEIA_API_BASE THEIA_MODEL THEIA_API_KEY (${ENVF}, valores no imprimidos)"
  else
    echo "    ✅ THEIA_API_BASE THEIA_MODEL THEIA_API_KEY (del entorno)"
  fi
else
  echo "    ⚠️  faltan:${sin_clave}"
  echo "       sin ellas la etapa theia se omite y las 7 columnas de anotacion salen NULL."
  echo "       Se ponen en ${ENVF} o en el entorno (no hace falta commitear nada)."
fi

exit "$faltan"
