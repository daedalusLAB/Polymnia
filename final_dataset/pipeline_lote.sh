#!/usr/bin/env bash
# pipeline_lote.sh — de un TSV de NewsScape a un dataset final, dejando todo el rastro
# para que lo depures a mano.
#
#   final_dataset/pipeline_lote.sh --tsv clips_nuevos.txt --lote lote3
#
# Al terminar, ./lote3/ contiene:
#   tsv/                                copia del TSV + SHA-256 (procedencia)
#   videos/raw/                         clips descargados
#   videos/masked/                      TalkNet-ASD (solo el hablante activo)
#   videos/1_person/                    admitidos por la criba de muñecas = lo que ve OpenPose
#   videos/descartados/<motivo>/        lo que se cayó por el camino, agrupado por motivo
#   dataset/OpenPose/                   JSON de pose por frame
#   dataset/{n}_persons/parquet_files/  un Parquet por clip (lo que clasifica el R)
#   munecas.csv                         score de muñecas de TODOS los clips, ordenados
#   <lote>.duckdb                       la base larga `multi_data` del pipeline
#   final/                              final_frames + final_videos (parquet y csv)
#                                       + manifest.json + DATASET.md (el README generado)
#   DICCIONARIO_COLUMNAS.md             qué significa cada columna, con la cobertura de ESTE lote
#   logs/                               la salida completa de cada etapa
#   informes/                           conteos, vídeos raros, cobertura de muñecas y
#                                       DICCIONARIO_COLUMNAS.csv (el diccionario en tabla)
#   RENDIMIENTO.md                      el embudo real de esta tanda, con costes
#
# Cada etapa comprueba SU propia salida. En este pipeline `exit 0` no significa nada:
# TalkNet-ASD rechaza clips sin generar fichero y la descarga solo mira el tamaño.
#
# Etapas (para reanudar con --desde / --hasta):
#   tsv descarga asd preseleccion openpose limpieza voz merge theia final
set -euo pipefail

# ------------------------------------------------------------------ opciones
RAIZ="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd -P)"

# Este directorio puede vivir DENTRO de Polymnia (final_dataset/ en el repo) o AL LADO
# de una copia de Polymnia. Se detecta; no hay que configurar nada.
if [ -x "$RAIZ/bin/polymnia" ]; then POLY_ROOT="$RAIZ"; else POLY_ROOT="$RAIZ/Polymnia"; fi
POLY="$POLY_ROOT/bin/polymnia"             # pone venv, R, Praat y scenedetect en el PATH
PY="$POLY_ROOT/.venv/bin/python"
DUCK="$POLY_ROOT/tools/bin/duckdb"
FDLOG="$POLY_ROOT/failed_downloads.log"    # download_clips lo escribe en el cwd del repo

# Los filtros de calidad viven en tools/ del repo. Se aceptan tambien en este
# directorio, para quien trabaje con una copia suelta del pipeline.
FILTROS="$POLY_ROOT/tools"
[ -f "$FILTROS/wrist_prescreen.py" ] || FILTROS="$RAIZ/final_dataset"

TSV=""; LOTE=""; OFFSET=2.0; SEARCHTERM="beginning_to_end"
MIN_SP_ANY=0.6; conf=0.7; FRAMES=15; DEVICE=0
# Los pesos no van en el repo. Manda la variable; si no, se aceptan los dos sitios donde
# han estado siempre (dentro del repo y al lado del checkout, como TalkNet-ASD).
# tools/preflight_entrega.sh usa la misma lista: si cambias uno, cambia el otro.
PESO="${POLYMNIA_YOLO_WEIGHTS:-}"
if [ -z "$PESO" ]; then
  for candidato in "$RAIZ/modelos/yolo11x-pose.pt" "$(cd "$RAIZ/.." 2>/dev/null && pwd -P)/modelos/yolo11x-pose.pt"; do
    if [ -f "$candidato" ]; then PESO="$candidato"; break; fi
  done
  # sin ninguno, se guarda la ruta propia del repo: es la que sale en el mensaje de error
  PESO="${PESO:-$RAIZ/modelos/yolo11x-pose.pt}"
fi
LIMIT=""; GATE=0; MIN_VISIBLE=80; EXTENDED=1; README=""; SKIP_THEIA=0
LIMPIEZA=1; CRITERIO_LIMPIEZA=todo; DICCIONARIO=1
MAX_SIN_VOZ_PCT=5
MAX_SIN_POSE_PCT=5
THEIA_WORKERS=${POLYMNIA_THEIA_WORKERS:-4}
TSV_SET=0; OFFSET_SET=0; SEARCHTERM_SET=0; LIMIT_SET=0; PRINT_CONFIG=0
DESDE=tsv; HASTA=final; FORCE=0

uso() {
  # El bloque de cabecera, entero, sin numeros de linea magicos: se corta solo donde acaba el
  # comentario (con sed '2,26p' cada linea anadida al arbol de arriba era una linea que faltaba).
  awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
  cat <<EOF

Opciones:
  --tsv F            (obligatorio) tab de NewsScape, sin cabecera, >=6 campos por línea
  --lote NOMBRE      carpeta de trabajo (por defecto: lote_AAAAMMDD_HHMM)
  --offset N         holgura por lado, misma en TODO el pipeline (defecto 2.0)
  --searchterm T     término del nombre de fichero (defecto beginning_to_end)
  --min-sp-any F     umbral de admisión de la criba de muñecas sobre sp_any (defecto 0.6)
                     sp_any = la MEJOR de las dos muñecas del hablante. Se admitió así
                     desde la auditoría de FREQ025: exigir las dos en el mismo frame
                     (sp_both >= 0.8) se dejó un clip que OpenPose valoró luego a 91,9 % /
                     100 % con gate=accepted. El umbral viejo sigue siendo el de --yolo-conf.
  --yolo-conf F      confianza de keypoint COCO del pre-filtro (defecto 0.7, calibrado)
  --frames N         frames muestreados por clip en la criba (defecto 15)
  --device N         GPU para ASD/OpenPose/criba (defecto 0)
  --peso RUTA        pesos yolo11-pose (defecto modelos/yolo11x-pose.pt)
  --limit N          usa solo las N primeras líneas del TSV (prueba barata)
  --gate             activa el gate de visibilidad en OpenPose (por defecto NO, ver nota)
  --min-visible-percent N   cobertura exigida por el gate (defecto 80, solo con --gate)
  --no-extended      dataset final en FEATURE_VERSION 2 (sin el bloque 3)
  --readme F         fichero a publicar como DATASET.md dentro de final/ en vez del generado
                     (cuidado: un README a mano trae números de otra tanda, ver nota)
  --skip-theia       no pedir anotaciones visuales a Theia
  --sin-diccionario  no generar DICCIONARIO_COLUMNAS ni README (final/ se queda sin DATASET.md)
  --sin-limpiar      NO limpiar candidatos fantasma (por defecto SÍ: sin eso, un solo frame
                     ambiguo manda el clip a 2_persons y el dataset sale vacío)
  --criterio-limpieza C  todo|torso|negro|nada (defecto todo; nada = solo informar)
  --max-sin-voz-pct F  umbral de clips descartados por falta de voz/prosodia que para el lote
                     (defecto 5; 0 lo desactiva). El clip descartado NO se publica y queda en
                     videos/descartados/sin_voz/ e informes/descartados_sin_voz.tsv
  --max-sin-pose-pct F umbral de clips que OpenPose dejó sin una sola detección (defecto 5;
                     0 lo desactiva). Se apartan en videos/descartados/sin_pose/ y
                     informes/descartados_sin_pose.tsv; solo aborta si no queda ninguno
  --theia-workers N  peticiones simultaneas a Theia (defecto 4, o \$POLYMNIA_THEIA_WORKERS).
                     Con varios lotes a la vez, baja a 1-2: el endpoint devuelve 429
  --desde ETAPA      reanudar desde una etapa asumiendo que las anteriores ya están
  --hasta ETAPA      parar después de una etapa
  --print-config     imprime el tsv, offset, searchterm y limite resueltos y sale (0),
                     sin crear ni tocar nada
  --force            permitir reutilizar una carpeta de lote ya existente
                     (lo que se sustituye se aparta a informes/_anterior/<hora>/, no se borra)

Con lo mínimo, una tanda completa es:
  polymnia lote --lote lotes/FREQ013
El TSV se busca por el nombre del lote (MULTIFLOW_TSVs/*/Recurrence_Tabulations/<LOTE>_Tab.txt,
POLYMNIA_TSV_DIR, la copia del propio lote) y el searchterm se deduce del nombre del lote.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tsv) TSV="$2"; TSV_SET=1; shift 2 ;;
    --lote) LOTE="$2"; shift 2 ;;
    --offset) OFFSET="$2"; OFFSET_SET=1; shift 2 ;;
    --searchterm) SEARCHTERM="$2"; SEARCHTERM_SET=1; shift 2 ;;
    --min-sp-any) MIN_SP_ANY="$2"; shift 2 ;;
    # --min-sp-both se retira, y NO se re-significa: quien lo pasa esperaba la regla estricta
    # de las dos muñecas, y aceptarselo interpretándolo como sp_any ensancharía la admisión
    # sin que nadie lo haya pedido. El ancho de admisión cambia qué clips hay en el dataset.
    --min-sp-both)
      echo "❌ --min-sp-both ya no existe: la criba admite por sp_any (la MEJOR de las dos" >&2
      echo "   muñecas del hablante), no por las dos a la vez. Usa --min-sp-any F." >&2
      echo "   Si de verdad quieres la regla estricta antigua no hay flag: sp_both sigue" >&2
      echo "   escribiendose en munecas.csv, filtra con él." >&2
      exit 1 ;;
    --yolo-conf) conf="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --peso) PESO="$2"; shift 2 ;;
    --limit) LIMIT="$2"; LIMIT_SET=1; shift 2 ;;
    --gate) GATE=1; shift ;;
    --min-visible-percent) MIN_VISIBLE="$2"; shift 2 ;;
    --no-extended) EXTENDED=0; shift ;;
    --readme) README="$2"; shift 2 ;;
    --skip-theia) SKIP_THEIA=1; shift ;;
    --sin-diccionario) DICCIONARIO=0; shift ;;
    --sin-limpiar) LIMPIEZA=0; shift ;;
    --criterio-limpieza) CRITERIO_LIMPIEZA="$2"; shift 2 ;;
    --max-sin-voz-pct) MAX_SIN_VOZ_PCT="$2"; shift 2 ;;
    --max-sin-pose-pct) MAX_SIN_POSE_PCT="$2"; shift 2 ;;
    --theia-workers) THEIA_WORKERS="$2"; shift 2 ;;
    --desde) DESDE="$2"; shift 2 ;;
    --hasta) HASTA="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --print-config) PRINT_CONFIG=1; shift ;;
    -h|--help) uso; exit 0 ;;
    *) echo "❌ opción desconocida: $1" >&2; uso; exit 1 ;;
  esac
done

[ -f "$TSV" ] || [ -z "$TSV" ] || { echo "❌ no existe el TSV: $TSV" >&2; exit 1; }
[ -z "$TSV" ] || TSV="$(readlink -f "$TSV")"
for ruta in "$POLY" "$PY" "$DUCK"; do
  [ -x "$ruta" ] || { echo "❌ falta $ruta" >&2; exit 1; }
done

ORDEN=(tsv descarga asd preseleccion openpose limpieza voz merge theia final)
idx() { local i; for i in "${!ORDEN[@]}"; do [ "${ORDEN[i]}" = "$1" ] && { echo "$i"; return; }; done; echo -1; }
I_DESDE=$(idx "$DESDE"); I_HASTA=$(idx "$HASTA")
[ "$I_DESDE" -ge 0 ] && [ "$I_HASTA" -ge 0 ] || { echo "❌ etapa desconocida (--desde/--hasta)" >&2; exit 1; }
hacer() { local i; i=$(idx "$1"); [ "$i" -ge "$I_DESDE" ] && [ "$i" -le "$I_HASTA" ]; }

[ -n "$LOTE" ] || LOTE="lote_$(date +%Y%m%d_%H%M)"
case "$LOTE" in /*) OUT="$LOTE" ;; *) OUT="$PWD/$LOTE" ;; esac
# el nombre de la base sale de la carpeta, no de --lote: con --lote /ruta/absoluta,
# "$OUT/${LOTE_NOMBRE}.duckdb" habria dado /ruta/absoluta//ruta/absoluta.duckdb
LOTE_NOMBRE="$(basename "$OUT")"
if [ -e "$OUT" ] && [ "$FORCE" -eq 0 ] && [ "$I_DESDE" -eq 0 ] && [ "$PRINT_CONFIG" -eq 0 ]; then
  echo "❌ $OUT ya existe. Usa otro --lote, o --force para reutilizarla." >&2; exit 1
fi
case "$OUT" in
  */entrega_*|*/entrega_*) echo "❌ no montes un lote dentro de una entrega ($OUT)" >&2; exit 1 ;;
esac

# ------------------------------------------- parametros recordados y derivados del lote
# Una sola orden basta: `--lote lotes/FREQ013`. Lo que no se pasa se deduce del nombre del
# lote; y si el lote ya existia, se recupera de informes/parametros.json (lo escribe la
# etapa 1) para que un reanudado no salga con otro --offset que el de la descarga.
PARAMS="$OUT/informes/parametros.json"
param() {  # param <clave>: valor guardado, o nada
  [ -f "$PARAMS" ] || return 0
  "$PY" -c 'import json,sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit
v = d.get(sys.argv[2])
print("" if v is None else v)' "$PARAMS" "$1" 2>/dev/null || true
}
if [ -f "$PARAMS" ]; then
  if [ "$OFFSET_SET" -eq 0 ]; then v="$(param offset)"; if [ -n "$v" ]; then OFFSET="$v"; OFFSET_SET=1; fi; fi
  if [ "$SEARCHTERM_SET" -eq 0 ]; then v="$(param searchterm)"; if [ -n "$v" ]; then SEARCHTERM="$v"; SEARCHTERM_SET=1; fi; fi
  if [ "$LIMIT_SET" -eq 0 ]; then v="$(param limit)"; if [ -n "$v" ]; then LIMIT="$v"; LIMIT_SET=1; fi; fi
fi
if [ -z "$TSV" ]; then
  # Los tabulados viven en MULTIFLOW_TSVs/<campaña>/Recurrence_Tabulations/: se busca en
  # cualquiera de las campañas (tanvir, ayesha, ...) en vez de clavar una carpeta.
  for candidato in "$OUT/tsv/${LOTE_NOMBRE}_Tab.txt" \
                   "${POLYMNIA_TSV_DIR:+${POLYMNIA_TSV_DIR}/${LOTE_NOMBRE}_Tab.txt}" \
                   $PWD/MULTIFLOW_TSVs/*/Recurrence_Tabulations/${LOTE_NOMBRE}_Tab.txt \
                   $RAIZ/../MULTIFLOW_TSVs/*/Recurrence_Tabulations/${LOTE_NOMBRE}_Tab.txt \
                   $RAIZ/MULTIFLOW_TSVs/*/Recurrence_Tabulations/${LOTE_NOMBRE}_Tab.txt \
                   $PWD/MULTIFLOW_TSVs/*/${LOTE_NOMBRE}_Tab.txt \
                   "$RAIZ/../MULTIFLOW_TSVs/${LOTE_NOMBRE}_Tab.txt" \
                   "$RAIZ/MULTIFLOW_TSVs/${LOTE_NOMBRE}_Tab.txt"; do
    if [ -n "$candidato" ] && [ -f "$candidato" ]; then TSV="$candidato"; break; fi
  done
fi
if [ "$SEARCHTERM_SET" -eq 0 ]; then
  SEARCHTERM="$(printf '%s' "$LOTE_NOMBRE" | tr 'A-Z' 'a-z')"
fi
if [ -z "$TSV" ]; then
  echo "❌ no encuentro el TSV de $LOTE_NOMBRE: pásalo con --tsv" >&2
  echo "   (busqué en \$OUT/tsv, \$POLYMNIA_TSV_DIR, ./MULTIFLOW_TSVs y ../MULTIFLOW_TSVs)" >&2
  exit 1
fi
[ -f "$TSV" ] || { echo "❌ no existe el TSV: $TSV" >&2; exit 1; }
TSV="$(readlink -f "$TSV")"

if [ "$PRINT_CONFIG" -eq 1 ]; then
  echo "lote:       $LOTE_NOMBRE"
  echo "tsv:        $TSV"
  echo "offset:     $OFFSET"
  echo "searchterm: $SEARCHTERM"
  echo "limit:      ${LIMIT:-(sin límite)}"
  echo "theia_workers: $THEIA_WORKERS"
  echo "sin_pose_max_pct: $MAX_SIN_POSE_PCT"
  echo "etapas:     ${ORDEN[*]:I_DESDE:I_HASTA-I_DESDE+1}"
  exit 0
fi

mkdir -p "$OUT"/{tsv,logs,informes} "$OUT"/videos/{raw,masked,1_person} \
       "$OUT"/videos/descartados/{asd_sin_mascara,preseleccion_sp_bajo,clasificado_multipersona,sin_voz}

# ------------------------------------------------------------------ utilidades
INICIO_TOTAL=$(date +%s)
MARCA=$(date +%s)
etapa() {
  echo; echo "══════════════════════════════════════════════════════════════"
  echo "▶ $1   ($(date '+%H:%M:%S'))"
  echo "══════════════════════════════════════════════════════════════"
  MARCA=$(date +%s)
}
coste() { printf '   ⏱  %s: %s s\n' "$1" "$(( $(date +%s) - MARCA ))"; }
conteo() {                       # reanudar una etapa no debe duplicar sus filas.
  local etapa="$1" metrica="$2" valor="$3" f="$OUT/informes/conteos.tsv" tmp
  # la clave es (etapa, metrica), no la etapa: una etapa registra varias metricas
  tmp="$(mktemp)"
  { [ -f "$f" ] && awk -F'\t' -v e="$etapa" -v m="$metrica" \
        'BEGIN{ORS="\n"} !($1 == e && $2 == m)' "$f"
    printf '%s\t%s\t%s\n' "$etapa" "$metrica" "$valor"; } > "$tmp"
  mv "$tmp" "$f"
}
APARTADO=""
apartar() {                      # apartar <ruta>: mueve un derivado viejo a informes/_anterior/
  [ -e "$1" ] || return 0        # (nada se borra; reanudar no debe mezclar dos pasadas)
  local rel="${1#"$OUT"/}" destino
  APARTADO="${APARTADO:-$OUT/informes/_anterior/$(date +%H%M%S)}"
  destino="$APARTADO/$rel"
  mkdir -p "$(dirname "$destino")"
  mv "$1" "$destino"
  echo "   apartado $rel -> ${destino#"$OUT"/}"
}
corre() {                       # corre <log> <cmd...>  (el pipe + pipefail preserva el fallo)
  local log="$1"; shift
  echo "   · log: ${log#$OUT/}"
  "$@" 2>&1 | tee "$log"
}
abortar() {
  echo >&2
  echo "❌ $1" >&2
  echo "   Etiqueta: $2   (reanuda con --desde $2 --force)" >&2
  echo "   Logs en $OUT/logs/" >&2
  exit 1
}
tsv_filtrado() {                       # el TSV efectivo, con --limit aplicado
  if [ -n "$LIMIT" ]; then head -n "$LIMIT" "$TSV"; else cat "$TSV"; fi
}

verificar_holgura() {                  # mide en la duracion del clip con que holgura se corto
  # La holgura no se puede deducir del nombre del fichero (lo escribe el propio
  # descargador con esa holgura). Se mide con ffprobe: duracion - ventana del TSV.
  # Se llama tambien en la etapa 1 para que un reanudado con --desde asd no se cuele
  # con clips cortados a otra holgura que la de esta pasada.
  [ -f "$OUT/tsv/efectivo.tsv" ] || return 0
  [ -d "$OUT/videos/raw" ] || return 0
  if [ -n "$(find "$OUT/videos/raw" -name '*.mp4' -print -quit 2>/dev/null)" ]; then
    corre "$OUT/logs/holgura.log" "$POLY" verificar-holgura \
          --videos "$OUT/videos/raw" --tab "$OUT/tsv/efectivo.tsv" \
          --offset "$OFFSET" --searchterm "$SEARCHTERM" && return 0
    # aqui no vale el balcon de abortar: reanudar no arregla un corte equivocado,
    # los clips ya estan en disco cortados como estan
    echo >&2
    echo "❌ los clips de $OUT/videos/raw no estan cortados a $OFFSET s por lado" >&2
    echo "   arriba esta la holgura medida. Elige y repite:" >&2
    echo "   a) construir con la holgura real: --offset <esa medida>  (no vuelve a bajar nada)" >&2
    echo "   b) recortar de verdad: vacia videos/raw y repite con el offset que quieras" >&2
    echo "   (un --desde no arregla esto; ver logs/holgura.log)" >&2
    exit 1
  fi
}

echo "lote:            $OUT"
echo "tsv:             $TSV"
echo "offset:          $OFFSET   searchterm: $SEARCHTERM"
echo "criba muñecas:   sp_any >= $MIN_SP_ANY  (la mejor de las dos muñecas; conf COCO $conf, $FRAMES frames, GPU $DEVICE)"
echo "gate visibilidad: $([ "$GATE" -eq 1 ] && echo "ON ($MIN_VISIBLE%)" || echo "OFF (nada se descarta por multi_person)")"
echo "bloque extendido: $([ "$EXTENDED" -eq 1 ] && echo "sí (FEATURE_VERSION 3)" || echo "no (v2)")"
echo "etapas:          ${ORDEN[*]:I_DESDE:I_HASTA-I_DESDE+1}"

# ------------------------------------------------- tool provenance (issue #14)
# Written once at run start, outside the stage list: a resumed run still has to say which
# tools it used. It goes through the wrapper because Praat and scenedetect only appear on
# the PATH the wrapper builds. Failing here is reported and never fatal: not knowing the
# versions is bad, aborting a six-hour lot over a version string is worse.
PROCEDENCIA="$OUT/informes/procedencia.json"
corre "$OUT/logs/00_procedencia.log" "$POLY" python "$POLY_ROOT/tools/tool_provenance.py" \
      --out "$PROCEDENCIA" --weights "$PESO" --root "$POLY_ROOT" --gpu \
  || echo "   ⚠️  provenance could not be written: the delivery will record that it is missing"

# ============================================================ 1. el TSV
if hacer tsv; then
  etapa "1/10  validar el tabulado"
  cp -f "$TSV" "$OUT/tsv/$(basename "$TSV")"
  tsv_filtrado > "$OUT/tsv/efectivo.tsv"
  (cd "$OUT/tsv" && sha256sum "efectivo.tsv" "$(basename "$TSV")" > SHA256SUMS)
  NLINES=$(wc -l < "$OUT/tsv/efectivo.tsv")
  if [ "$NLINES" -eq 0 ]; then abortar "el TSV no tiene líneas" tsv; fi
  corre "$OUT/logs/01_check_tab.log" "$POLY" check-tab "$OUT/tsv/efectivo.tsv" \
        --offset "$OFFSET" --show 6 || true
  # el divisor del msec depende del nombre del programa: 1000 si empieza por 2017, si no 100
  OUT_DIR="$OUT" "$PY" - <<'PY'
import os, collections
ruta = os.path.join(os.environ["OUT_DIR"], "tsv", "efectivo.tsv")
divisores = collections.Counter(); dur = []; malas = []
for i, linea in enumerate(open(ruta)):
    p = linea.rstrip("\n").split("\t")
    if len(p) < 6:
        malas.append((i + 1, len(p))); continue
    nombre = os.path.basename(p[0]).replace(".txt", "")
    d = 1000 if nombre.startswith("2017") else 100
    divisores[d] += 1
    dur.append((int(p[4]) + int(p[5]) / d) - (int(p[2]) + int(p[3]) / d))
print(f"   líneas leídas: {sum(divisores.values())}, incompletas: {len(malas)}")
print(f"   divisor del msec: {dict(divisores)}  <- 1000 solo si el programa empieza por 2017")
if dur:
    dur.sort()
    print(f"   duración marcada por el TSV: min {dur[0]:.2f}s  mediana {dur[len(dur)//2]:.2f}s  max {dur[-1]:.2f}s")
    print(f"   clip real = eso ± offset; si ves duraciones de minutos, el divisor no es el que creías")
if malas[:5]:
    print(f"   ⚠️  líneas con pocos campos: {malas[:5]}")
PY
  conteo tsv lineas_tsv "$NLINES"
  # Parametros de ESTA pasada: la etapa final y cualquier reanudado los recuperan de aqui.
  OUT_DIR="$OUT" P_TSV="$TSV" P_OFFSET="$OFFSET" P_SEARCHTERM="$SEARCHTERM" P_LIMIT="$LIMIT" "$PY" - <<'PY'
import hashlib, json, os
out = os.environ["OUT_DIR"]
def sha(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()
limite = os.environ.get("P_LIMIT") or ""
datos = {
    "offset": float(os.environ["P_OFFSET"]),
    "searchterm": os.environ["P_SEARCHTERM"],
    "limit": int(limite) if limite else None,
    "tsv": os.path.basename(os.environ.get("P_TSV") or ""),
    "tsv_sha256": sha(os.path.join(out, "tsv", "efectivo.tsv")),
}
with open(os.path.join(out, "informes", "parametros.json"), "w", encoding="utf-8") as fh:
    json.dump(datos, fh, indent=2, ensure_ascii=False)
print(f"   parámetros recordados: offset {datos['offset']:g}, searchterm {datos['searchterm']}")
PY
  verificar_holgura
  coste "tabulado"
fi

# ============================================================ 2. descarga
if hacer descarga; then
  etapa "2/10  descargar clips"
  T0_FALLOS="$(date '+%Y-%m-%d %H:%M:%S')"
  corre "$OUT/logs/02_download.log" "$POLY" download \
        --txt_file "$OUT/tsv/efectivo.tsv" --searchterm "$SEARCHTERM" \
        --output_dir "$OUT/videos/raw" --offset "$OFFSET" \
    || abortar "la descarga devolvió error" descarga
  RAW=$( { find "$OUT/videos/raw" -name '*.mp4' || true; } | wc -l)
  # El log de fallos es UNO para todo el repo (`uvrun` fija el cwd): con dos lotes bajando
  # a la vez comparten fichero, asi que se filtra por lote y por hora de inicio en vez de
  # por offset de bytes, que le copiaba a cada lote los fallos del otro.
  "$PY" "$POLY_ROOT/tools/filtrar_fallos_descarga.py" --log "$FDLOG" \
        --searchterm "$SEARCHTERM" --desde "$T0_FALLOS" \
        --out "$OUT/informes/failed_downloads_nuevos.log" || true
  NFAIL=$(wc -l < "$OUT/informes/failed_downloads_nuevos.log" 2>/dev/null || echo 0)
  # un HTML de error pasa el control de tamaño: hay que abrir los vídeos
  OUT_DIR="$OUT" "$PY" - <<'PY'
import cv2, glob, os
d = os.path.join(os.environ["OUT_DIR"], "videos", "raw")
raros = []; fps = set(); frames = []
for p in sorted(glob.glob(os.path.join(d, "*.mp4"))):
    magic = b"ftyp" in open(p, "rb").read(16)
    c = cv2.VideoCapture(p)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); f = c.get(cv2.CAP_PROP_FPS)
    ok, _ = c.read(); c.release()
    if f: fps.add(round(f, 3))
    if n > 0: frames.append(n)
    if not (magic and ok and n > 20): raros.append(os.path.basename(p))
with open(os.path.join(os.environ["OUT_DIR"], "informes", "videos_raros.txt"), "w") as fh:
    fh.write("\n".join(raros))
print(f"   mp4 en videos/raw: {len(glob.glob(os.path.join(d,'*.mp4')))}")
print(f"   fps detectado: {sorted(fps)}   frames: min {min(frames, default=0)} max {max(frames, default=0)}")
print(f"   vídeos que NO se abren o no son mp4: {len(raros)}" + (f"  -> informes/videos_raros.txt" if raros else ""))
PY
  echo "   fallos de descarga nuevos: $NFAIL  (el log original queda en $FDLOG, hace append)"
  conteo descarga clips_descargados "$RAW"
  coste "descarga"
  [ "$RAW" -gt 0 ] || abortar "no bajó ni un clip: mira logs/02_download.log e informes/failed_downloads_nuevos.log" descarga
  verificar_holgura
fi

# ============================================================ 3. TalkNet-ASD
if hacer asd; then
  etapa "3/10  hablante activo (raw -> masked)"
  corre "$OUT/logs/03_asd.log" "$POLY" asd \
        --input_dir "$OUT/videos/raw" --output_dir "$OUT/videos/masked" \
    || abortar "TalkNet-ASD devolvió error (¿PATH del wrapper? ¿POLYMNIA_ASD_PATH?)" asd
  MASKED=$( { find "$OUT/videos/masked" -name '*.mp4' || true; } | wc -l)
  ESCENA_FALSA=$(grep -c "single scene: False" "$OUT/logs/03_asd.log" || true)
  NO_CRITERIO=$(grep -c "doesn't meet criteria" "$OUT/logs/03_asd.log" || true)
  # ASD es un FILTRO: los que no cumplen no generan fichero y el proceso sale 0
  OUT_DIR="$OUT" "$PY" - <<'PY'
import glob, os, shutil
out = os.environ["OUT_DIR"]
dest = os.path.join(out, "videos", "descartados", "asd_sin_mascara")
crudos = {os.path.basename(p): p for p in glob.glob(os.path.join(out, "videos", "raw", "*.mp4"))}
masc = {os.path.basename(p) for p in glob.glob(os.path.join(out, "videos", "masked", "*.mp4"))}
perdidos = sorted(set(crudos) - masc)
for n in perdidos:
    shutil.copy2(crudos[n], os.path.join(dest, n))
print(f"   sin máscara (los rechazó ASD): {len(perdidos)}  -> videos/descartados/asd_sin_mascara/")
PY
  echo "   con escena única False: ${ESCENA_FALSA:-0}   sin criterio de hablante: ${NO_CRITERIO:-0}"
  conteo asd clips_con_mascara "$MASKED"
  coste "ASD"
  [ "$MASKED" -gt 0 ] || abortar "ASD no dejó ningún clip: la merma es de ASD, no de las muñecas" asd
fi

# ============================================================ 4. criba de muñecas
if hacer preseleccion; then
  etapa "4/10  criba de muñecas ANTES de OpenPose (que es lo caro)"
  [ -f "$PESO" ] || abortar "faltan los pesos de YOLO: $PESO (copia yolo11x-pose.pt ahí o pasa --peso)" preseleccion
  corre "$OUT/logs/04_prescreen.log" "$PY" "$FILTROS/wrist_prescreen.py" \
        --videos_folder "$OUT/videos/masked" --out_csv "$OUT/munecas.csv" \
        --weights "$PESO" --min-conf "$conf" --frames "$FRAMES" --device "$DEVICE" \
    || abortar "falló la criba de muñecas" preseleccion
  OUT_DIR="$OUT" MIN="$MIN_SP_ANY" "$PY" - <<'PY'
import os, shutil, pandas as pd
out = os.environ["OUT_DIR"]; min_sp = float(os.environ["MIN"])
d = pd.read_csv(os.path.join(out, "munecas.csv"))
if "sp_any" not in d.columns:
    # munecas.csv de una criba vieja: se deriva con la misma regla que escribe la criba
    # (max ignorando NaN; NaN solo si faltan las dos). Sin esto, reanudar un lote mixto
    # reventaria con un KeyError en vez de admitir lo que tocaba.
    d["sp_any"] = d[["sp_right", "sp_left"]].max(axis=1)
d = d.sort_values(["sp_any", "sp_both"], ascending=False)
selec = d[d.sp_any >= min_sp]
desc = d[d.sp_any < min_sp]
for _, r in selec.iterrows():
    shutil.copy2(r["path"], os.path.join(out, "videos", "1_person", r["video"]))
for _, r in desc.iterrows():
    shutil.copy2(r["path"], os.path.join(out, "videos", "descartados", "preseleccion_sp_bajo", r["video"]))
print(f"   admitidos {len(selec)} de {len(d)} con sp_any >= {min_sp}")
print("   los 8 mejores:")
for _, r in d.head(8).iterrows():
    print(f"     sp_any={r.sp_any:.2f} sp_both={r.sp_both:.2f} personas_max={r.max_people:g}  {r.video[:60]}")
if len(desc):
    print(f"   descartados: sp_any entre {desc.sp_any.min():.2f} y {desc.sp_any.max():.2f}")
print("   OJO: sp_any es un ranker del detector (YOLO), no una promesa de frames medibles.")
print("   Y una muneca vale: gesture_features.py calcula cada lado por separado (sp_both")
print("   sigue en munecas.csv como ranking). Que esto sea cierto y no una intuicion se")
print("   mide con tools/auditoria_criba.py sobre los clips que esta criba descarta.")
PY
  ADM=$( { find "$OUT/videos/1_person" -name '*.mp4' || true; } | wc -l)
  conteo preseleccion admitidos "$ADM"
  coste "criba"
  [ "$ADM" -gt 0 ] || abortar "la criba no admitió ningún clip: baja --min-sp-any (0.4-0.5) y repite --desde preseleccion" preseleccion
fi

# ============================================================ 5. OpenPose + Parquet
if hacer openpose; then
  etapa "5/10  OpenPose + dfMaker + clasificación por nº de personas"
  ARGS=(--videos_folder "$OUT/videos/1_person" --output_folder "$OUT")
  if [ "$GATE" -eq 1 ]; then
    ARGS+=(--filter-body-visibility --required-body-points 4,7
           --min-visible-percent "$MIN_VISIBLE" --min-keypoint-confidence 0.2)
  fi
  # --face_hands/--skeletons NO se pueden apagar pasando False (argos.py los declara sin type:
  # "False" llega como cadena y es True). Se dejan como en la entrega: cara y manos SÍ.
  # Al reanudar, los Parquet de la pasada anterior ya no representan estos JSON: argos
  # escribia ENCIMA de ellos y el R mezclaba dos clasificaciones.
  # Medido reanudando con --desde openpose: 20 Parquet en lugar de 4.
  for b in "$OUT"/dataset/*_persons; do apartar "$b"; done
  corre "$OUT/logs/05_openpose.log" "$POLY" argos "${ARGS[@]}" \
    || abortar "OpenPose/dfMaker devolvió error" openpose
  OUT_DIR="$OUT" "$PY" - <<'PY'
import glob, os, shutil
out = os.environ["OUT_DIR"]
for ruta in sorted(glob.glob(os.path.join(out, "dataset", "*_persons", "parquet_files", "*.parquet"))):
    bucket = os.path.basename(os.path.dirname(os.path.dirname(ruta)))
    print(f"   {bucket}: {len(glob.glob(os.path.join(os.path.dirname(ruta), '*.parquet')))} parquet")
    if bucket != "1_persons":
        destino = os.path.join(out, "videos", "descartados", "clasificado_multipersona")
        shutil.copy2(os.path.join(out, "videos", "1_person", os.path.basename(ruta).replace(".parquet", ".mp4")),
                     os.path.join(destino, os.path.basename(ruta).replace(".parquet", ".mp4")))
uno = glob.glob(os.path.join(out, "dataset", "1_persons", "parquet_files", "*.parquet"))
jsons = glob.glob(os.path.join(out, "dataset", "OpenPose", "*", "JSON_FILES"))
print(f"   clips con Parquet en 1_persons: {len(uno)}   clips con JSON de OpenPose: {len(jsons)}")
print("   los de 2_persons o más vienen de frames con segunda persona (o de un candidato que")
print("   fabrica la propia máscara de ASD): están en videos/descartados/clasificado_multipersona/")
PY
  TODOS=$( { find "$OUT/dataset" -name '*.parquet' 2>/dev/null || true; } | wc -l)
  conteo openpose parquet_en_cualquier_bucket "$TODOS"
  coste "OpenPose"
  # AQUI no se exige 1_persons: con los candidatos que fabrica la mascara, casi todo acaba
  # en 2_persons hasta que pasa la etapa de limpieza.
  [ "$TODOS" -gt 0 ] || abortar "OpenPose terminó pero no quedó ningún Parquet" openpose
fi

# ============================================================ 6. limpieza de candidatos
if hacer limpieza; then
  etapa "6/10  limpiar candidatos que no son cuerpos medibles"
  if [ "$LIMPIEZA" -eq 0 ]; then
    echo "   omitido por --sin-limpiar: un solo frame ambiguo manda el clip a 2_persons"
  else
    # La mascara de ASD fabrica candidatos de 2-3 puntos en su borde: OpenPose sobre el
    # RAW de Kimmel 08-25 y Tavis 05-20 da 1 persona en los 155 frames, y sobre el MASKED
    # da 2 en 12 y en 7. Como dfMaker hace people_id = indice del array y el R clasifica
    # con max(people_id), eso tira el clip entero al bucket 2_persons.
    corre "$OUT/logs/06_limpieza.log" "$PY" "$FILTROS/limpiar_fantasmas.py" \
          --lote "$OUT" --criterio "$CRITERIO_LIMPIEZA" \
      || abortar "falló la limpieza de candidatos" limpieza
    # La clasificacion sin limpiar ya no representa estos JSON. Si se reanuda justo aqui,
    # los buckets viejos tampoco valen: el mv anterior fallaba en silencio porque
    # _pre_limpieza ya existia y el bucket viejo se quedaba abajo contando como propio.
    for b in "$OUT"/dataset/*_persons; do apartar "$b"; done
    for jd in "$OUT"/dataset/OpenPose/*/JSON_FILES_limpio; do
      [ -d "$jd" ] || continue
      "$POLY" parquet --json_folder "$jd" --dataset_dir "$OUT/dataset" \
        >> "$OUT/logs/06_limpieza.log" 2>&1 \
        || abortar "el R no pudo reclasificar $jd" limpieza
    done
    # Clips que OpenPose dejo sin una sola deteccion: el R ya no revienta (deja
    # num_persons.txt=0), se apartan con motivo y el lote sigue. El umbral para cuando el
    # patron no son unos pocos clips sino OpenPose roto (o el lote se quedaria sin datos).
    SINPOSE_OUT=$("$PY" "$POLY_ROOT/tools/descartes_sin_pose.py" --lote "$OUT" \
                    --max-pct "$MAX_SIN_POSE_PCT" 2>&1) || {
      printf '%s\n' "$SINPOSE_OUT" | tee -a "$OUT/logs/06_limpieza.log" >&2
      abortar "demasiados clips sin pose (mira informes/descartados_sin_pose.tsv)" limpieza
    }
    printf '%s\n' "$SINPOSE_OUT" | tee -a "$OUT/logs/06_limpieza.log"
    conteo limpieza clips_sin_pose \
      "$(printf '%s\n' "$SINPOSE_OUT" | sed -n 's/^sin_pose=\([0-9][0-9]*\) .*/\1/p')"
  fi
  NPARQUET=$( { find "$OUT/dataset/1_persons/parquet_files" -name '*.parquet' 2>/dev/null || true; } | wc -l )
  MULTI=$( { find "$OUT/dataset" -maxdepth 1 -type d -name '*_persons' ! -name '1_persons' 2>/dev/null || true; } \
           | xargs -r -I{} sh -c 'ls {}/parquet_files/*.parquet 2>/dev/null | wc -l' \
           | awk '{s+=$1} END {print s+0}')
  echo "   Parquet en 1_persons: $NPARQUET   en otros buckets: $MULTI"
  echo "   (lo que se aparta de cada pasada queda en informes/_anterior/, nada se borra)"
  conteo limpieza parquet_1_person "$NPARQUET"
  conteo limpieza parquet_multipersona "$MULTI"
  coste "limpieza"
  [ "$NPARQUET" -gt 0 ] || abortar "no quedó ningún clip como 1_persons: mira informes/limpieza_fantasmas.json, o prueba --criterio-limpieza nada" limpieza
fi

# ============================================================ 7. voz
if hacer voz; then
  etapa "7/10  prosodia (Praat) + palabras por frame (WhisperX)"
  # dfMaker deja <clip>.parquet y, segun la version, tambien una carpeta por clip: si se
  # reanuda se juntan las dos estructuras y el recuento se infla (medido: 8 para 4 clips).
  apartar "$OUT/dataset/1_persons/parquet_speech_analysis"
  apartar "$OUT/dataset/1_persons/whisperx_framer_output"
  corre "$OUT/logs/07_voz.log" "$POLY" voice --input_folder "$OUT" --n_persons 1 \
    || echo "   ⚠️  voz devolvió aviso: ecos/apate escriben su log dentro de $OUT (ecos.log, apate.log)"
  # voice deja DOS estructuras en el mismo sitio: primero <clip>/<clip>_prosody.parquet y
  # 24 s despues la copia plana <clip>.parquet con su .prosody_compat_manifest.json. Contar
  # ficheros o mirar solo el primer nivel da 0 segun en que momento mires; se cuentan clips.
  PROS=$( { find "$OUT/dataset/1_persons/parquet_speech_analysis" -name '*.parquet' 2>/dev/null || true; } \
          | sed 's#.*/##; s#\.parquet$##; s#_prosody$##' | sort -u | wc -l)
  # apate deja las palabras en whisperx_framer_output como JSON por clip, no como Parquet
  WORDS=$( { find "$OUT/dataset/1_persons/whisperx_framer_output" -name '*.json' 2>/dev/null || true; } | wc -l)
  echo "   prosodia: $PROS parquet   palabras: $WORDS clips"
  [ "$WORDS" -gt 0 ] || echo "   ⚠️  sin palabras por frame: las columna words saldra NULL (ver apate.log)"
  conteo voz parquet_prosodia "$PROS"
  conteo voz clips_con_palabras "$WORDS"
  coste "voz"
fi

# ============================================================ 8. DuckDB
if hacer merge; then
  etapa "8/10  construir el DuckDB (tabla larga multi_data)"
  # Reanudar el merge sobre la base anterior revienta al insertar: multi_data nace con 28
  # columnas y theia le anade las 7 de anotacion visual, asi que la segunda pasada intenta
  # meter 28 valores en una tabla de 35 ("Binder Error: table multi_data has 35 columns but
  # 28 values were supplied") y, lo peor, Polymnia lo deja en exit 0.
  apartar "$OUT/${LOTE_NOMBRE}.duckdb"
  apartar "$OUT/${LOTE_NOMBRE}.duckdb.wal"
  MERGE_RC=0
  corre "$OUT/logs/08_merge.log" "$POLY" merge --input_folder "$OUT" --n_persons 1 \
        --db_name "$OUT/${LOTE_NOMBRE}.duckdb" \
        --discard-report "$OUT/informes/descartados_sin_voz.tsv" \
        --max-discard-pct "$MAX_SIN_VOZ_PCT" || MERGE_RC=$?
  # Evidencia del descarte, tambien cuando el merge aborta por umbral: el clip que no tiene
  # voz no se publica, pero su mp4 y su motivo quedan a la vista en el lote.
  OUT_DIR="$OUT" "$PY" - <<'PY'
import csv, os, shutil
out = os.environ["OUT_DIR"]
rep = os.path.join(out, "informes", "descartados_sin_voz.tsv")
if not os.path.isfile(rep):
    print("   descartes: 0 (hefesto no escribio reporte)")
    raise SystemExit
with open(rep, encoding="utf-8") as fh:
    filas = list(csv.DictReader(fh, delimiter="\t"))
destino = os.path.join(out, "videos", "descartados", "sin_voz")
os.makedirs(destino, exist_ok=True)
copiados = 0
for r in filas:
    clip = r.get("clip") or ""
    origen = os.path.join(out, "videos", "1_person", clip + ".mp4")
    if clip and os.path.isfile(origen):
        shutil.copy2(origen, os.path.join(destino, clip + ".mp4"))
        copiados += 1
print(f"   descartes: {len(filas)} clips sin voz -> videos/descartados/sin_voz/ ({copiados} mp4)")
for motivo in ("prosody_missing", "prosody_empty", "combined_empty"):
    n = sum(1 for r in filas if r.get("reason") == motivo)
    if n:
        print(f"     {motivo}: {n}")
PY
  SINVOZ=$( { [ -f "$OUT/informes/descartados_sin_voz.tsv" ] && tail -n +2 "$OUT/informes/descartados_sin_voz.tsv" || true; } | wc -l )
  conteo merge clips_sin_voz "$SINVOZ"
  [ "${MERGE_RC:-0}" -eq 0 ] || abortar "merge/hefesto devolvió error (mira logs/08_merge.log: umbral de descartes o fallo de inserción)" merge
  [ -f "$OUT/${LOTE_NOMBRE}.duckdb" ] || abortar "no se creó $OUT/${LOTE_NOMBRE}.duckdb" merge
  "$DUCK" -readonly "$OUT/${LOTE_NOMBRE}.duckdb" -c "SELECT count(*) AS filas, count(DISTINCT id) AS clips FROM multi_data"
  # SIEMPRE con type_points: en `points` conviven 4 modelos (cara, pose, dos manos)
  OUT_DIR="$OUT" LOTE_DB="${LOTE_NOMBRE}.duckdb" "$PY" - <<'PY'
import os, duckdb, pandas as pd
ruta = os.path.join(os.environ["OUT_DIR"], os.environ["LOTE_DB"])
con = duckdb.connect(ruta, read_only=True)
df = con.execute("""
  WITH f AS (
    SELECT id, count(DISTINCT frame) n FROM multi_data
    WHERE type_points='pose_keypoints' GROUP BY id)
  SELECT f.id, max(f.n) AS frames,
         round(100.0*sum(CASE WHEN p.points=4 AND p.c>=0.2 THEN 1 ELSE 0 END)/max(f.n),1) AS muneca_dcha_pct,
         round(100.0*sum(CASE WHEN p.points=7 AND p.c>=0.2 THEN 1 ELSE 0 END)/max(f.n),1) AS muneca_izq_pct
  FROM f JOIN multi_data p ON p.id = f.id
  WHERE p.type_points='pose_keypoints' AND p.points IN (4,7)
  GROUP BY f.id ORDER BY muneca_dcha_pct DESC
""").df()
dest = os.path.join(os.environ["OUT_DIR"], "informes", "cobertura_munecas.csv")
df.to_csv(dest, index=False)
ambas = df[(df.muneca_dcha_pct >= 60) & (df.muneca_izq_pct >= 60)]
print(f"   cobertura de muñeca por clip -> informes/cobertura_munecas.csv")
print(f"   clips con AMBAS muñecas >=60% de frames: {len(ambas)} de {len(df)}")
if len(ambas):
    for _, r in ambas.head(10).iterrows():
        print(f"     dcha {r.muneca_dcha_pct:5.1f}%  izq {r.muneca_izq_pct:5.1f}%  {r.id[:60]}")
PY
  CONTEO_DUCK=$("$DUCK" -readonly -noheader -csv "$OUT/${LOTE_NOMBRE}.duckdb" \
                 -c "SELECT count(DISTINCT id) FROM multi_data" 2>/dev/null | tr -d '[:space:]' || true)
  echo "   clips en la base: ${CONTEO_DUCK:-?}"
  conteo merge clips_en_duckdb "$CONTEO_DUCK"
  coste "merge"
fi

# ============================================================ 9. Theia
if hacer theia; then
  etapa "9/10  anotaciones visuales (Theia)"
  if [ "$SKIP_THEIA" -eq 1 ]; then
    # El motivo se escribe DENTRO del lote, no solo al stdout: un lote sin
    # anotaciones visuales no dejaba ni rastro dentro de el, y asi se fueron dos
    # tandas sin que nadie viera que la etapa 9 no habia llegado a ejecutarse.
    echo "omitido por --skip-theia: final_videos sale con las 7 columnas de anotacion a NULL" \
      | tee "$OUT/logs/09_theia.log"
    conteo theia clips_anotados 0
    conteo theia clips_con_error 0
    conteo theia motivo skip_theia_flag
  else
    # Theia corre por defecto y sin exportar nada: theia.py carga el .env de la raiz
    # del repo ANTES de resolver la configuracion. Aqui ya no se exporta ni se comprueba
    # el .env, y esta bien que no se compruebe: si falta la configuracion, la etapa debe
    # INTENTARSE y dejar el fallo escrito en logs/09_theia.log, no omitirse callada.
    # (Lo que habia antes fallaba dos veces. Una: el default de --api-key se resolvia en
    # el parser, antes de load_env, y la peticion salia con "Bearer none" -> 401. Dos: la
    # comprobacion era [ -f "$RAIZ/Polymnia/.env" ], pero RAIZ ya ES el repo Polymnia, asi
    # que preguntaba por Polymnia/Polymnia/.env, que no existe -> la etapa se saltaba SIEMPRE.)
    [ -f "$POLY_ROOT/.env" ] \
      || echo "   ⚠️  no hay $POLY_ROOT/.env: si tampoco exportas THEIA_API_BASE/THEIA_MODEL, la anotacion visual fallara"
    corre "$OUT/logs/09_theia.log" "$POLY" theia --input_folder "$OUT" --n_persons 1 \
          --db_name "$OUT/${LOTE_NOMBRE}.duckdb" --workers "$THEIA_WORKERS" \
      || echo "   ⚠️  Theia terminó con avisos (intenta reintentar con --desde theia --force)"
    # el resumen de theia es una linea propia; sin ella el 401 pasa desapercibido
    grep -o "\[theia\] ok=[0-9]* error=[0-9]*" "$OUT/logs/09_theia.log" | tail -1 || true
    grep -m1 -o "401[^)]*" "$OUT/logs/09_theia.log" \
      && echo "   ⚠️  401: el endpoint rechazo la clave (mira la linea 'clave:' de logs/09_theia.log: dice de donde salio)"
    grep -m1 -o "\[theia\] model=.*)" "$OUT/logs/09_theia.log" | tail -1 || true
    # theia inserta la fila tambien cuando falla, con la causa en `error`. Ojo: cuando va
    # BIEN deja error como cadena VACIA, no como NULL (medido: ok=4 y count(error IS NULL)=0),
    # asi que "IS NULL" a secas daria "0 anotados" en una pasada perfecta.
    SIN="error IS NULL OR trim(error) = ''"
    CON="error IS NOT NULL AND trim(error) <> ''"
    ANOT=$("$DUCK" -readonly -noheader -csv "$OUT/${LOTE_NOMBRE}.duckdb" \
             -c "SELECT count(*) FILTER (WHERE $SIN) FROM video_annotations" 2>/dev/null | tr -d '[:space:]' || true)
    FALL=$("$DUCK" -readonly -noheader -csv "$OUT/${LOTE_NOMBRE}.duckdb" \
             -c "SELECT count(*) FILTER (WHERE $CON) FROM video_annotations" 2>/dev/null | tr -d '[:space:]' || true)
    # Guarda: un lote con 0 anotaciones no se publica. En la tanda de octubre el endpoint
    # devolvio 403/429 y dos lotes salieron a final/ con las 7 columnas vacias (FREQ028_4,
    # FREQ029). Los clips ya anotados quedan en la base: reanudar no repite trabajo.
    "$PY" "$POLY_ROOT/tools/check_annotations.py" --db "$OUT/${LOTE_NOMBRE}.duckdb" \
      || abortar "ningún clip quedó anotado (¿clave o endpoint de Theia?): no se publica un dataset sin anotación. Lo ya anotado sigue en la base; reintenta con --desde theia --force" theia
    if [ "${FALL:-?}" = "0" ]; then
      echo "   anotacion visual: ${ANOT:-?} clips con etiquetas, 0 con error"
    else
      echo "   anotacion visual: ${ANOT:-?} clips con etiquetas, ${FALL:-?} con error (esos salen con las 7 columnas a NULL)"
    fi
    conteo theia clips_anotados "${ANOT:-?}"
    conteo theia clips_con_error "${FALL:-?}"
  fi
  coste "theia"
fi

# ============================================================ 10. dataset final
if hacer final; then
  etapa "10/10 dataset final (final_frames + final_videos + diccionario + README)"
  [ -e "$OUT/final" ] && [ "$FORCE" -eq 0 ] && abortar "ya existe $OUT/final (bórralo o usa --force)" final
  ARGS=(--db_name "$OUT/${LOTE_NOMBRE}.duckdb" --tab "$OUT/tsv/efectivo.tsv"
        --offset "$OFFSET" --searchterm "$SEARCHTERM" --provenance "$PROCEDENCIA")
  [ "$EXTENDED" -eq 1 ] && ARGS+=(--extended)

  # Regla de esta etapa: nada de lo que puede fallar se ejecuta YA despues de tocar la
  # entrega. El orden viejo era al reves: se apartaba `final/` y SOLO DESPUES corria la
  # pasada de medicion, asi que un abortazo dejaba el lote sin dataset publicado y sin nada
  # nuevo que publicar (ha pasado: un --searchterm equivocado hacia saltar esa pasada y el
  # `final/` entregado llevaba ya un rato en informes/_anterior/). Ahora lo fragile se
  # escribe a scratch oculto y el unico punto sin retorno es el rename del final.
  apartar "$OUT/.medicion"
  apartar "$OUT/.publicando"
  apartar "$OUT/.docs_nuevos"
  mkdir -p "$OUT/.docs_nuevos"
  # Basura de scratch: `mkdir .docs_nuevos` ocurre antes de lo que puede fallar, asi que un
  # abort dejaba el directorio (vacio) dentro del lote entregado, y el run siguiente lo
  # apartaba a informes/_anterior/ como si fuera un derivado real (ha pasado: hay uno vacio
  # ahi). Los tres son scratch por definicion: NUNCA son la entrega, que llega a su sitio por
  # rename. Por eso se pueden borrar al salir pase lo que pase. `OUT` es absoluta (caso de
  # arriba); el test extra es para que un OUT vacio no pueda convertirse en un `rm -rf /...`.
  trap '[ -n "${OUT:-}" ] && rm -rf "$OUT/.medicion" "$OUT/.publicando" "$OUT/.docs_nuevos" || true' EXIT

  # El README se publica DENTRO de final/ y entra en el sha256 del manifiesto, asi que tiene que
  # existir ANTES de publicar; y para hablar de cobertura tiene que leer los parquet ya calculados.
  # Se resuelve con una pasada de medicion a .medicion: mismas entradas, mismos numeros. Abajo se
  # compara byte a byte antes de fiarnos del documento, asi que no puede describir otro dataset.
  README_TMP="$OUT/.docs_nuevos/README_entrega.md"
  if [ "$DICCIONARIO" -eq 1 ]; then
    corre "$OUT/logs/10a_medicion.log" "$PY" "$RAIZ/final_dataset/final_dataset.py" \
      "${ARGS[@]}" --output_folder "$OUT/.medicion" \
      || abortar "la pasada de medicion falló (sin ella no hay diccionario; prueba --sin-diccionario). La entrega anterior sigue intacta" final
    corre "$OUT/logs/10b_diccionario.log" "$PY" "$RAIZ/final_dataset/diccionario_columnas.py" \
      --final "$OUT/.medicion" --csv "$OUT/.docs_nuevos/DICCIONARIO_COLUMNAS.csv" \
      --md "$OUT/.docs_nuevos/DICCIONARIO_COLUMNAS.md" --readme "$README_TMP" \
      || abortar "el diccionario falló. Si dice 'undocumented columns' es bueno: el código ha ganado una columna que nadie describió; documentala en gesture_features.py y relanza. La entrega anterior sigue intacta" final
    NDIC=$("$PY" -c "import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))" \
            "$OUT/.docs_nuevos/DICCIONARIO_COLUMNAS.csv")
    [ "${NDIC:-0}" -gt 0 ] || abortar "el diccionario salió vacío" final
    conteo final columnas_documentadas "$NDIC"
    echo "   diccionario: $NDIC columnas documentadas (se publican junto con el dataset)"
    echo "   README: generado con los números medidos en este lote"
  fi

  README_PUBLICAR="$README"
  [ -z "$README_PUBLICAR" ] && [ "$DICCIONARIO" -eq 1 ] && README_PUBLICAR="$README_TMP"
  [ -n "$README_PUBLICAR" ] && ARGS+=(--readme "$README_PUBLICAR")
  if [ -n "$README" ]; then
    echo "   ⚠️  publicando TU README como DATASET.md: nadie comprueba sus números. Los de la"
    echo "      tanda vieja (14,5% muñeca derecha, 3 clips bimanuales) no sirven para esta."
  fi
  [ -z "$README_PUBLICAR" ] && echo "   sin DATASET.md en final/ (--sin-diccionario sin README generado)"
  corre "$OUT/logs/10_final.log" "$PY" "$RAIZ/final_dataset/final_dataset.py" \
    "${ARGS[@]}" --output_folder "$OUT/.publicando" \
    || abortar "final_dataset.py devolvió error (la entrega anterior sigue en su sitio)" final

  if [ "$DICCIONARIO" -eq 1 ]; then
    for fichero in final_frames.parquet final_videos.parquet final_frames.csv final_videos.csv; do
      cmp -s "$OUT/.medicion/$fichero" "$OUT/.publicando/$fichero" \
        || abortar "$fichero difiere entre la medición y la publicación: el diccionario y el README describirían otro dataset. Nada publicado aun: la entrega anterior sigue en su sitio" final
    done
    echo "   comprobado: lo que se va a publicar es byte a byte lo que el diccionario describe"
  fi

  # --------------------------------------- punto sin retorno ---------------------------
  # A partir de aqui ya no hay nada que pueda fallar por motivos «normales» (medir,
  # diccionario, publicar y comparar ya pasaron con exito). Lo unico que se aparta es lo
  # que se sustituye, y se sustituye por lo que esta en scratch. Se separan dos rename
  # cortos para que la ventana sin entrega sea lo mas pequena posible.
  if [ "$DICCIONARIO" -eq 1 ]; then
    apartar "$OUT/informes/DICCIONARIO_COLUMNAS.csv"
    apartar "$OUT/DICCIONARIO_COLUMNAS.md"
    apartar "$OUT/informes/README_entrega.md"
    mv "$OUT/.docs_nuevos/DICCIONARIO_COLUMNAS.csv" "$OUT/informes/DICCIONARIO_COLUMNAS.csv"
    mv "$OUT/.docs_nuevos/DICCIONARIO_COLUMNAS.md" "$OUT/DICCIONARIO_COLUMNAS.md"
    mv "$README_TMP" "$OUT/informes/README_entrega.md"
    echo "   diccionario publicado: informes/DICCIONARIO_COLUMNAS.csv ($NDIC columnas)"
  fi
  apartar "$OUT/final"
  mv "$OUT/.publicando" "$OUT/final"
  rm -rf "$OUT/.medicion" "$OUT/.docs_nuevos"
  OUT_DIR="$OUT" "$PY" - <<'PY'
import json, os, pandas as pd
f = os.path.join(os.environ["OUT_DIR"], "final")
m = json.load(open(os.path.join(f, "manifest.json")))
map_ = m.get("clip_id_mapping", {})
f_ = m.get("features") or {}
print(f"   features: version {f_.get('version')}   extendidas: {f_.get('extended')}   "
      f"{len(f_.get('extended_columns') or [])} columnas del bloque 3")
print(f"   README publicado: {m.get('readme') or 'NINGUNO'}")
print(f"   lineas del tab: {map_.get('tab_rows')}   emparejados: {map_.get('matched')} "
      f"(exactos {map_.get('matched_exact_id')}, por centro de ventana {map_.get('matched_by_window_center')})"
      f"   sin emparejar: {len(map_.get('unmatched') or [])}")
for t in ("final_frames", "final_videos"):
    d = pd.read_parquet(os.path.join(f, t + ".parquet"))
    print(f"   {t}: {d.shape[0]} filas x {d.shape[1]} columnas")
if map_.get("unmatched") or (map_.get("matched_exact_id") or 0) < (map_.get("matched") or 0):
    print("   ⚠️  hay clips que no encajan por ID exacto: revisa que --offset coincide con la descarga")
PY
  coste "final"
fi

# ============================================================ rendimiento
TOTAL=$(( $(date +%s) - INICIO_TOTAL ))
{
  echo "# Rendimiento de $(basename "$OUT")"
  echo
  echo "Generado: $(date '+%Y-%m-%d %H:%M:%S') · coste total de esta pasada: ${TOTAL} s"
  echo
  echo "| etapa | métrica | valor |"
  echo "|---|---|---|"
  [ -f "$OUT/informes/conteos.tsv" ] && awk -F'\t' '{printf "| %s | %s | %s |\n", $1, $2, $3}' "$OUT/informes/conteos.tsv"
  echo
  echo "## Referencia del run original (100 líneas de TSV)"
  echo
  echo "| etapa | quedan |"
  echo "|---|---|"
  echo "| TSV maestro | 100 |"
  echo "| descarga | 100 |"
  echo "| TalkNet-ASD | 51 |"
  echo "| filtro YOLO de persona/manos | 39 |"
  echo "| clasificado 1_person por dfMaker | 29 |"
  echo "| con muñecas visibles | **4** |"
  echo
  echo "Es decir: ~4 % del master acaba dando datos de muñeca aprovechables. Si buscas N clips"
  echo "bimanuales, pide del orden de 15-25 x N líneas de TSV."
  echo
  echo "## Qué revisar a mano"
  echo
  echo "* \`informes/videos_raros.txt\` — mp4 que no se abren (un HTML de error pasa el control de tamaño)."
  echo "* \`informes/failed_downloads_nuevos.log\` — solo los fallos de ESTA pasada; el log original"
  echo "  vive en \`Polymnia/failed_downloads.log\` y hace append entre pasadas."
  echo "* \`munecas.csv\` — el score de todos los clips, incluidos los descartados."
  echo "* \`videos/descartados/<motivo>/\` — por qué se cayó cada clip."
  echo "* \`informes/descartados_sin_pose.tsv\` — los clips que OpenPose dejó sin una sola"
  echo "  detección (se apartan con su mp4 en videos/descartados/sin_pose/; umbral"
  echo "  \`--max-sin-pose-pct\`, 5 % por defecto)."
  echo "* \`informes/descartados_sin_voz.tsv\` — los clips sin voz/prosodia que hefesto no metió"
  echo "  en la base (clip, motivo, origen). El umbral \`--max-sin-voz-pct\` (defecto 5 %) para el"
  echo "  lote cuando el patrón no es un hablante mudo sino una extracción de voz rota."
  echo "* \`informes/cobertura_munecas.csv\` — la cobertura REAL de muñeca medida con OpenPose"
  echo "  (filtrando \`type_points='pose_keypoints'\`; sin ese filtro los puntos 4 y 7 son cejas y nudillos)."
  echo "* \`final/manifest.json\` — fórmulas, unidades, constantes y SHA-256 de cada fichero publicado."
  echo "* \`logs/09_theia.log\` — la anotación visual va por defecto (lee \`Polymnia/.env\` sin que
  exportes nada). Si sale 401/403 el lote **aborta** en esta etapa: no se publica un dataset sin
  anotación. Lo ya anotado queda en la base y se reintenta con \`--desde theia --force\`. Si el
  fichero dice «omitido por --skip-theia», es que se pidió así."
} > "$OUT/RENDIMIENTO.md"

echo
echo "══════════════════════════════════════════════════════════════"
echo "✔ listo en $OUT   (coste total ${TOTAL} s)"
echo "══════════════════════════════════════════════════════════════"
{ find "$OUT/videos/raw" "$OUT/videos/masked" "$OUT/videos/1_person" -name '*.mp4' 2>/dev/null || true; } \
  | sed "s|$OUT/||" | awk -F/ '{c[$1"/"$2]++} END {for (k in c) printf "   %-34s %s vídeos\n", k, c[k]}' | sort
echo "   $( { find "$OUT/dataset" -name '*.parquet' 2>/dev/null || true; } | wc -l) parquet en dataset/"
[ -f "$OUT/${LOTE_NOMBRE}.duckdb" ] && echo "   $(du -h "$OUT/${LOTE_NOMBRE}.duckdb" | cut -f1) ${LOTE_NOMBRE}.duckdb"
[ -d "$OUT/final" ] && echo "   dataset final en $OUT/final"
echo "   leo todo esto en $OUT/RENDIMIENTO.md"
