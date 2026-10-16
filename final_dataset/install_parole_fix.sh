#!/usr/bin/env bash
# Instala (o comprueba) el arreglo de la prosodia en el PAROLE que Polymnia guarda
# vendorizado bajo Polymnia/create_datasets/parole/.
#
#   ./install.sh              copia los ficheros canonicos y aparta los que habia
#   ./install.sh --comprobar  solo dice si lo instalado coincide con lo de aqui
#   ./install.sh --dest D     instalar sobre otro arbol (D tiene que tener scripts/)
#
# Por que hace falta un instalador: Polymnia tiene `create_datasets/` en .gitignore
# (linea 23, y la linea viene de upstream), asi que estos ficheros NO viven en ningun
# git. Sin una copia canonica en este repo, el arreglo solo existiria en el disco de
# esta maquina: se perderia con la maquina y no se podria revisar ni enviar arriba.
set -euo pipefail

AQUI="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)"   # final_dataset/
RAIZ="$(dirname "$AQUI")"
# Layout detection. This directory used to sit next to the checkout
# (trendsI/final_dataset beside trendsI/Polymnia) and now lives inside it as
# Polymnia/final_dataset. The parent directory name says nothing either way, so
# ask the tree where the wrapper is instead of assuming a name.
if [ -e "$RAIZ/bin/polymnia" ]; then
  DEST="$RAIZ"                          # Polymnia/final_dataset/ (repo layout since PR #8)
elif [ -e "$RAIZ/Polymnia/bin/polymnia" ]; then
  DEST="$RAIZ/Polymnia"                 # trendsI/final_dataset/ beside trendsI/Polymnia/
else
  DEST="$RAIZ/Polymnia"                 # unknown: the check below prints the path it looked for
fi
COMPROBAR=0

while [ $# -gt 0 ]; do
  case "$1" in
    --comprobar) COMPROBAR=1; shift ;;
    --dest) DEST="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "opcion desconocida: $1" >&2; exit 1 ;;
  esac
done

[ -d "$DEST/create_datasets/parole/scripts" ] || { echo "no existe $DEST/create_datasets/parole/scripts" >&2; exit 1; }

SCRIPTS="$DEST/create_datasets/parole/scripts"

# fichero_canonico -> destino relativo
#
# Only faithful upstream copies belong here: `--aplicar` overwrites the
# destination with what is in upstream_parole/, so anything listed here must be
# expected to be byte-identical to PAROLE forever.
#
# Polymnia/tools/test_prosody_interp.py is NOT a faithful copy and must not be
# listed. It is the same test, but maintained here on purpose: it resolves the
# prosody chain through this repository (with a fallback when PAROLE is not
# installed) and its `[skip]` messages name the paths a Polymnia user actually
# has. Its PAROLE twin lives in PAROLE's own tests/. Comparing bytes would be a
# lie -- and `--aplicar` would overwrite the better local version with the
# upstream one, which is what this entry used to do.
MAPA=(
  "upstream_parole/process_prosody.R:create_datasets/parole/scripts/process_prosody.R"
  "upstream_parole/prosody_align.R:create_datasets/parole/scripts/prosody_align.R"
)

# Reported for orientation only; never copied, never compared.
CO_MAINTAINED=("tools/test_prosody_interp.py")

sha() { sha256sum "$1" | cut -c1-16; }

if [ "$COMPROBAR" -eq 1 ]; then
  for f in "${CO_MAINTAINED[@]}"; do
    if [ -f "$DEST/$f" ]; then
      echo "  co-maintained  $f  (Polymnia keeps its own version; the PAROLE twin lives in PAROLE's tests/, not compared)"
    else
      echo "  missing        $f  (Polymnia's own test, tracked in git; not installed by this script)"
    fi
  done
fi

faltan=0
for par in "${MAPA[@]}"; do
  ORIG="$AQUI/${par%%:*}"
  DESTINO="$DEST/${par##*:}"
  [ -f "$ORIG" ] || { echo "falta la copia canonica: $ORIG" >&2; exit 1; }
  if [ "$COMPROBAR" -eq 1 ]; then
    if [ -f "$DESTINO" ] && [ "$(sha "$ORIG")" = "$(sha "$DESTINO")" ]; then
      echo "  igual   ${par##*:}  $(sha "$ORIG")"
    else
      echo "  DIFIERE ${par##*:}  (canonico $(sha "$ORIG"), instalado $( [ -f "$DESTINO" ] && sha "$DESTINO" || echo 'no esta' ))"
      faltan=1
    fi
  else
    if [ -f "$DESTINO" ] && [ "$(sha "$ORIG")" != "$(sha "$DESTINO")" ]; then
      APARTADO="$AQUI/upstream_parole/_apartado/$(date +%Y%m%d_%H%M%S)"
      mkdir -p "$APARTADO/$(dirname "${par##*:}")"
      mv "$DESTINO" "$APARTADO/${par##*:}"
      echo "  apartado ${par##*:} -> ${APARTADO#${RAIZ}/}/${par##*:}"
    fi
    mkdir -p "$(dirname "$DESTINO")"
    cp "$ORIG" "$DESTINO"
    echo "  instalado ${par##*:}  $(sha "$DESTINO")"
  fi
done

if [ "$COMPROBAR" -eq 1 ]; then
  [ "$faltan" -eq 0 ] && echo "lo instalado coincide con la copia canonica"
  exit "$faltan"
fi

echo
echo "ahora: (cd $DEST && .venv/bin/python tools/test_prosody_interp.py)   # debe salir 8/8"
