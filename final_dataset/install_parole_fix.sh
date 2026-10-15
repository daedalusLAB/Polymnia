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
RAIZ="$(dirname "$AQUI")"                                                    # raiz de trendsI
DEST="$RAIZ/Polymnia"
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
MAPA=(
  "upstream_parole/process_prosody.R:create_datasets/parole/scripts/process_prosody.R"
  "upstream_parole/prosody_align.R:create_datasets/parole/scripts/prosody_align.R"
  "upstream_parole/test_prosody_interp.py:tools/test_prosody_interp.py"
)

sha() { sha256sum "$1" | cut -c1-16; }
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
