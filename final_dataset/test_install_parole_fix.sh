#!/usr/bin/env bash
# Tests for install_parole_fix.sh: which tree it installs into, and what it refuses to do.
#
# Why this file exists: `final_dataset/` moved inside the repository (Polymnia/final_dataset/),
# and the installer kept resolving its target as `<parent>/Polymnia`. Inside the repo that
# parent *is* Polymnia, so the command stopped working with
# "no existe .../Polymnia/Polymnia/create_datasets/parole/scripts" -- the exact path a fresh
# clone hits when it follows the README. These checks pin both layouts, and pin the refusal
# too, so a green run cannot come from a check that stopped testing anything.
#
# Output: one line per check; exit 0 if all pass, 1 otherwise.
set -uo pipefail

AQUI="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd -P)"
HECHO=0
FALLOS=0

check() {  # check <nombre> <0|1> [detalle]
  HECHO=$((HECHO + 1))
  if [ "$2" -eq 0 ]; then
    echo "[ok]    $1"
  else
    FALLOS=$((FALLOS + 1))
    echo "[FAIL]  $1${3:+  -- $3}"
  fi
}

# The installer only needs itself plus its canonical copies; everything else is scaffolding.
plantar() {  # plantar <directorio_con_final_dataset>
  mkdir -p "$1/upstream_parole"
  cp "$AQUI/install_parole_fix.sh" "$1/install_parole_fix.sh"
  cp "$AQUI/upstream_parole/process_prosody.R" \
     "$AQUI/upstream_parole/prosody_align.R" \
     "$AQUI/upstream_parole/test_prosody_interp.py" "$1/upstream_parole/"
}

marcar_repo() {  # marcar_repo <raiz_de_polymnia>  makes the tree look like a Polymnia checkout
  mkdir -p "$1/bin" "$1/create_datasets/parole/scripts" "$1/tools"
  printf '#!/usr/bin/env bash\ntrue\n' > "$1/bin/polymnia"
  chmod +x "$1/bin/polymnia"
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# --- layout 1: final_dataset/ inside the repository (the layout since PR #8) -------------
POLY="$tmp/en_repo/Polymnia"
plantar "$POLY/final_dataset"
marcar_repo "$POLY"
sal="$(bash "$POLY/final_dataset/install_parole_fix.sh" 2>&1)"; rc=$?
check "repo layout: installs without error" $rc "$(echo "$sal" | tail -1)"
n=$(find "$POLY/create_datasets/parole/scripts" "$POLY/tools" -type f 2>/dev/null | wc -l)
check "repo layout: the three files land in the repo" $([ "$n" -eq 3 ]; echo $?) "hay $n de 3"
check "repo layout: nothing is created under Polymnia/Polymnia" \
  $([ ! -e "$POLY/Polymnia" ]; echo $?) "se creo Polymnia/Polymnia"
sal="$(bash "$POLY/final_dataset/install_parole_fix.sh" --comprobar 2>&1)"; rc=$?
check "repo layout: --comprobar agrees with the canonical copies" $rc "$(echo "$sal" | tail -1)"

# A real difference must still be red, otherwise the agreement above proves nothing.
printf '# tampered\n' >> "$POLY/create_datasets/parole/scripts/prosody_align.R"
sal="$(bash "$POLY/final_dataset/install_parole_fix.sh" --comprobar 2>&1)"; rc=$?
check "repo layout: a tampered file is reported as DIFIERE" \
  $([ "$rc" -ne 0 ] && echo "$sal" | grep -q "DIFIERE"; echo $?) "$(echo "$sal" | tail -1)"
sal="$(bash "$POLY/final_dataset/install_parole_fix.sh" 2>&1)"; rc=$?
apartado="$(find "$POLY/final_dataset/upstream_parole/_apartado" -name prosody_align.R 2>/dev/null | head -1)"
check "repo layout: reinstalling parks the previous file under _apartado/" \
  $([ "$rc" -eq 0 ] && [ -n "$apartado" ]; echo $?) "no aparecio el archivo apartado"

# --- layout 2: final_dataset/ next to the checkout (the layout it was written for) ------
HER="$tmp/hermano"
plantar "$HER/final_dataset"
marcar_repo "$HER/Polymnia"
sal="$(bash "$HER/final_dataset/install_parole_fix.sh" 2>&1)"; rc=$?
check "sibling layout: installs without error" $rc "$(echo "$sal" | tail -1)"
n=$(find "$HER/Polymnia/create_datasets/parole/scripts" "$HER/Polymnia/tools" -type f 2>/dev/null | wc -l)
check "sibling layout: files land in the neighbouring checkout" $([ "$n" -eq 3 ]; echo $?) "hay $n de 3"
check "sibling layout: nothing leaked into final_dataset/" \
  $([ ! -e "$HER/final_dataset/create_datasets" ] \
    && [ ! -e "$HER/final_dataset/tools/test_prosody_interp.py" ]; echo $?)

# --- an explicit --dest wins over the probe ---------------------------------------------
OTRA="$tmp/override"
plantar "$OTRA/final_dataset"
marcar_repo "$OTRA/Polymnia"
mkdir -p "$OTRA/otro_sitio/create_datasets/parole/scripts" "$OTRA/otro_sitio/tools"
sal="$(bash "$OTRA/final_dataset/install_parole_fix.sh" --dest "$OTRA/otro_sitio" 2>&1)"; rc=$?
n=$(find "$OTRA/otro_sitio" -type f 2>/dev/null | wc -l)
check "--dest overrides the layout probe" $([ "$rc" -eq 0 ] && [ "$n" -eq 3 ]; echo $?) \
  "$(echo "$sal" | tail -1)"

# --- refusal: a checkout with no PAROLE must stay an error, naming the path -------------
SIN="$tmp/sin_parole/Polymnia"
plantar "$SIN/final_dataset"
mkdir -p "$SIN/bin" && printf '#!/usr/bin/env bash\ntrue\n' > "$SIN/bin/polymnia" && chmod +x "$SIN/bin/polymnia"
sal="$(bash "$SIN/final_dataset/install_parole_fix.sh" 2>&1)"; rc=$?
check "no PAROLE: exits non-zero" $([ "$rc" -ne 0 ]; echo $?) "salio $rc"
check "no PAROLE: names the directory it looked for" \
  $(echo "$sal" | grep -q "create_datasets/parole/scripts"; echo $?) "$(echo "$sal" | head -1)"

echo
echo "$((HECHO - FALLOS))/$HECHO checks passed"
[ "$FALLOS" -eq 0 ] || echo "$FALLOS FAILED"
exit $((FALLOS > 0))
