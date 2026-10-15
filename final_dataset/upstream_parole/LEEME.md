# Arreglo de la prosodia de PAROLE (vendorizado en `Polymnia/create_datasets/parole/`)

## Que esta mal, en el codigo que produce los datos

`create_datasets/parole/scripts/process_prosody.R` aterriza las pistas de Praat (1-10 ms)
en la rejilla de fotogramas (~33 ms) con **una unica spline cubica** sobre todos los
puntos no-NA, evaluada en **todos** los fotogramas. Tres defectos, cada uno con su cifra:

1. **Rellena el silencio.** `valid <- !is.na(values)` tira los huecos *antes* de ajustar,
   y luego evalua en todos los timestamps: con 4 puntos validos en todo el clip, el 100 %
   de los fotogramas sale con numero. En un clip real de 5,2 s, 1715 de 5129 frames de
   pitch son silencio (Praat lo escribe como candidato `mute`, `frequency = 0`).
2. **Extrapola fuera de la pista y no se puede apagar**: `splinefun` no admite el
   argumento `extrapolation` (comprobado: `unused argument`). Con `To Formant (burg):
   0.001, 5, 5500` no puede salir ninguna frecuencia por encima de 5500 Hz, y en los
   datos entregados habia una F5 de **29.144.321 Hz** y otra de **-658.034 Hz**. Con
   `To Harmonicity (cc)` el suelo de Praat es **-200 dB**; se midieron **-246,7 dB** en
   una entrega y **-1193,6 dB** en este clip de prueba.
3. **Convierte "no medido" en un valor fisico plausible**: `df_final$pitch[df_final$pitch
   < 10] <- 0` pinta de 0 la extrapolacion. En una entrega, **1408 de 4553 fotogramas
   (30,9 %) decian `pitch == 0`**, y la media de pitch salia 125,3 Hz en vez de 182,5 Hz.

Y no habia ninguna bandera de procedencia: las munecas publican `right_interpolated` y su
contador; la voz no publicaba nada, asi que NULL, silencio y extrapolacion vestian igual.

## Que hace el arreglo

`prosody_align.R` (nuevo) sustituye la spline por `align_track()`: parte cada pista en
**tramos** de muestras consecutivas -un hueco mayor que `MAX_GAP_S` rompe el tramo- y,
dentro de cada tramo, interpola **lineal**. Fuera de un tramo, NULL. Con paso de pista de
1-10 ms frente a fotogramas de 33 ms la lineal es practicamente una lectura directa, y
tiene la propiedad que la cubica no tiene: ningun valor sale del rango medido.

* `MAX_GAP_S = 0,1 s`, declarado y justificado: medido en un clip real, las rachas de
  silencio del pitch tienen mediana 70 ms, p90 238 ms y maximo 359 ms. 0,1 s son ~3
  fotogramas a 30 fps: se puentea lo que sigue siendo la misma emision, se deja NULL
  donde hay silencio de verdad.
* `MIN_TRACK_POINTS = 2`: con una sola marca de tiempo no hay nada que interpolar.
* `marcar_silencio_pitch()`: el `frequency = 0` de Praat es ausencia de pista voiced, no
  una frecuencia. Llega al alineador como NULL. Se va el centinela `< 10 -> 0`.
* `voice_present` (nueva columna logica): TRUE solo donde hay pitch medido. Es el
  companero de `right_n_speed` en la parte de voz.

El contrato con `hefesto.py` se respeta: sigue habiendo **una fila por fotograma**.

## Medido en un clip real, misma entrada de Praat, codigo viejo contra nuevo

CNN Newsroom, 155 fotogramas, 5,2 s (ficheros de Praat generados con el
`extract_prosody.praat` del repo):

| senal | antes | despues |
|---|---|---|
| `pitch` | 100 % cubierto, 0,0..339,0 Hz | 78,1 %, **145,6..339,0 Hz** |
| `harmonicity` | **-1193,6 .. 37,3 dB** | **-200,0 .. 20,5 dB** (el suelo de Praat y su techo real) |
| `formant.5_frequency` | **-44.218 .. 19.793 Hz** | **3966 .. 5368 Hz** |
| formantes fuera del techo del analisis (>5500 Hz) | 40 fotogramas | 0 |
| `harmonicity` bajo el suelo de Praat (<-200) | 33 fotogramas | 0 |
| `pitch == 0` | 49 fotogramas | 0 |
| media de `pitch` | 136,0 Hz | **199,8 Hz** |
| `voice_present` | no existia | 121/155 fotogramas con voz medida |

## Y medido dentro del pipeline, sobre los mismos clips

El lote de prueba (`/tmp/lote_smoke`, 4 clips, 620 fotogramas) se regenero con
`pipeline_lote.sh --desde voz --force`, o sea con Praat, el merge y el dataset final otra vez
sobre las MISMAS pistas:

| senal | antes (spline) | ahora (tramos) |
|---|---|---|
| `pitch == 0` en `final_frames` | 194 fotogramas | **0** |
| `harmonicity` por debajo de -200 dB | 90 fotogramas | **0** |
| formantes fuera de 0-5500 Hz | 143 valores | **0** |
| media de `pitch` | 128,3 Hz | **188,4 Hz** |
| `voice_present` | no existia | 485 con pista, 131 sin pista, 4 sin pista de pitch |

Y el README que genera el pipeline escribe el regimen medido (`tramos` o `spline`) en su tabla de
voz, asi que una entrega no puede decir que esta arreglada sin que se compruebe sobre sus
propios numeros.

## Por que hace falta un instalador

Polymnia tiene `create_datasets/` en `.gitignore` (linea 23, y esa linea **viene de
upstream**), y su `README.md` dice que ahi dentro va
[daedalusLAB/parole](https://github.com/daedalusLAB/parole). O sea: el codigo que produce
la prosodia **no esta en ningun git** dentro de esta instalacion. Sin copia canonica aqui,
el arreglo solo existiria en el disco de esta maquina.

```bash
final_dataset/install_parole_fix.sh               # instala (aparta lo anterior a _apartado/)
final_dataset/install_parole_fix.sh --comprobar   # dice si lo instalado coincide (sha256)
```

Copia canonica de los tres ficheros: `upstream_parole/`. El test vive en
`Polymnia/tools/test_prosody_interp.py` (ese directorio SIES versionado) y cubre 8 casos:
silencio largo sin pitch, nada despues del fin de la pista, todo valor dentro del rango de
su pista, el 0 de Praat no se publica, hueco corto puenteado con pitch real, una fila por
fotograma, `voice_present` coherente, y que la intensidad medida no se NULL-ee de mas.

## Mergeado arriba: https://github.com/daedalusLAB/parole/pull/1

**Mergeado el 2026-10-14 en `main`** (commit `87d0f29`), desde `raul-agent:fix/prosody-align-spans`
contra `main` (4 ficheros, +401/-25): `scripts/process_prosody.R`, `scripts/prosody_align.R`,
`tests/test_prosody_interp.py` (2/8 con el codigo anterior, 8/8 con este) y `CHANGELOG.md` con el
formato bilingue EN/ES que ya usaban. Comprobado tras el merge: el `process_prosody.R` y el
`prosody_align.R` de `main` son **identicos byte a byte** a la copia vendorizada aqui y a la
copia canonica de este directorio, asi que no hay nada que reinstalar.

Detalle del CI, para no llevarse a engano: el run del PR (`main.yml`) **no llego a ejecutar el
pipeline**. Estaba en `action_required` (PR desde fork) y GitHub lo marco como fallo al cerrarse
el PR; en el merge, el run que si corre es el del push a `main`. Y `validation.yml` falla en 0 s
en cualquier rama: su YAML es invalido.

Dos cosas que se encontraron al mirar el repo arriba y que van en la descripcion del PR:

* `.github/workflows/` esta en el `.gitignore` de PAROLE (linea 23), asi que **no** he tocado
  CI. El test se suelto: `python tests/test_prosody_interp.py` (~10 s, `Rscript` + `pandas`).
* `.github/workflows/validation.yml` **no es YAML valido** (linea 50: el `uses:` del paso
  `Checkout` con 2 espacios en vez de 6). `main.yml` si lo es. Es anterior a este PR.

`patches/prosody-align-sin-extrapolacion.patch` sigue aqui porque es lo que se aplico (contra el
`process_prosody.R` original, que en `main` esta identico: comprobado con `diff`) y porque sirve
para regenerar el arreglo si algun dia PAROLE se reinstala limpio:

```bash
git clone git@github.com:daedalusLAB/parole.git && cd parole
git apply --check ~/miscosas/trendsI/final_dataset/upstream_parole/patches/prosody-align-sin-extrapolacion.patch
git apply ~/miscosas/trendsI/final_dataset/upstream_parole/patches/prosody-align-sin-extrapolacion.patch
```

El test de arriba esta adaptado al layout de PAROLE (`scripts/`); el de aqui
(`test_prosody_interp.py`) apunta a `create_datasets/parole/scripts/`, que es donde vive en
Polymnia. Son la misma comprobacion con dos rutas distintas: si el de PAROLE cambia, se refleja
aqui a mano y `install_parole_fix.sh --comprobar` lo pilla.

### Si algun dia sale `DIFIERE` con PAROLE ya actualizado

Desde el merge, instalar PAROLE limpio desde upstream **ya trae el arreglo**, asi que un
`DIFIERE` puede ser buena noticia (subiste a una version mas nueva) o mala (perdiste el arreglo).
Se distingue en un segundo:

```bash
grep -c align_track Polymnia/create_datasets/parole/scripts/process_prosody.R
#  1 o mas -> esta alineado por tramos: va bien, solo hay que refrescar la copia canonica
#  0        -> volvio atras: instala con `final_dataset/install_parole_fix.sh`
```

## Variante guardada: pinear Praat a 6.4.30 (`patches/parole-instalador-pin-6430.patch`)

El PR https://github.com/daedalusLAB/parole/pull/2 deja **por defecto la ultima release**, que hoy
es Praat 7.0.02. Nuestros `.Pitch` salen de **Praat 6.4.30** (`Polymnia/create_datasets/parole/env/praat`,
6.4.30 de abril de 2025), asi que existe una variante que invierte el default: `PRAAT_TAG` fijado a
`v6.4.30`, con `PRAAT_TAG=latest` para volver a la descubierta por la API y `PRAAT_URL=<url>` para
poner la URL a mano. Esta **guardada y no enviada** (decicion del usuario: dejar el PR tal cual).

```bash
# sobre la cabaza del PR #2 (commit 6d23534); comprobar antes de aplicar
git apply --check final_dataset/upstream_parole/patches/parole-instalador-pin-6430.patch
```

Medido en esta maquina con las cuatro rutas, descarga real:

| invocacion | instala |
| --- | --- |
| sin variables | `praat6430_linux-intel64-barren.tar.gz` -> `Praat 6.4.30 (April 29 2025)` |
| `PRAAT_TAG=latest` | via API -> `praat7002_linux-x64v3-barren.tar.gz` -> `Praat 7.0.02` |
| `PRAAT_TAG=v7.0.02` | nombre construido sin API -> `Praat 7.0.02` |
| `PRAAT_TAG=v9.9.99` | sale 1 con «no hay asset barren» y las dos salidas alternativas |

Dos razones de peso para el pin, aparte de la reproducibilidad: el camino normal ya no depende de
`api.github.com` (sin token, 60 peticiones/hora por IP) y evita el asset `x64v3` de 7.x, que exige
AVX2 y en una maquina vieja revienta con *illegal instruction* diciendo que Praat «esta instalado».
