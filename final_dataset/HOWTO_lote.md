# HOWTO — lanzar el pipeline contra tu propio TSV

De un `.txt` tabulado de NewsScape a `final_frames` + `final_videos`, con una carpeta de
depuración al lado. Los tiempos y rendimientos de esta hoja están **medidos en esta máquina**
(RTX 4090), no estimados.

Todo ocurre dentro de una carpeta de lote. No escribe sobre tus entradas, y al reanudar lo que
sustituye se aparta a `informes/_anterior/<hora>/` en vez de borrarse.

---

## 0. Comprobaciones antes de gastar GPU

```bash
# Tu carpeta de trabajo: la que contiene Polymnia/ y donde nacen las carpetas de lote.
cd "$TRABAJO"                        # p. ej. ~/proyectos/trendsI

Polymnia/bin/polymnia doctor          # GPU, OpenPose, TalkNet-ASD, WhisperX, Praat, R
                                      #   + bloque «entrega»: pesos de la criba y .env de Theia
final_dataset/install_parole_fix.sh --comprobar   # la prosodia arreglada sigue instalada
nvidia-smi --query-gpu=memory.used --format=csv   # que esté libre el device 0
curl -s -o /dev/null -w "%{http_code}\n" \
  "https://gallo.case.edu/cgi-bin/snippets/newsscape_mp4_snippet.cgi?file=x&start=0&end=1"
```

El `--comprobar` del medio no es ceremonia: el código que produce la prosodia vive en
`Polymnia/create_datasets/parole/`, y Polymnia tiene `create_datasets/` en `.gitignore`. O sea
que ese código **no está en ningún git**: si alguien reinstala PAROLE o trae el árbol de otra
máquina, el arreglo desaparece en silencio y el lote sale con la spline vieja (pitch a 0 en un
tercio de los fotogramas y valores que Praat no puede producir). Si sale `DIFIERE`, se
restaura con `final_dataset/install_parole_fix.sh` y se mira `final_dataset/upstream_parole/LEEME.md`.

El último debe dar `307` (redirige al caché). Si da error de conexión, Gallo está caído y no
hay nada que lanzar: todo lo demás depende de las descargas.

**La clave de Theia no hace falta tocarla.** El script ya exporta `THEIA_API_BASE`,
`THEIA_API_KEY` y `THEIA_MODEL` desde `Polymnia/.env` antes de llamar a Theia, y eso es
necesario: `theia.py` lee el default de `--api-key` **antes** de cargar el `.env`, así que sin
exportarla la petición sale con `Bearer none` y el endpoint contesta `401`. (Si lanzas `polymnia
theia` a mano sin exportar nada, tropiezas con eso; medido: `error=1` sin exportar, `ok=1`
exportando.)

Si aun así ves `401` en el log, entonces sí es la clave o el endpoint:

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $(sed -n 's/^THEIA_API_KEY=//p' Polymnia/.env)" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$(sed -n 's/^THEIA_MODEL=//p' Polymnia/.env)\",\"messages\":[{\"role\":\"user\",\"content\":\"di ok\"}],\"max_tokens\":4}" \
  "$(sed -n 's/^THEIA_API_BASE=//p' Polymnia/.env)/chat/completions"
```

Prueba una generación de verdad. `/models` no vale: contestaba `200` mientras la generación
devolvía `401`. Y si la clave está muerta, **lanza igual**: solo pierdes las 7 columnas de
anotación visual y se recupera en 1 minuto (`--desde theia --force`).

## 1. Valida el TSV sin descargar nada

```bash
TSV=mi_tsv.txt
Polymnia/bin/polymnia check-tab $TSV --offset 2.0 --show 6
```

Qué tienes que querer ver:

* líneas leídas = las de tu fichero, `incompletas: 0`;
* duraciones del orden de **0,1–2 s** (el TSV marca la frase; el clip real es eso ± 2,5 s);
* el **divisor del `msec`**: es 1000 si el nombre del programa empieza por `2017`, y 100 en
  caso contrario. Si ves duraciones de minutos, el divisor no es el que creías y vas a
  descargar clips de otro trozo del vídeo.

Las líneas con campos de menos **no se apartan a ningún sitio**: se quedan en
`tsv/efectivo.tsv` y la etapa 1 te dice cuántas son y cuáles (`incompletas: N` y
`⚠️ líneas con pocos campos: [(2, 5), …]`, en `logs/01_check_tab.log`). Descárgalas
limpias o quítalas tú del TSV; si no, esas líneas se pierden en el camino sin que el recuento
final se queje.

## 2. Prueba barata primero (8 clips, ~3 minutos)

```bash
final_dataset/pipeline_lote.sh --tsv $TSV --lote lote3 --limit 8 --hasta preseleccion
```

Eso hace tabulado + descargas + hablante activo + criba de muñecas, y para antes de OpenPose
(que es lo caro). Al terminar:

```bash
cat lote3/RENDIMIENTO.md                          # el embudo: líneas → descargados → con máscara → admitidos
head -5 lote3/munecas.csv                          # score de muñeca de TODOS los clips, ordenados
ls lote3/videos/descartados/*/ | head              # quién se cayó y por cuál motivo
```

Con eso decides si el TSV merece la GPU. Referencia medida en este corpus: de 71 líneas
nuevas, 22 pasaron el hablante activo (31 %) y 5 de esas 22 pasaron la criba de muñecas
(23 %). En el run original de 100 líneas: 51 → 39 → 29 clips, y solo 4 con muñecas medibles.
**Si te salen menos de ~5 clips por 100 líneas, el problema es el TSV, no el pipeline.**

## 3. La tanda completa

```bash
final_dataset/pipeline_lote.sh --tsv $TSV --lote lote3
```

Puedes dejarlo en segundo plano (`nohup ... > fuera.log 2>&1 &`); cada etapa escribe su log
dentro del lote, así que no dependes de la ventana.

Costes medidos por clip: descarga 3,3 s · hablante activo 7,8 s · criba 0,8 s · OpenPose
14,3 s (solo sobre los admitidos) · limpieza 1,5 s · voz 7,8 s · DuckDB y dataset final ~1 s.
Los 4 clips de prueba: **144 s** de punta a punta. Para 100 líneas, lo que manda es ASD sobre
las 100 (~13 min) más OpenPose sobre los admitidos.

Flags que vas a usar de verdad:

| flag | nota |
|---|---|
| `--limit N` | las N primeras líneas. La prueba barata |
| `--desde ETAPA --force` | reanudar. Etapas: `tsv descarga asd preseleccion openpose limpieza voz merge theia final` |
| `--hasta ETAPA` | parar después de una etapa |
| `--min-sp-both 0.8` / `--yolo-conf 0.7` | umbrales de la criba. Bajar `--min-sp-both` a 0.6 te da más clips con peores muñecas |
| `--no-extended` | features versión 2 (sin el bloque 3) |
| `--readme MI.md` | se publica como `final/DATASET.md` y entra en el SHA-256 |
| `--skip-theia` | si sabes que la clave está mal |

## 4. Qué mirar al terminar, en este orden

```bash
sed -n '1,20p' lote3/RENDIMIENTO.md
cat lote3/informes/cobertura_munecas.csv
ls lote3/final/
grep -m1 "voice chain regime" lote3/final/DATASET.md
Polymnia/.venv/bin/python - <<'PY'
import json; m=json.load(open("lote3/final/manifest.json"))
f=m["features"]; c=m["clip_id_mapping"]
print("features:", f["version"], "| extendidas:", f["extended"])
print("lineas del tab:", c["tab_rows"], "emparejados:", c["matched"],
      "(exactos", c["matched_exact_id"], ") sin emparejar:", len(c["unmatched"]))
print("holgura real de los clips:", c["processed_padding_s"])
PY
```

El `grep` del medio contesta en que cadena de prosodia estás. `**tramos**` es la arreglada: la
voz es NULL donde Praat no seguía pista, `voice_present` existe y no hay valores fuera de las
cotas. `**spline**` significa que el lote salió del código viejo (el `--comprobar` del punto 0
no se corrió, o salió `DIFIERE`): los números de voz de ese lote no son medidas por fotograma.
El mismo grep te dice cuantos fotogramas tienen voz medida, que es lo que hay que reportar junto
a cualquier estadístico de voz.

Señales de que algo va mal, y qué significa:

| ves | qué es |
|---|---|
| `emparejados` < `lineas del tab`, o `processed_padding_s` ≠ `[2.0]` | el `--offset` de esta pasada no coincide con la descarga. Vuelve a bajar con el mismo offset, o construye con el offset que consta en el manifiesto |
| el pipeline corta en la etapa 1 o 2 con *«no están cortados a N s por lado»* | los clips que ya había en `videos/raw` se bajaron con otra holgura que la de esta pasada. Lo dice `logs/holgura.log`, con la holgura medida: construye con `--offset <medida>` o vacía `videos/raw` y vuelve a bajar. Reanudar con `--desde` no lo arregla: los clips ya están cortados |
| `Parquet en 1_persons: 0` | la etapa 6 no salvó ningún clip: `informes/limpieza_fantasmas.json` dice por qué se fue cada candidato |
| `clips en la base: N` menor que los Parquet | el DuckDB no tragó todo; mira `logs/08_merge.log` (el paso `hefesto` deja exit 0 aunque fallen inserts) |
| `anotación visual: 0 clips con etiquetas, N con error` | el `error` del log dice por qué. Un `401` **con la clave ya exportada** sí es credencial o endpoint; arreglado eso, `--desde theia --force` anota solo los que faltan |
| `cobertura_munecas.csv` con dos columnas por debajo de 60 % | clips sin muñecas medibles: no te van a dar features bimanuales, por mucho que OpenPose los aceptara |
| `voice chain regime \| **spline**` | el lote salió del PAROLE sin arreglar. Comprueba con `install_parole_fix.sh --comprobar`, instala, y regenera con `--desde voz --force` |
| `manifest.json` con `"readme": null` | alguien corrió la etapa final con `--sin-diccionario` (o el lote es de antes). Regenera como dice el punto 4.1 |
| `difere entre la medición y la publicación` | la etapa final aborta a propósito: el diccionario y el README se midieron sobre unos parquet que no son los publicados. Relanza `--desde final --force` |

### 4.1 El diccionario de columnas (esto es lo que se le da a la clase)

Sale **solo**, al final de la etapa 10. Tres señales de que salió: la linea `diccionario: N
columnas documentadas` (283 sin `voice_present`, 284 con él), la linea `comprobado: lo publicado
es byte a byte lo que el diccionario
describe`, y `final/DATASET.md` existiendo con `"readme": "DATASET.md"` en el manifiesto. Lo que
produce, en este orden de utilidad para un alumno:

| fichero | qué es |
|---|---|
| `final/DATASET.md` | el README: portada con los números de esta entrega + diccionario completo. Es lo que se manda con los parquet |
| `DICCIONARIO_COLUMNAS.md` (raíz del lote) | el mismo documento, sin portada, para leerlo con el lote delante |
| `informes/DICCIONARIO_COLUMNAS.csv` | una fila por columna, para filtrar y ordenar |

Cada fila lleva qué mide, la fórmula y las unidades, el tipo, y **cuántos clips de este lote
tienen dato de verdad** en esa columna. Ese último número es la razón de existir del fichero:
`cos_phase_mean` está en el 89,7 % de los clips de `OZBEKSMALL` y en el 10,3 % de la entrega
vieja; la columna existe en las dos, el significado no es el mismo.

Para un lote viejo, o uno que se corrió con `--sin-diccionario`, se genera a mano:

```bash
Polymnia/.venv/bin/python final_dataset/diccionario_columnas.py \
    --final lote3/final \
    --csv lote3/informes/DICCIONARIO_COLUMNAS.csv \
    --md  lote3/DICCIONARIO_COLUMNAS.md
```

Tres cosas que conviene saber de este paso:

* **No se escribe dentro de `final/` a mano.** Esa carpeta es un paquete atómico: su
  `manifest.json` lista los sha256 de lo que había al publicarla. El generador **se niega** si le
  pasas un `--readme` dentro de la entrega que describe. Lo único que entra ahí es `DATASET.md`, y
  entra por `final_dataset.py --readme` antes de publicar.
* **Si el generador se cae, es una buena señal.** Su guard se niega a publicar si aparece una
  columna que nadie documentó (`undocumented columns: ...`). Un README incompleto que no lo
  dice es peor que ningún README. La salida de la columna nueva va en
  `diccionario_columnas.py`, en las tablas `MIDE_*` / `FORMULA_PROPIA`, citando la función que
  la calcula.
* Las fórmulas no están redactadas a mano: se leen de las constantes de `gesture_features.py`
  (`FEATURE_UNITS`, `EXTENDED_FEATURE_UNITS`) y de la función que produce cada columna, así que
  si el código cambia, el diccionario cambia al regenerarlo. Está comprobado en
  `final_dataset/test_diccionario_columnas.py` (`TOTAL: todo OK`).

## 5. Reanudar sin miedo

```bash
final_dataset/pipeline_lote.sh --tsv $TSV --lote lote3 --desde openpose --force
```

Lo que había antes de esa etapa se va a `lote3/informes/_anterior/<hora>/` con la misma
estructura de carpetas. Está comprobado que dos reanudados seguidos dan el mismo número de
clips y de Parquet. Si quieres comparar lo que cambió, mira `_anterior` con `diff -r`.

## 6. Dos cosas que NO debes tocar

* **El `--offset` es uno solo para toda la tanda.** Cambiarlo a mitad hace que los IDs de los
  clips ya procesados no coincidan con tu TSV y el dataset final se quede huérfano.
* **No pongas `--gate` todavía.** Medido: con el filtro de visibilidad de `argos` (muñecas
  ≥80 %), los 5 clips que habían pasado la criba fueron rechazados **todos** con el motivo
  `multi_person`, y no por falta de cobertura: 4 de ellos tenían 85–93 % de muñecas. El
  rechaza viene de candidatos que fabrica la propia máscara de ASD en su borde, y decidir si
  eso es "1 persona" o no es una decisión de contrato que está pendiente. La etapa 6 del
  script es el apaño honesto y trazable mientras tanto.
* **No publiques un `DATASET.md` de otra tanda** con `--readme` sin recalcular sus números de
  cobertura. Un README con cifras de otro lote es peor que ningún README.

## 7. Si algo peta, por dónde mirar

```bash
ls lote3/logs/                       # 01_check_tab … 10_final (más 10a_medicion y
                                     # 10b_diccionario), uno por etapa
grep -iE "error|traceback|❌" lote3/logs/*.log | head
tail -30 Polymnia/failed_downloads.log     # va al cwd del repo y hace APPEND entre tandas
ls lote3/videos/descartados/* | head       # el motivo está en el nombre de la carpeta
```

`failed_downloads.log` es el que más engaña: si no lo has limpiado, verás fallos de tandas
anteriores y creerás que Gallo está caído.
