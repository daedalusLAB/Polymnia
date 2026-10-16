# Montar un dataset final desde un TSV nuevo

Ruta completa desde un `.txt` tabulado de NewsScape hasta `final_frames` + `final_videos`.
Cada paso dice **qué comprobar**: en este pipeline un `exit 0` no prueba nada (hay etapas
que fallan en silencio y etapas que filtran sin decirlo).

Variables que usamos todo el rato:

```bash
# Tu carpeta de trabajo: la que contiene Polymnia/ y donde nacen las carpetas de lote.
cd "$TRABAJO"                        # p. ej. ~/proyectos/trendsI
TSV=mi_tsv.txt                 # tu TSV, sin cabecera
LOTE=lote3                     # carpeta de trabajo de esta tanda
OFFSET=2.0                     # holgura por lado; DEBE ser la misma en todo el pipeline
POLY=Polymnia/bin/polymnia     # wrapper: pone el venv, R, Praat y scenedetect en el PATH
PY=Polymnia/.venv/bin/python
```

El wrapper **no es opcional** en las etapas de GPU: llamado suelto, TalkNet-ASD no
encuentra `scenedetect` ni `python`, no produce nada y **devuelve exit 0**.

---

## El camino corto: un comando y una carpeta

```bash
# Tu carpeta de trabajo: la que contiene Polymnia/ y donde nacen las carpetas de lote.
cd "$TRABAJO"                        # p. ej. ~/proyectos/trendsI
final_dataset/pipeline_lote.sh --tsv mi_tsv.txt --lote lote3
```

Hace los 7 pasos de abajo en orden y deja **una sola carpeta** con todo lo que hace falta
para depurar a mano. Medido con 4 clips desde cero: **144 s** (OpenPose es lo que manda,
unos 14 s por clip).

| dentro de `lote3/` | qué es |
|---|---|
| `tsv/efectivo.tsv` | el TSV tal y como lo tragó el pipeline |
| `videos/raw/` | las descargas de Gallo |
| `videos/masked/` | lo que saca TalkNet-ASD (solo el hablante activo) |
| `videos/1_person/` | los clips que pasaron la criba de muñecas: **esto es lo que ve OpenPose** |
| `videos/descartados/<motivo>/` | lo que se quedó fuera, agrupado por motivo (`asd_sin_mascara`, `preseleccion_sp_bajo`, `clasificado_multipersona`) |
| `dataset/OpenPose/<clip>/JSON_FILES` | salida cruda de OpenPose |
| `dataset/OpenPose/<clip>/JSON_FILES_limpio` | la misma salida sin los candidatos que no son cuerpos (etapa 6) |
| `dataset/1_persons/parquet_files/` | un Parquet por clip (lo que hace el `dfMaker` de `multimolang`) |
| `dataset/1_persons/parquet_speech_analysis/` | prosodia (Praat) |
| `dataset/1_persons/whisperx_framer_output/` | palabras por fotograma |
| `<lote>.duckdb` | la tabla larga `multi_data` más `video_annotations` |
| `final/` | `final_frames` y `final_videos` (Parquet + CSV), su `manifest.json` y **`DATASET.md`** (el README generado) |
| `informes/` | `cobertura_munecas.csv`, `limpieza_fantasmas.json`, `conteos.tsv`, `DICCIONARIO_COLUMNAS.csv` y `_anterior/` |
| `logs/` | un log por etapa, numerado (`05_openpose.log`, `06_limpieza.log`, …) |
| `RENDIMIENTO.md` | coste y conteos de cada etapa, en una tabla |
| `DICCIONARIO_COLUMNAS.md` | una fila por columna de las dos tablas, con cobertura medida en **este** lote. **Lo genera la etapa final**, no hay que pedirlo |

Los flags que importan (el resto, `--help`):

| flag | para qué |
|---|---|
| `--limit N` | solo las N primeras líneas del TSV: la prueba barata |
| `--offset N` | holgura por lado, **la misma en todo el pipeline**. Defecto **2.0**, el mismo que usa el constructor suelto `final_dataset.py`. Lotes ya cortados con otra (los 29 clips entregados y OZBEKALL se bajaron con **2.5**) exigen que lo pases explícito: la etapa 1 y la 2 lo miden con `ffprobe` y cortan el run si no coincide |
| `--desde` / `--hasta ETAPA` | reanudar o cortar. Etapas: `tsv descarga asd preseleccion openpose limpieza voz merge theia final` |
| `--force` | reutilizar la carpeta del lote. Lo que se sustituye se **aparta** a `informes/_anterior/<hora>/`, no se borra. En la etapa `final` el apartadsolo ocurre al final de todo (medir → diccionario → publicar a scratch → comparar byte a byte): un abort deja la entrega anterior en su sitio, no el lote sin `final/` |
| `--min-sp-any` / `--yolo-conf` / `--frames` | umbrales de la criba de muñecas (0.6 / 0.7 / 15). `--min-sp-any` decide sobre `sp_any` (la MEJOR de las dos muñecas); `--min-sp-both` **ya no existe** y el pipeline se niega en vez de re-significarlo |
| `--gate` | además el filtro de visibilidad de `argos` con `--min-visible-percent` (defecto 80). **Por defecto NO** |
| `--sin-limpiar` / `--criterio-limpieza` | etapa 6: desactivarla, o cambiar su criterio (`todo`/`torso`/`negro`/`nada`) |
| `--skip-theia` / `--no-extended` / `--readme F` | anotaciones visuales, versión de features 2 en vez de 3, y tu README en vez del generado |
| `--sin-diccionario` | no genera `DICCIONARIO_COLUMNAS` ni README: `final/` queda **sin** `DATASET.md` y el manifiesto sale con `"readme": None` |
| `--device` / `--peso` / `--searchterm` | GPU, pesos YOLO y término del nombre de fichero. El `--searchterm` tiene que ser **el mismo con el que se descargó** el lote o la etapa final no casa ni un id (el defecto `beginning_to_end` casi nunca lo es) |

Qué **no** hace el script por ti: el *contenido* de un README escrito a mano. El generado sale
con los números medidos en el lote; si pasas `--readme F` con el de otra tanda, sus números de
 cobertura son mentira (el pipeline te lo avisa al publicarlo).

### Qué hay que mirar en cada etapa (un `exit 0` no prueba nada)

| etapa | señal de que va bien | si sale mal |
|---|---|---|
| `descarga` | `videos/raw` = nº de líneas del TSV | `Polymnia/failed_downloads.log` (va al `cwd` del repo y hace **append** entre tandas) |
| `asd` | `videos/masked` > 0 | `videos/descartados/asd_sin_mascara`; lo rechaza `Has single scene: False` |
| `preseleccion` | `admitidos` en `RENDIMIENTO.md` | `videos/descartados/preseleccion_sp_bajo`, con el score de cada clip en `munecas.csv` (`sp_any` decide, `sp_both` es ranking) |
| `openpose` | nº de Parquet ≈ nº de clips | los Parquet que ves **aún no son el veredicto**: espera a la etapa 6 |
| `limpieza` | `Parquet en 1_persons` > 0 y `en otros buckets` pequeño | `informes/limpieza_fantasmas.json` dice por qué se fue cada candidato |
| `voz` | `prosodia: N parquet` y `palabras: N clips` | `apate.log` y `ecos.log`, que caen **dentro del lote** |
| `merge` | `clips en la base: N` | `hefesto` puede dejar `exit 0` con inserts fallidos: mira `08_merge.log` |
| `theia` | `anotación visual: N clips con etiquetas, 0 con error` | el `error` de `09_theia.log` dice por qué; con las 7 columnas a NULL el resto del dataset sigue sirviendo |
| `final` | `final_frames`/`final_videos`, `emparejados: N (exactos N)`, `README publicado: DATASET.md` y `comprobado: lo que se va a publicar es byte a byte…` | si sale `ningun clip de este tab casa con los N ids`, lee el mensaje: imprime el `--searchterm` usado y dos ids de muestra, y casi siempre ESE es el fallo (no «faltan etapas»: eso lo supuse yo y me costó media hora y una entrega apartada). Si hay no-exactos, el `--offset` no coincide con la descarga; si peta con `undocumented columns`, el código ganó una columna que nadie describió: hay que describirla, no saltarse el guard |

### La etapa 6 decide cuántos clips sobreviven

`dfMaker` numera los candidatos por orden de aparición y `max_people_classification.R`
clasifica el clip por `max(people_id)`: **un solo fotograma con un candidato de adorno**
manda el clip entero a `2_persons`. Y la máscara de TalkNet-ASD fabrica candidatos de 2-3
puntos en su borde. Medido con OpenPose sobre el vídeo crudo: Kimmel 08-25 y Tavis 05-20
tienen **una** persona en los 155 fotogramas; sobre el enmascarado aparecen dos en 12 y en 7.

`final_dataset/limpiar_fantasmas.py` separa una cosa de otra por **persistencia**: un
candidato ambiguo que aparece en menos del 25 % de los fotogramas es ruido del borde de la
máscara y se poda; uno que se mantiene es una segunda persona de verdad y **no se toca**
(eso es una decisión de contrato, no de código). También caen los candidatos que se apoyan
sobre el negro de la máscara. Nunca toca los JSON originales: escribe `JSON_FILES_limpio` y
deja el porqué de cada candidato en `informes/limpieza_fantasmas.json`. En la tanda de prueba
los 4 clips quedaron en `1_persons`; sin esta etapa, 0.

```bash
# a mano, sobre un lote ya pasado por OpenPose:
Polymnia/.venv/bin/python final_dataset/limpiar_fantasmas.py --lote lote3
Polymnia/.venv/bin/python final_dataset/test_limpiar_fantasmas.py   # 19 comprobaciones
```

### Lo que se queda fuera del lote, y por qué no importa

TalkNet-ASD trabaja en `TalkNet-ASD/output/<clip>/` (CSV del hablante activo y vídeos
temporales) mientras corre, y **se lo limpia él mismo**: al acabar, `ls TalkNet-ASD/output`
está vacío. No hay nada que redirectar; si quieres verlo, mira el log (`03_asd.log`) que va
al lote. Los pesos están en `modelos/yolo11x-pose.pt` y la clave de Theia en `Polymnia/.env`.

---

## Paso 0 — el TSV

Formato esperado: 6+ campos separados por tabulador, sin cabecera.

```
1 programa.txt   2 id_clip(t__...)   3 start_sec   4 start_msec   5 end_sec   6 end_msec   [7 contexto]
```

**Trampa del divisor**: el `msec` se divide entre **1000 si el nombre del programa empieza
por `2017`**, y entre **100 en caso contrario** (`download_clips.py:26`). Un TSV de 2018 o
posterior usa 100, así que los tiempos salen desplazados si esperabas milisegundos.
Compruébalo antes de gastar nada:

```bash
$POLY check-tab $TSV --offset $OFFSET --show 6      # dry-run del tabulado, sin descargar
# y, si quieres verlo con tus propias reglas:
$PY - <<'PY'
import os
for i, line in enumerate(open("mi_tsv.txt")):
    p = line.rstrip("\n").split("\t")
    if len(p) < 6: print(f"linea {i+1}: solo {len(p)} campos"); continue
    name = os.path.basename(p[0]).replace(".txt","")
    div = 1000 if name.startswith("2017") else 100
    s = int(p[2]) + int(p[3])/div; e = int(p[4]) + int(p[5])/div
    if i < 5: print(f"{name[:48]:48s} {s:9.3f} -> {e:9.3f}  dur={e-s:5.2f}s  divisor={div}")
PY
```

Qué quieres ver: duraciones del orden de **0,1–2 s** (el TSV marca la frase; el clip real
son ±`OFFSET` alrededor). Si ves duraciones de minutos, el divisor no es el que creías.

---

## Paso 1 — descargar los clips

```bash
mkdir -p $LOTE
$POLY download --txt_file $TSV --searchterm beginning_to_end \
    --output_dir $PWD/$LOTE/videos/raw --offset $OFFSET
```

**Qué comprobar** (el script solo mira `tamaño > 0`; un HTML de error pasa):

```bash
ls $LOTE/videos/raw/*.mp4 | wc -l                      # cuantos bajaron
wc -c $LOTE/failed_downloads.log 2>/dev/null            # 0 = sin fallos registrados
$PY - <<'PY'
import cv2, glob, os
for p in sorted(glob.glob("lote3/videos_raw/*.mp4")):       # cambia lote3 si cambias el lote
    magic = b"ftyp" in open(p,"rb").read(16)
    c = cv2.VideoCapture(p); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); fps = c.get(cv2.CAP_PROP_FPS)
    ok, _ = c.read(); c.release()
    if not (magic and ok and n > 20): print("RARO:", os.path.basename(p), magic, ok, n)
print("listo")
PY
```

Esperado: fps **29,970**, ~151–182 frames para clips de ±2,5 s, y ningún `RARO`.
El nombre de fichero resultante es `programa_inicio_fin_searchterm.mp4`; **ese nombre es la
clave** con la que el dataset encajará con la base de datos. Si cambias `--searchterm` o
`--offset`, cambia la clave.

---

## Paso 2 — TalkNet-ASD (hablante activo, `raw` → `masked`)

```bash
mkdir -p $LOTE/videos/masked
$POLY asd --input_dir $PWD/$LOTE/videos/raw --output_dir $PWD/$LOTE/videos/masked \
    2>&1 | tee $LOTE/asd.log
grep -c "single scene: True"  $LOTE/asd.log     # clips que cumplen el criterio
grep -c "doesn't meet criteria" $LOTE/asd.log   # clips que ASD tira
ls $LOTE/videos/masked/*.mp4 | wc -l            # los que realmente hay
```

**ASD es un filtro, no una conversión**: exige escena única y score medio positivo de
hablante activo. Los que no cumplen **no generan fichero** y el proceso sigue con exit 0.
Cuenta la merma explícitamente; es tuya, no de las muñecas.

Coste medido: ~6,9 s por clip que pasa (29 clips en 3m20s en una RTX 4090).

Si quieres verificar que tu ASD es el mismo que produjo `entrega_1p`, pasa los 29
`entrega_1p/videos_raw` por ASD y compara SHA-256 con `entrega_1p/videos_masked`: debe dar
29/29 idénticos.

---

## Paso 3 — criba de muñecas (ANTES de OpenPose, que es lo caro)

El repo tiene `polymnia filter` para esto y **no sirve para gestos**: exige una sola persona
en cuadro. En el run original se llevó 12 de los 51 clips que ASD había aprobado, y los que
se llevó son precisamente los planos con dos personas donde el presentador sí mueve las manos
(`Access Hollywood`, `CNN Newsroom`). Para un dataset de gestos eso es descartar justo lo que
buscas. Aquí se alimenta `videos/1_person` desde la lista del pre-filtro, no desde `filter`.

No uses `polymnia filter` / `is_there_a_person_in_the_video.py` para esto: prueba
`xy != [0,0]` en vez de confianza, y en COCO un keypoint **ocluido existe con coordenada**,
así que aprueba muñecas que OpenPose luego rechaza. Además vuelca 15 JPG por vídeo.

Usa el pre-filtro calibrado:

```bash
$PY final_dataset/wrist_prescreen.py \
    --videos_folder $LOTE/videos/masked \
    --out_csv $LOTE/munecas.csv \
    --weights /tmp/run50/.scratch/weights/yolo11x-pose.pt \
    --min-conf 0.7 --frames 15 --device 0
```

Criterio de admisión: **`sp_any >= 0.6`** — `sp_any` es la fracción de los 15 frames muestreados
en la que el cuerpo con la cara más confiable tiene **al menos una** muñeca con `conf >= 0.7`
(`max(sp_right, sp_left)`).

Por qué una muñeca y no las dos, y por qué esto es un cambio de dataset: la regla anterior
era `sp_both >= 0.8` (las dos muñecas en el **mismo** frame). Se auditó con OpenPose sobre los
16 clips que esa regla descartó en FREQ025, y se dejó uno que OpenPose valoró luego a
**91,9 % / 100 %** de cobertura con `gate=accepted`. Irrecuperable: la criba corre antes de
OpenPose. La matriz de los 20 clips del lote, con la verdad medida por OpenPose:

| regla | admite | TP | FP | FN | recall |
|---|---|---|---|---|---|
| `sp_both >= 0.8` (la de la entrega FREQ025) | 4 | 3 | 1 | 1 | 75 % |
| `sp_any >= 0.6` (la actual) | 5 | 4 | 1 | 0 | 100 % |

Cuesta **+1 clip de GPU sobre 20** y no cuela ningún falso nuevo. Admitir una muñeca no
baja la calidad de las features: `gesture_features.py` calcula derecha e izquierda por
separado y tolera que falte una; lo que no puede fabricar son las dos.

Lo que **no** arregla esto: el único falso positivo que queda (`WKYC 102.005`, `sp_both` justo
0,8 y 69,5 % real en la muñeca derecha) tampoco se quita bajando `--yolo-conf` — medido sobre
los 29 clips de calibración: con 0,5 y con 0,2 entran los mismos útiles y además algún basura.

El umbral de confianza del detector sigue calibrado igual:

| conf | clips buenos retenidos | clips malos que se cuelan |
|---|---|---|
| 0,2 | 4/4 | 4/25 |
| 0,5 | 4/4 | 3/25 |
| **0,7** | **4/4** | **1/25** |

`sp_any` es la columna que decide; `sp_both` se sigue escribiendo como ranking y como
-evidencia de lo que hacía la regla anterior. `both` es la versión estricta (solo frames con
exactamente una persona) y solo tiene sentido sobre `masked`.

Estas cifras no son fe: se re-miden con `tools/auditoria_criba.py`, que pasa OpenPose sobre
los clips que la criba tiró. Sobre una entrega ya construida con la regla estricta, audita
con `--min-sp-both 0.8` (si no, le cambias la regla y mides otra población). Y son de UN lote
con n=20: el intervalo del falso negativo es ancho (0,2–30,9 %), así que en un corpus nuevo
la primera tanda merece auditoría propia antes de fiarse del umbral.

Copiar los admitidos a la carpeta que consume OpenPose:

```bash
mkdir -p $LOTE/videos/1_person
$PY - <<'PY'
import pandas as pd, shutil
d = pd.read_csv("lote3/munecas.csv")            # cambia lote3
sel = d[d.sp_any >= 0.6]
for _, r in sel.iterrows(): shutil.copy(r.path, f"lote3/videos/1_person/{r.video}")
print(f"admitidos {len(sel)} de {len(d)}")
PY
```

---

## Paso 4 — OpenPose + dfMaker, con el gate de visibilidad

```bash
$POLY argos --videos_folder $PWD/$LOTE/videos/1_person --output_folder $PWD/$LOTE \
    --filter-body-visibility --required-body-points 4,7 \
    --min-visible-percent 80 --min-keypoint-confidence 0.2
```

El gate mide los puntos BODY_25 **4 (muñeca dcha) y 7 (muñeca izq)** sobre el JSON que
OpenPose acaba de escribir, y rechaza **antes** de dfMaker: el clip se queda, su Parquet va
a cuarentena y el informe queda en:

```bash
for f in $LOTE/dataset/body_visibility/*.json; do
  $PY -c "import json,sys; d=json.load(open('$f')); print('$f'.split('/')[-1][:46], d['decision'], d['accepted'])"
done
```

Con `--min-visible-percent 80` te quedas solo con los clips donde la muñeca está medible
casi todo el rato. Si te sale una tanda demasiado pequeña, baja a 60 y anota el cambio:
el umbral define el dataset.

---

## Paso 5 — voz, base de datos y anotación visual

```bash
$POLY voice  --input_folder $PWD/$LOTE --n_persons 1          # ecos (prosodia) + apate (WhisperX)
$POLY merge  --input_folder $PWD/$LOTE --n_persons 1 --db_name $PWD/$LOTE/$LOTE.duckdb
$POLY theia  --input_folder $PWD/$LOTE --n_persons 1 --db_name $PWD/$LOTE/$LOTE.duckdb
```

`merge` es lo que construye el DuckDB con la tabla larga `multi_data`; `theia` añade las 7
etiquetas visuales (las "anotaciones" del dataset). Si `theia` se cae a mitad, tiene `--limit`
y `--attempts`.

**Qué comprobar antes de llamar a eso "dataset"**:

```bash
$POLY duckdb $LOTE/$LOTE.duckdb -c "
SELECT count(*) filas, count(DISTINCT id) clips FROM multi_data"
# visibilidad REAL de muneca: OBLIGATORIO filtrar type_points
$POLY duckdb $LOTE/$LOTE.duckdb -c "
WITH f AS (SELECT id, count(DISTINCT frame) n FROM multi_data WHERE type_points='pose_keypoints' GROUP BY id)
SELECT f.id, round(100.0*sum(CASE WHEN p.points=4 AND p.c>=0.2 THEN 1 ELSE 0 END)/max(f.n),1) AS dcha,
       round(100.0*sum(CASE WHEN p.points=7 AND p.c>=0.2 THEN 1 ELSE 0 END)/max(f.n),1) AS izq
FROM f JOIN multi_data p ON p.id=f.clip WHERE p.type_points='pose_keypoints' AND p.points IN (4,7)
GROUP BY f.id ORDER BY 2 DESC"
```

**La trampa que me he comido esta misma semana**: `multi_data` mete en la misma columna
`points` cuatro modelos (`face_keypoints`, `pose_keypoints`, `hand_right_keypoints`,
`hand_left_keypoints`). Un `WHERE points IN (4,7)` sin `type_points='pose_keypoints'` suma
cejas y nudillos y te dice que la muñeca está visible en el 100 % de los frames. **No es
verdad**. Siempre con el filtro de `type_points`.

---

## Paso 6 — el dataset final

```bash
cd final_dataset
$PY final_dataset.py \
    --db_name ../$LOTE/$LOTE.duckdb \
    --tab ../$TSV \
    --output_folder ../entrega_${LOTE}_final \
    --offset $OFFSET \
    --extended \
    --readme DATASET_v3.md
```

Qué hace y por qué falla:

* Abre el DuckDB en **solo lectura** y **se niega** si `--output_folder` ya existe. No borres
  carpetas de entregas anteriores para "que pase": publica en una nueva.
* `--offset` es **2.0 por defecto**, el mismo del pipeline. En lotes descargados con otra
  holgura (los 29 clips entregados y OZBEKALL se cortaron con **2.5**) sin este flag no
  encaja ni un ID por coincidencia exacta. Comprueba en el manifiesto:
  `clip_id_mapping.matched_exact_id` debe ser igual al número de clips, y
  `processed_padding_s` debe valer `[2.0]` (o el valor que pasaras).
* `--extended` añade el bloque FEATURE_VERSION 3 (134 columnas de secuencia + 6 de frame).
  Sin la bandera, la salida es la versión 2.
* La matemática de muñeca se importa de `Polymnia/gesture_features.py`; no la dupliques.
* `manifest.json` guarda fórmulas, unidades, constantes, columnas caídas y el SHA-256 de cada
  fichero publicado (el propio manifiesto queda fuera del hash, y el motivo está declarado).

No lances el primero encima de la entrega buena. Prueba en `/tmp`:

```bash
$PY final_dataset.py --db_name ../$LOTE/$LOTE.duckdb --tab ../$TSV \
    --output_folder /tmp/prueba_$LOTE --offset $OFFSET --extended
```

## Paso 7 — el README, que es donde miente la gente

`--readme DATASET_v3.md` **copia el fichero que le pases tal cual** y lo hashea dentro del
paquete. `DATASET_v3.md` describe la cobertura medida de los **29 clips viejos** (14,5 % de
frames con muñeca derecha, 6 clips con datos aprovechables, 3 con bimanual). Publicarlo con
una tanda nueva es publicar números falsos.

Antes de entregar, repasa contra tus ficheros reales:

* filas × columnas de `final_frames` y `final_videos` (los imprime el propio script)
* recuento de frames con muñeca válida por mano, y cuantos clips tienen `>= 10` frames
* cuantos clips publican `n_cos_phase >= 10` (bimanual) y sus valores
* los ejemplos de código del README: ejecuta el SQL y el pandas **contra los ficheros que
  acabas de publicar**, no contra los anteriores

### El diccionario columna por columna sale solo; el README también

El README cuenta el dataset en prosa; lo que necesita alguien que va a abrir
`final_videos.parquet` sin ti es una fila por columna. Eso se **genera**, no se escribe, y lo
genera la etapa `final` del pipeline: deja `informes/DICCIONARIO_COLUMNAS.csv`, el mismo
documento en `DICCIONARIO_COLUMNAS.md` (raíz del lote) y lo publica como `DATASET.md` dentro de
`final/`, donde entra en los sha256 del manifiesto.

Para regenerarlo a mano sobre una entrega vieja (o sobre un lote que se corrió con
`--sin-diccionario`):

```bash
Polymnia/.venv/bin/python final_dataset/diccionario_columnas.py \
    --final lote3/final \
    --csv lote3/informes/DICCIONARIO_COLUMNAS.csv \
    --md  lote3/DICCIONARIO_COLUMNAS.md \
    --readme lote3/informes/README_entrega.md   # solo si vas a re-publicar con DATASET.md
```

Por qué generado y no redactado:

* las fórmulas y unidades se leen de `FEATURE_UNITS` y `EXTENDED_FEATURE_UNITS` en
  `gesture_features.py`, y cada columna cita la función que la calcula. Redactado a mano, el
  documento se desincroniza del código en cuanto alguien toca una medida.
* cada fila lleva **cuantos clips de este lote tienen dato** en esa columna, medido sobre el
  Parquet entregado. Es el número que un README escrito a mano no puede tener.
* si el código estrena una columna sin documentar, el generador **se cae** en vez de publicar
  un hueco. Está probado en `final_dataset/test_diccionario_columnas.py`.

Medido con las dos entregas que hay ahora mismo: `cos_phase_mean` publica en el **89,7 %** de
los clips de `OZBEKSMALL` y en el **10,3 %** de la entrega vieja, y la mediana de
`right_mean_speed` es 1,184 frente a 0,626. Mismas 210 columnas, dos datasets que no significan
lo mismo. Sin esa columna de cobertura, un alumno compara los dos y saca conclusiones sobre la
metodología cuando lo que cambió fue la suerte con las muñecas.

No lo metas dentro de `final/` a mano: esa carpeta es un paquete atómico y su `manifest.json`
tiene los sha256 de lo que había al publicarla. El CSV va en `informes/` y el MD en la raiz del
lote; el único fichero que entra en `final/` es `DATASET.md`, y entra **antes** de publicar,
pasando por `final_dataset.py --readme` (que es lo que hace el pipeline). El propio generador
se niega a escribir dentro de la entrega que está describiendo.

Y el orden no es capricho: el README tiene que existir antes de publicar (porque se hash
con el resto) pero habla de cobertura (y para eso necesita los parquet ya hechos). El pipeline
lo resuelve con una **pasada de medición** a `.medicion`: calcula los parquet, saca de ahi
diccionario y README, publica `final/` de una sola vez, y al final compara `final_*.parquet`
y `final_*.csv` byte a byte entre las dos pasadas. Si divergieran, aborta: un documento que
describe otros números es lo que se quiere evitar. Si ves `.medicion` por ahí es que la etapa
se interrumpió a mitad; se aparta sola al relanzar.

---

## Orden de costes, por si te preguntan por la GPU

| etapa | coste medido |
|---|---|
| descarga | ~1 s + 2 s de cortesía por clip |
| ASD | ~6,9 s por clip que pasa |
| pre-filtro YOLO | ~2 min para 29 clips (15 frames cada uno) |
| OpenPose (`--face_hands True`) | lo caro; para eso existe el pre-filtro y el gate |

## Trampas ya pisadas (todas verificadas en este repo)
* **`--face_hands False` no desactiva cara y manos.** `argos.py` declara el argumento con
  `default=True` y sin `type`, así que `False` llega como la **cadena** `"False"`, que es
  `True` en Python. El `run50` pidió desactivarlo y su `dataset.duckdb` tiene 418 670 filas
  de `face_keypoints` y 125 601 de cada mano. Si quieres OpenPose solo con cuerpo, hay que
  arreglar el flag (`action`/`BooleanOptionalAction`), no pasarle `False`.
* **`--offset` no coincide entre etapas**: `download_clips.py` y `asd --second` usan 2,5 y
  `final_dataset.py` viene a 2,0. Un despiste y ningún clip encaja por ID exacto.
* **`exit 0` no significa nada** en ASD ni en la descarga: ASD rechaza sin generar fichero, y
  la descarga solo mira el tamaño. Cuenta ficheros y abre los vídeos.
* **`type_points`**: la columna `points` la comparten cuatro modelos. Sin
  `WHERE type_points='pose_keypoints'` los índices 4 y 7 son cejas y nudillos.

* **`vad` no es BOOLEANO en todas las bases.** En `1p.duckdb` sí; en las que saca el pipeline
  de hoy es entero y está todo a NULL. `bool_or(vad)` peta con `Binder Error`, así que el
  constructor ahora castea según el tipo declarado (`TRY_CAST`) y publica NULL si no hay
  medida. Comprobado: la entrega de los 29 clips sigue saliendo byte a byte igual.
* **Al reanudar, los derivados viejos se mezclaban con los nuevos** (20 Parquet en vez de 4,
  4 clips fantasma en `2_persons`, y `merge` petando porque `theia` había ensanchado
  `multi_data` de 28 a 35 columnas). Ahora cada etapa aparta lo suyo a
  `informes/_anterior/<hora>/` antes de correr. Dos reanudados seguidos dan el mismo número.
* **La voz vive en `final_frames`, no en `final_videos`.** Medido en las tres entregas que hay
  (`OZBEKSMALL`, la v2 y la v3 viejas): `final_frames` lleva `pitch`, `intensity`,
  `harmonicity`, `formant.1..5_frequency`, `vad` y `words`; `final_videos` no lleva **ni una**.
  O sea que cualquier análisis voz-gesto se hace fotograma a fotograma en `final_frames`; si
  alguien quiere «la voz de un clip» se lo tiene que inventar él, agregando, y eso ya no está
  en el contrato de la entrega.
* **La voz viene de otra cadena, y hasta hace nada no era una medida por fotograma.**
  `process_prosody.R` cogía la pista de Praat y la re-muestreaba con `splinefun(method = "fmm")`
  evaluada en el `frame_timestamp` de CADA fotograma (`interpolate_spline`), con mínimo de 4
  puntos válidos por clip: todos los fotogramas recibían número, incluidos los de silencio. Y
  `pitch` llevaba centinela: `df_final$pitch[df_final$pitch < 10] <- 0`.
  **Eso está arreglado aguas arriba** (`prosody_align.R`: tramos + lineal + NULL fuera de
  pista + `voice_present`), el arreglo está instalado en esta máquina y **ya está mergeado en
  PAROLE: https://github.com/daedalusLAB/parole/pull/1** (commit `87d0f29` del 2026-10-14; la
  copia vendorizada de aquí coincide byte a byte con su `main`). Medido sobre los
  MISMOS 620 fotogramas del lote de prueba, antes y después de volver a correr `voz`:
  `pitch == 0` de 194 a 0, `harmonicity` por debajo del suelo de Praat (−200 dB) de 90 a 0,
  formantes fuera de 0-5500 Hz de 143 a 0, y la media de `pitch` de 128,3 Hz a 188,4 Hz.
  Lo que NO se puede recuperar: las entregas ya publicadas salieron con la spline, así que en
  `OZBEKSMALL` y en la v2/v3 viejas siguen ahí `pitch == 0` en 1408/4553 y 1553/4433
  fotogramas, HNR de −246,7 dB y un F5 de 29.144.321 Hz. Para esas entregas: filtrar por la
  banda de pitch (60-600 Hz) antes de cualquier estadístico y no llamar silencio al 0.
  Cómo se distingue una entrega de otra sin acordarse: el README que genera el pipeline mide
  las cotas y escribe el régimen (`tramos` o `spline`) en la tabla de voz.
* **El código que produce la prosodia no está en ningún git.** Va vendorizado en
  `Polymnia/create_datasets/parole/`, y Polymnia tiene `create_datasets/` en `.gitignore` (línea
  23, heredada de upstream). Un `clone` limpio o un reinstalar PAROLE deja el arreglo fuera sin
  avisar. Por eso `final_dataset/upstream_parole/` guarda la copia canónica, el parche para
  enviar arriba y un instalador: `final_dataset/install_parole_fix.sh --comprobar` sale 1 si lo
  instalado ha derivado. Es el primer chequeo del HOWTO antes de lanzar un lote.
* **`dfMaker` escribe la prosodia con dos estructuras** (primero `<clip>/<clip>_prosody.parquet`
  y unos 24 s después la copia plana `<clip>.parquet`). Cuenta clips, no ficheros.
* **`polymnia theia` lee el `.env` él solo: no exportes nada.** Precedencia flag > entorno >
  `.env`, y el log de la etapa dice `clave:.env` / `clave:entorno` / `clave:sin clave` sin escribir
  nunca la clave. Dos cosas que estaban mal y estaban documentadas aquí como si fueran el
  contrato: (1) `theia.py` fijaba el default de `--api-key` con `os.environ.get("THEIA_API_KEY",
  "none")` al construir el parser, antes de `load_env`, y `"none"` es una cadena verdad, así que
  mandaba `Bearer none` → 401; (2) `pipeline_lote.sh` comprobaba `[ -f "$RAIZ/Polymnia/.env" ]`,
  pero `$RAIZ` **ya es** el repo `Polymnia`, así que preguntaba por `Polymnia/Polymnia/.env` y la
  etapa 9 se saltaba **siempre** — un lote salía sin las 7 columnas de anotación y sin ninguna
  traza dentro del lote. Arreglado en `theia.py` (con tests en `tools/test_theia.py`) y en la
  etapa 9, que ahora corre por defecto, escribe el motivo de un omitido en `logs/09_theia.log` y
  lo deja en `conteos.tsv`. Y ojo con la métrica: cuando Theia va **bien** deja la columna `error`
  como cadena vacía, no como NULL.
* **El scratch de TalkNet-ASD está fuera del lote pero es transitorio**: `TalkNet-ASD/output/`
  queda vacío al acabar.
* **El diccionario y el README vienen con la etapa final, pero tienen truco de orden.** `DATASET.md`
  se hash dentro de `final/`, o sea que tiene que existir *antes* de publicar; y a la vez habla de
  cobertura, o sea que necesita los parquet ya calculados. Por eso la etapa hace **dos pasadas**
  (`logs/10a_medicion.log` y `logs/10_final.log`) y termina comparando los cuatro ficheros byte a
  byte: si vieras `difere entre la medición y la publicación`, aborta, y bien. Si te saltas la
  etapa con `--sin-diccionario`, el manifiesto vuelve a salir con `"readme": None` y las
  columnas documentadas (283, o 284 si el lote trae `voice_present`) no están escritas en ningún
  sitio: el manifiesto solo documenta **familias** (3 de 210
  nombres literales). Y si el generador peta con `undocumented columns`, la columna nueva va en
  las tablas `MIDE_*` / `FORMULA_PROPIA` de `diccionario_columnas.py` citando la función que la
  calcula: el guard está para eso, no para saltárselo.

## Rendimiento real de la cadena (medido sobre los logs del run_bte)

| etapa | clips que quedan |
|---|---|
| TSV master | 100 |
| descarga | 100 |
| TalkNet-ASD (escena única + score positivo) | **51** |
| filtro YOLO de persona/manos | 39 |
| `1_person` clasificada por dfMaker | 29 |
| con muñecas visibles (OpenPose, puntos 4 y 7) | **4** |

Es decir: **un 4 % del master acaba dando datos de muñecas aprovechables**. Si buscas N clips
bimanuales, pide del orden de 15-25 × N líneas de TSV. Y la merma no es solo de las muñecas:
ASD se lleva 49 de 100 por sí solo. Ese rendimiento es lo que justifica el pre-filtro de
muñecas antes de OpenPose, no una preferencia estética.

### El gate de visibilidad rechaza por `multi_person` antes de mirar la cobertura

`body_visibility.py:360` corta con **un solo frame** que tenga más de una persona. Medido en
`lote2`: 4 clips con **ambas muñecas al 85-93 % de cobertura** fueron rechazados por tener
solo el **5-9 %** de los frames con segunda persona. Si te pasa, no es que fallen las muñecas:

```bash
$PY - <<'PY'
import json, glob
for f in sorted(glob.glob("lote2/dataset/body_visibility/*.json")):
    d = json.load(open(f)); pts = {p["point"]: p for p in d["points"]}
    print(f"{d['clip'][:44]:44s} {d['decision']:13s} multi={d['frames_multi_person']:3d}/{d['expected_frames']}"
          f"  dcha={pts[4]['coverage_percent']:5.1f}%  izq={pts[7]['coverage_percent']:5.1f}%")
PY
```

Y **no hay identidad en los JSON**: `person_id` sale `[-1]` en los 167 fotogramas con persona
(OpenPose no corre con `--identification`). Cualquier regla que quiera "quedarse con el
hablante" en un frame con dos cuerpos tiene que calcular la correspondencia ella sola; no la
des regala OpenPose.

### La confianza de YOLO no es comparable al `score` de OpenPose

Sobre la misma muñeca, en los mismos frames: YOLO `conf 0.95-0.98` frente a OpenPose
`score 0.52-0.78`. El `--min-conf 0.7` del pre-filtro es un umbral **del detector**, no una
equivalencia con `--min-keypoint-confidence 0.2`. Sirve para rankear, no para prometer
recuento de frames.

### Las manos izquierda/derecha están bien etiquetadas (no hace falta repetir el control)

Emparejando COCO 10 ↔ BODY_25 4 y COCO 9 ↔ BODY_25 7 en los mismos frames: desvío medio
**1,8 px**. Emparejando cruzado: **146 px** en un frame de 640. El mapeo del pre-filtro es
correcto.

## Reglas que no se rompen

* El DuckDB del pipeline es **entrada**: nunca se escribe sobre él, y las entregas anteriores
  no se sobrescriben.
* La carpeta de salida nueva, o nada.
* NULL no es cero: un frame sin muñeca no es un frame con muñeca quieta.
* Nada de normalizar entre clips ni scores que dependan del conjunto: un valor debe significar
  lo mismo en tu tanda que en la mía.
* Los features bimanuales solo donde las dos muñecas son válidas, con columnas contadoras
  (`right_n_speed`, `left_n_speed`, `n_cos_phase`) que digan cuánta base tiene cada número.
* Cada columna nueva: fórmula, unidades, referencia y un test con trayectoria conocida.
