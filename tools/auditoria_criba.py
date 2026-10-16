#!/usr/bin/env python3
"""Auditoría sombra de la criba de muñecas: OpenPose sobre lo que la criba tiró.

Por qué existe: `tools/wrist_prescreen.py` decide ANTES de OpenPose y su error es
de un solo sentido - un clip que descarta no vuelve a medirse nunca, así que el
falso negativo es **estructuralmente invisible** para el pipeline. Sin medir, la
calibración «`sp_both` retiene 4 de 4» era compatible con un recall del ~40 % (dos
colas, 95 %): se podían estar tirando casi la mitad de los clips válidos sin que nadie
lo viera. Este script existe para no tener que opinar sobre eso.

Medido ya sobre el censo de los 16 clips que la regla estricta descartó en FREQ025:
**1 de 16** habría pasado el gate (95 % exacto 0,2-30,9 %). Ese único clip es el que
motivó que la admisión pasara a `sp_any >= 0.6` (ver `tools/wrist_prescreen.py`), y es
el motivo por el que la auditoría audita la regla que se aplico a CADA lote
(`--criterio` / `--umbral`, o `--min-sp-both 0.8` para una entrega ya construida con la
regla estricta) en lugar de suponer la de hoy: cambiar la regla en el auditores medir
otra población y salir con una cifra convincente y equivocada. n=1 lote: el intervalo
es ancho a propósito.

Qué hace: coge los clips que `munecas.csv` dejó por debajo del umbral, les pasa el
OpenPose que nunca les corrió, y mide cuántos habrían superado el criterio de
muñecas. Los clips descartados siguen enteros en `videos/descartados/preseleccion_sp_bajo/`
(y en `videos/masked/`), así que la auditoría no rescata nada: solo cuenta.

Qué responde, en el orden en que duele:

* `pasa_coverage` - cuántos descartados tienen la cobertura de muñeca (BODY_25 4 y 7,
  c >= 0,2) que exige el gate. Cada uno de esos es un dato perdido por la criba.
* `una_mano` - cuántos tienen UNA muñeca por encima del umbral. Estos no son ruido:
  `gesture_features.py` deja las features de una mano con la otra a NULL, así que
  admitirlos es una decisión de contrato, no de calidad.
* `pasa_al` - a qué porcentaje de cobertura habrían pasado (40 / 60 / 80), para poder
  hablar de umbrales sin improvisar.
* intervalos exactos de Clopper-Pearson, porque con 16 clips un porcentaje suelto
  no es un hecho, y en los extremos (0 de 16, 4 de 4) la aproximación normal miente.
  Y el `sp_any` (la mejor de las dos muñecas) se reporta aparte: es la métrica que
  `gesture_features.py` tolera, porque una mano con la otra a NULL sigue dando features.

Lo que NO hace: no escribe en `dataset/` del lote, no toca `videos/`, no reabre la
criba, no decide el umbral. El JSON de OpenPose queda bajo
`informes/auditoria_criba/dataset/OpenPose/<clip>/JSON_FILES` para que sea
comprobable, y `limpiar_fantasmas.py` (que recorre `<lote>/dataset/OpenPose`) no lo
ve porque está colgado de `informes/`.

Los lentes puros (selección, veredicto, intervalos) no importan ni OpenPose ni cv2:
se prueban en `tools/test_auditoria_criba.py` con matrices fabricadas.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

REQUIRED_POINTS = (4, 7)  # BODY_25: muñeca derecha, muñeca izquierda
# Estados que cuentan como medicion VALIDA. `reutilizado` lo es de verdad: OpenPose ya
# corrio en una pasada anterior y la cobertura se releera del mismo JSON, solo que sin
# pagar GPU. Contar solo `medido` hacia que reanudar una auditoria ESCRIBIERA en ceros un
# informe que estaba bien, y que ademas saliera 1 gritando "no se ha medido NINGUN clip".
ESTADOS_MEDIDOS = ("medido", "reutilizado")
# Costes medidos en RTX 4090 con clips de FREQ025 (~4,5 s, ~140 frames). Son DOS cifras a
# proposito: la auditoria corre solo el modelo de cuerpo, y el pipeline corre ademas cara y
# manos (argos.py declara --face_hands sin type=, asi que en la practica siempre van
# encendidas). Usar la cifra del pipeline prometi6 229 s para una pasada que tard6 86 s.
SEGUNDOS_POR_CLIP = 5.4  # esta herramienta: sin cara ni manos (86 s / 16 clips medidos)
SEGUNDOS_POR_CLIP_CARA_MANOS = 14.3  # config del pipeline: --face --hand encendidos


def segundos_por_clip(cara_manos):
    """El coste por clip de la configuracion que se va a usar, no de una cualquiera."""
    return SEGUNDOS_POR_CLIP_CARA_MANOS if cara_manos else SEGUNDOS_POR_CLIP
CAMPOS = ["video", "sp_both", "sp_any", "sp_right", "sp_left", "max_people", "frames",
          "op_dcha_pct", "op_izq_pct", "gate", "pasa_coverage", "pasa_gate", "una_mano",
          "pasa_al", "frames_multi_person", "frames_no_person", "frames_missing_json",
          "estado", "error", "segundos"]


# ------------------------------------------------------------- selección ----

def _num(valor, por_defecto=float("nan")):
    """float() tolerante con el CSV (cadenas vacías, NaN, basura)."""
    try:
        return float(valor)
    except (TypeError, ValueError):
        return por_defecto


def read_munecas(ruta):
    """Filas de `munecas.csv` con los scores como números. `path` es la ruta del .mp4."""
    with open(ruta, newline="", encoding="utf-8") as handle:
        filas = []
        for fila in csv.DictReader(handle):
            for campo in ("sp_both", "sp_any", "sp_right", "sp_left", "right", "left",
                          "both", "n_sampled", "max_people", "reported_frames"):
                if campo in fila:
                    fila[campo] = _num(fila[campo])
            filas.append(fila)
    return filas


def sp_any(fila):
    """Mejor de las dos muñecas del hablante.

    Manda la columna `sp_any` si la criba la escribió: es lo que de verdad decidió la
    admisión. Si no está (los `munecas.csv` de antes de esta regla) se deriva con la misma
    norma: max ignorando NaN, NaN solo si faltan las dos. La auditoría tiene que medir la
    regla con la que se construyó ESA entrega, no la que esté de moda hoy.
    """
    escrito = _num(fila.get("sp_any"))
    if escrito == escrito:
        return escrito
    valores = [_num(fila.get("sp_right")), _num(fila.get("sp_left"))]
    buenos = [v for v in valores if v == v]  # sin NaN
    return max(buenos) if buenos else float("nan")


def score(fila, criterio):
    """El número con el que la regla de admisión juzgó este clip."""
    return sp_any(fila) if criterio == "sp_any" else _num(fila.get("sp_both"))


# Defecto de cada criterio: sp_both admitió con 0,8 hasta la auditoría de FREQ025; sp_any
# admite con 0,6 en pipeline_lote.sh. Son los umbrales con los que se hicieron las
# entregas, no una preferencia estética.
UMBRALES = {"sp_any": 0.6, "sp_both": 0.8}


def resolver_regla(criterio=None, umbral=None, min_sp_both=None):
    """(criterio, umbral) sin adivinar: lo dicho manda, y lo contradictorio se rechaza.

    `--min-sp-both X` no es un sinónimo de `--umbral X`: afirma que la entrega se construyó
    con la regla estricta, así que arrastra el criterio. Aceptar banderas contradictorias
    produciría una auditoría de una regla que nadie aplicó.
    """
    if min_sp_both is not None:
        if umbral is not None or (criterio is not None and criterio != "sp_both"):
            raise ValueError("--min-sp-both afirma sp_both; no lo mezcles con --umbral/--criterio")
        return "sp_both", float(min_sp_both)
    criterio = criterio or "sp_any"
    if criterio not in UMBRALES:
        raise ValueError(f"criterio desconocido: {criterio} (sp_any | sp_both)")
    return criterio, UMBRALES[criterio] if umbral is None else float(umbral)


def clips_con_datos(filas, criterio):
    """Cuántas filas tienen número usable para el criterio (0 = este CSV no se puede auditar).

    Con `sp_any` sobre un CSV que solo trae `sp_both` (los 29 de calibración, p. ej.) todas
    las filas darían NaN, y NaN cuenta «descartado»: la auditoría se pondría a medir el
    corpus entero creyendo que mide los descartados. Mejor decirlo antes de gastar GPU.
    """
    return sum(1 for fila in filas if score(fila, criterio) == score(fila, criterio))


def select_rejected(filas, umbral, criterio="sp_any"):
    """Las filas que la regla de admisión dejó fuera: `<criterio> < umbral` (NaN cuenta fuera).

    La criba copia a `descartados/` lo que no llega al umbral, y un score NaN (clip
    no decodificable) tampoco se habría admitido, así que entra en la auditoría con
    su propio estado.
    """
    fuera = []
    for fila in filas:
        if not (score(fila, criterio) >= umbral):  # False con NaN: a propósito
            fila["sp_any"] = sp_any(fila)
            fuera.append(fila)
    return fuera


def stratified_sample(filas, muestra, criterio="sp_any"):
    """`muestra` clips repartidos por el score, determinista. 0 o más que el total = todos.

    No es un muestreo aleatorio: lo que decide está justo debajo del umbral, y un
    sorteo sobre una tanda donde casi todo puntúa 0,0 gastaría la GPU en clips
    visiblemente muertos. Se ordena por `sp_both` de mayor a menor y se anda la lista
    a pasos iguales, así que la muestra cubre la frontera y el fondo a la vez.
    """
    total = len(filas)
    n = int(muestra)
    if n <= 0 or n >= total:
        return list(filas)
    orden = sorted(filas, key=lambda f: (-(_num(score(f, criterio), float("-inf"))),
                                         str(f.get("video", ""))))
    if n == 1:
        return [orden[0]]
    pasos = [(i * (total - 1)) / (n - 1) for i in range(n)]
    elegidos, vistos = [], set()
    for paso in pasos:
        indice = int(round(paso))
        while indice in vistos and indice < total - 1:
            indice += 1
        vistos.add(indice)
        elegidos.append(orden[indice])
    return elegidos


# ------------------------------------------------------------- veredicto ----

def veredicto(pcts, percent, sensibilidades=(40.0, 60.0)):
    """Qué habría pasado con este clip, dado su cobertura por punto.

    `pcts` es {punto BODY_25: porcentaje de frames visible}. Se exigen las DOS
    muñecas para `pasa_coverage` porque ese es el criterio actual del gate; `una_mano`
    es la versión unilateral, que es la que las features de una sola mano admiten.
    Los dos son porcentajes sobre TODOS los frames del clip, no sobre los medidos.
    """
    valores = [p for p in pcts.values() if p == p]
    out = {"pasa_coverage": False, "una_mano": False, "pasa_al": ""}
    if valores:
        out["pasa_coverage"] = all(v >= percent for v in valores)
        out["una_mano"] = max(valores) >= percent
        pasan = [s for s in sorted(sensibilidades) if min(valores) >= s]
        out["pasa_al"] = ",".join(f"{s:g}" for s in pasan)
    return out


def _binom_cdf(k, n, p):
    """P(X <= k) de una binomial(n, p), suma directa. Vale para las n de esta auditoría."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    for i in range(int(k) + 1):
        total += math.comb(n, i) * p ** i * (1.0 - p) ** (n - i)
    return min(1.0, total)


def ic_exacto(acierto, total, alfa=0.05):
    """Intervalo de Clopper-Pearson (bidireccional) de una proporción. (inf, sup).

    Exacto, no aproximado, y a propósito: la pregunta que se hace con una muestra de
    ~16 clips es del tipo "4/4 retenidos" o "0/16 colados", y en esos extremos la
    aproximación normal y hasta Wilson se ponen optimistas: Wilson dice que retener 4/4
    es compatible con un recall >= 71 %, y lo exacto es >= 40 % (dos colas; >= 47 % con
    una sola). Esa es la frase que hay que escribir en un RUNBOOK. Con k=0 el límite
    superior sigue siendo positivo: "no apareció ninguno" no es "no los hay".
    """
    n, k = int(total), int(acierto)
    if n <= 0:
        return (float("nan"), float("nan"))
    k = max(0, min(n, k))
    alfa /= 2.0  # dos colas: 0,025 por lado para un 95 %

    def raiz_creciente(f, objetivo):
        """p en [0,1] con f(p) = objetivo, sabiendo que f es MONOTONA CRECIENTE en p.

        La direccion importa y es lo que tiene que quedar dicho aqui: la funcion del
        limite inferior crece con p y la del superior decrece, y usar el mismo criterio
        para las dos devuelve 100 % para un 4/4 (que es exactamente lo que devolvia).
        """
        inf, sup = 0.0, 1.0
        for _ in range(100):
            medio = (inf + sup) / 2.0
            if f(medio) < objetivo:
                inf = medio
            else:
                sup = medio
        return (inf + sup) / 2.0

    # limite inferior: P(X >= k | p) = alfa/2. Crece con p.
    if k == 0:
        inferior = 0.0
    else:
        inferior = raiz_creciente(lambda p: 1.0 - _binom_cdf(k - 1, n, p), alfa)
    # limite superior: P(X <= k | p) = alfa/2. DEcrece con p, asi que se busca sobre -f.
    if k == n:
        superior = 1.0
    else:
        superior = raiz_creciente(lambda p: -_binom_cdf(k, n, p), -alfa)
    return (inferior, superior)


def abortar_por_fallos_repetidos(estados, umbral=3):
    """True si los ultimos `umbral` clips fallaron exactamente igual.

    Tres fallos identicos seguidos no son casualidad del corpus: es el entorno (rutas, GPU,
    pesos). Seguir lanzando OpenPose solo gasta minutos y tapa el mensaje del primero, que
    es el unico que dice que estaba pasando.
    """
    if len(estados) < umbral:
        return False
    ultimos = estados[-umbral:]
    return len(set(ultimos)) == 1 and ultimos[0] not in ESTADOS_MEDIDOS


def fila_no_intentada(fila, motivo):
    """Clip de la muestra al que no llegamos: sigue apareciendo como 'no intentado'.

    No puede recibir el veredicto de un clip no medido, y no puede desaparecer del recuento:
    si falta, el informe insinua que se audito menos de lo que se habia planeado.
    """
    fila_out = {"video": str(fila.get("video", "")), "estado": "no intentado", "error": motivo,
                "op_dcha_pct": float("nan"), "op_izq_pct": float("nan"), "gate": "",
                "frames": 0, "frames_multi_person": 0, "frames_no_person": 0,
                "frames_missing_json": 0, "segundos": 0.0,
                "pasa_coverage": False, "una_mano": False, "pasa_gate": False, "pasa_al": ""}
    for campo in ("sp_both", "sp_any", "sp_right", "sp_left", "max_people"):
        fila_out[campo] = fila.get(campo, float("nan"))
    return fila_out


def resumen(filas, percent, sensibilidades=(40.0, 60.0)):
    """El recuento de la auditoría. Las proporciones son sobre lo AUDITADO.

    Auditado incluye lo reaprovechado de una pasada anterior (`ESTADOS_MEDIDOS`): son
    medidas reales, lo unico que no pagaron fue la GPU esta vez.
    """
    medidos = [f for f in filas if f.get("estado") in ESTADOS_MEDIDOS]
    n = len(medidos)
    pasa = sum(1 for f in medidos if f.get("pasa_coverage"))
    una = sum(1 for f in medidos if f.get("una_mano"))
    gate = sum(1 for f in medidos if f.get("pasa_gate"))
    multi = sum(1 for f in medidos if _num(f.get("frames_multi_person")) > 0)
    por_sens = {}
    for s in sensibilidades:
        por_sens[s] = sum(1 for f in medidos if str(int(s)) in str(f.get("pasa_al", "")).split(","))
    return {"auditados": n,
            "descartados_en_total": len(filas),
            "pasa_coverage": pasa,
            "pasa_coverage_ic": ic_exacto(pasa, n),
            "una_mano": una,
            "una_mano_ic": ic_exacto(una, n),
            "pasa_gate": gate,
            "con_frames_multi_person": multi,
            "sin_medir": [f for f in filas if f.get("estado") not in ESTADOS_MEDIDOS],
            "por_sensibilidad": por_sens,
            "segundos": sum(_num(f.get("segundos"), 0.0) for f in filas)}


# --------------------------------------------------------------- abrir el ----

def _ruta_json(out_root, clip):
    return os.path.join(out_root, "dataset", "OpenPose", clip, "JSON_FILES")


def audit_clip(ruta_video, out_root, config, openpose_path, model_pose,
               cara_manos=False, force=False, sensibilidades=(40.0, 60.0)):
    """Un clip descartado: OpenPose donde nunca corrió, y su cobertura medida.

    Devuelve la fila del informe. Nunca lanza por un problema del clip: un OpenPose
    que no arranca o un `data_error` del gate son estados, no excepciones, para que
    un clip raro no deje la auditoría a medias.

    `sensibilidades` tiene que llegar HASTA AQUI, no solo al resumen: es `veredicto` quien
    escribe `pasa_al`, y `resumen` se limita a contar etiquetas que ya estan ahi. Si aqui
    se usan los defaults, un `--sensibilidad 50,70` informa ceros con exit 0.
    """
    import argos
    import body_visibility

    # OpenPose se ejecuta con cwd=/opt/openpose (asi resuelve sus modelos), luego toda ruta
    # que le pasamos en la linea de comandos se interpreta DESDE AHÍ. Con un --lote
    # relativo, argos creaba los directorios en mi cwd pero openpose.bin buscaba
    # ../lotes/... dentro de /opt/openpose y moria con "Could not create directory":
    # rc=255 en 0,2 s por clip, con los 16 directorios vacios como unica pista. El pipeline
    # real se libra porque siempre pasa rutas absolutas; aqui no dependemos de quien invoque.
    ruta_video = os.path.abspath(ruta_video)
    out_root = os.path.abspath(out_root)

    clip = argos.clip_name(ruta_video)
    fila = {"video": os.path.basename(ruta_video), "estado": "medido", "error": "",
            "op_dcha_pct": float("nan"), "op_izq_pct": float("nan"), "gate": "",
            "frames": 0, "frames_multi_person": 0, "frames_no_person": 0,
            "frames_missing_json": 0, "segundos": 0.0}
    if not os.path.isfile(ruta_video):
        fila.update(estado="sin mp4", error=f"no existe {ruta_video}")
        return veredicto({}, config.min_visible_percent, sensibilidades), fila

    json_dir = _ruta_json(out_root, clip)
    esperados, error_conteo = argos._frame_count(ruta_video)
    if error_conteo:
        fila.update(estado="sin conteo de frames", error=error_conteo)
        return veredicto({}, config.min_visible_percent, sensibilidades), fila
    fila["frames"] = esperados

    ya_medido = os.path.isdir(json_dir) and any(
        nombre.endswith(".json") for nombre in os.listdir(json_dir))
    t0 = time.time()
    if ya_medido and not force:
        fila["estado"] = "reutilizado"  # la auditoría es reanudable: no repite GPU
    else:
        # `call=None` es la costura de test de argos, no un defecto de fabrica que se
        # arregle solo: quien llama tiene que pasar el runner de produccion, si no la
        # llamada revienta dentro de argos con "'NoneType' object is not callable".
        # `_default_call` es ademas lo que preserva el `cwd=openpose_path` con el que
        # OpenPose resuelve sus modelos.
        rc = argos.run_openpose(ruta_video, out_root, openpose_path, cara_manos, False,
                                model_pose=model_pose, call=argos._default_call())
        if rc != 0:
            fila.update(estado="openpose fallo", error=f"openpose devolvio rc={rc}")
            fila["segundos"] = round(time.time() - t0, 1)
            return veredicto({}, config.min_visible_percent, sensibilidades), fila

    decision = body_visibility.evaluate_body_visibility(json_dir, esperados, config,
                                                        clip=clip).as_dict()
    fila["segundos"] = round(time.time() - t0, 1)
    fila["gate"] = decision.get("decision", "")
    fila["frames_multi_person"] = decision.get("frames_multi_person", 0)
    fila["frames_no_person"] = decision.get("frames_no_person", 0)
    fila["frames_missing_json"] = decision.get("frames_missing_json", 0)
    if fila["gate"] == "data_error":
        fila.update(estado="data_error", error=str(decision.get("error") or "")[:300])
        return veredicto({}, config.min_visible_percent, sensibilidades), fila

    pcts = {p["point"]: p["coverage_percent"] for p in decision.get("points", [])}
    fila["op_dcha_pct"] = round(pcts.get(4, float("nan")), 1)
    fila["op_izq_pct"] = round(pcts.get(7, float("nan")), 1)
    v = veredicto(pcts, config.min_visible_percent, sensibilidades)
    v["pasa_gate"] = fila["gate"] == "accepted"  # el gate rechaza ademas por multi_person
    return v, fila


# ------------------------------------------------------------------ informe ----

def escribe_csv(filas, destino):
    """`NaN` se escribe vacio, no como el texto `nan`.

    Es un detalle deliberado y distinto de `wrist_prescreen.write_csv`, que deja `nan`:
    en este informe una celda vacia significa "no medido", y `estado` ya dice por que.
    Pandas lee las dos cosas igual; para el ojo humano, vacio no se confunde con un dato.
    """
    with open(destino, "w", newline="", encoding="utf-8") as handle:
        escritor = csv.DictWriter(handle, fieldnames=CAMPOS, extrasaction="ignore")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({k: ("" if isinstance(v, float) and v != v
                                   else f"{v:.4f}" if isinstance(v, float) else v)
                               for k, v in fila.items()})


def escribe_resumen(filas, r, destino, args):
    inf, sup = r["pasa_coverage_ic"]
    i_una, s_una = r["una_mano_ic"]
    lineas = [f"# Auditoría sombra de la criba de muñecas — {args.lote_nombre}", "",
              f"Generado con `tools/auditoria_criba.py`. **Regla auditada: `{args.regla}`** "
              f"(los clips por debajo de esa regla son los que se auditan), con gate de "
              f"cobertura `{args.percent:g} %` sobre BODY_25 "
              f"4 y 7 con `c >= {args.min_keypoint_confidence:g}`).",
              "",
              f"La regla importa: una auditoría con `sp_both >= 0,8` mide otra población que "
              f"una con `sp_any >= 0,6`. Si este lote se construyó con la regla estricta, "
              f"re-audítalo con `--min-sp-both 0.8`.",
              "",
              f"| métrica | valor |",
              f"|---|---|",
              f"| descartados por la criba | {r['descartados_en_total']} |",
              f"| auditados con OpenPose | {r['auditados']} |",
              f"| **habrían pasado el gate de muñecas** | **{r['pasa_coverage']} de "
              f"{r['auditados']}** (95 % exacto {inf * 100:.0f}–{sup * 100:.0f} %) |",
              f"| tendrían UNA muñeca medible | {r['una_mano']} de {r['auditados']} "
              f"(95 % exacto {i_una * 100:.0f}–{s_una * 100:.0f} %) |",
              f"| los aceptaría el gate tal como está | {r['pasa_gate']} (el gate rechaza "
              f"además cualquier frame con 2 personas) |",
              f"| auditados con frames multi-persona | {r['con_frames_multi_person']} |",
              f"| sin medir | {len(r['sin_medir'])} |",
              f"| coste de la auditoría | {r['segundos']:.0f} s |",
              ""]
    if r["por_sensibilidad"]:
        lineas += ["| cobertura exigida | habrían pasado |", "|---|---|"]
        lineas += [f"| ≥ {s:g} % de las dos muñecas | {c} |"
                   for s, c in sorted(r["por_sensibilidad"].items())]
        lineas += [""]
    lineas += ["## Clip por clip", "",
               "| clip | sp_both | sp_any | muñeca dcha | muñeca izq | gate | pasa | una |",
               "|---|---|---|---|---|---|---|---|"]
    for fila in sorted(filas, key=lambda f: -(max(_num(f.get("op_dcha_pct"), 0.0),
                                                  _num(f.get("op_izq_pct"), 0.0)))):
        lineas.append(f"| `{str(fila.get('video', ''))[:52]}` | {_fmt(fila.get('sp_both'))} | "
                      f"{_fmt(fila.get('sp_any'))} | {_fmt(fila.get('op_dcha_pct'), '%')} | "
                      f"{_fmt(fila.get('op_izq_pct'), '%')} | {fila.get('gate', '')} | "
                      f"{'SÍ' if fila.get('pasa_coverage') else 'no'} | "
                      f"{'SÍ' if fila.get('una_mano') else 'no'} |")
    if r["sin_medir"]:
        lineas += ["", "### Sin medir", ""]
        lineas += [f"* `{str(f.get('video', ''))[:52]}` — {f.get('estado')}: "
                   f"{str(f.get('error', ''))[:140]}" for f in r["sin_medir"]]
    lineas += ["", "## Cómo leerlo", "",
               "Un descartado que habría pasado el gate es un dato perdido: la criba lo",
               "eliminó antes de OpenPose y nada de lo que viene después puede rescatarlo.",
               "El coste de un falso positivo era una llamada de GPU; el de un falso",
               "negativo es el clip entero. Con pocos positivos auditados el intervalo es",
               "ancho a propósito: es una hipótesis sobre tu corpus, no una tasa.", ""]
    with open(destino, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lineas))


def _fmt(valor, sufijo=""):
    v = _num(valor)
    if v != v:
        return "—"
    return f"{v:g}{sufijo}"


# --------------------------------------------------------------------- cli ----

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lote", required=True, help="carpeta del lote (con munecas.csv)")
    p.add_argument("--munecas", default=None, help="por defecto <lote>/munecas.csv")
    p.add_argument("--criterio", choices=("sp_any", "sp_both"), default=None,
                   help="metrica con la que se admitio este lote (defecto sp_any; sp_both "
                        "es la regla estricta con la que se construyo FREQ025)")
    p.add_argument("--umbral", type=float, default=None,
                   help="umbral del criterio (defecto 0,6 para sp_any y 0,8 para sp_both)")
    p.add_argument("--min-sp-both", dest="min_sp_both", type=float, default=None,
                   help="atajo para auditar una entrega hecha con la regla estricta: "
                        "equivale a --criterio sp_both --umbral F")
    p.add_argument("--sample", type=int, default=0,
                   help="cuantos descartados auditar, repartidos por score (0 = todos)")
    p.add_argument("--percent", type=float, default=80.0,
                   help="cobertura que exige el gate, % de frames (defecto 80)")
    p.add_argument("--sensibilidad", default="40,60",
                   help="umbrales extra de sensibilidad, separados por comas")
    p.add_argument("--min-keypoint-confidence", dest="min_keypoint_confidence", type=float,
                   default=0.2, help="confianza minima del keypoint (defecto 0,2)")
    p.add_argument("--required-points", dest="required_points", default="4,7",
                   help="puntos BODY_25 que cuentan (defecto 4,7 = las dos munecas)")
    p.add_argument("--openpose-path", default="/opt/openpose")
    p.add_argument("--model-pose", default="BODY_25",
                   help="BODY_25 a proposito: los indices 4 y 7 solo significan muneca ahi")
    p.add_argument("--cara-manos", action="store_true",
                   help="ademas cara y manos de OpenPose (no cambia las munecas: cuesta tiempo)")
    p.add_argument("--force", action="store_true",
                   help="re-auditar lo que ya tenga JSON (por defecto se reutiliza)")
    p.add_argument("--out_dir", default=None,
                   help="por defecto <lote>/informes/auditoria_criba")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="ensena el plan y el coste estimado; no toca la GPU")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    munecas = args.munecas or os.path.join(args.lote, "munecas.csv")
    if not os.path.isfile(munecas):
        print(f"❌ no existe {munecas}; la auditoria necesita el score de TODOS los clips",
              file=sys.stderr)
        return 1
    puntos = tuple(int(x) for x in args.required_points.split(",") if x.strip())
    sensibilidades = tuple(float(x) for x in args.sensibilidad.split(",") if x.strip())
    args.lote_nombre = os.path.basename(os.path.normpath(args.lote))
    args.munecas = munecas
    try:
        args.criterio, args.umbral = resolver_regla(args.criterio, args.umbral, args.min_sp_both)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    args.regla = f"{args.criterio} >= {args.umbral:g}"

    todas = read_munecas(munecas)
    if not clips_con_datos(todas, args.criterio):
        # NaN cuenta «descartado», asi que sin datos la auditoria mediria TODO el lote
        # creyendo que mide los descartados. Es el caso de un CSV viejo sin sp_right/sp_left.
        print(f"❌ {os.path.basename(munecas)} no tiene datos para auditar con "
              f"`{args.criterio}` (ni sp_any ni sp_right/sp_left). Audita con "
              f"`--criterio sp_both`, que es lo que ese CSV registra.", file=sys.stderr)
        return 1
    fuera = select_rejected(todas, args.umbral, args.criterio)
    plan = stratified_sample(fuera, args.sample, args.criterio)
    print(f"📋 {len(todas)} clips en {os.path.basename(munecas)}, "
          f"{len(fuera)} descartados con la regla `{args.regla}`, "
          f"{len(plan)} a auditar")
    if not plan:
        print("   nada que auditar: la criba no descartó ningún clip")
        return 0
    coste = segundos_por_clip(args.cara_manos)
    print(f"   estimación de GPU: {len(plan) * coste:.0f} s "
          f"({coste} s/clip medidos en RTX 4090, "
          f"{'con' if args.cara_manos else 'sin'} cara y manos)")
    for fila in plan:
        print(f"     sp_both={_fmt(fila.get('sp_both'))} sp_any={_fmt(fila.get('sp_any'))} "
              f"{str(fila.get('video', ''))[:62]}")
    if args.dry_run:
        print("   (dry-run: no he llamado a OpenPose)")
        return 0

    out_root = args.out_dir or os.path.join(args.lote, "informes", "auditoria_criba")
    os.makedirs(out_root, exist_ok=True)
    import body_visibility
    config = body_visibility.VisibilityConfig.from_args(puntos, args.percent,
                                                        args.min_keypoint_confidence)

    filas, estados_vistos, abortado = [], [], False
    for i, fila in enumerate(plan, 1):
        if abortado:
            filas.append(fila_no_intentada(fila, "abortado tras 3 fallos identicos "
                                                 "seguidos (mira los clips de arriba)"))
            continue
        ruta = fila.get("path") or os.path.join(args.lote, "videos", "masked",
                                                str(fila.get("video", "")))
        v, medida = audit_clip(ruta, out_root, config, args.openpose_path, args.model_pose,
                               cara_manos=args.cara_manos, force=args.force,
                               sensibilidades=sensibilidades)
        medida.update({k: fila.get(k) for k in ("sp_both", "sp_any", "sp_right", "sp_left",
                                                "max_people")})
        medida.update({"pasa_coverage": v["pasa_coverage"], "una_mano": v["una_mano"],
                       "pasa_gate": v.get("pasa_gate", False), "pasa_al": v["pasa_al"]})
        filas.append(medida)
        print(f"   [{i}/{len(plan)}] {medida['estado']:14s} "
              f"dcha={_fmt(medida.get('op_dcha_pct'), '%'):>6s} "
              f"izq={_fmt(medida.get('op_izq_pct'), '%'):>6s} "
              f"{'PASA' if medida['pasa_coverage'] else ('una' if medida['una_mano'] else '-'):4s} "
              f"{str(fila.get('video', ''))[:52]}")
        escribe_csv(filas, os.path.join(out_root, "auditoria.csv"))  # avance persistente
        estados_vistos.append(medida["estado"])
        if abortar_por_fallos_repetidos(estados_vistos) and i < len(plan):
            abortado = True
            print(f"\n⚠️  3 fallos identicos seguidos ({medida['estado']}: "
                  f"{str(medida['error'])[:90]}): eso es el entorno, no los clips. Paro aqui "
                  f"y dejo {len(plan) - i} sin intentar.", file=sys.stderr)

    if abortado:
        escribe_csv(filas, os.path.join(out_root, "auditoria.csv"))  # con los no intentados


    r = resumen(filas, args.percent, sensibilidades)
    escribe_csv(filas, os.path.join(out_root, "auditoria.csv"))
    escribe_resumen(filas, r, os.path.join(out_root, "RESUMEN.md"), args)
    inf, sup = r["pasa_coverage_ic"]
    print(f"\n✅ {os.path.join(out_root, 'RESUMEN.md')}")
    print(f"   habrían pasado el gate de muñecas: {r['pasa_coverage']} de {r['auditados']} "
          f"(95 % exacto {inf * 100:.0f}–{sup * 100:.0f} %) -> esos son datos perdidos")
    print(f"   con UNA muñeca medible:            {r['una_mano']} de {r['auditados']}")
    print(f"   el gate tal como está aceptaría:   {r['pasa_gate']} "
          f"(rechaza cualquier frame con 2 personas)")
    if r["sin_medir"]:
        print(f"   sin medir: {len(r['sin_medir'])} (mira el CSV)")
    print(f"   coste: {r['segundos']:.0f} s")
    if r["auditados"] == 0:
        # Sin un solo clip medido la auditoria no puede afirmar nada, y un 0 aqui se
        # confundiria con "descartados sanos" en cualquier log del pipeline.
        print("\n❌ no se ha medido NINGUN clip: la auditoria no dice nada. Mira la columna "
              "error de auditoria.csv (normalmente rutas o GPU).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
