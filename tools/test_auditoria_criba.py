#!/usr/bin/env python3
"""Tests de la lógica pura de `auditoria_criba.py`, sin OpenPose ni GPU.

La auditoría existe para medir un error que el pipeline no puede ver: los clips que
la criba tiró y que habrían pasado el gate de muñecas. Si la selección o el veredicto
están mal, el informe sale con una cifra convincente y equivocada, y eso es peor que
no medir. Aquí se prueban las cuatro decisiones que se pueden equivocar en silencio:

  1. quién está descartado  -> el límite del umbral y los scores NaN
  2. qué se muestrea         -> que la muestra cubra la frontera y el fondo, y sea
                                determinista (dos pasadas, la misma lista)
  3. qué cuenta como pasar   -> las dos muñecas vs UNA, y los porcentajes de sensibilidad
  4. qué dice el intervalo   -> Clopper-Pearson contra la forma cerrada de la beta, en
                                los extremos (0 de 16, 16 de 16), que es donde las
                                aproximaciones mienten

No se importa OpenPose, cv2 ni GPU: `audit_clip` importa argos y body_visibility dentro
de la función, así que el módulo se importa limpio.
"""
import csv
import os
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

from auditoria_criba import (CAMPOS, UMBRALES, _num, abortar_por_fallos_repetidos,  # noqa: E402
                             clips_con_datos, escribe_csv, fila_no_intentada, ic_exacto,
                             read_munecas, resolver_regla, resumen, score, select_rejected,
                             sp_any, stratified_sample, veredicto)

FAIL = []
TOTAL = 0


def check(nombre, ok, detalle=""):
    global TOTAL
    TOTAL += 1
    print(f"[{'ok ' if ok else 'FAIL'}] {nombre}" + ("" if ok else f"  {detalle}"))
    if not ok:
        FAIL.append(nombre)


def clip(nombre, sp_both="", sp_right="", sp_left="", path=None):
    return {"video": nombre + ".mp4", "sp_both": sp_both, "sp_right": sp_right,
            "sp_left": sp_left, "max_people": "1",
            "path": path or f"/lote/videos/masked/{nombre}.mp4"}


# ------------------------------------------------------------- selección ----

def test_el_limite_no_es_descartado():
    """El umbral es inclusivo: lo auditado es lo que quedó POR DEBAJO, no lo que empata.

    Se prueba con `sp_both` explicito. Con el criterio nuevo el mismo fixture daría otro
    resultado a propósito: `pordebajo` tiene una muneca a 0,8 y la otra a 0,0, y eso con
    `sp_any` ES admisible. Ese es el cambio de regla, no un descuido del test.
    """
    filas = [clip("admitidojusto", 0.8, 0.8, 0.8), clip("pordebajo", 0.7999, 0.8, 0.0),
             clip("por encima", 0.8001, 0.8, 0.8)]
    fuera = select_rejected(filas, 0.8, "sp_both")
    check("el empates con el umbral NO se audita (se admitio)",
          [f["video"] for f in fuera] == ["pordebajo.mp4"],
          str([f["video"] for f in fuera]))
    check("la regla antigua es la que deja fuera ese clip de una sola muneca",
          [f["video"] for f in select_rejected(filas, UMBRALES["sp_any"], "sp_any")] == [],
          "con sp_any >= 0,6 los tres son admisibles: una muneca llega")


def test_una_muneca_cambia_de_bando_segun_el_criterio():
    """El caso que motivó el cambio, con los numeros reales del KABC 1087.

    sp_both=0,667 (fuera de la regla estricta) pero sp_any=1,0. OpenPose le dio luego
    91,9 % / 100 % con gate=accepted: era un clip bueno tirado antes de poder medirlo.
    """
    kabc = [clip("kabc", 0.6667, 0.6667, 1.0)]
    check("con la regla estricta el KABC estaba descartado",
          [f["video"] for f in select_rejected(kabc, 0.8, "sp_both")] == ["kabc.mp4"])
    check("con la regla nueva el KABC se admite (ya no hay nada que auditar de él)",
          select_rejected(kabc, 0.6, "sp_any") == [])


def test_sp_any_manda_sobre_lo_derivado():
    """Si la criba escribio sp_any, se audita ESA: es lo que decidio la admision."""
    fila = {"video": "x.mp4", "sp_any": "0.45", "sp_right": "1.0", "sp_left": "1.0"}
    check("la columna escrita gana a la derivada de las munecas",
          sp_any(fila) == 0.45, str(sp_any(fila)))
    check("sin columna escrita se deriva de las dos munecas",
          sp_any({"video": "y", "sp_right": "0.3", "sp_left": "0.9"}) == 0.9)


def test_resolver_regla_no_adivina():
    """Las banderas que se contradicen no producen una auditoria de una regla inexistente."""
    check("por defecto se audita la regla actual", resolver_regla() == ("sp_any", 0.6),
          str(resolver_regla()))
    check("--min-sp-both arrastra el criterio estricto",
          resolver_regla(min_sp_both=0.8) == ("sp_both", 0.8), str(resolver_regla(min_sp_both=0.8)))
    check("el umbral explicito manda sobre el defecto del criterio",
          resolver_regla(criterio="sp_both", umbral=0.6) == ("sp_both", 0.6))
    for nombre, kwargs in [("min-sp-both con umbral", {"umbral": 0.6, "min_sp_both": 0.8}),
                           ("min-sp-both con criterio ajeno",
                            {"criterio": "sp_any", "min_sp_both": 0.8}),
                           ("criterio inventado", {"criterio": "sp_derecha"})]:
        try:
            resolver_regla(**kwargs)
            check(f"{nombre} se niega en vez de mezclar", False, "->acepto")
        except ValueError:
            check(f"{nombre} se niega en vez de mezclar", True)


def test_sin_datos_del_criterio_no_se_audit_nada():
    """Un CSV sin munecas por separado no se puede auditar con sp_any.

    NaN cuenta «descartado», asi que sin datos la auditoria mediria TODO el lote creyendo
    que mide los descartados. Es el caso de prescreen_validation_29.csv, que solo tiene
    sp_both. Mejor negarse antes de gastar GPU.
    """
    sin_munecas = [{"video": "a", "sp_both": "0.8"}, {"video": "b", "sp_both": "0.0"}]
    check("sin sp_right/sp_left no hay datos para sp_any",
          clips_con_datos(sin_munecas, "sp_any") == 0,
          str(clips_con_datos(sin_munecas, "sp_any")))
    check("pero si los hay para sp_both (por eso existe --criterio)",
          clips_con_datos(sin_munecas, "sp_both") == 2)
    mixto = sin_munecas + [{"video": "c", "sp_both": "0.1", "sp_right": "0.9"}]
    check("basta una fila con munecas para que el criterio sea auditable",
          clips_con_datos(mixto, "sp_any") == 1, str(clips_con_datos(mixto, "sp_any")))


def test_la_muestra_sigue_la_metrica_que_decide():
    """Mismas filas, distinto orden segun el criterio: la muestra debe cubrir la frontera DE ESA regla.

    OJO con el size: pidiendo muestra >= total no se prueba nada, porque el codigo ataja y
    devuelve TODOS en orden de entrada (no hay orden cuando se mide todo). Hace falta una
    muestra estrictamente mas pequena que el lote.
    """
    filas = [clip("una_sola", 0.0, 1.0, 0.0), clip("dos_regulares", 0.7, 0.7, 0.7),
             clip("baja", 0.5, 0.5, 0.5), clip("muerto", 0.0, 0.0, 0.0)]
    con_sp_any = [f["video"] for f in stratified_sample(filas, 2, "sp_any")]
    con_sp_both = [f["video"] for f in stratified_sample(filas, 2, "sp_both")]
    check("con sp_any el de una sola muneca encabeza la muestra",
          con_sp_any[0] == "una_sola.mp4", str(con_sp_any))
    check("con sp_both encabeza el de las dos regulares",
          con_sp_both[0] == "dos_regulares.mp4", str(con_sp_both))
    check("y el de una sola muneca cae al fondo con la regla estricta",
          con_sp_both[-1] == "una_sola.mp4", str(con_sp_both))
    check("con muestra >= total no hay orden que probar (ataja y devuelve todos)",
          [f["video"] for f in stratified_sample(filas, 4, "sp_any")] == [f["video"] for f in filas],
          "el orden de entrada se conserva a proposito")


def test_nan_cuenta_como_descartado():
    """Un score NaN (clip no decodificable) tampoco se habría admitido."""
    filas = [clip("sindato"), clip("muerto", 0.0, 0.0, 0.0)]
    fuera = select_rejected(filas, 0.8)
    check("NaN entra en la auditoria", len(fuera) == 2, str([f["video"] for f in fuera]))
    check("NaN se queda NaN, no se convierte en 0",
          _num(fuera[0]["sp_both"]) != _num(fuera[0]["sp_both"]))


def test_sp_any_es_la_mejor_muneca():
    check("sp_any = max(dcha, izq)", sp_any(clip("a", 0.0, 0.133, 1.0)) == 1.0)
    check("sp_any ignora un NaN y usa la otra", sp_any(clip("b", "", "", 0.6)) == 0.6)
    check("sp_any sin ninguna muñeca es NaN", sp_any(clip("c")) != sp_any(clip("c")))


def test_read_munecas_coacciona_y_conserva_ruta():
    with tempfile.TemporaryDirectory() as td:
        ruta = os.path.join(td, "munecas.csv")
        with open(ruta, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=["video", "sp_both", "sp_right", "sp_left",
                                                   "max_people", "reported_frames", "path"])
            w.writeheader()
            w.writerow({"video": "a.mp4", "sp_both": "0.8000", "sp_right": "1.0000",
                        "sp_left": "0.8000", "max_people": "1", "reported_frames": "140",
                        "path": "/lote/videos/masked/a.mp4"})
            w.writerow({"video": "b.mp4", "sp_both": "", "sp_right": "", "sp_left": "",
                        "max_people": "3", "reported_frames": "",
                        "path": "/lote/videos/masked/b.mp4"})
        filas = read_munecas(ruta)
        check("los scores salen numeros", isinstance(filas[0]["sp_both"], float)
              and filas[0]["sp_both"] == 0.8, repr(filas[0]["sp_both"]))
        check("una celda vacia es NaN, no 0",
              _num(filas[1]["sp_both"]) != _num(filas[1]["sp_both"])
              and _num(filas[1]["reported_frames"]) != _num(filas[1]["reported_frames"]),
              repr(filas[1]))
        check("leer no inventa sp_any: lo calcula quien lo pide",
              "sp_any" not in filas[1] and sp_any(filas[1]) != sp_any(filas[1]), repr(filas[1]))
        check("la ruta del mp4 se conserva", filas[0]["path"].endswith("a.mp4"), filas[0]["path"])


# --------------------------------------------------------------- muestra ----

def test_muestra_vacia_es_todo():
    filas = [clip(f"c{i}", i / 10.0) for i in range(10)]
    check("sample=0 audita todos", len(stratified_sample(filas, 0)) == 10)
    check("sample mayor que el total audita todos", len(stratified_sample(filas, 99)) == 10)


def test_muestra_cubre_frontera_y_fondo():
    """Lo que decide está justo debajo del umbral; el sorteo gastaria GPU en muertos.

    Scores todos distintos a propósito: con empates, el orden de desempate es el nombre
    y `muerto9` sale despues de `muerto12`, que es una trampa del test, no del codigo.
    """
    filas = [clip("frontera", 0.7333)] + [clip(f"p{i:02d}", (13 - i) / 100.0) for i in range(13)]
    elegidos = stratified_sample(filas, 4)
    nombres = [f["video"] for f in elegidos]
    check("la muestra pilla el mejor score (la frontera)", "frontera.mp4" in nombres,
          str(nombres))
    check("la muestra pilla el peor de la lista (el fondo)", nombres[-1] == "p12.mp4",
          str(nombres))
    check("la muestra no repite clips", len(set(nombres)) == len(nombres), str(nombres))
    check("la muestra es determinista",
          [f["video"] for f in stratified_sample(filas, 4)] == nombres)
    check("un solo clip elegible = el mejor",
          stratified_sample(filas, 1)[0]["video"] == "frontera.mp4")


# ------------------------------------------------------------- veredicto ----

def test_veredicto_bimanual_y_unilateral():
    v = veredicto({4: 92.1, 7: 90.7}, 80.0)
    check("las dos por encima del umbral -> pasa", v["pasa_coverage"] and v["una_mano"], str(v))
    v = veredicto({4: 100.0, 7: 12.0}, 80.0)
    check("una mano buena NO pasa el gate de dos munecas", not v["pasa_coverage"], str(v))
    check("una mano buena cuenta como una_mano", v["una_mano"], str(v))
    check("el umbral es inclusivo (justo 80 pasa)",
          veredicto({4: 80.0, 7: 80.0}, 80.0)["pasa_coverage"])


def test_veredicto_sensibilidades():
    v = veredicto({4: 69.5, 7: 99.3}, 80.0, sensibilidades=(40.0, 60.0))
    check("pasa_al lista los umbrales que superan AMBAS", v["pasa_al"] == "40,60", v["pasa_al"])
    v = veredicto({4: 45.0, 7: 99.0}, 80.0, sensibilidades=(40.0, 60.0))
    check("una muneca por debajo del minimo corta la lista", v["pasa_al"] == "40", v["pasa_al"])
    v = veredicto({4: 10.0, 7: 10.0}, 80.0, sensibilidades=(40.0, 60.0))
    check("por debajo de todo, lista vacia", v["pasa_al"] == "", v["pasa_al"])


def test_veredicto_sin_datos_no_es_un_cero_silencioso():
    v = veredicto({}, 80.0)
    check("sin cobertura medida no se afirma nada",
          v["pasa_coverage"] is False and v["una_mano"] is False and v["pasa_al"] == "", str(v))
    check("un punto NaN no arrastra al otro",
          veredicto({4: 90.0, 7: float("nan")}, 80.0)["pasa_coverage"] is True,
          "solo se afirma sobre lo medido")


# -------------------------------------------------------------- intervalo ----

def test_intervalo_contra_la_beta():
    """Clopper-Pearson tiene forma cerrada con la beta: se compara, no se confía."""
    try:
        from scipy.stats import beta
    except Exception:
        print("[skip] intervalo contra la beta: sin scipy en este entorno")
        return
    for k, n in ((0, 16), (2, 16), (5, 20), (4, 4), (16, 16), (1, 4)):
        inf, sup = ic_exacto(k, n)
        esperado_inf = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
        esperado_sup = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
        check(f"IC {k}/{n} coincide con la beta",
              abs(inf - esperado_inf) < 1e-4 and abs(sup - esperado_sup) < 1e-4,
              f"[{inf:.4f},{sup:.4f}] vs [{esperado_inf:.4f},{esperado_sup:.4f}]")


def test_intervalo_en_los_extremos():
    inf, sup = ic_exacto(0, 16)
    check("0 de 16: inferior 0", inf == 0.0, str(inf))
    check("0 de 16: el superior NO es 0 (no vistos no es inexistentes)", 0.15 < sup < 0.25,
          f"{sup:.4f}")
    inf, sup = ic_exacto(16, 16)
    check("16 de 16: superior 1", sup == 1.0, str(sup))
    check("16 de 16: inferior muy por debajo de 1", 0.7 < inf < 0.9, f"{inf:.4f}")
    inf, sup = ic_exacto(4, 4)
    check("4 de 4: la frase honesta es >= ~40 %, no >= ~71 %",
          0.35 < inf < 0.45, f"{inf:.4f}")
    check("n=0 no es un numero", ic_exacto(0, 0)[0] != ic_exacto(0, 0)[0])


# ---------------------------------------------------------------- informe ----

def test_resumen_cuenta_lo_medido_no_lo_intento():
    filas = [
        dict(clip("a", 0.0), estado="medido", pasa_coverage=True, una_mano=True,
             pasa_gate=True, pasa_al="40,60", frames_multi_person="0", segundos="12"),
        dict(clip("b", 0.0), estado="medido", pasa_coverage=False, una_mano=True,
             pasa_gate=False, pasa_al="40", frames_multi_person="7", segundos="11"),
        dict(clip("c", 0.0), estado="openpose fallo", pasa_coverage=False, una_mano=False,
             pasa_gate=False, pasa_al="", frames_multi_person="0", segundos="2",
             error="rc=1"),
    ]
    r = resumen(filas, 80.0)
    check("auditados solo lo medido", r["auditados"] == 2 and r["descartados_en_total"] == 3,
          str(r))
    check("1 de 2 habrian pasado", r["pasa_coverage"] == 1, str(r))
    check("2 de 2 tenian al menos una muneca", r["una_mano"] == 2, str(r))
    check("el gate aceptaria 1 (el otro tiene frames multi-persona)", r["pasa_gate"] == 1,
          str(r))
    check("lo sin medir se nombra, no se ignora",
          [f["video"] for f in r["sin_medir"]] == ["c.mp4"], str(r["sin_medir"]))
    check("el porcentaje de sensibilidad se suma", r["por_sensibilidad"][40.0] == 2
          and r["por_sensibilidad"][60.0] == 1, str(r["por_sensibilidad"]))


def test_csv_escribe_todas_las_columnas_del_informe():
    with tempfile.TemporaryDirectory() as td:
        ruta = os.path.join(td, "auditoria.csv")
        escribe_csv([dict(clip("a", 0.0), estado="medido", op_dcha_pct=91.5,
                          op_izq_pct=float("nan"), pasa_coverage=True)], ruta)
        leido = list(csv.DictReader(open(ruta, encoding="utf-8")))
        check("el CSV tiene las columnas pactadas", list(leido[0].keys()) == CAMPOS,
              str(list(leido[0].keys())))
        check("un float se escribe con decimales", leido[0]["op_dcha_pct"] == "91.5000",
              leido[0]["op_dcha_pct"])
        check("un NaN no se cuela como texto 'nan'", leido[0]["op_izq_pct"] == "",
              repr(leido[0]["op_izq_pct"]))


# ------------------------------------------------------------- cableado ----

class _Config:
    min_visible_percent = 80.0


def _modulos_falsos(record, cobertura=None, rc=0, decision="accepted", error=None):
    """`argos` y `body_visibility` de pega, inyectados en sys.modules.

    El fake de `run_openpose` reproduce el contrato real: `call=None` es la costura de
    test de argos y petar si no es llamable, que es justo como se rompio la primera
    pasada de esta auditoria (TypeError dentro de argos, en el clip 1, con la GPU sin
    haberse enterado). Un fake que aceptara cualquier cosa no fijaria nada.
    """
    import types

    argos_falso = types.ModuleType("argos")
    argos_falso.clip_name = lambda v: os.path.basename(v)[:-4] if v.endswith(".mp4") \
        else os.path.basename(v)
    argos_falso._frame_count = lambda v: (140, record.get("error_conteo"))

    def run_openpose(video, out_root, openpose_path, face_hands, skeletons,
                     model_pose=None, call=None):
        if not callable(call):
            raise TypeError("'NoneType' object is not callable")  # como argos.py:67
        record["llamado"] = {"face_hands": face_hands, "skeletons": skeletons,
                             "model_pose": model_pose, "openpose_path": openpose_path,
                             "video": video}
        return rc

    argos_falso.run_openpose = run_openpose
    argos_falso._default_call = lambda: (lambda argv, cwd=None, quiet=False: 0)

    bv_falso = types.ModuleType("body_visibility")

    class _Decision:
        def __init__(self):
            cob = cobertura or {}
            self._d = {"decision": decision, "error": error,
                       "frames_multi_person": 3, "frames_no_person": 1,
                       "frames_missing_json": 0,
                       "points": [{"point": 4, "coverage_percent": cob.get(0, float("nan"))},
                                  {"point": 7, "coverage_percent": cob.get(7, float("nan"))}]}

        def as_dict(self):
            return self._d

    def evaluate(json_dir, expected_frames, config, clip=None):
        record["medido_con"] = {"json_dir": json_dir, "expected_frames": expected_frames,
                                "clip": clip}
        return _Decision()

    bv_falso.evaluate_body_visibility = evaluate
    guardados = {k: sys.modules.get(k) for k in ("argos", "body_visibility")}
    sys.modules["argos"], sys.modules["body_visibility"] = argos_falso, bv_falso
    return guardados


def _restaura(guardados):
    for clave, modulo in guardados.items():
        if modulo is None:
            sys.modules.pop(clave, None)
        else:
            sys.modules[clave] = modulo


def test_audit_clip_pasa_un_runner_llamable():
    """El bug de la primera pasada: `call=None` hacia petar argos en el clip 1."""
    from auditoria_criba import audit_clip
    with tempfile.TemporaryDirectory() as td:
        video = os.path.join(td, "videos", "masked", "clipX.mp4")
        os.makedirs(os.path.dirname(video))
        open(video, "wb").close()
        record = {}
        guardados = _modulos_falsos(record, cobertura={0: 92.1, 7: 90.7})
        try:
            v, fila = audit_clip(video, os.path.join(td, "out"), _Config(), "/opt/openpose",
                                 "BODY_25")
            check("llega a 'medido' en vez de reventar", fila["estado"] == "medido", str(fila))
            check("pasa un runner llamable a run_openpose",
                  record.get("llamado") is not None, str(record))
            check("pindea BODY_25 (los indices 4 y 7 solo significan muneca ahi)",
                  record["llamado"]["model_pose"] == "BODY_25", str(record["llamado"]))
            check("no pide cara ni manos por defecto (no cambian las munecas: cuestan)",
                  record["llamado"]["face_hands"] is False
                  and record["llamado"]["skeletons"] is False, str(record["llamado"]))
            check("mide contra los frames del video, no contra los JSON",
                  record["medido_con"]["expected_frames"] == 140, str(record["medido_con"]))
            check("el JSON queda colgado de out_root, nunca de dataset/ del lote",
                  record["medido_con"]["json_dir"].endswith(
                      os.path.join("dataset", "OpenPose", "clipX", "JSON_FILES")),
                  record["medido_con"]["json_dir"])
            check("la cobertura del fake llega al veredicto",
                  fila["op_dcha_pct"] == 92.1 and v["pasa_coverage"] is True, str((fila, v)))
        finally:
            _restaura(guardados)


def test_audit_clip_no_rompe_por_un_problema_del_clip():
    """OpenPose que no arranca y data_error son estados, no excepciones."""
    from auditoria_criba import audit_clip
    with tempfile.TemporaryDirectory() as td:
        video = os.path.join(td, "clipY.mp4")
        open(video, "wb").close()
        guardados = _modulos_falsos({}, rc=1)
        try:
            v, fila = audit_clip(video, os.path.join(td, "out"), _Config(), "/opt/openpose",
                                 "BODY_25")
            check("openpose que falla es un estado", fila["estado"] == "openpose fallo"
                  and "rc=1" in fila["error"], str(fila))
            check("un clip roto no se cuenta como aprobado", v["pasa_coverage"] is False, str(v))
        finally:
            _restaura(guardados)
        guardados = _modulos_falsos({}, decision="data_error", error="skeleton no es BODY_25")
        try:
            v, fila = audit_clip(video, os.path.join(td, "out"), _Config(), "/opt/openpose",
                                 "BODY_25")
            check("data_error se conserva con su causa", fila["estado"] == "data_error"
                  and "BODY_25" in fila["error"], str(fila))
        finally:
            _restaura(guardados)
        guardados = _modulos_falsos({})
        try:
            v, fila = audit_clip(os.path.join(td, "noexiste.mp4"), os.path.join(td, "out"),
                                 _Config(), "/opt/openpose", "BODY_25")
            check("mp4 que no existe es un estado", fila["estado"] == "sin mp4", str(fila))
        finally:
            _restaura(guardados)


def test_audit_clip_reanuda_sin_repetir_gpu():
    """Lo ya medido se reutiliza; con --force se repite. Una auditoria de 4 min no se relanza a ciegas."""
    from auditoria_criba import audit_clip
    with tempfile.TemporaryDirectory() as td:
        video = os.path.join(td, "clipZ.mp4")
        open(video, "wb").close()
        jsons = os.path.join(td, "out", "dataset", "OpenPose", "clipZ", "JSON_FILES")
        os.makedirs(jsons)
        # hace falta un .json de verdad: una carpeta vacia NO es "ya medido", y es
        # justo el estado que dejo la pasada que petio a medias
        open(os.path.join(jsons, "clipZ_000000000000_keypoints.json"), "w").close()
        record = {}
        guardados = _modulos_falsos(record, cobertura={0: 50.0, 7: 50.0})
        try:
            v, fila = audit_clip(video, os.path.join(td, "out"), _Config(), "/opt/openpose",
                                 "BODY_25")
            check("con JSON previos no vuelve a OpenPose",
                  fila["estado"] == "reutilizado" and "llamado" not in record, str(fila))
            check("y aun asi se mide", fila["op_dcha_pct"] == 50.0, str(fila))
            v, fila = audit_clip(video, os.path.join(td, "out"), _Config(), "/opt/openpose",
                                 "BODY_25", force=True)
            check("--force si que repite la GPU", "llamado" in record, str(record))
        finally:
            _restaura(guardados)
        # y una carpeta vacia NO es "ya medido": es lo que dej6 la pasada que pet6
        vacia = os.path.join(td, "out2", "dataset", "OpenPose", "clipZ", "JSON_FILES")
        os.makedirs(vacia)
        record = {}
        guardados = _modulos_falsos(record, cobertura={0: 10.0, 7: 10.0})
        try:
            v, fila = audit_clip(video, os.path.join(td, "out2"), _Config(), "/opt/openpose",
                                 "BODY_25")
            check("una carpeta JSON vacia no se confunde con medido",
                  fila["estado"] == "medido" and "llamado" in record, str(fila))
        finally:
            _restaura(guardados)


def test_rutas_relativas_llegan_absolutas_a_openpose():
    """El bug de la segunda pasada: rc=255 en los 16 clips porque OpenPose corre en /opt/openpose.

    `run_openpose` ejecuta con cwd=/opt/openpose, asi que una --write_json relativa se
    resolvia dentro de /opt/openpose y no en el lote. Los tests de arriba usan rutas
    absolutas y no lo pillaban: este usa un --lote relativo, que es lo que escribi yo.
    """
    from auditoria_criba import audit_clip
    cwd_original = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        try:
            os.chdir(td)
            os.makedirs("lote/videos/masked")
            video = "lote/videos/masked/clipR.mp4"
            open(video, "wb").close()
            record = {}
            guardados = _modulos_falsos(record, cobertura={0: 85.0, 7: 85.0})
            try:
                v, fila = audit_clip(video, "lote/informes/auditoria_criba", _Config(),
                                     "/opt/openpose", "BODY_25")
                check("el mp4 llega absoluto a openpose.bin",
                      os.path.isabs(record["llamado"]["video"]), record["llamado"]["video"])
                check("el mp4 absoluto sigue apuntando al mismo sitio",
                      os.path.realpath(record["llamado"]["video"]) == os.path.realpath(video),
                      record["llamado"]["video"])
                check("el JSON se pide dentro del lote, no dentro de /opt/openpose",
                      os.path.isabs(record["medido_con"]["json_dir"])
                      and os.path.realpath(record["medido_con"]["json_dir"]).startswith(
                          os.path.realpath(os.path.join(td, "lote"))),
                      record["medido_con"]["json_dir"])
                check("y el informe sale medido", fila["estado"] == "medido", str(fila))
            finally:
                _restaura(guardados)
        finally:
            os.chdir(cwd_original)


def test_aborta_tras_tres_fallos_identios():
    """Un entorno roto se detecta en el clip 3, no en el 16 (y el 1 gasta 3 s, no 4 min)."""
    check("dos fallos iguales aun no abortan (puede ser un clip raro)",
          abortar_por_fallos_repetidos(["openpose fallo", "openpose fallo"]) is False)
    check("tres iguales si", abortar_por_fallos_repetidos(
        ["openpose fallo"] * 3) is True)
    check("fallos distintos no abortan", abortar_por_fallos_repetidos(
        ["sin mp4", "openpose fallo", "data_error"]) is False)
    check("medir no es un fallo repetido", abortar_por_fallos_repetidos(["medido"] * 5) is False)
    check("reutilizar tampoco", abortar_por_fallos_repetidos(["reutilizado"] * 5) is False)
    check("el umbral se puede bajar", abortar_por_fallos_repetidos(
        ["sin conteo de frames"] * 2, umbral=2) is True)
    check("una serie larga con un bueno en medio no aborta", abortar_por_fallos_repetidos(
        ["openpose fallo", "openpose fallo", "medido", "openpose fallo",
         "openpose fallo"]) is False)


def test_no_intentado_sigue_apareciendo():
    """Lo que no llegamos a medir no desaparece del informe ni se cuenta como medido."""
    fila = fila_no_intentada(dict(clip("abandono", 0.5, 0.5, 0.0), sp_any=1.0), "abortado")
    check("estado no intentado", fila["estado"] == "no intentado", str(fila))
    check("conserva los scores de la criba", fila["sp_both"] == 0.5 and fila["sp_any"] == 1.0
          and fila["sp_right"] == 0.5 and fila["sp_left"] == 0.0, str(fila))
    check("no se inventa un veredicto", fila["pasa_coverage"] is False
          and fila["una_mano"] is False and fila["pasa_al"] == "", str(fila))
    check("tiene las columnas del CSV", all(campo in fila for campo in CAMPOS),
          str([c for c in CAMPOS if c not in fila]))
    medido = dict(clip("si", 0.5), estado="medido", pasa_coverage=True, una_mano=True,
                  pasa_gate=False, op_dcha_pct=90.0, op_izq_pct=90.0, pasa_al="60,80")
    r = resumen([medido, fila], 80.0)
    check("lo no intentado no entra en el denominador", r["auditados"] == 1, str(r["auditados"]))
    check("pero si en 'sin medir'", len(r["sin_medir"]) == 1
          and r["sin_medir"][0]["estado"] == "no intentado", str(r["sin_medir"]))


def test_el_coste_estimado_es_el_de_la_configuracion():
    """El dry-run prometi6 229 s para una pasada que tardo 86 s: la cifra era de otra config."""
    from auditoria_criba import (SEGUNDOS_POR_CLIP, SEGUNDOS_POR_CLIP_CARA_MANOS,
                                segundos_por_clip)
    check("sin cara ni manos se estima lo medido aqui (86 s / 16 clips)",
          segundos_por_clip(False) == SEGUNDOS_POR_CLIP == 5.4,
          str(segundos_por_clip(False)))
    check("con cara y manos se estima la config del pipeline",
          segundos_por_clip(True) == SEGUNDOS_POR_CLIP_CARA_MANOS == 14.3,
          str(segundos_por_clip(True)))
    check("apagar cara y manos barata de verdad el plan (no es la misma pasada)",
          segundos_por_clip(False) < segundos_por_clip(True) * 0.6,
          f"{segundos_por_clip(False)} vs {segundos_por_clip(True)}")


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        try:
            globals()[fn]()
        except Exception as e:  # noqa: BLE001 - un test que peta es un fallo, no un silencio
            check(f"{fn} (excepcion)", False, repr(e))
    if FAIL:
        print(f"\nTOTAL: {TOTAL - len(FAIL)} / {TOTAL}   fallos: {', '.join(FAIL)}")
        sys.exit(1)
    print(f"\nTOTAL: {TOTAL} / {TOTAL}")
    sys.exit(0)
