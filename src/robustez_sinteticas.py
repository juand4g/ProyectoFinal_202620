"""
Robustez de las alternativas del metodo unificado ante ruido y numero de
puntos, con curvas sinteticas de periodo conocido.

Alternativas:
  Decision de rama (binaria vs. pulsante):
    - skew(mag) > 0.5 (analitica, sin P0)
    - skew corregida por ruido > 0.5 (analitica): el ruido gaussiano diluye
      la skewness como g_obs = g_int * (s2 / (s2 + e2))^(3/2), con s2 la
      varianza de la senal y e2 la media de err^2; se despeja g_int.
    - regresion logistica P(binaria) > 0.7 (supervisada, entrenada con las
      2000 curvas OGLE de decision_binaria_pulsante.py)
    - oraculo (clase real), como referencia
  Selector del multiplo en la rama de binarias:
    - x2 + BIC (analitico, H0=4)
    - gradient boosting (supervisado, entrenado con las 1000 binarias OGLE de
      selector_multiplo_binarias.py)
  Rama de pulsantes: P0 directo.

Curvas sinteticas:
  - Plantillas: estrellas OGLE que NO estan en ninguna muestra de
    entrenamiento, con N >= MIN_PUNTOS_PLANTILLA, amplitud/error mediano >=
    MIN_SNR_PLANTILLA y buen plegado en el periodo de catalogo (dispersion
    de los residuos <= MAX_DISPERSION_PLANTILLA errores; descarta doble modo,
    Blazhko fuerte y periodos dudosos). Cefeidas solo de modo simple (F, 1, 2).
    Plantilla = mediana por bin de fase, interpolada circularmente.
  - Periodo verdadero: P_cat * U(0.85, 1.15), para no heredar la relacion
    exacta entre el periodo original y su cadencia.
  - Cadencia: tiempos reales de OTRA estrella OGLE de la misma clase.
  - Ruido gaussiano con sigma = amplitud pico a pico / SNR; err = sigma.
  - Cada plantilla se simula en todas las configuraciones SNR x N (mismo
    periodo, misma cadencia, misma fase): comparaciones pareadas.

Cache append-safe en Robustez_Sinteticas/cache_sinteticas.csv.
"""

import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from decision_binaria_pulsante import (
    CACHE_CSV as CACHE_DECISION, INDICADORES, indicadores_forma,
)
from etapa1_cefeidas_rrlyrae import CACHE_CSV as CACHE_PULSANTES, TIPOS, cargar_catalogo as catalogo_pulsantes
from refinamiento_rank_product_bic import N_PROCESOS
from refinamiento_rank_product_enmascarado_bic import CACHE_CSV as CACHE_HIBRIDO, periodos_base
from refinamiento_subarmonicos_bic import (
    N_MAX, PHOT_DIR, RANDOM_SEED, acierto, bic_fourier, cargar_catalogo as catalogo_binarias, n_hibrido,
)
from selector_multiplo_binarias import (
    CACHE_INDICADORES, construir_tabla, indicadores_curva, modelos,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "Robustez_Sinteticas"
CACHE_CSV = OUT_DIR / "cache_sinteticas.csv"
CACHE_PLANTILLAS = OUT_DIR / "plantillas.csv"
CACHE_SKEW_OGLE = OUT_DIR / "cache_skew_ogle.csv"
RESUMEN_TXT = OUT_DIR / "resumen_sinteticas.txt"

N_PLANTILLAS = {"Binaria": 120, "Cefeida": 60, "RR Lyrae": 60}
SNRS = [50, 15, 8, 4]
N_OBJETIVO = [0, 100]  # 0 = todos los puntos de la cadencia
MIN_PUNTOS_PLANTILLA = 400
MIN_SNR_PLANTILLA = 25
MAX_DISPERSION_PLANTILLA = 2.0
MIN_PUNTOS_CADENCIA = 100
P_MAX_PLANTILLA = 100.0
H0 = 4
UMBRAL_SKEW = 0.5
UMBRAL_LOGISTICA = 0.7
CLASES = ["Binaria", "Cefeida", "RR Lyrae"]

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- plantillas
def leer(ruta):
    t, mag, err = np.loadtxt(ruta, unpack=True)
    return t, mag, np.where(err > 0, err, np.median(err[err > 0]))


def plantilla(t, mag, periodo):
    """Mediana por bin de fase (bins segun N), con bins vacios interpolados.
    Devuelve (centros, valores) o None si cubre menos del 90% de la fase."""
    n_bins = int(np.clip(t.size // 5, 50, 150))
    fase = ((t - t.min()) / periodo) % 1.0
    idx = np.minimum((fase * n_bins).astype(int), n_bins - 1)
    centros = (np.arange(n_bins) + 0.5) / n_bins
    valores = np.array([np.median(mag[idx == b]) if np.any(idx == b) else np.nan for b in range(n_bins)])
    llenos = ~np.isnan(valores)
    if llenos.mean() < 0.9:
        return None
    valores = np.interp(centros, np.r_[centros[llenos] - 1, centros[llenos], centros[llenos] + 1],
                        np.tile(valores[llenos], 3))
    return centros, valores - np.median(valores)


def evaluar_plantilla(c, v, fase):
    return np.interp(fase, np.r_[c - 1, c, c + 1], np.tile(v, 3))


def candidatos_por_clase():
    """(ruta, P_cat, subtipo) de estrellas fuera de toda muestra de entrenamiento."""
    usados = set(pd.read_csv(CACHE_HIBRIDO)["id"]) | set(pd.read_csv(CACHE_PULSANTES)["id"])
    cat_b = catalogo_binarias()
    out = {"Binaria": [(PHOT_DIR / f"{i}.dat", float(cat_b.at[i, "P"]), cat_b.at[i, "Type"])
                       for i in sorted(p.stem for p in PHOT_DIR.glob("*.dat"))
                       if i in cat_b.index and i not in usados]}
    for clase, (carpeta, catalogo_txt) in TIPOS.items():
        cat = catalogo_pulsantes(catalogo_txt)
        out[clase] = [(carpeta / f"{i}.dat", float(cat.at[i, "P_1"]), str(cat.at[i, "Mode"]))
                      for i in sorted(p.stem for p in carpeta.glob("*.dat"))
                      if i in cat.index and i not in usados]
    return out


def elegir_plantillas():
    if CACHE_PLANTILLAS.exists():
        return pd.read_csv(CACHE_PLANTILLAS)
    filas = []
    for clase, candidatos in candidatos_por_clase().items():
        random.seed(RANDOM_SEED)
        random.shuffle(candidatos)
        elegidas = []
        cadencias = []
        for ruta, periodo, subtipo in tqdm(candidatos, desc=f"Plantillas {clase}", unit="estrella"):
            if len(elegidas) >= N_PLANTILLAS[clase] and len(cadencias) >= N_PLANTILLAS[clase]:
                break
            try:
                t, mag, err = leer(ruta)
            except (ValueError, OSError):
                continue
            if t.size >= MIN_PUNTOS_CADENCIA and len(cadencias) < N_PLANTILLAS[clase]:
                cadencias.append(str(ruta))
            if (len(elegidas) >= N_PLANTILLAS[clase] or t.size < MIN_PUNTOS_PLANTILLA
                    or periodo > P_MAX_PLANTILLA or (clase == "Cefeida" and subtipo not in {"F", "1", "2"})):
                continue
            pl = plantilla(t, mag, periodo)
            if pl is None:
                continue
            amplitud = pl[1].max() - pl[1].min()
            if amplitud / np.median(err) < MIN_SNR_PLANTILLA:
                continue
            resid = mag - np.median(mag) - evaluar_plantilla(*pl, ((t - t.min()) / periodo) % 1.0)
            if np.std(resid) / np.median(err) > MAX_DISPERSION_PLANTILLA:
                continue
            elegidas.append({"clase": clase, "subtipo": subtipo, "plantilla": str(ruta),
                             "periodo_plantilla": periodo, "amplitud_plantilla": amplitud})
        # cadencia de OTRA estrella de la misma clase
        for k, fila in enumerate(elegidas):
            j = k + 1
            while cadencias[j % len(cadencias)] == fila["plantilla"]:
                j += 1
            fila["cadencia"] = cadencias[j % len(cadencias)]
        filas += elegidas
    df = pd.DataFrame(filas)
    rng = np.random.default_rng(RANDOM_SEED)
    df["periodo_verdadero"] = df["periodo_plantilla"] * rng.uniform(0.85, 1.15, len(df))
    df["fase0"] = rng.uniform(0, 1, len(df))
    df["id_plantilla"] = np.arange(len(df))
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(CACHE_PLANTILLAS, index=False)
    return df


# ---------------------------------------------------------------- simulacion
def curva_sintetica(spec):
    t_p, mag_p, _ = leer(spec["plantilla"])
    c, v = plantilla(t_p, mag_p, spec["periodo_plantilla"])
    t = np.loadtxt(spec["cadencia"], usecols=0)
    rng = np.random.default_rng(int(spec["semilla"]))
    if spec["n_objetivo"] and t.size > spec["n_objetivo"]:
        t = np.sort(rng.choice(t, int(spec["n_objetivo"]), replace=False))
    fase = ((t - t.min()) / spec["periodo_verdadero"] + spec["fase0"]) % 1.0
    sigma = (v.max() - v.min()) / spec["snr"]
    mag = 15.0 + evaluar_plantilla(c, v, fase) + rng.normal(0, sigma, t.size)
    return t, mag, np.full(t.size, sigma)


def skew_corregida(mag, err):
    var_senal = np.var(mag) - np.mean(err ** 2)
    if var_senal <= 0:
        return np.nan
    return float(skew(mag) * (np.var(mag) / var_senal) ** 1.5)


def analizar(spec):
    warnings.filterwarnings("ignore")
    t, mag, err = curva_sintetica(spec)
    fila = dict(spec) | {"n_puntos": t.size, "baseline_d": t.max() - t.min(),
                         "periodo_catalogo": spec["periodo_verdadero"]}
    for metodo, p in periodos_base(t, mag, err).items():
        fila[f"periodo_{metodo}"] = p
    p0 = fila["periodo_rp_c"]
    for n in range(1, N_MAX + 1):
        fila[f"bic_rp_c_h{H0}_n{n}"] = (bic_fourier(t, mag, err, 1.0 / (n * p0), n * H0)
                                        if n * p0 <= fila["baseline_d"] else np.nan)
    fila |= indicadores_forma(t, mag, err, p0) or {"skew_mag": float(skew(mag))}
    fila["skew_corregida"] = skew_corregida(mag, err)
    fila |= {f"sel_{k}": v for k, v in indicadores_curva(t, mag, err, p0).items()}
    return fila


def especificaciones(plantillas):
    specs = []
    for _, p in plantillas.iterrows():
        for snr in SNRS:
            for n_obj in N_OBJETIVO:
                specs.append(p.to_dict() | {
                    "id": f"S{p['id_plantilla']:04d}_snr{snr}_n{n_obj}", "snr": snr, "n_objetivo": n_obj,
                    "semilla": RANDOM_SEED + 1000 * int(p["id_plantilla"]) + 10 * snr + n_obj})
    return specs


# ---------------------------------------------------------------- evaluacion
def entrenar_selector():
    hib = pd.read_csv(CACHE_HIBRIDO)
    ind = pd.read_csv(CACHE_INDICADORES)
    _, X, n_real = construir_tabla(hib, ind)
    gb = modelos()["Gradient boosting"](0).fit(X[n_real > 0], n_real[n_real > 0])
    return gb, list(ind.columns), list(X.columns)


def matriz_logistica(d):
    return d[INDICADORES + ["n_puntos"]].assign(log_p0=np.log10(d["p0"])).fillna(0.0).to_numpy()


def entrenar_logistica():
    d = pd.read_csv(CACHE_DECISION)
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    return modelo.fit(matriz_logistica(d), (d["clase"] == "Binaria").to_numpy())


def skew_ogle():
    """Skewness cruda y corregida de las 2000 curvas OGLE (referencia)."""
    if CACHE_SKEW_OGLE.exists():
        return pd.read_csv(CACHE_SKEW_OGLE)
    d = pd.read_csv(CACHE_DECISION)
    carpetas = {"Binaria": PHOT_DIR} | {c: carpeta for c, (carpeta, _) in TIPOS.items()}
    filas = []
    for _, r in tqdm(d.iterrows(), total=len(d), desc="Skew OGLE", unit="estrella"):
        t, mag, err = leer(carpetas[r["clase"]] / f"{r['id']}.dat")
        filas.append({"id": r["id"], "clase": r["clase"], "skew_mag": float(skew(mag)),
                      "skew_corregida": skew_corregida(mag, err)})
    out = pd.DataFrame(filas)
    out.to_csv(CACHE_SKEW_OGLE, index=False)
    return out


def evaluar(cache):
    """Tabla con las decisiones de rama y los aciertos de cada selector.
    Devuelve (df, decisiones, ok_selector, ok_p0)."""
    cache = cache.astype({"snr": int, "n_objetivo": int, "id_plantilla": int})
    gb, cols_ind, cols_X = entrenar_selector()
    logistica = entrenar_logistica()

    ind = cache[["id"] + [f"sel_{c}" for c in cols_ind if c != "id"]]
    ind.columns = cols_ind
    columnas_base = ["id", "periodo_rp_c", "periodo_ent_c", "periodo_ls", "baseline_d", "n_puntos",
                     "periodo_catalogo"] + [f"bic_rp_c_h{H0}_n{n}" for n in range(1, N_MAX + 1)]
    _, X, _ = construir_tabla(cache[columnas_base], ind)  # merge por id conserva el orden
    df = cache.copy()
    df["n_gb"] = gb.predict(X[cols_X])
    df["n_hib"] = n_hibrido(df, H0, "bic_rp_c")
    df["p0"] = df["periodo_rp_c"]
    df["prob_bin"] = logistica.predict_proba(matriz_logistica(df))[:, 1]
    es_bin = (df["clase"] == "Binaria").to_numpy()

    decisiones = {
        "oraculo": es_bin,
        "skew>0.5": df["skew_mag"].to_numpy() > UMBRAL_SKEW,
        "skew corr.>0.5": df["skew_corregida"].fillna(-np.inf).to_numpy() > UMBRAL_SKEW,
        "logistica>0.7": df["prob_bin"].to_numpy() > UMBRAL_LOGISTICA,
    }
    P = df["periodo_catalogo"].to_numpy()
    p0 = df["p0"].to_numpy()
    ok_selector = {"x2+BIC": acierto(df["n_hib"].to_numpy() * p0, P),
                   "GB": acierto(df["n_gb"].to_numpy() * p0, P)}
    ok_p0 = acierto(p0, P)
    return df, decisiones, ok_selector, ok_p0


def resumen(cache):
    df, decisiones, ok_selector, ok_p0 = evaluar(cache)
    es_bin = (df["clase"] == "Binaria").to_numpy()
    P = df["periodo_catalogo"].to_numpy()

    lineas = []
    w = lineas.append
    w("ROBUSTEZ DE LAS ALTERNATIVAS DEL METODO UNIFICADO - CURVAS SINTETICAS")
    w("=" * 78)
    pl = cache.drop_duplicates("id_plantilla")
    w(f"Plantillas: " + ", ".join(f"{c} {n}" for c, n in pl["clase"].value_counts().items())
      + f" (binarias por tipo: {pl.loc[pl['clase'] == 'Binaria', 'subtipo'].value_counts().to_dict()}; "
      f"Cefeidas por modo: {pl.loc[pl['clase'] == 'Cefeida', 'subtipo'].value_counts().to_dict()}).")
    w(f"Configuraciones: SNR (amplitud pico a pico / sigma) {SNRS} x N {{todos, 100}}. "
      f"Total {len(df)} curvas. Acierto = error < 1% respecto del periodo verdadero.")
    w("Rama pulsante: P0 directo. Entrenamiento de los supervisados: solo OGLE.")
    w("")

    w("1. Etapa 1 (P0 = rank_product condicional) y techo de cada selector con decision perfecta")
    w(f"  {'SNR':>4} {'N':>6}  {'Bin x2+BIC':>11}{'Bin GB':>8}{'Cef P0':>8}{'RRL P0':>8}{'Cef LS':>8}{'RRL LS':>8}")
    ok_ls = acierto(df["periodo_ls"].to_numpy(), P)
    for (snr, n_obj), g in df.groupby(["snr", "n_objetivo"], sort=False):
        i = g.index
        celdas = [ok_selector["x2+BIC"][i][es_bin[i]].mean(), ok_selector["GB"][i][es_bin[i]].mean()]
        for c in ["Cefeida", "RR Lyrae"]:
            celdas.append(ok_p0[i][(df.loc[i, "clase"] == c).to_numpy()].mean())
        for c in ["Cefeida", "RR Lyrae"]:
            celdas.append(ok_ls[i][(df.loc[i, "clase"] == c).to_numpy()].mean())
        w(f"  {snr:4d} {('todos' if n_obj == 0 else n_obj):>6}  {celdas[0]:11.1%}"
          + "".join(f"{x:8.1%}" for x in celdas[1:]))
    w("")

    w("2. Error de la decision de rama (pulsantes -> rama binaria / binarias -> rama pulsante)")
    w(f"  {'SNR':>4} {'N':>6}  " + "".join(f"{k:>20}" for k in decisiones if k != "oraculo"))
    for (snr, n_obj), g in df.groupby(["snr", "n_objetivo"], sort=False):
        i = g.index
        celdas = []
        for k, ruta in decisiones.items():
            if k == "oraculo":
                continue
            r = ruta[i]
            celdas.append(f"{np.mean(r[~es_bin[i]]):5.1%} / {np.mean(~r[es_bin[i]]):5.1%}")
        w(f"  {snr:4d} {('todos' if n_obj == 0 else n_obj):>6}  " + "".join(f"{c:>20}" for c in celdas))
    w("")

    w("3. Acierto final por clase (Binaria / Cefeida / RR Lyrae)")
    combos = [(dk, sk) for dk in decisiones for sk in ok_selector]
    for (snr, n_obj), g in df.groupby(["snr", "n_objetivo"], sort=False):
        i = g.index
        w(f"  SNR {snr}, N {'todos' if n_obj == 0 else n_obj}:")
        for dk, sk in combos:
            ok = np.where(decisiones[dk][i], ok_selector[sk][i], ok_p0[i])
            w(f"    {dk:16s} + {sk:7s}" + "".join(
                f"{ok[(df.loc[i, 'clase'] == c).to_numpy()].mean():9.1%}" for c in CLASES))
    w("")

    w("4. Promedio sobre las 8 configuraciones y caida de SNR 50 a SNR 4 (N todos)")
    w(f"  {'Combinacion':26s}" + "".join(f"{c:>10}" for c in CLASES) + "   caida Bin  caida Cef  caida RRL")
    alto = (df["snr"] == max(SNRS)) & (df["n_objetivo"] == 0)
    bajo = (df["snr"] == min(SNRS)) & (df["n_objetivo"] == 0)
    for dk, sk in combos:
        ok = np.where(decisiones[dk], ok_selector[sk], ok_p0)
        prom = [ok[(df["clase"] == c).to_numpy()].mean() for c in CLASES]
        caida = [ok[(alto & (df["clase"] == c)).to_numpy()].mean() - ok[(bajo & (df["clase"] == c)).to_numpy()].mean()
                 for c in CLASES]
        w(f"  {dk + ' + ' + sk:26s}" + "".join(f"{x:10.1%}" for x in prom)
          + "".join(f"{x * 100:10.1f}pp" for x in caida))
    w("")

    og = skew_ogle()
    b = (og["clase"] == "Binaria").to_numpy()
    w("5. Referencia OGLE: la regla de skewness sobre las 2000 curvas reales")
    for col in ["skew_mag", "skew_corregida"]:
        r = og[col].fillna(-np.inf).to_numpy() > UMBRAL_SKEW
        w(f"  {col:15s} > {UMBRAL_SKEW}: pulsantes -> rama binaria {np.mean(r[~b]):5.1%}, "
          f"binarias -> rama pulsante {np.mean(~r[b]):5.1%}")

    texto = "\n".join(lineas)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    plantillas = elegir_plantillas()
    print(plantillas.groupby("clase").size().to_dict(), flush=True)
    specs = especificaciones(plantillas)
    cache = pd.read_csv(CACHE_CSV) if CACHE_CSV.exists() else pd.DataFrame(columns=["id"])
    hechos = set(cache["id"])
    pendientes = [s for s in specs if s["id"] not in hechos]

    filas = []
    with ProcessPoolExecutor(max_workers=N_PROCESOS) as pool:
        futuros = [pool.submit(analizar, s) for s in pendientes]
        for k, futuro in enumerate(tqdm(as_completed(futuros), total=len(futuros),
                                        desc="Sinteticas", unit="curva")):
            filas.append(futuro.result())
            if (k + 1) % 50 == 0:
                cache = pd.concat([cache, pd.DataFrame(filas)], ignore_index=True)
                cache.to_csv(CACHE_CSV, index=False)
                filas = []
    if filas:
        cache = pd.concat([cache, pd.DataFrame(filas)], ignore_index=True)
        cache.to_csv(CACHE_CSV, index=False)

    ids = {s["id"] for s in specs}
    resumen(cache[cache["id"].isin(ids)].reset_index(drop=True))


if __name__ == "__main__":
    main()
