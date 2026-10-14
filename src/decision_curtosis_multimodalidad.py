"""
Alternativas a la skewness corregida para la decision de rama (binaria vs.
pulsante) del metodo unificado: curtosis, combinaciones skewness/curtosis y
multimodalidad del histograma de magnitudes (14/10/2026).

Indicadores (solo dependen de las magnitudes y sus errores; ninguno usa P0):
  - g1c: skewness corregida por ruido (la del metodo actual).
  - g2, g2c: curtosis en exceso, cruda y corregida por ruido. El cuarto
    cumulante del ruido gaussiano es cero, de modo que kappa4 no cambia y
    g2_int = g2_obs (s2_obs / s2_senal)^2, con s2_senal = var(mag) - <err^2>
    (analogo a la correccion de la skewness).
  - BC, BCc: coeficiente de bimodalidad (g1^2 + 1) / (g2 + 3), que combina
    skewness y curtosis (vale 1 para una distribucion de dos puntos, 2/3
    para una sinusoide, 5/9 para una uniforme y 1/3 para una gaussiana).
  - frontera: g2c - g1c^2 + 2 >= 0 (desigualdad de Pearson); vale 0 para
    una distribucion de dos niveles, como una curva constante con eclipses
    breves.
  - n_modos: numero de modos del estimador de densidad por nucleo (KDE) de
    las magnitudes con el ancho de banda de Silverman (prominencia >= 10%
    del maximo).
  - h_crit: ancho de banda critico de Silverman (1981), el mayor ancho de
    banda (en unidades de la desviacion estandar) con el que la KDE aun tiene
    al menos dos modos prominentes. Corregido por ruido: el ruido gaussiano
    equivale a ensanchar el nucleo, h_int = sqrt(h_crit^2 + <err^2>/s2) /
    sqrt(s2_senal / s2); si h_crit < h_Silverman / 2 los modos se atribuyen
    al muestreo y h_int = 0.

Conjuntos (cada uno con su referencia y su rama de binarias ya calculada):
  - OGLE (banda I): 3000 binarias + 1500 Cefeidas + 1500 RR Lyrae de
    decision_binaria_pulsante.py; originales = primeras 1000 + 500 + 500.
  - Sinteticas: 5760 curvas de robustez_sinteticas.py (se regeneran con la
    misma semilla).
  - TESS: 720 estrellas de prueba_tess.py (acierto al 1%).

Protocolo: el umbral de cada indicador (y los dos de las reglas
combinadas) se fija sobre las curvas OGLE ORIGINALES maximizando el acierto
balanceado (promedio de Binaria, Cefeida y RR Lyrae tras la ruta); luego se
evalua sin cambios en las OGLE agregadas, las sinteticas y TESS. Para medir la
dependencia del conjunto (banda, instrumento, cadencia) se reporta tambien
el umbral optimo de cada conjunto y cuanto se pierde por usar el de OGLE.

Caches: Decision_Rama/cache_momentos_{ogle,sinteticas,tess}.csv.
Resumen: Decision_Rama/resumen_curtosis_multimodalidad.txt.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decision_binaria_pulsante import CACHE_CSV as CACHE_DECISION
from etapa1_cefeidas_rrlyrae import N_POR_TIPO_BASE, TIPOS
from refinamiento_rank_product_bic import N_PROCESOS
from refinamiento_subarmonicos_bic import (
    MUESTRA_CSV, N_MUESTRA_BASE, PHOT_DIR, acierto, n_hibrido,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "Decision_Rama"
RESUMEN_TXT = OUT_DIR / "resumen_curtosis_multimodalidad.txt"
H0 = 4
CLASES = ["Binaria", "Cefeida", "RR Lyrae"]
N_REJILLA_KDE = 2048
H_REJILLA = np.geomspace(0.02, 2.0, 60)  # anchos de banda (en desviaciones estandar)
PROMINENCIA_MODO = 0.10
UMBRAL_SKEW_ACTUAL = 0.5

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ indicadores
def modos_kde(z, h):
    """Numero de modos con prominencia >= PROMINENCIA_MODO del maximo de la
    KDE gaussiana de z (estandarizado) con ancho de banda h, por histograma
    fino convolucionado."""
    lo, hi = z.min() - 4 * h, z.max() + 4 * h
    hist, bordes = np.histogram(z, bins=N_REJILLA_KDE, range=(lo, hi))
    dens = gaussian_filter1d(hist.astype(float), h / (bordes[1] - bordes[0]), mode="constant")
    picos, _ = find_peaks(np.r_[0.0, dens, 0.0], prominence=PROMINENCIA_MODO * dens.max())
    return len(picos)


def indicadores(mag, err):
    mag = np.asarray(mag, float)
    var, e2 = np.var(mag), np.mean(np.asarray(err, float) ** 2)
    g1, g2 = float(skew(mag)), float(kurtosis(mag))
    vs = var - e2
    f = var / vs if vs > 0 else np.nan
    g1c, g2c = g1 * f ** 1.5, g2 * f ** 2
    beta2c = g2c + 3
    bcc = (g1c ** 2 + 1) / beta2c if np.isfinite(beta2c) and beta2c > 0 else np.nan
    z = (mag - np.median(mag)) / np.sqrt(var)
    iqr = np.subtract(*np.percentile(z, [75, 25]))
    h_silverman = 0.9 * min(1.0, iqr / 1.34) * z.size ** -0.2
    con_dos = [h for h in H_REJILLA if modos_kde(z, h) >= 2]
    h_crit = max(con_dos) if con_dos else 0.0
    # por debajo de ~h_Silverman/2 los modos son de muestreo, no de la senal
    real = h_crit >= 0.5 * h_silverman
    h_crit_c = (np.sqrt(h_crit ** 2 + e2 / var) / np.sqrt(vs / var) if real and vs > 0
                else (0.0 if not real else np.nan))
    return {"g1": g1, "g1c": g1c, "g2": g2, "g2c": g2c,
            "BC": (g1 ** 2 + 1) / (g2 + 3), "BCc": bcc, "frontera": g2c - g1c ** 2 + 2,
            "n_modos": modos_kde(z, h_silverman), "h_crit": h_crit, "h_crit_c": h_crit_c}


# ------------------------------------------------------------------ conjuntos
def leer(ruta):
    t, mag, err = np.loadtxt(ruta, unpack=True)
    return t, mag, np.where(err > 0, err, np.median(err[err > 0]))


def _ogle(fila):
    carpeta = PHOT_DIR if fila["clase"] == "Binaria" else TIPOS[fila["clase"]][0]
    _, mag, err = leer(carpeta / f"{fila['id']}.dat")
    return {"id": fila["id"]} | indicadores(mag, err)


def _sintetica(spec):
    from robustez_sinteticas import curva_sintetica
    _, mag, err = curva_sintetica(spec)
    return {"id": spec["id"]} | indicadores(mag, err)


def _tess(fila):
    from prueba_tess import archivo
    _, mag, err = np.loadtxt(archivo(fila["nombre"]), unpack=True)
    return {"nombre": fila["nombre"]} | indicadores(mag, np.where(err > 0, err, np.median(err[err > 0])))


def calcular(nombre, filas, funcion, clave):
    """Cache append-safe de los indicadores de un conjunto."""
    ruta = OUT_DIR / f"cache_momentos_{nombre}.csv"
    cache = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=[clave])
    pendientes = [f for f in filas if f[clave] not in set(cache[clave])]
    if pendientes:
        print(f"{nombre}: {len(pendientes)} curvas", flush=True)
        with ProcessPoolExecutor(max_workers=N_PROCESOS) as pool:
            nuevos = list(pool.map(funcion, pendientes, chunksize=20))
        cache = pd.concat([cache, pd.DataFrame(nuevos)], ignore_index=True)
        cache.to_csv(ruta, index=False)
    return cache


def conjunto_ogle():
    d = pd.read_csv(CACHE_DECISION)
    ind = calcular("ogle", d[["id", "clase"]].to_dict("records"), _ogle, "id")
    d = d.merge(ind, on="id")
    orig = set(pd.read_csv(MUESTRA_CSV)["id"][:N_MUESTRA_BASE])
    for tipo in ["Cefeida", "RR Lyrae"]:
        orig |= set(pd.read_csv(BASE_DIR / "Etapa1_Pulsantes" / f"muestra_{tipo.replace(' ', '_')}.csv")["id"]
                    [:N_POR_TIPO_BASE])
    P, p0 = d["periodo_catalogo"].to_numpy(), d["p0"].to_numpy()
    d["ok_bin"] = acierto(d["n_hib"].to_numpy() * p0, P)
    d["ok_pul"] = acierto(p0, P)
    d["conjunto"] = np.where(d["id"].isin(orig), "OGLE originales", "OGLE agregadas")
    return d


def conjunto_sinteticas():
    from robustez_sinteticas import CACHE_CSV
    d = pd.read_csv(CACHE_CSV).astype({"snr": int, "n_objetivo": int, "id_plantilla": int, "semilla": int})
    specs = d[["id", "plantilla", "periodo_plantilla", "cadencia", "periodo_verdadero", "fase0", "snr",
               "n_objetivo", "semilla"]].to_dict("records")
    ind = calcular("sinteticas", specs, _sintetica, "id")
    d = d.drop(columns=[c for c in ind.columns if c in d.columns and c != "id"]).merge(ind, on="id")
    P, p0 = d["periodo_catalogo"].to_numpy(), d["periodo_rp_c"].to_numpy()
    d["ok_bin"] = acierto(n_hibrido(d, H0, "bic_rp_c") * p0, P)
    d["ok_pul"] = acierto(p0, P)
    d["conjunto"] = "Sinteticas"
    return d


def conjunto_tess():
    from prueba_tess import CACHE_CSV, MUESTRA_CSV as MUESTRA_TESS
    d = pd.read_csv(CACHE_CSV)
    d = d[d["nombre"].isin(set(pd.read_csv(MUESTRA_TESS)["nombre"]))].reset_index(drop=True)
    ind = calcular("tess", d[["nombre"]].to_dict("records"), _tess, "nombre")
    d = d.merge(ind, on="nombre")
    P, p0 = d["periodo_referencia"].to_numpy(), d["periodo_rp_c"].to_numpy()
    d["ok_bin"] = acierto(n_hibrido(d, H0, "bic_rp_c") * p0, P)
    d["ok_pul"] = acierto(p0, P)
    d["conjunto"] = "TESS"
    return d


# ------------------------------------------------------------------ reglas
# Cada indicador: (columna, sentido); sentido +1 = binaria si valor > umbral.
# Para la multimodalidad el sentido no se supone: se fija con el AUC en la
# calibracion (OGLE originales), ver sentido_calibrado.
INDICADORES_1D = {
    "skew corregida g1c": ("g1c", +1),
    "skew cruda g1": ("g1", +1),
    "curtosis corregida g2c": ("g2c", +1),
    "curtosis cruda g2": ("g2", +1),
    "BC corregido": ("BCc", +1),
    "BC crudo": ("BC", +1),
    "frontera g2c-g1c^2+2": ("frontera", -1),
    "h_crit corregido": ("h_crit_c", None),
    "h_crit crudo": ("h_crit", None),
    "n_modos (Silverman)": ("n_modos", None),
}


def valores(d, col, sentido):
    """Valor orientado (mayor = mas binaria); NaN -> rama pulsante."""
    return np.nan_to_num(sentido * d[col].to_numpy(float), nan=-np.inf)


def sentido_calibrado(cal, col):
    v = np.nan_to_num(cal[col].to_numpy(float), nan=-1e12)
    return +1 if roc_auc_score(cal["clase"] == "Binaria", v) >= 0.5 else -1


def tasas(d, ruta):
    """Acierto por clase tras la ruta y errores de la decision."""
    ok = np.where(ruta, d["ok_bin"], d["ok_pul"])
    es_bin = (d["clase"] == "Binaria").to_numpy()
    r = {c: ok[(d["clase"] == c).to_numpy()].mean() for c in CLASES}
    r["balanceado"] = np.nanmean([r[c] for c in CLASES])
    r["p->b"] = ruta[~es_bin].mean()
    r["b->p"] = (~ruta[es_bin]).mean()
    return r


def umbral_optimo(d, v):
    cand = np.unique(np.nanpercentile(v[np.isfinite(v)], np.linspace(0.5, 99.5, 400)))
    bal = [tasas(d, v > u)["balanceado"] for u in cand]
    return cand[int(np.argmax(bal))], max(bal)


def reglas_combinadas(cal):
    """Reglas de dos indicadores ajustadas sobre `cal` (OGLE originales).
    Devuelve {nombre: funcion(d) -> ruta booleana} y una descripcion."""
    reglas = {}
    global INDICADORES_1D_POR_COL
    INDICADORES_1D_POR_COL = {col: sentido for col, sentido in INDICADORES_1D.values()}

    def par(col_a, col_b, operador, nombre):
        sa, sb = (INDICADORES_1D_POR_COL[c] for c in (col_a, col_b))
        va, vb = valores(cal, col_a, sa), valores(cal, col_b, sb)
        ua = np.unique(np.nanpercentile(va[np.isfinite(va)], np.linspace(1, 99, 60)))
        ub = np.unique(np.nanpercentile(vb[np.isfinite(vb)], np.linspace(1, 99.5, 80)))
        _, a, b = max((tasas(cal, operador(va > a, vb > b))["balanceado"], a, b) for a in ua for b in ub)
        reglas[f"{nombre} (a={sa * a:.2f}, b={sb * b:.2f})"] =             lambda d, a=a, b=b: operador(valores(d, col_a, sa) > a, valores(d, col_b, sb) > b)

    par("g1c", "g2c", np.logical_or, "g1c > a  O  g2c > b")
    par("g1c", "g2c", np.logical_and, "g1c > a  Y  g2c > b")
    par("g1c", "h_crit_c", np.logical_or, "g1c > a  O  h_crit_c (sentido) b")
    par("g1c", "h_crit_c", np.logical_and, "g1c > a  Y  h_crit_c (sentido) b")
    par("g2c", "h_crit_c", np.logical_and, "g2c > a  Y  h_crit_c (sentido) b")

    X = lambda d: np.nan_to_num(d[["g1c", "g2c"]].to_numpy(float), nan=0.0).clip(-50, 50)
    log = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    log.fit(X(cal), (cal["clase"] == "Binaria").to_numpy())
    prob_cal = log.predict_proba(X(cal))[:, 1]
    u, _ = umbral_optimo(cal, prob_cal)
    reglas[f"logistica(g1c, g2c) > {u:.2f}"] = lambda d, u=u: log.predict_proba(X(d))[:, 1] > u
    return reglas


# ------------------------------------------------------------------ resumen
def resumen(datos):
    lineas = []
    w = lineas.append
    cal = datos["OGLE originales"]
    for nombre, (col, sentido) in list(INDICADORES_1D.items()):
        if sentido is None:
            INDICADORES_1D[nombre] = (col, sentido_calibrado(cal, col))
    w("DECISION DE RAMA: SKEWNESS vs. CURTOSIS vs. MULTIMODALIDAD (14/10/2026)")
    w("=" * 90)
    w("Umbrales fijados en OGLE originales (max. acierto balanceado = promedio Bin/Cef/RRL tras la")
    w("ruta) y aplicados sin cambios al resto. Bal = acierto balanceado; p->b = pulsantes enviadas a")
    w("la rama de binarias; b->p = binarias enviadas a la rama de pulsantes.")
    w("Conjuntos: " + ", ".join(f"{k} {len(d)}" for k, d in datos.items()))
    w("")

    w("1. Referencias")
    for nombre, f in [("rama perfecta (clase real)", lambda d: (d["clase"] == "Binaria").to_numpy()),
                      ("todo a pulsante", lambda d: np.zeros(len(d), bool)),
                      (f"ACTUAL: g1c > {UMBRAL_SKEW_ACTUAL}", lambda d: valores(d, "g1c", +1) > UMBRAL_SKEW_ACTUAL)]:
        w(f"  {nombre}")
        for k, d in datos.items():
            t = tasas(d, f(d))
            w(f"    {k:16s} Bin {t['Binaria']:6.1%}  Cef {t['Cefeida']:6.1%}  RRL {t['RR Lyrae']:6.1%}  "
              f"Bal {t['balanceado']:6.1%}  p->b {t['p->b']:5.1%}  b->p {t['b->p']:5.1%}")
    w("")

    w("2. AUC de cada indicador (binaria vs. pulsante) con el sentido fijado en OGLE originales")
    w("   (AUC < 0.5 en otro conjunto = el indicador se invierte alli)")
    w(f"  {'Indicador':24s}" + "".join(f"{k:>17}" for k in datos))
    for nombre, (col, s) in INDICADORES_1D.items():
        celdas = []
        for d in datos.values():
            v = valores(d, col, s)
            v = np.where(np.isfinite(v), v, -1e12)
            celdas.append(roc_auc_score(d["clase"] == "Binaria", v))
        w(f"  {nombre + (' (-)' if s < 0 else ''):24s}" + "".join(f"{x:17.3f}" for x in celdas))
    w("")

    w("3. Medianas por clase (OGLE todas / TESS)")
    og = pd.concat([datos["OGLE originales"], datos["OGLE agregadas"]])
    for col in ["g1c", "g2c", "BCc", "h_crit_c", "n_modos"]:
        w(f"  {col:9s} OGLE " + ", ".join(f"{c} {og.loc[og['clase'] == c, col].median():6.2f}" for c in CLASES)
          + "  | TESS " + ", ".join(f"{g} {d.loc[d['grupo'] == g, col].median():5.2f}"
                                     for d in [datos['TESS']] for g in d['grupo'].unique()))
    w("  Fraccion con n_modos >= 2: " + "; ".join(
        f"{k}: " + ", ".join(f"{c} {(d.loc[d['clase'] == c, 'n_modos'] >= 2).mean():.0%}" for c in CLASES)
        for k, d in datos.items()))
    w("")

    w("4. Reglas de un indicador con umbral fijado en OGLE originales")
    w(f"  {'Regla':38s}{'conjunto':>17}{'Bin':>8}{'Cef':>8}{'RRL':>8}{'Bal':>8}{'p->b':>7}{'b->p':>7}"
      f"{'u_opt propio':>14}{'Bal opt':>9}")
    filas_resumen = []
    reglas = {}
    for nombre, (col, s) in INDICADORES_1D.items():
        u, _ = umbral_optimo(cal, valores(cal, col, s))
        reglas[f"{nombre} {'>' if s > 0 else '<'} {s * u:.3g}"] = (lambda d, col=col, s=s, u=u: valores(d, col, s) > u,
                                                                   col, s)
    reglas[f"skew corregida g1c > {UMBRAL_SKEW_ACTUAL} (ACTUAL)"] = (
        lambda d: valores(d, "g1c", +1) > UMBRAL_SKEW_ACTUAL, "g1c", +1)
    for nombre, (f, col, s) in reglas.items():
        for j, (k, d) in enumerate(datos.items()):
            t = tasas(d, f(d))
            u_opt, bal_opt = umbral_optimo(d, valores(d, col, s))
            w(f"  {nombre if j == 0 else '':38s}{k:>17}{t['Binaria']:8.1%}{t['Cefeida']:8.1%}{t['RR Lyrae']:8.1%}"
              f"{t['balanceado']:8.1%}{t['p->b']:7.1%}{t['b->p']:7.1%}{s * u_opt:14.3g}{bal_opt:9.1%}")
            filas_resumen.append({"regla": nombre, "conjunto": k, **t, "bal_opt": bal_opt})
    w("  u_opt propio / Bal opt = umbral que seria optimo en ESE conjunto y su acierto balanceado")
    w("  (la diferencia con Bal mide cuanto depende la regla del conjunto).")
    w("")

    w("5. Reglas combinadas (ajustadas en OGLE originales)")
    for nombre, f in reglas_combinadas(cal).items():
        w(f"  {nombre}")
        for k, d in datos.items():
            t = tasas(d, f(d))
            w(f"    {k:16s} Bin {t['Binaria']:6.1%}  Cef {t['Cefeida']:6.1%}  RRL {t['RR Lyrae']:6.1%}  "
              f"Bal {t['balanceado']:6.1%}  p->b {t['p->b']:5.1%}  b->p {t['b->p']:5.1%}")
            filas_resumen.append({"regla": nombre, "conjunto": k, **t})
    w("")

    w("6. Resumen: acierto balanceado por conjunto y dispersion entre conjuntos (fuera de OGLE orig.)")
    r = pd.DataFrame(filas_resumen)
    tabla = r.pivot_table(index="regla", columns="conjunto", values="balanceado", sort=False)[list(datos)]
    fuera = tabla.drop(columns="OGLE originales")
    tabla["min fuera"] = fuera.min(axis=1)
    tabla["rango"] = fuera.max(axis=1) - fuera.min(axis=1)
    w(tabla.sort_values("min fuera", ascending=False).to_string(float_format=lambda x: f"{100 * x:6.1f}"))
    w("")
    w("7. Sinteticas por SNR y N: acierto balanceado de las mejores reglas")
    sin = datos["Sinteticas"]
    claves = [k for k in reglas if k.startswith(("skew corregida g1c > 0.5", "curtosis corregida", "BC corregido",
                                                 "h_crit corregido", "n_modos"))]
    w(f"  {'SNR':>4}{'N':>7}  " + "".join(f"{k[:22]:>24}" for k in claves))
    for (snr, n), g in sin.groupby(["snr", "n_objetivo"], sort=False):
        w(f"  {snr:4d}{('todos' if n == 0 else n):>7}  "
          + "".join(f"{tasas(g, reglas[k][0](g))['balanceado']:24.1%}" for k in claves))
    w("")
    w("8. TESS por grupo: fraccion enviada a la rama de binarias")
    tess = datos["TESS"]
    w(f"  {'Regla':38s}" + "".join(f"{g[:14]:>16}" for g in tess["grupo"].unique()))
    for k in claves:
        ruta = reglas[k][0](tess)
        w(f"  {k[:38]:38s}" + "".join(f"{ruta[(tess['grupo'] == g).to_numpy()].mean():16.1%}"
                                      for g in tess["grupo"].unique()))

    texto = "\n".join(lineas)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)
    return reglas


def cargar_conjuntos():
    OUT_DIR.mkdir(exist_ok=True)
    og = conjunto_ogle()
    return {"OGLE originales": og[og["conjunto"] == "OGLE originales"].reset_index(drop=True),
            "OGLE agregadas": og[og["conjunto"] == "OGLE agregadas"].reset_index(drop=True),
            "Sinteticas": conjunto_sinteticas(), "TESS": conjunto_tess()}


if __name__ == "__main__":
    resumen(cargar_conjuntos())
