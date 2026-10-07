"""
Decision binaria vs. pulsante a partir de la forma de la curva plegada en
P0 (rank_product condicional), para unificar el metodo: P0 directo si la
estrella parece pulsante, refinamiento hibrido (x2 + BIC) si parece binaria.

Idea fisica: una binaria eclipsante plegada en P_orb/2 es casi simetrica en
el tiempo (el eclipse se ve igual al entrar y al salir); una pulsante
fundamental (RRab, Cefeida F) sube rapido y baja lento.

No recalcula la busqueda global: usa los P0 ya guardados en
  - Refinamiento_Subarmonicos/cache_rank_product_enmascarado_bic.csv
    (1000 binarias, semilla 2026, con los BIC del hibrido)
  - Etapa1_Pulsantes/cache_etapa1_pulsantes.csv (500 Cefeidas + 500 RR Lyrae)

Por estrella se ajusta (minimos cuadrados ponderados) una serie de Fourier
m(fase) = a0 + sum_k A_k cos(2 pi k fase + phi_k), k = 1..H_FORMA, en P0, y se
calculan (ninguno usa el catalogo):
  - asimetria: sqrt(min_c sum_k A_k^2 sin^2(phi_k + 2 pi k c) / sum_k A_k^2),
    la fraccion rms de la parte impar respecto del eje de simetria optimo c.
    0 = curva simetrica en el tiempo, 1 = totalmente antisimetrica.
  - |sin(phi21)|, |sin(phi31)| (phi_k1 = phi_k - k phi_1; simetrica -> 0).
  - R21 = A2/A1, R31 = A3/A1.
  - fraccion de subida (de minimo a maximo de brillo, en fase) y su
    distancia a 0.5.
  - asimetria estadistica (skewness) de las magnitudes.
  - amplitud pico a pico del modelo.
Para las pulsantes ademas se calculan los BIC de n = 1..6 (H0 = 4), para
saber que haria la rama de binarias si la decision las enviara ahi.

Cache append-safe en Etapa1_Pulsantes/cache_decision_forma.csv.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from etapa1_cefeidas_rrlyrae import CACHE_CSV as CACHE_PULSANTES, OUT_DIR, TIPOS
from refinamiento_rank_product_enmascarado_bic import CACHE_CSV as CACHE_BINARIAS
from refinamiento_subarmonicos_bic import (
    N_MAX, PHOT_DIR, RANDOM_SEED, acierto, bic_fourier, n_hibrido,
)

CACHE_CSV = OUT_DIR / "cache_decision_forma.csv"
RESUMEN_TXT = OUT_DIR / "resumen_decision_forma.txt"

H_FORMA = 4  # armonicos del ajuste de forma en P0
H0 = 4  # el del metodo hibrido
MIN_PUNTOS_POR_PARAMETRO = 3
N_GRILLA = 1000
META_PULSANTES = 0.02  # fraccion maxima de pulsantes enviadas a la rama de binarias
INDICADORES = ["asimetria", "sin_phi21", "sin_phi31", "R21", "R31",
               "dist_subida", "skew_mag", "amplitud"]

warnings.filterwarnings("ignore")


def ajuste_fourier(t, mag, err, periodo, h):
    """Amplitudes A_k y fases phi_k (convencion coseno) y a0."""
    fase = 2 * np.pi * (t - t.min()) / periodo
    k = np.arange(1, h + 1)
    X = np.column_stack([np.ones_like(t), np.cos(np.outer(fase, k)), np.sin(np.outer(fase, k))])
    w = 1.0 / err
    coef = np.linalg.lstsq(X * w[:, None], mag * w, rcond=None)[0]
    a, b = coef[1:h + 1], coef[h + 1:]
    # a cos + b sin = A cos(x + phi), con A cos(phi) = a, A sin(phi) = -b
    return coef[0], np.hypot(a, b), np.arctan2(-b, a)


def indicadores_forma(t, mag, err, periodo):
    h = min(H_FORMA, (t.size // MIN_PUNTOS_POR_PARAMETRO - 1) // 2)
    if h < 1:
        return None
    a0, A, phi = ajuste_fourier(t, mag, err, periodo, h)
    k = np.arange(1, h + 1)
    c = np.linspace(0, 1, N_GRILLA, endpoint=False)
    impar = (A[None, :] ** 2 * np.sin(phi[None, :] + 2 * np.pi * np.outer(c, k)) ** 2).sum(axis=1)
    x = np.linspace(0, 1, N_GRILLA, endpoint=False)
    modelo = a0 + (A[None, :] * np.cos(2 * np.pi * np.outer(x, k) + phi[None, :])).sum(axis=1)
    fase_debil, fase_brillante = x[np.argmax(modelo)], x[np.argmin(modelo)]  # magnitudes
    subida = (fase_brillante - fase_debil) % 1.0
    phi_k1 = lambda j: phi[j - 1] - j * phi[0] if h >= j else np.nan
    return {
        "asimetria": float(np.sqrt(impar.min() / np.sum(A ** 2))),
        "sin_phi21": float(np.abs(np.sin(phi_k1(2)))),
        "sin_phi31": float(np.abs(np.sin(phi_k1(3)))),
        "R21": float(A[1] / A[0]) if h >= 2 else np.nan,
        "R31": float(A[2] / A[0]) if h >= 3 else np.nan,
        "subida": float(subida),
        "dist_subida": float(abs(subida - 0.5)),
        "skew_mag": float(skew(mag)),
        "amplitud": float(modelo.max() - modelo.min()),
    }


def leer(ruta):
    t, mag, err = np.loadtxt(ruta, unpack=True)
    return t, mag, np.where(err > 0, err, np.median(err[err > 0]))


def tabla_base():
    """Una fila por estrella con P0, periodo de catalogo y n del hibrido
    (binarias: de la cache; pulsantes: se calcula aqui)."""
    bin_ = pd.read_csv(CACHE_BINARIAS)
    bin_ = pd.DataFrame({
        "id": bin_["id"], "clase": "Binaria", "subtipo": bin_["tipo_morfologico"],
        "periodo_catalogo": bin_["periodo_catalogo"], "p0": bin_["periodo_rp_c"],
        "n_hib": n_hibrido(bin_, H0, "bic_rp_c"),
        "ruta": [str(PHOT_DIR / f"{i}.dat") for i in bin_["id"]],
    })
    pul = pd.read_csv(CACHE_PULSANTES)
    carpetas = {tipo: carpeta for tipo, (carpeta, _) in TIPOS.items()}
    pul = pd.DataFrame({
        "id": pul["id"], "clase": pul["tipo"], "subtipo": pul["modo"],
        "periodo_catalogo": pul["periodo_catalogo"], "p0": pul["periodo_rp_c"],
        "n_hib": np.nan,
        "ruta": [str(carpetas[tp] / f"{i}.dat") for tp, i in zip(pul["tipo"], pul["id"])],
    })
    return pd.concat([bin_, pul], ignore_index=True)


def n_hibrido_pulsante(t, mag, err, p0):
    baseline = t.max() - t.min()
    fila = {f"bic_h{H0}_n{n}": (bic_fourier(t, mag, err, 1.0 / (n * p0), n * H0)
                                if n * p0 <= baseline else np.nan)
            for n in range(1, N_MAX + 1)}
    return int(n_hibrido(pd.DataFrame([fila]), H0, "bic")[0])


def calcular():
    base = tabla_base()
    cache = pd.read_csv(CACHE_CSV) if CACHE_CSV.exists() else pd.DataFrame(columns=["id"])
    hechos = set(cache["id"])
    filas = []
    for _, r in base[~base["id"].isin(hechos)].iterrows():
        t, mag, err = leer(r["ruta"])
        forma = indicadores_forma(t, mag, err, r["p0"]) or {}
        n_hib = r["n_hib"] if r["clase"] == "Binaria" else n_hibrido_pulsante(t, mag, err, r["p0"])
        filas.append({**r.drop("ruta").to_dict(), "n_hib": n_hib, "n_puntos": t.size, **forma})
        if len(filas) % 200 == 0:
            print(f"  {len(filas)} estrellas", flush=True)
    if filas:
        cache = pd.concat([cache, pd.DataFrame(filas)], ignore_index=True)
        cache.to_csv(CACHE_CSV, index=False)
    return cache[cache["id"].isin(set(base["id"]))].reset_index(drop=True)


def aciertos_ruteados(d, a_binaria):
    """Acierto si la decision manda a la rama de binarias (n_hib * P0) o a la
    de pulsantes (P0)."""
    P = d["periodo_catalogo"].to_numpy()
    p0 = d["p0"].to_numpy()
    return np.where(a_binaria, acierto(d["n_hib"].to_numpy() * p0, P), acierto(p0, P))


def resumen(d):
    lineas = []
    w = lineas.append
    es_bin = (d["clase"] == "Binaria").to_numpy()
    w("DECISION BINARIA vs. PULSANTE POR LA FORMA DE LA CURVA PLEGADA EN P0")
    w("=" * 78)
    w(f"P0 = rank_product condicional (caches existentes). Ajuste de Fourier con {H_FORMA} armonicos "
      f"en P0. Binarias: {es_bin.sum()}; pulsantes: {(~es_bin).sum()} "
      f"({', '.join(f'{c} {n}' for c, n in d.loc[~es_bin, 'clase'].value_counts().items())}).")
    w("")
    w("Referencias (decision perfecta / siempre una rama):")
    for nombre, ruta in [("oraculo (clase real)", es_bin), ("todo a pulsante (P0)", np.zeros_like(es_bin)),
                         ("todo a binaria (hibrido)", np.ones_like(es_bin))]:
        ok = aciertos_ruteados(d, ruta)
        w(f"  {nombre:26s}" + "   ".join(f"{c} {ok[d['clase'] == c].mean():6.1%}"
                                         for c in ["Binaria", "Cefeida", "RR Lyrae"]))
    w("")

    w("1. Poder de separacion de cada indicador (AUC binaria vs. pulsante; 0.5 = nada, 1 = perfecto)")
    w(f"  {'Indicador':14s}{'AUC':>7}{'mediana bin.':>14}{'mediana Cef.':>14}{'mediana RRL':>13}")
    for ind in INDICADORES:
        v = d[ind].to_numpy()
        ok = ~np.isnan(v)
        auc = roc_auc_score(es_bin[ok], -v[ok])
        w(f"  {ind:14s}{max(auc, 1 - auc):7.3f}{d.loc[es_bin, ind].median():14.3f}"
          f"{d.loc[d['clase'] == 'Cefeida', ind].median():14.3f}{d.loc[d['clase'] == 'RR Lyrae', ind].median():13.3f}"
          f"{'' if auc >= 0.5 else '   (mayor en binarias)'}")
    w("")

    w(f"2. Regla sin entrenamiento: binaria si asimetria < umbral")
    w(f"  {'umbral':>8}{'pulsantes a rama bin.':>23}{'binarias a rama puls.':>23}"
      f"{'Binaria':>10}{'Cefeida':>10}{'RR Lyrae':>10}")
    asim = d["asimetria"].fillna(1.0).to_numpy()
    for u in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        ruta = asim < u
        ok = aciertos_ruteados(d, ruta)
        w(f"  {u:8.2f}{np.mean(ruta[~es_bin]):23.1%}{np.mean(~ruta[es_bin]):23.1%}"
          + "".join(f"{ok[d['clase'] == c].mean():10.1%}" for c in ["Binaria", "Cefeida", "RR Lyrae"]))
    # Umbral calibrado para mandar <= META_PULSANTES de pulsantes a la rama
    # de binarias, validado en dos mitades (se calibra en una y se evalua en la otra).
    rng = np.random.default_rng(RANDOM_SEED)
    mitad = rng.random(len(d)) < 0.5
    w(f"  Umbral calibrado (percentil {META_PULSANTES:.0%} de la asimetria en pulsantes), validado en mitades:")
    for cal, ev in [(mitad, ~mitad), (~mitad, mitad)]:
        u = np.percentile(asim[cal & ~es_bin], 100 * META_PULSANTES)
        ruta = asim < u
        ok = aciertos_ruteados(d, ruta)
        w(f"    umbral {u:.3f}: pulsantes a rama bin. {np.mean(ruta[ev & ~es_bin]):.1%}, binarias a rama "
          f"puls. {np.mean(~ruta[ev & es_bin]):.1%}; "
          + "   ".join(f"{c} {ok[ev & (d['clase'] == c).to_numpy()].mean():6.1%}"
                       for c in ["Binaria", "Cefeida", "RR Lyrae"]))
    w("")

    w("3. Decision supervisada (regresion logistica sobre todos los indicadores, CV 5x estratificada)")
    X = d[INDICADORES + ["n_puntos"]].assign(log_p0=np.log10(d["p0"])).fillna(0.0).to_numpy()
    prob = np.zeros(len(d))
    for entreno, prueba in StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED).split(X, es_bin):
        modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        modelo.fit(X[entreno], es_bin[entreno])
        prob[prueba] = modelo.predict_proba(X[prueba])[:, 1]
    w(f"  AUC fuera de muestra: {roc_auc_score(es_bin, prob):.3f}")
    for umbral in [0.5, 0.7, 0.9]:
        ruta = prob > umbral
        ok = aciertos_ruteados(d, ruta)
        w(f"  P(binaria) > {umbral:.1f}: pulsantes a rama bin. {np.mean(ruta[~es_bin]):.1%}, binarias a rama puls. "
          f"{np.mean(~ruta[es_bin]):.1%}; "
          + "   ".join(f"{c} {ok[d['clase'] == c].mean():6.1%}" for c in ["Binaria", "Cefeida", "RR Lyrae"]))
    w("")

    w("4. Quienes quedan del lado equivocado con la regla de asimetria (umbral 0.15)")
    ruta = asim < 0.15
    for clase in ["Cefeida", "RR Lyrae"]:
        g = d[(d["clase"] == clase) & ruta]
        w(f"  {clase} enviadas a rama de binarias: {len(g)}"
          + (f"; por modo: {g['subtipo'].value_counts().to_dict()}" if clase == "Cefeida" else
             f"; P_cat mediana {g['periodo_catalogo'].median():.3f} d (RRc tipicas P < 0.45 d): "
             f"{np.mean(g['periodo_catalogo'] < 0.45):.0%} con P < 0.45 d"))
    g = d[es_bin & ~ruta]
    w(f"  Binarias enviadas a rama de pulsantes: {len(g)}; por tipo: {g['subtipo'].value_counts().to_dict()}")
    w(f"  Pulsantes que la rama de binarias igual acierta (n_hib = 1): "
      f"{np.mean(d.loc[~es_bin, 'n_hib'] == 1):.1%}")
    w("")

    w("5. Regla sin entrenamiento por asimetria estadistica de las magnitudes (no usa P0)")
    w("  Binaria si skew(mag) > umbral: los eclipses son excursiones breves hacia magnitudes")
    w("  mayores (cola debil); las pulsantes pasan mas tiempo cerca del minimo de brillo.")
    skew_mag = d["skew_mag"].to_numpy()
    for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ruta = skew_mag > u
        ok = aciertos_ruteados(d, ruta)
        w(f"  skew > {u:4.2f}: pulsantes a rama bin. {np.mean(ruta[~es_bin]):5.1%}, binarias a rama puls. "
          f"{np.mean(~ruta[es_bin]):5.1%}; "
          + "   ".join(f"{c} {ok[d['clase'] == c].mean():6.1%}" for c in ["Binaria", "Cefeida", "RR Lyrae"]))
    g = d[es_bin & (skew_mag <= 0.5)]
    w(f"  Binarias con skew <= 0.5: {len(g)}; por tipo: {g['subtipo'].value_counts().to_dict()}")

    texto = "\n".join(lineas)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    resumen(calcular())
