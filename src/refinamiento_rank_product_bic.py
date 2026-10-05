"""
Refinamiento por subarmonicos con BIC (refinamiento_subarmonicos_bic.py)
aplicado sobre el periodo de rank_product en lugar del de Lomb-Scargle, para
ver si las dos mejoras se suman.

Misma muestra de 1000 binarias (semilla 2026). Para cada estrella se calcula,
sobre UNA grilla uniforme en frecuencia (0.1-100 d, como en
prueba_nocturna_binarias.py):
  - Lomb-Scargle (maxima potencia)
  - minima entropia (L=K=7, sin enmascarar alias diarios, igual que la
    prueba nocturna)
  - rank_product = percentil(LS) * percentil(-entropia)
La grilla es el doble de fina que la de la prueba nocturna (N_FREQ=100 000,
paso ~1e-4 d^-1 ~ 1/(L*T) para T ~ 1500 d), porque con 50 000 puntos el paso
es comparable al ancho de los minimos de entropia.

Despues, sobre el periodo de cada metodo base se evalua el BIC de los
subarmonicos f0/n (n=1..6), con la misma funcion y la misma regla hibrida
(x2 por defecto; n>=3 solo si mejora el BIC de n=2). Cache append-safe;
el calculo por estrella se reparte en varios procesos.
"""

import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from entropypf import get_entropies
from scipy.stats import rankdata
from tqdm import tqdm

from refinamiento_subarmonicos_bic import (
    H0_VALORES, MIN_PUNTOS, N_MAX, N_MUESTRA, OUT_DIR, PHOT_DIR, RANDOM_SEED,
    acierto, bic_fourier, cargar_catalogo, n_hibrido,
)

CACHE_CSV = OUT_DIR / "cache_rank_product_bic.csv"
RESUMEN_TXT = OUT_DIR / "resumen_rank_product_bic.txt"

P_MIN, P_MAX = 0.1, 100.0
N_FREQ = 100_000
L, K = 7, 7
TAMANO_BLOQUE = 20_000  # periodos por llamada a get_entropies (limita memoria)
N_PROCESOS = 6
METODOS = {"ls": "Lomb-Scargle", "ent": "Minima entropia", "rp": "rank_product"}


def periodos_base(t, mag, err):
    frecuencia = np.linspace(1.0 / P_MAX, 1.0 / P_MIN, N_FREQ)
    potencia = LombScargle(t, mag, err).power(frecuencia)
    entropia = np.concatenate([
        get_entropies(t, mag, 1.0 / frecuencia[i:i + TAMANO_BLOQUE], L=L, K=K)
        for i in range(0, N_FREQ, TAMANO_BLOQUE)
    ])
    rango = (rankdata(potencia) / N_FREQ) * (rankdata(-entropia) / N_FREQ)
    return {
        "ls": 1.0 / frecuencia[np.argmax(potencia)],
        "ent": 1.0 / frecuencia[np.argmin(entropia)],
        "rp": 1.0 / frecuencia[np.argmax(rango)],
    }


def analizar_estrella(id_estrella, tipo, periodo_catalogo):
    warnings.filterwarnings("ignore")
    try:
        t, mag, err = np.loadtxt(PHOT_DIR / f"{id_estrella}.dat", unpack=True)
    except (ValueError, OSError):
        return None
    if t.size < MIN_PUNTOS:
        return None
    err = np.where(err > 0, err, np.median(err[err > 0]))
    baseline = t.max() - t.min()

    fila = {"id": id_estrella, "tipo_morfologico": tipo, "n_puntos": t.size,
            "baseline_d": baseline, "periodo_catalogo": periodo_catalogo}
    for metodo, p0 in periodos_base(t, mag, err).items():
        fila[f"periodo_{metodo}"] = p0
        for h0 in H0_VALORES:
            for n in range(1, N_MAX + 1):
                bic = bic_fourier(t, mag, err, 1.0 / (n * p0), n * h0) if n * p0 <= baseline else np.nan
                fila[f"bic_{metodo}_h{h0}_n{n}"] = bic
    return fila


def resumen(cache):
    P = cache["periodo_catalogo"].to_numpy()
    lineas = []
    w = lineas.append
    w("RESUMEN - REFINAMIENTO POR SUBARMONICOS (BIC) SOBRE CADA METODO BASE")
    w("=" * 78)
    w(f"Binarias: {len(cache)} (semilla {RANDOM_SEED}); grilla uniforme de {N_FREQ} "
      f"frecuencias, {P_MIN}-{P_MAX} d; entropia L=K={L}, sin enmascarar alias.")
    w("Hibrido: x2 por defecto; n>=3 solo si su BIC es menor que el de n=2.\n")

    w(f"{'Metodo base':18s}{'directo':>9}{'siempre x2':>12}"
      + "".join(f"{f'hib H0={h}':>11}" for h in H0_VALORES) + f"{'oraculo x1-x6':>15}")
    for metodo, nombre in METODOS.items():
        p0 = cache[f"periodo_{metodo}"].to_numpy()
        oraculo = np.zeros(len(cache), dtype=bool)
        for n in range(1, N_MAX + 1):
            oraculo |= acierto(n * p0, P)
        celdas = [acierto(p0, P).mean(), acierto(2 * p0, P).mean()]
        celdas += [acierto(n_hibrido(cache, h, f"bic_{metodo}") * p0, P).mean() for h in H0_VALORES]
        w(f"{nombre:18s}{celdas[0]:9.1%}{celdas[1]:12.1%}"
          + "".join(f"{c:11.1%}" for c in celdas[2:]) + f"{oraculo.mean():15.1%}")

    # Comparacion estrella por estrella con H0=4 (mejor H0 del experimento con LS 0.1-3000 d)
    ok_ls = acierto(n_hibrido(cache, 4, "bic_ls") * cache["periodo_ls"].to_numpy(), P)
    ok_rp = acierto(n_hibrido(cache, 4, "bic_rp") * cache["periodo_rp"].to_numpy(), P)
    w("\nHibrido H0=4, rank_product vs. Lomb-Scargle, estrella por estrella:")
    w(f"  ambos aciertan {np.mean(ok_ls & ok_rp):6.1%}   solo LS {np.mean(ok_ls & ~ok_rp):6.1%}   "
      f"solo rank_product {np.mean(~ok_ls & ok_rp):6.1%}   ninguno {np.mean(~ok_ls & ~ok_rp):6.1%}")
    misma = np.abs(cache["periodo_rp"] - cache["periodo_ls"]) / cache["periodo_ls"] < 0.01
    w(f"  rank_product coincide con LS (1%) en {misma.mean():.1%} de las estrellas")

    w("\nPor periodo catalogado (hibrido H0=4):")
    df = pd.DataFrame({"ls": ok_ls, "rp": ok_rp, "rango": pd.cut(P, [0, 1, 3, 10, 30, 100, 1e4])})
    for rango, g in df.groupby("rango", observed=True):
        w(f"  {str(rango):16s} n={len(g):4d}   LS {g['ls'].mean():6.1%}   rank_product {g['rp'].mean():6.1%}")

    texto = "\n".join(lineas)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)


def main():
    catalogo = cargar_catalogo()
    disponibles = sorted(p.stem for p in PHOT_DIR.glob("*.dat"))
    disponibles = [i for i in disponibles if i in catalogo.index]
    random.seed(RANDOM_SEED)
    muestra = random.sample(disponibles, N_MUESTRA)  # misma muestra que refinamiento_subarmonicos_bic.py

    cache = pd.read_csv(CACHE_CSV) if CACHE_CSV.exists() else pd.DataFrame()
    procesados = set(cache["id"]) if not cache.empty else set()
    pendientes = [i for i in muestra if i not in procesados]

    filas_nuevas = []
    with ProcessPoolExecutor(max_workers=N_PROCESOS) as pool:
        futuros = [pool.submit(analizar_estrella, i, catalogo.at[i, "Type"], float(catalogo.at[i, "P"]))
                   for i in pendientes]
        for k, futuro in enumerate(tqdm(as_completed(futuros), total=len(futuros),
                                        desc="rank_product + BIC", unit="estrella")):
            fila = futuro.result()
            if fila is not None:
                filas_nuevas.append(fila)
            if filas_nuevas and (k + 1) % 50 == 0:
                cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
                cache.to_csv(CACHE_CSV, index=False)
                filas_nuevas = []

    if filas_nuevas:
        cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
        cache.to_csv(CACHE_CSV, index=False)

    resumen(cache[cache["id"].isin(muestra)].reset_index(drop=True))


if __name__ == "__main__":
    main()
