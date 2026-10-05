"""
Igual que refinamiento_rank_product_bic.py, pero tratando los alias diarios
de la entropia, que sin ningun tratamiento cae en 0.499 o 0.997 d en ~99% de
las binarias (experimento 9 de la bitacora) y reduce rank_product a
Lomb-Scargle. Se usa L=10 (experimento 5). Se comparan dos tratamientos:

1. Mascara OFICIAL (caracterizacion_entropia.py): se excluyen todas las
   fracciones m/n de un dia (n <= 6, +/-0.03 d). Cubre ~61% de la grilla en
   frecuencia. Dentro de rank_product se prueba:
     - rp_excl: frecuencias enmascaradas excluidas (rank_product = 0).
     - rp_neut: percentil de entropia neutro (0.5) en las enmascaradas.
2. Entropia CONDICIONAL H(mag|fase) = H(fase, mag) - H(fase), sin mascara
   (Graham et al. 2013, MNRAS 434, 2629). Con ~1 dato por noche, plegar en
   P = m/n dias deja las fases en solo m valores; eso baja H(fase, mag) pero
   tambien H(fase), y la resta cancela el efecto. Antes se probaron mascaras
   mas estrechas (m/n con m chico, dia solar y sidereo) y cada una dejaba
   aparecer minimos espurios en otras fracciones (1/7, 7/2, 7 d...).

H(fase) se obtiene con la misma get_entropies pasando magnitudes constantes
(todo cae en un solo bin de magnitud), con el mismo origen de fase t0 y la
misma normalizacion log(L*K). Los percentiles de entropia se calculan solo
sobre las frecuencias libres. Despues se aplica a cada metodo base la misma
regla hibrida de subarmonicos con BIC. Misma muestra de 1000 binarias
(semilla 2026).
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

from caracterizacion_entropia import ALIAS_DIURNO_EPS, generar_alias_diurnos
from refinamiento_rank_product_bic import N_FREQ, N_PROCESOS, P_MAX, P_MIN, TAMANO_BLOQUE
from refinamiento_subarmonicos_bic import (
    H0_VALORES, MIN_PUNTOS, N_MAX, N_MUESTRA, OUT_DIR, PHOT_DIR, RANDOM_SEED,
    acierto, bic_fourier, cargar_catalogo, n_hibrido,
)

CACHE_CSV = OUT_DIR / "cache_rank_product_enmascarado_bic.csv"
RESUMEN_TXT = OUT_DIR / "resumen_rank_product_enmascarado_bic.txt"

L, K = 10, 7
METODOS = {"ls": "Lomb-Scargle", "ent": "Entropia m.ofic.",
           "rp_excl": "rank_prod excl.", "rp_neut": "rank_prod neutro",
           "ent_c": "Entropia condic.", "rp_c": "rank_prod condic."}

FRECUENCIA = np.linspace(1.0 / P_MAX, 1.0 / P_MIN, N_FREQ)
CENTROS = generar_alias_diurnos(P_MIN, P_MAX)
_periodos = 1.0 / FRECUENCIA
MASCARA = np.zeros(N_FREQ, dtype=bool)
for _c in CENTROS:
    MASCARA |= np.abs(_periodos - _c) < ALIAS_DIURNO_EPS


def entropias_conjunta_y_fase(t, mag):
    """S(fase, mag) y S(fase), normalizadas por log(L*K). get_entropies toma
    como origen de fase el punto mas cercano al percentil 2 de magnitud; con
    magnitudes constantes tomaria el primer punto, asi que se reordena para
    que ese punto quede primero y ambas llamadas usen el mismo t0."""
    i0 = int(np.argmin(np.abs(mag - np.percentile(mag, 2))))
    orden = np.r_[i0, np.delete(np.arange(t.size), i0)]
    t, mag = t[orden], mag[orden]
    constante = np.zeros_like(mag)
    conjunta, fase = [], []
    for i in range(0, N_FREQ, TAMANO_BLOQUE):
        periodos = 1.0 / FRECUENCIA[i:i + TAMANO_BLOQUE]
        conjunta.append(get_entropies(t, mag, periodos, L=L, K=K))
        fase.append(get_entropies(t, constante, periodos, L=L, K=K))
    return np.concatenate(conjunta), np.concatenate(fase)


def periodos_base(t, mag, err):
    potencia = LombScargle(t, mag, err).power(FRECUENCIA)
    entropia, entropia_fase = entropias_conjunta_y_fase(t, mag)
    condicional = entropia - entropia_fase
    rango_ls = rankdata(potencia) / N_FREQ

    def rango_entropia(s, mascara, relleno):
        """Percentil de -s calculado solo sobre las frecuencias libres; las
        enmascaradas reciben el valor de relleno."""
        rango = np.full(N_FREQ, relleno)
        rango[~mascara] = rankdata(-s[~mascara]) / np.sum(~mascara)
        return rango

    sin_mascara = np.zeros(N_FREQ, dtype=bool)
    return {
        "ls": 1.0 / FRECUENCIA[np.argmax(potencia)],
        "ent": 1.0 / FRECUENCIA[np.argmin(np.where(MASCARA, np.inf, entropia))],
        "rp_excl": 1.0 / FRECUENCIA[np.argmax(rango_ls * rango_entropia(entropia, MASCARA, 0.0))],
        "rp_neut": 1.0 / FRECUENCIA[np.argmax(rango_ls * rango_entropia(entropia, MASCARA, 0.5))],
        "ent_c": 1.0 / FRECUENCIA[np.argmin(condicional)],
        "rp_c": 1.0 / FRECUENCIA[np.argmax(rango_ls * rango_entropia(condicional, sin_mascara, 0.0))],
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


def en_mascara(periodos):
    return (np.abs(np.asarray(periodos)[:, None] - CENTROS[None, :]) < ALIAS_DIURNO_EPS).any(axis=1)


def resumen(cache):
    P = cache["periodo_catalogo"].to_numpy()
    lineas = []
    w = lineas.append
    w("RESUMEN - rank_product CON ENTROPIA ENMASCARADA O CONDICIONAL + REFINAMIENTO BIC")
    w("=" * 78)
    w(f"Binarias: {len(cache)} (semilla {RANDOM_SEED}); grilla uniforme de {N_FREQ} frecuencias, "
      f"{P_MIN}-{P_MAX} d; entropia L={L}, K={K}.")
    w(f"Mascara oficial: fracciones m/n de dia (n<=6), +/-{ALIAS_DIURNO_EPS} d; cubre "
      f"{MASCARA.mean():.1%} de la grilla en frecuencia y {np.mean(en_mascara(1.0 / FRECUENCIA[FRECUENCIA < 1])):.1%} "
      f"de la grilla con P > 1 d.")
    w(f"Binarias con P_cat dentro de la mascara: {en_mascara(P).mean():.1%}; "
      f"con P_cat/2 dentro: {en_mascara(P / 2).mean():.1%}")
    w("Hibrido: x2 por defecto; n>=3 solo si su BIC es menor que el de n=2.")
    w("")

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

    ok = {m: acierto(n_hibrido(cache, 4, f"bic_{m}") * cache[f"periodo_{m}"].to_numpy(), P) for m in METODOS}
    w("")
    w("Hibrido H0=4, cada rank_product vs. Lomb-Scargle, estrella por estrella:")
    for m in ["rp_excl", "rp_neut", "rp_c"]:
        w(f"  {METODOS[m]:18s} ambos {np.mean(ok['ls'] & ok[m]):6.1%}   solo LS {np.mean(ok['ls'] & ~ok[m]):6.1%}   "
          f"solo rank_product {np.mean(~ok['ls'] & ok[m]):6.1%}   ninguno {np.mean(~ok['ls'] & ~ok[m]):6.1%}")
    for m in ["rp_neut", "rp_c"]:
        w(f"  Union LS o {METODOS[m]} (oraculo entre los dos): {np.mean(ok['ls'] | ok[m]):.1%}")

    w("")
    for m in ["ent", "ent_c"]:
        razon = cache[f"periodo_{m}"].to_numpy() / P
        w(f"Razon P/P_cat, {METODOS[m]} directa: " + "   ".join(
            f"~{e} {np.mean(np.abs(razon - v) / v < 0.01):6.1%}"
            for e, v in [("x1", 1), ("1/2", 0.5), ("1/3", 1 / 3), ("2", 2)]))
    w(f"Entropia condicional que cae dentro de la mascara oficial: "
      f"{en_mascara(cache['periodo_ent_c'].to_numpy()).mean():.1%} (la mascara cubre {MASCARA.mean():.1%} de la grilla)")

    w("")
    w("Por periodo catalogado (hibrido H0=4):")
    df = pd.DataFrame({m: ok[m] for m in METODOS} | {"rango": pd.cut(P, [0, 1, 3, 10, 30, 100, 1e4])})
    for rango, g in df.groupby("rango", observed=True):
        w(f"  {str(rango):16s} n={len(g):4d}   " + "   ".join(f"{m} {g[m].mean():6.1%}" for m in METODOS))

    texto = "\n".join(lineas)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)


def main():
    catalogo = cargar_catalogo()
    disponibles = sorted(p.stem for p in PHOT_DIR.glob("*.dat"))
    disponibles = [i for i in disponibles if i in catalogo.index]
    random.seed(RANDOM_SEED)
    muestra = random.sample(disponibles, N_MUESTRA)

    cache = pd.read_csv(CACHE_CSV) if CACHE_CSV.exists() else pd.DataFrame()
    procesados = set(cache["id"]) if not cache.empty else set()
    pendientes = [i for i in muestra if i not in procesados]

    filas_nuevas = []
    with ProcessPoolExecutor(max_workers=N_PROCESOS) as pool:
        futuros = [pool.submit(analizar_estrella, i, catalogo.at[i, "Type"], float(catalogo.at[i, "P"]))
                   for i in pendientes]
        for k, futuro in enumerate(tqdm(as_completed(futuros), total=len(futuros),
                                        desc="rank_product enmascarado/condicional + BIC", unit="estrella")):
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
