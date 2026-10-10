"""
Refinamiento por subarmonicos con seleccion de modelo por BIC, como segunda
etapa despues de Lomb-Scargle, para binarias eclipsantes.

Motivacion (ver bitacora_experimentos_alias_binarias.txt): Lomb-Scargle casi
siempre encuentra ALGUN armonico n*f_orb de la frecuencia orbital (n=2 en la
mayoria, pero n=3,4,6,... en las binarias de periodo largo, con eclipses
angostos en fase). El problema no es solo el x2 sino decidir n sin usar el
catalogo.

Metodo, para cada estrella:
  1. f0 = frecuencia de maxima potencia de LS (misma configuracion que
     caracterizacion_lomb_scargle.py: 0.1-3000 d, samples_per_peak=5).
  2. Candidatos f0/n, n = 1..N_MAX (solo si n/f0 <= linea base).
  3. A cada candidato se le ajusta (minimos cuadrados ponderados) una serie
     de Fourier con H = n*H0 armonicos. Asi todos los modelos alcanzan la
     misma frecuencia maxima H0*f0 y el modelo en f0 queda contenido en el
     de f0/n (anidados cuando n divide a n').
  4. Se elige el n de menor BIC = N ln(RSS_w/N) + k ln N, con k = 2H+1. Se
     usa la forma con varianza desconocida para que el resultado no dependa
     de una mala calibracion global de los errores fotometricos de OGLE.

Se guarda el BIC de cada (n, H0) en el cache, de modo que el barrido de H0
se evalua sin recalcular. Muestra NUEVA de N_MUESTRA binarias (semilla
propia), con LS recalculado aqui. Cache append-safe. La muestra se amplio
de 1000 a 3000 el 09/10/2026 (las 1000 originales son el prefijo; ver
muestra_anidada) y queda congelada en Refinamiento_Subarmonicos/
muestra_binarias.csv. El techo (oraculo) usa multiplos hasta x10.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ogle_collection"
PHOT_DIR = DATA_DIR / "binaries_phot"
CATALOGO = DATA_DIR / "binaries.txt"
OUT_DIR = BASE_DIR / "Refinamiento_Subarmonicos"
CACHE_CSV = OUT_DIR / "cache_refinamiento_bic.csv"
RESUMEN_TXT = OUT_DIR / "resumen_refinamiento_bic.txt"

N_MUESTRA_BASE = 1000  # muestra original (04/10/2026)
N_MUESTRA = 3000  # ampliada el 09/10/2026; las 1000 originales son su prefijo
RANDOM_SEED = 2026  # distinta de la muestra oficial (42) y de la ampliada (123)
MUESTRA_CSV = OUT_DIR / "muestra_binarias.csv"
PLANTILLAS_CSV = BASE_DIR / "Robustez_Sinteticas" / "plantillas.csv"
N_PLANTILLAS_ORIGINALES = 240  # plantillas de las sinteticas anteriores a la ampliacion

P_MIN, P_MAX = 0.1, 3000.0
SAMPLES_PER_PEAK = 5
MIN_PUNTOS = 10

N_MAX = 6  # subarmonicos f0/1 ... f0/N_MAX que explora la regla de seleccion
N_TECHO = 10  # multiplos x1 ... x10 para el techo (oraculo), desde el 09/10/2026
H0_VALORES = [2, 3, 4, 6, 8]  # armonicos del modelo en f0 (barrido)
MIN_PUNTOS_POR_PARAMETRO = 3  # se descarta un candidato si N < 3*k
UMBRAL = 0.01


def cargar_catalogo():
    df = pd.read_csv(CATALOGO, sep="\t", comment="#").set_index("ID")
    return df[["Type", "P"]]


def muestra_anidada(disponibles, n_base, n_total, semilla, excluir=()):
    """Muestra de n_total elementos cuyos primeros n_base son exactamente
    random.sample(disponibles, n_base) con la semilla dada (la muestra
    original). El resto se extrae, con otra semilla, de los no elegidos que
    no esten en `excluir`."""
    random.seed(semilla)
    base = random.sample(disponibles, n_base)
    if n_total <= n_base:
        return base[:n_total]
    vetados = set(base) | set(excluir)
    resto = [i for i in disponibles if i not in vetados]
    return base + random.Random(semilla + 1).sample(resto, n_total - n_base)


def muestra_congelada(ruta, generar):
    """Lee la lista de IDs de `ruta`; si no existe, la genera y la guarda,
    para que la muestra no dependa de cambios posteriores en los archivos."""
    if ruta.exists():
        return pd.read_csv(ruta)["id"].tolist()
    ids = generar()
    ruta.parent.mkdir(exist_ok=True)
    pd.DataFrame({"id": ids}).to_csv(ruta, index=False)
    return ids


def estrellas_plantilla_originales():
    """Estrellas usadas como plantilla o cadencia en las sinteticas originales:
    las ampliaciones de las muestras OGLE las excluyen para no mezclar ambas."""
    if not PLANTILLAS_CSV.exists():
        return set()
    pl = pd.read_csv(PLANTILLAS_CSV)
    pl = pl[pl["id_plantilla"] < N_PLANTILLAS_ORIGINALES]
    return {Path(r).stem for r in pd.concat([pl["plantilla"], pl["cadencia"]])}


def muestra_binarias(disponibles):
    return muestra_congelada(MUESTRA_CSV, lambda: muestra_anidada(
        disponibles, N_MUESTRA_BASE, N_MUESTRA, RANDOM_SEED, estrellas_plantilla_originales()))


def n_real_techo(p0, P):
    """Menor n en 1..N_TECHO con n*P0 dentro del 1% de P (0 si ninguno)."""
    n_real = np.zeros(len(P), dtype=int)
    for n in range(N_TECHO, 0, -1):
        n_real[acierto(n * np.asarray(p0), P)] = n
    return n_real


def periodo_ls(t, mag, err):
    ls = LombScargle(t, mag, err)
    frecuencia, potencia = ls.autopower(
        minimum_frequency=1.0 / P_MAX, maximum_frequency=1.0 / P_MIN,
        samples_per_peak=SAMPLES_PER_PEAK,
    )
    return 1.0 / frecuencia[np.argmax(potencia)]


def bic_fourier(t, mag, err, frecuencia, n_armonicos):
    """BIC de una serie de Fourier de n_armonicos en la frecuencia dada,
    ajustada por minimos cuadrados ponderados."""
    k = 2 * n_armonicos + 1
    if t.size < MIN_PUNTOS_POR_PARAMETRO * k:
        return np.nan
    fase = 2 * np.pi * frecuencia * (t - t.min())
    j = np.arange(1, n_armonicos + 1)
    argumento = np.outer(fase, j)
    X = np.hstack([np.ones((t.size, 1)), np.cos(argumento), np.sin(argumento)])
    w = 1.0 / err
    coef, *_ = np.linalg.lstsq(X * w[:, None], mag * w, rcond=None)
    rss = np.sum(((mag - X @ coef) * w) ** 2)
    return t.size * np.log(rss / t.size) + k * np.log(t.size)


def analizar_estrella(id_estrella, tipo, periodo_catalogo):
    try:
        t, mag, err = np.loadtxt(PHOT_DIR / f"{id_estrella}.dat", unpack=True)
    except (ValueError, OSError):
        return None
    if t.size < MIN_PUNTOS:
        return None
    err = np.where(err > 0, err, np.median(err[err > 0]))
    baseline = t.max() - t.min()

    p_ls = periodo_ls(t, mag, err)
    f0 = 1.0 / p_ls
    fila = {
        "id": id_estrella, "tipo_morfologico": tipo, "n_puntos": t.size,
        "baseline_d": baseline, "periodo_catalogo": periodo_catalogo, "periodo_ls": p_ls,
    }
    for h0 in H0_VALORES:
        for n in range(1, N_MAX + 1):
            bic = np.nan
            if n * p_ls <= baseline:
                bic = bic_fourier(t, mag, err, f0 / n, n * h0)
            fila[f"bic_h{h0}_n{n}"] = bic
    return fila


def n_elegido(cache, h0):
    columnas = [f"bic_h{h0}_n{n}" for n in range(1, N_MAX + 1)]
    bics = cache[columnas].to_numpy()
    todo_nan = np.all(np.isnan(bics), axis=1)
    n = np.full(len(cache), 1)
    n[~todo_nan] = np.nanargmin(bics[~todo_nan], axis=1) + 1
    return n


def n_hibrido(cache, h0, prefijo="bic"):
    """x2 por defecto (prior fisico: dos eclipses por orbita); se sube a
    n>=3 solo si el mejor de esos candidatos tiene menor BIC que n=2."""
    columnas = [f"{prefijo}_h{h0}_n{n}" for n in range(1, N_MAX + 1)]
    bics = np.where(cache[columnas].isna(), np.inf, cache[columnas].to_numpy())
    n_alto = np.argmin(bics[:, 2:], axis=1) + 3
    bic_alto = bics[np.arange(len(cache)), n_alto - 1]
    n = np.where(bic_alto < bics[:, 1], n_alto, 2)
    return np.where(np.isinf(bics[:, 1]), 1, n)  # 2*P_LS mayor que la linea base


def acierto(periodo, periodo_catalogo):
    return np.abs(periodo - periodo_catalogo) / periodo_catalogo < UMBRAL


def resumen(cache):
    P = cache["periodo_catalogo"].to_numpy()
    p_ls = cache["periodo_ls"].to_numpy()
    lineas = []
    w = lineas.append

    # n "verdadero": multiplo entero de P_LS que reproduce el catalogo (0 = ninguno)
    n_real = n_real_techo(p_ls, P)

    w(f"RESUMEN - REFINAMIENTO POR SUBARMONICOS CON BIC")
    w("=" * 78)
    w(f"Binarias analizadas: {len(cache)} (semilla {RANDOM_SEED}); "
      f"mediana de puntos: {int(np.median(cache['n_puntos']))}")
    w(f"Candidatos f0/n, n=1..{N_MAX}; modelo de Fourier con H = n*H0 armonicos.\n")

    w("REFERENCIAS (misma muestra)")
    w(f"  LS directo (x1):                     {acierto(p_ls, P).mean():6.1%}")
    w(f"  Siempre x2:                          {acierto(2 * p_ls, P).mean():6.1%}")
    w(f"  Oraculo x1-x2 (usa catalogo):        {(n_real >= 1).__and__(n_real <= 2).mean():6.1%}")
    w(f"  Oraculo x1-x{N_MAX} (usa catalogo):        {((n_real >= 1) & (n_real <= N_MAX)).mean():6.1%}")
    w(f"  Oraculo x1-x{N_TECHO} (usa catalogo):       {(n_real >= 1).mean():6.1%}\n")

    w("SELECCION POR BIC (sin usar el catalogo)")
    mejor_h0, mejor_tasa = None, -1
    for h0 in H0_VALORES:
        n = n_elegido(cache, h0)
        tasa = acierto(n * p_ls, P).mean()
        if tasa > mejor_tasa:
            mejor_h0, mejor_tasa = h0, tasa
        w(f"  H0={h0}:  aciertos {tasa:6.1%}   n elegido: "
          + "  ".join(f"x{k}={np.mean(n == k):.1%}" for k in range(1, N_MAX + 1)))

    n = n_elegido(cache, mejor_h0)
    ok = acierto(n * p_ls, P)
    w(f"\nDETALLE PARA EL MEJOR H0 = {mejor_h0}")
    w("  Matriz n_real (filas, 0 = sin multiplo simple) vs. n elegido (columnas):")
    w("            " + "".join(f"{f'x{k}':>7}" for k in range(1, N_MAX + 1)) + "   total")
    for r in range(0, N_TECHO + 1):
        m = n_real == r
        if m.sum() == 0:
            continue
        w(f"    real {r}: " + "".join(f"{np.sum(m & (n == k)):7d}" for k in range(1, N_MAX + 1))
          + f"   {m.sum():5d}")

    resolubles = n_real > 0
    w(f"\n  Aciertos entre las resolubles con algun x1-x{N_TECHO}: {ok[resolubles].mean():.1%} "
      f"(n={resolubles.sum()})")

    w("\n  Por tipo morfologico (NC / C=W UMa / ELL):")
    for tipo, g in cache.assign(ok=ok, x2=acierto(2 * p_ls, P)).groupby("tipo_morfologico"):
        w(f"    {tipo:4s} n={len(g):4d}   BIC {g['ok'].mean():6.1%}   siempre x2 {g['x2'].mean():6.1%}")

    w("\n  Por periodo catalogado:")
    bins = [0, 1, 3, 10, 30, 100, 1e4]
    df = pd.DataFrame({"ok": ok, "x2": acierto(2 * p_ls, P), "or": n_real > 0,
                       "rango": pd.cut(P, bins)})
    for rango, g in df.groupby("rango", observed=True):
        w(f"    {str(rango):16s} n={len(g):4d}   BIC {g['ok'].mean():6.1%}   "
          f"siempre x2 {g['x2'].mean():6.1%}   oraculo x1-x{N_TECHO} {g['or'].mean():6.1%}")

    w("\nVARIANTE HIBRIDA: x2 por defecto; n>=3 solo si su BIC es menor que el de n=2")
    w("(nunca se elige x1; sin umbral ajustado, delta BIC = 0)")
    for h0 in H0_VALORES:
        n = n_hibrido(cache, h0)
        w(f"  H0={h0}:  aciertos {acierto(n * p_ls, P).mean():6.1%}")

    texto = "\n".join(lineas)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)


def main():
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(exist_ok=True)

    catalogo = cargar_catalogo()
    disponibles = sorted(p.stem for p in PHOT_DIR.glob("*.dat"))
    disponibles = [i for i in disponibles if i in catalogo.index]
    muestra = muestra_binarias(disponibles)

    cache = pd.read_csv(CACHE_CSV) if CACHE_CSV.exists() else pd.DataFrame()
    procesados = set(cache["id"]) if not cache.empty else set()

    filas_nuevas = []
    for id_estrella in tqdm(muestra, desc="Refinamiento BIC", unit="estrella"):
        if id_estrella in procesados:
            continue
        fila = analizar_estrella(id_estrella, catalogo.at[id_estrella, "Type"],
                                 float(catalogo.at[id_estrella, "P"]))
        if fila is not None:
            filas_nuevas.append(fila)

    if filas_nuevas:
        cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
        cache.to_csv(CACHE_CSV, index=False)

    resumen(cache[cache["id"].isin(muestra)].reset_index(drop=True))


if __name__ == "__main__":
    main()
