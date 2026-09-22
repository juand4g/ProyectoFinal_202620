"""
Prueba exploratoria: Box Least Squares (astropy.timeseries.BoxLeastSquares)
como alternativa a Lomb-Scargle para binarias eclipsantes. A diferencia de
LS (que ajusta una sinusoide), BLS ajusta un pulso tipo caja -- forma mucho
mas parecida a un eclipse real -- por lo que en principio no deberia sufrir
el mismo sesgo hacia el armonico doble (P/2) que ya documentamos para LS y
para minima entropia.

Busqueda LOCAL alrededor de P_LS (no una busqueda global 0.1-3000 d desde
cero): para cada estrella se buscan periodos en [0.5*P_LS, 5*P_LS] y
duraciones relativas de 1% a 30% de P_LS (acorde al ancho de eclipse medido
en estimar_ancho_eclipse_fase.py). Es un chequeo intencionalmente comparable
a discriminador_entropia_multiplos_ls.py: se le da a BLS la misma vecindad
de candidatos que a LS/entropia, para ver si el propio criterio de BLS
resuelve el alias x2 sin necesitar una correccion externa.

Nota tecnica: BLS exige max(duracion) < min(periodo); por eso la busqueda es
local (por estrella) y no global de una sola vez sobre las 500 binarias con
periodos de 0.1 a 3000 d -- una grilla de duracion fija no puede ser valida
simultaneamente para periodos tan dispares.

Reaprovecha el cache oficial de Lomb-Scargle (periodo_ls, periodo_catalogo)
solo para elegir la vecindad de busqueda. Cache propio, append-safe.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from entropypf import get_test_periods
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ogle_collection" / "binaries_phot"
CACHE_LS = BASE_DIR / "Lomb-Scargle" / "cache_resultados_ls.csv"
CACHE_PROPIO = BASE_DIR / "Lomb-Scargle" / "cache_bls_binarias.csv"

P_NUM_LOCAL = 3000  # grilla de periodos LOCAL (no la global de 0.1-3000d)
VENTANA_MIN, VENTANA_MAX = 0.5, 5.0  # relativo a P_LS
DURACION_MIN_REL, DURACION_MAX_REL = 0.01, 0.30  # relativo a P_LS
N_DURACIONES = 8
UMBRAL = 0.01
MULTIPLOS_ALIAS = [1, 2, 3, 4]  # para la correccion de alias, igual que en los otros experimentos

COLUMNAS = ["id", "periodo_catalogo", "periodo_ls", "periodo_bls", "duracion_bls",
            "error_relativo", "periodo_bls_alias", "error_relativo_alias"]


def mejor_alias(periodo, periodo_catalogo, multiplos=MULTIPLOS_ALIAS):
    candidatos = [periodo * n for n in multiplos]
    errores = [abs(p - periodo_catalogo) / periodo_catalogo for p in candidatos]
    i = int(np.argmin(errores))
    return candidatos[i], errores[i]


def main():
    cache_ls = pd.read_csv(CACHE_LS)
    binarias = cache_ls[cache_ls["tipo"] == "Binaria"].dropna(subset=["periodo_ls", "periodo_catalogo"])

    if CACHE_PROPIO.exists():
        cache = pd.read_csv(CACHE_PROPIO)
    else:
        cache = pd.DataFrame(columns=COLUMNAS)
    procesados = set(cache["id"])

    filas_nuevas = []
    for _, fila in tqdm(binarias.iterrows(), total=len(binarias), desc="BLS", unit="estrella"):
        id_estrella = fila["id"]
        if id_estrella in procesados:
            continue
        try:
            t, mag, err = np.loadtxt(DATA_DIR / f"{id_estrella}.dat", unpack=True)
        except (ValueError, OSError):
            continue
        if t.size < 10:
            continue

        periodo_ls = float(fila["periodo_ls"])
        periodo_catalogo = float(fila["periodo_catalogo"])
        flujo = -mag  # BLS busca caidas (dips); en magnitud el eclipse es un aumento, se invierte

        periodos = get_test_periods(VENTANA_MIN * periodo_ls, VENTANA_MAX * periodo_ls, P_NUM_LOCAL)
        duraciones = np.geomspace(DURACION_MIN_REL * periodo_ls, DURACION_MAX_REL * periodo_ls, N_DURACIONES)

        modelo = BoxLeastSquares(t, flujo, dy=err)
        try:
            pg = modelo.power(periodos, duraciones)
        except Exception:
            continue
        i_mejor = int(np.argmax(pg.power))
        periodo_bls = float(pg.period[i_mejor])
        duracion_bls = float(pg.duration[i_mejor])

        error_relativo = abs(periodo_bls - periodo_catalogo) / periodo_catalogo
        periodo_alias, error_alias = mejor_alias(periodo_bls, periodo_catalogo)

        filas_nuevas.append({
            "id": id_estrella, "periodo_catalogo": periodo_catalogo, "periodo_ls": periodo_ls,
            "periodo_bls": periodo_bls, "duracion_bls": duracion_bls,
            "error_relativo": error_relativo,
            "periodo_bls_alias": periodo_alias, "error_relativo_alias": error_alias,
        })

    if filas_nuevas:
        cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
        cache.to_csv(CACHE_PROPIO, index=False)

    print(f"\nMuestra: {len(cache)} binarias. Busqueda LOCAL BLS en [0.5,5]*P_LS, "
          f"duraciones relativas [{DURACION_MIN_REL:.0%},{DURACION_MAX_REL:.0%}]*P_LS.\n")

    acierto_directo = (cache["error_relativo"] < UMBRAL).mean() * 100
    acierto_alias = (cache["error_relativo_alias"] < UMBRAL).mean() * 100
    print(f"BLS directo (sin correccion de alias):        {acierto_directo:.1f}%")
    print(f"BLS + correccion de alias (x1-x4, como LS):   {acierto_alias:.1f}%")
    print(f"\nPara comparar (de experimentos anteriores, mismas 500 binarias):")
    print(f"  Lomb-Scargle directo:                        3.8%")
    print(f"  Lomb-Scargle + alias (x1-x3):                78.0%")
    print(f"  Minima entropia directo (L=7 oficial):        14.6%")
    print(f"  Minima entropia + alias (L=7 oficial):        32.0%")
    print(f"  Baseline 'siempre LS x2' (sin ningun metodo nuevo): 70.8%")


if __name__ == "__main__":
    main()
