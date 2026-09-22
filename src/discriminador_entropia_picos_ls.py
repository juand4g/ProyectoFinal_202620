"""
Version 2 del discriminador de discriminador_entropia_multiplos_ls.py: en
vez de evaluar la entropia en el punto EXACTO de cada multiplo (P_LS, 2P_LS,
3P_LS, 4P_LS), busca el minimo local real de entropia en una ventana angosta
alrededor de cada uno (los frecuenciogramas de entropia tienen minimos muy
angostos y pronunciados -- ver Seccion "Picos falsos..." de la tesis -- asi
que evaluar en el punto exacto del multiplo puede caer en un "hombro" en vez
de en el minimo real cercano). Se compara el mejor pico de cada una de las 4
ventanas y se elige el de menor entropia.

Reaprovecha el cache oficial de Lomb-Scargle. Cache propio, append-safe.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from entropypf import get_entropies
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ogle_collection" / "binaries_phot"
CACHE_LS = BASE_DIR / "Lomb-Scargle" / "cache_resultados_ls.csv"
CACHE_PROPIO = BASE_DIR / "Lomb-Scargle" / "cache_discriminador_entropia_picos.csv"

L, K = 10, 7
MULTIPLOS = [1, 2, 3, 4]
VENTANA_REL = 0.02  # +/- 2% del candidato
N_PUNTOS_VENTANA = 201
UMBRAL = 0.01

COLUMNAS = (
    ["id", "periodo_catalogo", "periodo_ls"]
    + [f"entropia_pico_x{n}" for n in MULTIPLOS]
    + [f"periodo_pico_x{n}" for n in MULTIPLOS]
    + ["multiplo_elegido", "periodo_elegido", "error_relativo_elegido", "acierto"]
)


def picos_por_multiplo(t, mag, periodo_ls):
    ventanas = []
    for n in MULTIPLOS:
        centro = periodo_ls * n
        ventanas.append(np.linspace(centro * (1 - VENTANA_REL), centro * (1 + VENTANA_REL), N_PUNTOS_VENTANA))
    periodos_todos = np.concatenate(ventanas)
    entropias_todas = get_entropies(t, mag, periodos_todos, L=L, K=K)

    picos_periodo, picos_entropia = [], []
    for i, n in enumerate(MULTIPLOS):
        ini, fin = i * N_PUNTOS_VENTANA, (i + 1) * N_PUNTOS_VENTANA
        sub_p, sub_e = periodos_todos[ini:fin], entropias_todas[ini:fin]
        j = int(np.argmin(sub_e))
        picos_periodo.append(sub_p[j])
        picos_entropia.append(sub_e[j])
    return picos_periodo, picos_entropia


def main():
    cache_ls = pd.read_csv(CACHE_LS)
    binarias = cache_ls[cache_ls["tipo"] == "Binaria"].dropna(subset=["periodo_ls", "periodo_catalogo"])

    if CACHE_PROPIO.exists():
        cache = pd.read_csv(CACHE_PROPIO)
    else:
        cache = pd.DataFrame(columns=COLUMNAS)
    procesados = set(cache["id"])

    filas_nuevas = []
    for _, fila in tqdm(binarias.iterrows(), total=len(binarias), desc="discriminador-picos", unit="estrella"):
        id_estrella = fila["id"]
        if id_estrella in procesados:
            continue
        try:
            t, mag, err = np.loadtxt(DATA_DIR / f"{id_estrella}.dat", unpack=True)
        except (ValueError, OSError):
            continue

        periodo_ls = float(fila["periodo_ls"])
        periodo_catalogo = float(fila["periodo_catalogo"])

        picos_periodo, picos_entropia = picos_por_multiplo(t, mag, periodo_ls)
        i_elegido = int(np.argmin(picos_entropia))
        multiplo_elegido = MULTIPLOS[i_elegido]
        periodo_elegido = picos_periodo[i_elegido]
        error_relativo_elegido = abs(periodo_elegido - periodo_catalogo) / periodo_catalogo

        fila_nueva = {"id": id_estrella, "periodo_catalogo": periodo_catalogo, "periodo_ls": periodo_ls}
        for n, ent, per in zip(MULTIPLOS, picos_entropia, picos_periodo):
            fila_nueva[f"entropia_pico_x{n}"] = float(ent)
            fila_nueva[f"periodo_pico_x{n}"] = float(per)
        fila_nueva["multiplo_elegido"] = multiplo_elegido
        fila_nueva["periodo_elegido"] = periodo_elegido
        fila_nueva["error_relativo_elegido"] = error_relativo_elegido
        fila_nueva["acierto"] = error_relativo_elegido < UMBRAL
        filas_nuevas.append(fila_nueva)

    if filas_nuevas:
        cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
        cache.to_csv(CACHE_PROPIO, index=False)

    print(f"\nMuestra: {len(cache)} binarias. L={L}, K={K}. Ventana +/-{VENTANA_REL:.0%} alrededor de cada multiplo, "
          f"{N_PUNTOS_VENTANA} puntos/ventana, se toma el minimo local real dentro de cada una.\n")

    tasa_acierto = cache["acierto"].mean() * 100
    print(f"Tasa de acierto del discriminador (mejor PICO de entropia entre las 4 ventanas): {tasa_acierto:.1f}%\n")

    print("Distribucion del multiplo elegido:")
    print((cache["multiplo_elegido"].value_counts(normalize=True).sort_index() * 100).to_string(float_format=lambda x: f"{x:.1f}%"))

    pc = cache["periodo_catalogo"].to_numpy()
    pls = cache["periodo_ls"].to_numpy()
    for n in MULTIPLOS:
        err_n = np.abs(pls * n - pc) / pc
        print(f"Baseline 'siempre elegir x{n}' (sin discriminador): {(err_n < UMBRAL).mean()*100:.1f}%")

    errores_reales = np.abs(pls[:, None] * np.array(MULTIPLOS)[None, :] - pc[:, None]) / pc[:, None]
    acierto_oraculo = (np.min(errores_reales, axis=1) < UMBRAL)
    print(f"\nTecho (oraculo, usando el catalogo): {acierto_oraculo.mean()*100:.1f}%")

    multiplo_real_idx = np.argmin(errores_reales, axis=1)
    multiplo_real = np.array(MULTIPLOS)[multiplo_real_idx]
    print("\nMatriz de confusion (multiplo real necesario, entre los que SI acertaron con algun candidato) vs elegido:")
    df_conf = pd.DataFrame({
        "multiplo_real": multiplo_real[acierto_oraculo],
        "multiplo_elegido": cache["multiplo_elegido"].to_numpy()[acierto_oraculo],
    })
    print(pd.crosstab(df_conf["multiplo_real"], df_conf["multiplo_elegido"], margins=True))


if __name__ == "__main__":
    main()
