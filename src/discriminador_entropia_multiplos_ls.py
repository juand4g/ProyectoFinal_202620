"""
Prueba exploratoria: para cada binaria, se toma el periodo hallado por
Lomb-Scargle (P_LS) y se evalua la entropia de Shannon (entropypf, L=10,
K=7 -- el L elegido en barrido_L_entropia_binarias.py) de la curva de luz
plegada en P_LS, 2*P_LS, 3*P_LS y 4*P_LS. Se elige como periodo final el
multiplo de MENOR entropia (sin usar el periodo catalogado para nada: es
un discriminador no supervisado) y se compara contra el periodo catalogado
para ver si acerto.

Reaprovecha el cache oficial de Lomb-Scargle (periodo_ls, periodo_catalogo)
-- no se recalcula Lomb-Scargle. Cache propio, append-safe.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from entropypf import get_entropies
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ogle_collection" / "binaries_phot"
CACHE_LS = BASE_DIR / "Lomb-Scargle" / "cache_resultados_ls.csv"
CACHE_PROPIO = BASE_DIR / "Lomb-Scargle" / "cache_discriminador_entropia_multiplos.csv"

L, K = 10, 7
MULTIPLOS = [1, 2, 3, 4]
UMBRAL = 0.01

COLUMNAS = (
    ["id", "periodo_catalogo", "periodo_ls"]
    + [f"entropia_x{n}" for n in MULTIPLOS]
    + ["multiplo_elegido", "periodo_elegido", "error_relativo_elegido", "acierto"]
)


def main():
    cache_ls = pd.read_csv(CACHE_LS)
    binarias = cache_ls[cache_ls["tipo"] == "Binaria"].dropna(subset=["periodo_ls", "periodo_catalogo"])

    if CACHE_PROPIO.exists():
        cache = pd.read_csv(CACHE_PROPIO)
    else:
        cache = pd.DataFrame(columns=COLUMNAS)
    procesados = set(cache["id"])

    filas_nuevas = []
    for _, fila in tqdm(binarias.iterrows(), total=len(binarias), desc="discriminador", unit="estrella"):
        id_estrella = fila["id"]
        if id_estrella in procesados:
            continue
        try:
            t, mag, err = np.loadtxt(DATA_DIR / f"{id_estrella}.dat", unpack=True)
        except (ValueError, OSError):
            continue

        periodo_ls = float(fila["periodo_ls"])
        periodo_catalogo = float(fila["periodo_catalogo"])
        candidatos = periodo_ls * np.array(MULTIPLOS, dtype=float)

        entropias = get_entropies(t, mag, candidatos, L=L, K=K)
        i_elegido = int(np.argmin(entropias))
        multiplo_elegido = MULTIPLOS[i_elegido]
        periodo_elegido = candidatos[i_elegido]
        error_relativo_elegido = abs(periodo_elegido - periodo_catalogo) / periodo_catalogo

        fila_nueva = {"id": id_estrella, "periodo_catalogo": periodo_catalogo, "periodo_ls": periodo_ls}
        for n, ent in zip(MULTIPLOS, entropias):
            fila_nueva[f"entropia_x{n}"] = float(ent)
        fila_nueva["multiplo_elegido"] = multiplo_elegido
        fila_nueva["periodo_elegido"] = periodo_elegido
        fila_nueva["error_relativo_elegido"] = error_relativo_elegido
        fila_nueva["acierto"] = error_relativo_elegido < UMBRAL
        filas_nuevas.append(fila_nueva)

    if filas_nuevas:
        cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
        cache.to_csv(CACHE_PROPIO, index=False)

    print(f"\nMuestra: {len(cache)} binarias. L={L}, K={K} fijos para evaluar entropia en cada multiplo.\n")

    tasa_acierto = cache["acierto"].mean() * 100
    print(f"Tasa de acierto del discriminador (elige el multiplo de menor entropia): {tasa_acierto:.1f}%\n")

    print("Distribucion del multiplo elegido por el discriminador:")
    print((cache["multiplo_elegido"].value_counts(normalize=True).sort_index() * 100).to_string(float_format=lambda x: f"{x:.1f}%"))

    # Baselines de comparacion: "siempre adivinar x2", "siempre x1" (sin discriminador)
    pc = cache["periodo_catalogo"].to_numpy()
    pls = cache["periodo_ls"].to_numpy()
    for n in MULTIPLOS:
        err_n = np.abs(pls * n - pc) / pc
        print(f"Baseline 'siempre elegir x{n}' (sin discriminador): {(err_n < UMBRAL).mean()*100:.1f}%")

    # Ground truth: que multiplo hacia falta realmente (entre los 4 candidatos, el de menor error vs catalogo)
    errores_reales = np.abs(pls[:, None] * np.array(MULTIPLOS)[None, :] - pc[:, None]) / pc[:, None]
    multiplo_real_idx = np.argmin(errores_reales, axis=1)
    multiplo_real = np.array(MULTIPLOS)[multiplo_real_idx]
    acierto_oraculo = (np.min(errores_reales, axis=1) < UMBRAL)

    print(f"\nTecho: si el discriminador SIEMPRE eligiera el mejor de los 4 candidatos (oraculo, usando el catalogo): "
          f"{acierto_oraculo.mean()*100:.1f}%")

    print("\nMatriz de confusion (multiplo real necesario, entre los que SI acertaron con algun candidato) vs elegido por el discriminador:")
    df_conf = pd.DataFrame({
        "multiplo_real": multiplo_real[acierto_oraculo],
        "multiplo_elegido": cache["multiplo_elegido"].to_numpy()[acierto_oraculo],
    })
    print(pd.crosstab(df_conf["multiplo_real"], df_conf["multiplo_elegido"], margins=True))


if __name__ == "__main__":
    main()
