"""
Caracterizacion de exactitud y precision del metodo de minima entropia de
Shannon (entropypf; Cincotta, Mendez & Nunez 1995) para la busqueda de
periodos en Binarias, Cefeidas Clasicas y RR Lyrae del catalogo fotometrico
OGLE (banda I, tiempo en HJD).

Analogo a caracterizacion_lomb_scargle.py: misma semilla aleatoria y mismos
rangos de busqueda de periodo por tipo, para permitir comparar ambos metodos
sobre exactamente las mismas estrellas.

Los resultados se guardan en un CSV que funciona como cache: al volver a
ejecutar el script, las estrellas ya procesadas no se vuelven a calcular.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from entropypf import find_best_period, get_entropies, get_test_periods
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PARAMETROS
# ---------------------------------------------------------------------------
N_BINARIAS = 5
N_CEFEIDAS = 5
N_RRLYRAE = 5

RANDOM_SEED = 42  # misma semilla que caracterizacion_lomb_scargle.py -> misma muestra

# Rango de busqueda de periodos (dias) para cada tipo de estrella.
P_MIN_BINARIAS, P_MAX_BINARIAS = 0.1, 3000.0
P_MIN_CEFEIDAS, P_MAX_CEFEIDAS = 0.2, 250.0
P_MIN_RRLYRAE, P_MAX_RRLYRAE = 0.1, 1.2

P_NUM = 10_000  # numero de periodos de prueba en la grilla no uniforme
L, K = 7, 7  # dimensiones de la grilla fase-magnitud
ALIASES = (1,)  # enmascara armonicos del dia sideral (0.5 d, 1/3 d, ...)
N_CANDIDATOS = 3
MIN_PUNTOS = 10  # curvas con menos puntos que esto se descartan

GENERAR_GRAFICAS = True
N_EJEMPLOS_PERIODOGRAMA = 3  # numero de periodogramas de ejemplo a graficar por tipo

DATA_DIR = Path(__file__).resolve().parent.parent / "ogle_collection"
OUT_DIR = Path(__file__).resolve().parent.parent / "Entropia"
CACHE_CSV = OUT_DIR / "cache_resultados_entropia.csv"
PLOTS_DIR = OUT_DIR / "graficas"

TIPOS = {
    "Binaria": {
        "carpeta": DATA_DIR / "binaries_phot",
        "catalogo": DATA_DIR / "binaries.txt",
        "col_periodo": "P",
        "n_muestra": N_BINARIAS,
        "p_min": P_MIN_BINARIAS,
        "p_max": P_MAX_BINARIAS,
    },
    "Cefeida": {
        "carpeta": DATA_DIR / "classical_cefeids_phot",
        "catalogo": DATA_DIR / "classical_cefeids.txt",
        "col_periodo": "P_1",
        "n_muestra": N_CEFEIDAS,
        "p_min": P_MIN_CEFEIDAS,
        "p_max": P_MAX_CEFEIDAS,
    },
    "RR Lyrae": {
        "carpeta": DATA_DIR / "rrlyrae_phot",
        "catalogo": DATA_DIR / "rrlyrae.txt",
        "col_periodo": "P_1",
        "n_muestra": N_RRLYRAE,
        "p_min": P_MIN_RRLYRAE,
        "p_max": P_MAX_RRLYRAE,
    },
}

COLUMNAS_CACHE = [
    "tipo", "id", "n_puntos", "baseline_d",
    "periodo_catalogo", "periodo_entropia", "entropia_minima",
    "error_relativo", "periodo_entropia_alias", "error_relativo_alias",
]


def cargar_catalogo(path_txt, col_periodo):
    df = pd.read_csv(path_txt, sep="\t", comment="#")
    df = df.set_index("ID")
    return df[col_periodo]


def calcular_periodo_entropia(t, mag, p_min, p_max):
    periodos, entropias = find_best_period(
        t, mag, p_min, p_max, P_NUM,
        L=L, K=K, n_candidates=N_CANDIDATOS, aliases=ALIASES,
    )
    return float(periodos[0]), float(entropias[0])


def mejor_alias(periodo_entropia, periodo_catalogo):
    """El metodo de minima entropia tambien puede confundir el periodo real
    con sus armonicos (P/2, 2P, P/3, 3P), en particular en binarias
    eclipsantes con minimos de profundidad similar."""
    candidatos = [periodo_entropia, periodo_entropia / 2, periodo_entropia * 2,
                  periodo_entropia / 3, periodo_entropia * 3]
    errores = [abs(p - periodo_catalogo) / periodo_catalogo for p in candidatos]
    i = int(np.argmin(errores))
    return candidatos[i], errores[i]


def graficar_periodograma(t, mag, p_min, p_max, periodo_catalogo, periodo_entropia, tipo, id_estrella, outfile):
    periodo = get_test_periods(p_min, p_max, P_NUM)
    entropia = get_entropies(t, mag, periodo, L=L, K=K)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(periodo, entropia, lw=0.7, color="darkorange")
    ax.axvline(periodo_catalogo, color="green", ls="--", label=f"P catalogo = {periodo_catalogo:.5f} d")
    ax.axvline(periodo_entropia, color="red", ls=":", label=f"P entropia minima = {periodo_entropia:.5f} d")
    ax.set_xscale("log")
    ax.invert_yaxis()  # entropia minima = mejor periodo
    ax.set_xlabel("Periodo (d)")
    ax.set_ylabel("Entropia de Shannon normalizada")
    ax.set_title(f"{tipo} - {id_estrella}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def graficar_resumen(cache):
    validos = cache.dropna(subset=["error_relativo"])
    if validos.empty:
        return

    tipos_lista = sorted(validos["tipo"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datos_boxplot = [validos.loc[validos["tipo"] == t, "error_relativo"] for t in tipos_lista]
    axes[0].boxplot(datos_boxplot, showfliers=False)
    axes[0].set_xticks(range(1, len(tipos_lista) + 1))
    axes[0].set_xticklabels(tipos_lista)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Error relativo |P_entropia - P_cat| / P_cat")
    axes[0].set_title("Exactitud y precision por tipo de estrella (minima entropia)")

    colores = {"Binaria": "tab:orange", "Cefeida": "tab:blue", "RR Lyrae": "tab:green"}
    for tipo in tipos_lista:
        grupo = validos[validos["tipo"] == tipo]
        axes[1].scatter(
            grupo["periodo_catalogo"], grupo["periodo_entropia"],
            s=8, alpha=0.4, label=tipo, color=colores.get(tipo),
        )
    lims = [validos["periodo_catalogo"].min(), validos["periodo_catalogo"].max()]
    axes[1].plot(lims, lims, color="black", lw=0.8, ls="--", label="P_entropia = P_cat")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Periodo catalogo (d)")
    axes[1].set_ylabel("Periodo minima entropia (d)")
    axes[1].set_title("Periodo recuperado vs. periodo catalogado")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "resumen_error_relativo.png", dpi=150)
    plt.close(fig)


def imprimir_resumen(cache):
    validos = cache.dropna(subset=["error_relativo"])
    if validos.empty:
        print("No hay resultados validos todavia.")
        return

    print("\nResumen por tipo de estrella (error relativo del periodo, minima entropia):")
    resumen = validos.groupby("tipo")["error_relativo"].agg(
        mediana="median", media="mean", desviacion="std", n="count",
    )
    resumen["tasa_acierto_1pct"] = validos.groupby("tipo")["error_relativo"].apply(lambda s: (s < 0.01).mean())
    resumen["tasa_acierto_alias_1pct"] = validos.groupby("tipo")["error_relativo_alias"].apply(lambda s: (s < 0.01).mean())
    print(resumen.to_string(float_format=lambda x: f"{x:.4f}"))


def main():
    random.seed(RANDOM_SEED)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_CSV.exists():
        cache = pd.read_csv(CACHE_CSV)
    else:
        cache = pd.DataFrame(columns=COLUMNAS_CACHE)

    procesados = set(zip(cache["tipo"], cache["id"]))
    filas_nuevas = []

    for tipo, cfg in TIPOS.items():
        periodos_catalogo = cargar_catalogo(cfg["catalogo"], cfg["col_periodo"])
        archivos = sorted(cfg["carpeta"].glob("*.dat"))
        ids_disponibles = [f.stem for f in archivos if f.stem in periodos_catalogo.index]

        n = min(cfg["n_muestra"], len(ids_disponibles))
        muestra = random.sample(ids_disponibles, n)

        n_ejemplos_graficados = 0

        for id_estrella in tqdm(muestra, desc=tipo, unit="estrella"):
            if (tipo, id_estrella) in procesados:
                continue

            path_dat = cfg["carpeta"] / f"{id_estrella}.dat"
            try:
                t, mag, err = np.loadtxt(path_dat, unpack=True)
            except (ValueError, OSError) as exc:
                warnings.warn(f"No se pudo leer {path_dat}: {exc}")
                continue

            if t.size < MIN_PUNTOS:
                filas_nuevas.append({col: np.nan for col in COLUMNAS_CACHE} | {
                    "tipo": tipo, "id": id_estrella, "n_puntos": int(t.size),
                })
                procesados.add((tipo, id_estrella))
                continue

            periodo_catalogo = float(periodos_catalogo.loc[id_estrella])

            periodo_entropia, entropia_minima = calcular_periodo_entropia(
                t, mag, cfg["p_min"], cfg["p_max"]
            )

            error_relativo = abs(periodo_entropia - periodo_catalogo) / periodo_catalogo
            periodo_alias, error_alias = mejor_alias(periodo_entropia, periodo_catalogo)

            filas_nuevas.append({
                "tipo": tipo,
                "id": id_estrella,
                "n_puntos": int(t.size),
                "baseline_d": float(t.max() - t.min()),
                "periodo_catalogo": periodo_catalogo,
                "periodo_entropia": periodo_entropia,
                "entropia_minima": entropia_minima,
                "error_relativo": error_relativo,
                "periodo_entropia_alias": periodo_alias,
                "error_relativo_alias": error_alias,
            })
            procesados.add((tipo, id_estrella))

            if GENERAR_GRAFICAS and n_ejemplos_graficados < N_EJEMPLOS_PERIODOGRAMA:
                nombre = f"periodograma_{tipo.replace(' ', '_')}_{id_estrella}.png"
                graficar_periodograma(
                    t, mag, cfg["p_min"], cfg["p_max"], periodo_catalogo, periodo_entropia,
                    tipo, id_estrella, PLOTS_DIR / nombre,
                )
                n_ejemplos_graficados += 1

    if filas_nuevas:
        cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
        cache.to_csv(CACHE_CSV, index=False)

    if GENERAR_GRAFICAS:
        graficar_resumen(cache)

    imprimir_resumen(cache)


if __name__ == "__main__":
    main()
