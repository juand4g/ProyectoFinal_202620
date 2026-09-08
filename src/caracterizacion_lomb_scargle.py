"""
Caracterizacion de exactitud y precision del metodo Lomb-Scargle (astropy)
para la busqueda de periodos en Binarias, Cefeidas Clasicas y RR Lyrae
del catalogo fotometrico OGLE (banda I, tiempo en HJD).

Los resultados se guardan en un CSV que funciona como cache: al volver a
ejecutar el script, las estrellas ya procesadas no se vuelven a calcular.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from tqdm import tqdm

from graficas_periodograma import (
    agregar_lineas_alias, handle_punto, marcar_punto_borde, panel_curva_luz,
)

# ---------------------------------------------------------------------------
# PARAMETROS
# ---------------------------------------------------------------------------
N_BINARIAS = 500
N_CEFEIDAS = 500
N_RRLYRAE = 500

RANDOM_SEED = 42  # fija la muestra aleatoria para que la cache sea reutilizable

# Rango de busqueda de periodos (dias) para cada tipo de estrella.
P_MIN_BINARIAS, P_MAX_BINARIAS = 0.1, 3000.0
P_MIN_CEFEIDAS, P_MAX_CEFEIDAS = 0.2, 250.0
P_MIN_RRLYRAE, P_MAX_RRLYRAE = 0.1, 1.2

SAMPLES_PER_PEAK = 5  # densidad de la grilla de frecuencias (astropy autopower)
MIN_PUNTOS = 10  # curvas con menos puntos que esto se descartan

GENERAR_GRAFICAS = True
N_EJEMPLOS_PERIODOGRAMA = 3  # numero de periodogramas de ejemplo a graficar por tipo

DATA_DIR = Path(__file__).resolve().parent.parent / "ogle_collection"
OUT_DIR = Path(__file__).resolve().parent.parent / "Lomb-Scargle"
CACHE_CSV = OUT_DIR / "cache_resultados_ls.csv"
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
    "periodo_catalogo", "periodo_ls", "potencia_ls",
    "error_relativo", "periodo_ls_alias", "error_relativo_alias",
]


def cargar_catalogo(path_txt, col_periodo):
    df = pd.read_csv(path_txt, sep="\t", comment="#")
    df = df.set_index("ID")
    return df[col_periodo]


def calcular_periodo_ls(t, mag, err, p_min, p_max, samples_per_peak=5):
    ls = LombScargle(t, mag, err)
    frecuencia, potencia = ls.autopower(
        minimum_frequency=1.0 / p_max,
        maximum_frequency=1.0 / p_min,
        samples_per_peak=samples_per_peak,
    )
    i_mejor = np.argmax(potencia)
    periodo_ls = 1.0 / frecuencia[i_mejor]
    return periodo_ls, potencia[i_mejor], frecuencia, potencia


def mejor_alias(periodo_ls, periodo_catalogo):
    """El metodo Lomb-Scargle suele confundir el periodo real con sus
    armonicos (P/2, 2P, P/3, 3P), en particular en binarias eclipsantes.
    Se reporta tambien el alias mas cercano al periodo catalogado."""
    candidatos = [periodo_ls, periodo_ls / 2, periodo_ls * 2, periodo_ls / 3, periodo_ls * 3]
    errores = [abs(p - periodo_catalogo) / periodo_catalogo for p in candidatos]
    i = int(np.argmin(errores))
    return candidatos[i], errores[i]


def graficar_periodograma(t, mag, frecuencia, potencia, periodo_catalogo, periodo_ls, tipo, id_estrella, outfile):
    """Figura de 2 filas: el frecuenciograma de Lomb-Scargle ocupa toda la
    fila superior, y la curva de luz real (plegada con el periodo catalogado)
    y la curva de luz encontrada (plegada con el periodo de Lomb-Scargle) se
    muestran en paneles separados, uno al lado del otro, en la fila inferior
    -- nunca superpuestas en un mismo panel. El periodo catalogado y el de
    Lomb-Scargle se marcan en el periodograma con parejas de triangulos en el
    borde del recuadro, para no tapar el pico con una linea vertical."""
    f_catalogo = 1.0 / periodo_catalogo
    f_ls = 1.0 / periodo_ls

    fig = plt.figure(figsize=(8.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_periodograma = fig.add_subplot(gs[0, :])
    ax_lc_cat = fig.add_subplot(gs[1, 0])
    ax_lc_enc = fig.add_subplot(gs[1, 1])

    ax_periodograma.plot(frecuencia, potencia, lw=0.8, color="steelblue")
    marcar_punto_borde(ax_periodograma, f_catalogo, color="green")
    marcar_punto_borde(ax_periodograma, f_ls, color="red")
    ax_periodograma.set_xscale("log")
    ax_periodograma.set_xlabel("Frecuencia (d$^{-1}$)")
    ax_periodograma.set_ylabel("Potencia LS")
    ax_periodograma.legend(handles=[
        handle_punto("green", f"f catalogo = {f_catalogo:.5f} d$^{{-1}}$ (P = {periodo_catalogo:.5f} d)"),
        handle_punto("red", f"f Lomb-Scargle = {f_ls:.5f} d$^{{-1}}$ (P = {periodo_ls:.5f} d)"),
    ], loc="best")

    panel_curva_luz(ax_lc_cat, t, mag, periodo_catalogo, color="green",
                     titulo="Curva real")
    panel_curva_luz(ax_lc_enc, t, mag, periodo_ls, color="red",
                     titulo="Curva encontrada")

    fig.suptitle(f"{tipo} - {id_estrella}")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def regenerar_ejemplos(cache):
    """Vuelve a graficar los primeros N_EJEMPLOS_PERIODOGRAMA de cada tipo
    presentes en el cache, releyendo su fotometria original y recalculando
    el periodograma. Se hace como paso separado (no dentro del bucle
    principal) para que las graficas siempre reflejen el codigo de graficado
    vigente, incluso en corridas donde esas estrellas ya estaban en cache y
    por tanto no se reprocesan."""
    for tipo, cfg in TIPOS.items():
        candidatas = cache[(cache["tipo"] == tipo) & cache["periodo_ls"].notna()]
        ids_ejemplo = candidatas["id"].head(N_EJEMPLOS_PERIODOGRAMA).tolist()
        for id_estrella in ids_ejemplo:
            fila = candidatas[candidatas["id"] == id_estrella].iloc[0]
            path_dat = cfg["carpeta"] / f"{id_estrella}.dat"
            t, mag, err = np.loadtxt(path_dat, unpack=True)
            _, _, frecuencia, potencia = calcular_periodo_ls(
                t, mag, err, cfg["p_min"], cfg["p_max"], SAMPLES_PER_PEAK
            )
            nombre = f"periodograma_{tipo.replace(' ', '_')}_{id_estrella}.png"
            graficar_periodograma(
                t, mag, frecuencia, potencia,
                float(fila["periodo_catalogo"]), float(fila["periodo_ls"]),
                tipo, id_estrella, PLOTS_DIR / nombre,
            )


def graficar_resumen(cache):
    validos = cache.dropna(subset=["error_relativo"])
    if validos.empty:
        return

    tipos_lista = sorted(validos["tipo"].unique())
    fig = plt.figure(figsize=(8.5, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.7, 1])
    ax_box = fig.add_subplot(gs[0])
    ax_scatter = fig.add_subplot(gs[1])

    datos_boxplot = [validos.loc[validos["tipo"] == t, "error_relativo"] for t in tipos_lista]
    ax_box.boxplot(datos_boxplot, showfliers=False)
    ax_box.set_xticks(range(1, len(tipos_lista) + 1))
    ax_box.set_xticklabels(tipos_lista)
    ax_box.set_yscale("log")
    ax_box.set_ylabel("Error relativo del periodo")
    ax_box.set_title("Exactitud y precision por tipo de estrella")

    colores = {"Binaria": "tab:orange", "Cefeida": "tab:blue", "RR Lyrae": "tab:green"}
    f_catalogo = 1.0 / validos["periodo_catalogo"]
    f_ls = 1.0 / validos["periodo_ls"]
    for tipo in tipos_lista:
        mascara = validos["tipo"] == tipo
        ax_scatter.scatter(
            f_catalogo[mascara], f_ls[mascara],
            s=8, alpha=0.4, label=tipo, color=colores.get(tipo),
        )
    lims = [f_catalogo.min(), f_catalogo.max()]
    ax_scatter.plot(lims, lims, color="black", lw=0.8, ls="--", label="f_LS = f_cat")
    agregar_lineas_alias(ax_scatter, lims)
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Frecuencia catalogo (d$^{-1}$)")
    ax_scatter.set_ylabel("Frecuencia Lomb-Scargle (d$^{-1}$)")
    ax_scatter.set_title("Frecuencia recuperada vs. frecuencia catalogada")
    ax_scatter.legend(loc="lower right")

    fig.savefig(PLOTS_DIR / "resumen_error_relativo.png", dpi=150)
    plt.close(fig)


def imprimir_resumen(cache):
    validos = cache.dropna(subset=["error_relativo"])
    if validos.empty:
        print("No hay resultados validos todavia.")
        return

    print("\nResumen por tipo de estrella (error relativo del periodo):")
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

            periodo_ls, potencia_ls, frecuencia, potencia = calcular_periodo_ls(
                t, mag, err, cfg["p_min"], cfg["p_max"], SAMPLES_PER_PEAK
            )

            error_relativo = abs(periodo_ls - periodo_catalogo) / periodo_catalogo
            periodo_alias, error_alias = mejor_alias(periodo_ls, periodo_catalogo)

            filas_nuevas.append({
                "tipo": tipo,
                "id": id_estrella,
                "n_puntos": int(t.size),
                "baseline_d": float(t.max() - t.min()),
                "periodo_catalogo": periodo_catalogo,
                "periodo_ls": periodo_ls,
                "potencia_ls": float(potencia_ls),
                "error_relativo": error_relativo,
                "periodo_ls_alias": periodo_alias,
                "error_relativo_alias": error_alias,
            })
            procesados.add((tipo, id_estrella))

    if filas_nuevas:
        cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
        cache.to_csv(CACHE_CSV, index=False)

    if GENERAR_GRAFICAS:
        regenerar_ejemplos(cache)
        graficar_resumen(cache)

    imprimir_resumen(cache)


if __name__ == "__main__":
    main()
