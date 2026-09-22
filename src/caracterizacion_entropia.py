"""
Caracterizacion de exactitud y precision del metodo de minima entropia de
Shannon (entropypf; Cincotta, Mendez & Nunez 1995) para la busqueda de
periodos en Binarias, Cefeidas Clasicas y RR Lyrae del catalogo fotometrico
OGLE (banda I, tiempo en HJD).

Analogo a caracterizacion_lomb_scargle.py: misma semilla aleatoria (misma
muestra de estrellas) y mismos rangos de busqueda para Cefeidas y RR Lyrae.
Para Binarias el rango se acota respecto al usado en Lomb-Scargle (ver nota
junto a P_MAX_BINARIAS) porque con el rango completo (hasta 3000 d) el metodo
de minima entropia, tal como esta documentado (aliases=(1,365), L=K=7),
converge sistematicamente a un alias estacional de ~350-370 d en vez del
periodo orbital real -- se verifico que el efecto persiste incluso con
grillas mas finas (L=K=20), es decir, no es un problema de resolucion sino
del propio metodo frente a curvas de luz dispersas de un solo sitio con
brechas estacionales.

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
from scipy.signal import find_peaks
from tqdm import tqdm

from graficas_periodograma import (
    agregar_eje_periodo, agregar_lineas_alias, handle_punto, marcar_punto_borde,
    panel_curva_luz,
)

# ---------------------------------------------------------------------------
# PARAMETROS
# ---------------------------------------------------------------------------
N_BINARIAS = 500
N_CEFEIDAS = 500
N_RRLYRAE = 500

RANDOM_SEED = 42  # misma semilla que caracterizacion_lomb_scargle.py -> misma muestra

# Rango de busqueda de periodos (dias) para cada tipo de estrella. Cefeidas y
# RR Lyrae usan el mismo rango que caracterizacion_lomb_scargle.py. Binarias
# NO: con el rango completo (0.1-3000 d, igual que en Lomb-Scargle) el metodo
# de minima entropia converge de forma sistematica a ~350-370 d (alias
# estacional) en vez del periodo orbital real -- ver docstring del modulo.
# Se acota a 100 d, que cubre holgadamente la mediana del catalogo (3.27 d) y
# la inmensa mayoria de las binarias eclipsantes.
P_MIN_BINARIAS, P_MAX_BINARIAS = 0.1, 100.0
P_MIN_CEFEIDAS, P_MAX_CEFEIDAS = 0.2, 250.0
P_MIN_RRLYRAE, P_MAX_RRLYRAE = 0.1, 1.2

P_NUM = 10_000  # numero de periodos de prueba en la grilla no uniforme
L, K = 7, 7  # dimensiones de la grilla fase-magnitud
ALIASES = (1, 365)  # enmascara armonicos del dia sideral y del alias anual
N_CANDIDATOS = 3
MIN_PUNTOS = 10  # curvas con menos puntos que esto se descartan

# Fotometria terrestre de un solo sitio (~1 dato por noche) produce agrupa-
# miento espurio de fase (entropia baja) no solo en P=1 d, 0.5 d y 1/3 d
# (lo unico que cubre aliases=(1,...) de entropypf), sino en CUALQUIER
# fraccion simple m/n de un dia: se verifico empiricamente que la busqueda
# en binarias convergia a P=0.749 d = 3/4 d, que ningun alias estandar
# cubre. Se enmascaran entonces todas las fracciones m/n con denominador
# n <= ALIAS_DIURNO_QMAX dentro del rango de busqueda -- solo para binarias,
# que son las unicas con periodos reales de pocos dias sobre baselines de
# anios (Cefeidas y RR Lyrae ya funcionan bien con el manejo estandar).
ALIAS_DIURNO_QMAX = 6
ALIAS_DIURNO_EPS = 0.03  # dias, ventana de exclusion alrededor de cada m/n

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


def generar_alias_diurnos(p_min, p_max, q_max=ALIAS_DIURNO_QMAX):
    """Centros de exclusion en todas las fracciones simples m/n de un dia
    (n <= q_max) dentro de [p_min, p_max] -- ver nota junto a ALIAS_DIURNO_QMAX."""
    centros = set()
    for n in range(1, q_max + 1):
        m = 1
        while m / n <= p_max + ALIAS_DIURNO_EPS:
            p = m / n
            if p >= p_min - ALIAS_DIURNO_EPS:
                centros.add(round(p, 6))
            m += 1
    return np.array(sorted(centros))


def calcular_periodo_entropia_binaria(t, mag, p_min, p_max):
    """Version para binarias: enmascara fracciones simples de un dia ademas
    de aplicar find_peaks sobre la grilla de entropypf (ver ALIAS_DIURNO_QMAX)."""
    periodos_prueba = get_test_periods(p_min, p_max, P_NUM)

    centros = generar_alias_diurnos(p_min, p_max)
    mascara = (np.abs(periodos_prueba[:, None] - centros[None, :]) < ALIAS_DIURNO_EPS).any(axis=1)
    periodos_prueba = periodos_prueba[~mascara]

    entropias = get_entropies(t, mag, periodos_prueba, L=L, K=K)

    picos, _ = find_peaks(-entropias, prominence=0.01, distance=30)
    if len(picos) == 0:
        picos = [int(np.argmin(entropias))]

    candidatos_p = periodos_prueba[picos]
    candidatos_s = entropias[picos]
    orden = np.argsort(candidatos_s)[:N_CANDIDATOS]
    return float(candidatos_p[orden][0]), float(candidatos_s[orden][0])


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
    """Figura de 2 filas: el frecuenciograma de entropia ocupa toda la fila
    superior, y la curva de luz real (plegada con el periodo catalogado) y la
    curva de luz encontrada (plegada con el periodo de minima entropia) se
    muestran en paneles separados, uno al lado del otro, en la fila inferior
    -- nunca superpuestas en un mismo panel. La entropia se grafica tal cual:
    el minimo global es un minimo real de la curva, sin invertir el eje
    vertical para simularlo como un pico. El periodo catalogado y el de
    minima entropia se marcan en el periodograma con parejas de triangulos en
    el borde del recuadro, para no tapar el minimo con una linea vertical."""
    periodo = get_test_periods(p_min, p_max, P_NUM)
    entropia = get_entropies(t, mag, periodo, L=L, K=K)
    frecuencia = 1.0 / periodo
    f_catalogo = 1.0 / periodo_catalogo
    f_entropia = 1.0 / periodo_entropia

    fig = plt.figure(figsize=(8.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_periodograma = fig.add_subplot(gs[0, :])
    ax_lc_cat = fig.add_subplot(gs[1, 0])
    ax_lc_enc = fig.add_subplot(gs[1, 1])

    ax_periodograma.plot(frecuencia, entropia, lw=0.8, color="darkorange")
    marcar_punto_borde(ax_periodograma, f_catalogo, color="green")
    marcar_punto_borde(ax_periodograma, f_entropia, color="red")
    ax_periodograma.set_xscale("log")
    ax_periodograma.set_xlabel("Frecuencia (d$^{-1}$)")
    ax_periodograma.set_ylabel("Entropia normalizada")
    agregar_eje_periodo(ax_periodograma)
    ax_periodograma.legend(handles=[
        handle_punto("green", f"f catalogo = {f_catalogo:.5f} d$^{{-1}}$ (P = {periodo_catalogo:.5f} d)"),
        handle_punto("red", f"f entropia minima = {f_entropia:.5f} d$^{{-1}}$ (P = {periodo_entropia:.5f} d)"),
    ], loc="best")

    panel_curva_luz(ax_lc_cat, t, mag, periodo_catalogo, color="green",
                     titulo="Curva real")
    panel_curva_luz(ax_lc_enc, t, mag, periodo_entropia, color="red",
                     titulo="Curva encontrada")

    fig.suptitle(f"{tipo} - {id_estrella}")
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def regenerar_ejemplos(cache):
    """Vuelve a graficar los primeros N_EJEMPLOS_PERIODOGRAMA de cada tipo
    presentes en el cache, releyendo su fotometria original. Se hace como
    paso separado (no dentro del bucle principal) para que las graficas
    siempre reflejen el codigo de graficado vigente, incluso en corridas
    donde esas estrellas ya estaban en cache y por tanto no se reprocesan."""
    for tipo, cfg in TIPOS.items():
        candidatas = cache[(cache["tipo"] == tipo) & cache["periodo_entropia"].notna()]
        ids_ejemplo = candidatas["id"].head(N_EJEMPLOS_PERIODOGRAMA).tolist()
        for id_estrella in ids_ejemplo:
            fila = candidatas[candidatas["id"] == id_estrella].iloc[0]
            path_dat = cfg["carpeta"] / f"{id_estrella}.dat"
            t, mag, err = np.loadtxt(path_dat, unpack=True)
            nombre = f"periodograma_{tipo.replace(' ', '_')}_{id_estrella}.png"
            graficar_periodograma(
                t, mag, cfg["p_min"], cfg["p_max"],
                float(fila["periodo_catalogo"]), float(fila["periodo_entropia"]),
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
    ax_box.set_title("Exactitud y precision por tipo de estrella (minima entropia)")

    colores = {"Binaria": "tab:orange", "Cefeida": "tab:blue", "RR Lyrae": "tab:green"}
    f_catalogo = 1.0 / validos["periodo_catalogo"]
    f_entropia = 1.0 / validos["periodo_entropia"]
    for tipo in tipos_lista:
        mascara = validos["tipo"] == tipo
        ax_scatter.scatter(
            f_catalogo[mascara], f_entropia[mascara],
            s=8, alpha=0.4, color=colores.get(tipo),
        )
    lims = [f_catalogo.min(), f_catalogo.max()]
    ax_scatter.plot(lims, lims, color="black", lw=0.8, ls="--")
    agregar_lineas_alias(ax_scatter, lims)
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Frecuencia catalogo (d$^{-1}$)")
    ax_scatter.set_ylabel("Frecuencia minima entropia (d$^{-1}$)")
    ax_scatter.set_title("Frecuencia recuperada vs. frecuencia catalogada")
    # Sin leyenda dentro de los ejes (tapaba puntos): el significado de
    # colores y lineas se explica en el caption de la figura (ver Tesis).

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

            if tipo == "Binaria":
                periodo_entropia, entropia_minima = calcular_periodo_entropia_binaria(
                    t, mag, cfg["p_min"], cfg["p_max"]
                )
            else:
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

    if filas_nuevas:
        cache = pd.concat([cache, pd.DataFrame(filas_nuevas)], ignore_index=True)
        cache.to_csv(CACHE_CSV, index=False)

    if GENERAR_GRAFICAS:
        regenerar_ejemplos(cache)
        graficar_resumen(cache)

    imprimir_resumen(cache)


if __name__ == "__main__":
    main()
