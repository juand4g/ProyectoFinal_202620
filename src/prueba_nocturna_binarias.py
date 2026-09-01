"""
Corrida larga (pensada para dejar corriendo toda la noche, ~8 h estimadas)
sobre MUCHAS binarias eclipsantes, comparando 4 metodos de busqueda de
periodo -- continuacion de combinacion_binarias_muestra.py (que con n=40
encontro que "producto" nunca empeora a Lomb-Scargle solo y a veces lo
mejora, y que "rank_product" tiene mas upside pero tambien mas riesgo):

  - Lomb-Scargle (astropy)
  - minima entropia (entropypf)
  - producto:      LS_norm(f) * (1 - entropia_norm(f))
  - rank_product:  percentil(LS)(f) * percentil(-entropia)(f)

Se deja fuera cociente / suma_zscores / emparejamiento_picos (ya probados
en combinacion_binarias_muestra.py, con peor o igual desempeño que estos 4).

Misma convencion de siempre: todo el trabajo (grilla, combinacion de curvas)
ocurre en FRECUENCIA; el periodo solo aparece (1) como entrada obligatoria
de get_entropies y (2) al reportar el resultado final de cada metodo.

Por cada estrella y metodo se guarda tambien el MULTIPLICADOR de alias que
minimiza el error contra el periodo catalogado (1, 1/2, 2, 1/3 o 3 -- p.ej.
multiplicador=3 significa que hubo que multiplicar el periodo encontrado por
3 para que coincidiera con el catalogo). Al final se generan:
  - un boxplot de error relativo (con alias) por metodo,
  - un histograma de la distribucion de multiplicadores por metodo,
  - una grafica de periodo encontrado (crudo, sin corregir) vs el
    multiplicador que necesito, por metodo.

CACHE: los resultados se van agregando (append) a un CSV cada
GUARDAR_CADA estrellas, no solo al final -- si el proceso se corta a medias
(o se interrumpe con Ctrl+C), lo ya procesado queda guardado en disco, y al
volver a correr el script las estrellas ya vistas no se recalculan.

Graficas de ejemplo (periodograma completo) solo se guardan para las
primeras N_EJEMPLOS_PLOT estrellas -- con miles de estrellas no tiene
sentido (ni cabe en disco) guardar una imagen por estrella.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from entropypf import get_entropies
from scipy.stats import rankdata
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PARAMETROS
# ---------------------------------------------------------------------------
# Calibrado con combinacion_binarias_muestra.py: ~1.8 s/estrella con estos
# mismos parametros (N_FREQ, L, K) en esta maquina. 16000 * 1.8 s ~= 8 h.
# Si tu maquina es mas rapida/lenta, ajusta N_MUESTRA proporcionalmente
# (el ETA de la barra de progreso de tqdm te lo confirma en los primeros
# minutos de la corrida real).
N_MUESTRA = 16_000
RANDOM_SEED = 42
ESTRELLAS_FIJAS = ["OGLE-LMC-ECL-07968", "OGLE-LMC-ECL-01807"]  # ya vistas en pruebas anteriores

P_MIN, P_MAX = 0.1, 100.0  # mismo rango acotado usado para binarias en los scripts anteriores
N_FREQ = 50_000  # grilla UNIFORME en frecuencia (ciclos/dia)
L, K = 7, 7

MIN_PUNTOS = 10
GUARDAR_CADA = 100  # estrellas entre cada volcado a disco

N_EJEMPLOS_PLOT = 8

MULTIPLICADORES = [1.0, 0.5, 2.0, 1.0 / 3.0, 3.0]
ETIQUETA_MULTIPLICADOR = {1.0: "x1 (sin alias)", 0.5: "x1/2", 2.0: "x2", 1.0 / 3.0: "x1/3", 3.0: "x3"}
ORDEN_MULTIPLICADORES = [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0]

DATA_DIR = Path(__file__).resolve().parent.parent / "ogle_collection"
OUT_DIR = Path(__file__).resolve().parent.parent / "Prueba_Nocturna_Binarias"
CACHE_CSV = OUT_DIR / "cache_binarias_nocturno.csv"
PLOTS_DIR = OUT_DIR / "graficas"

COLUMNAS_CACHE = [
    "id", "metodo", "n_puntos", "baseline_d", "periodo_catalogo",
    "periodo_encontrado", "error_relativo",
    "multiplicador_alias", "periodo_alias", "error_relativo_alias",
]

COLORES_METODO = {"Lomb-Scargle": "steelblue", "1 - entropia": "darkorange",
                   "producto": "seagreen", "rank_product": "purple"}


def cargar_catalogo():
    df = pd.read_csv(DATA_DIR / "binaries.txt", sep="\t", comment="#")
    return df.set_index("ID")["P"]


def normalizar(x):
    x_min, x_max = np.min(x), np.max(x)
    if x_max <= x_min:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def mejor_alias(periodo_encontrado, periodo_catalogo):
    candidatos = [(m, periodo_encontrado * m) for m in MULTIPLICADORES]
    errores = sorted(
        (abs(p - periodo_catalogo) / periodo_catalogo, m, p) for m, p in candidatos
    )
    error_alias, multiplicador, periodo_alias = errores[0]
    return multiplicador, periodo_alias, error_alias


def analizar_estrella(id_estrella, periodo_catalogo, t, mag, err):
    frecuencia = np.linspace(1.0 / P_MAX, 1.0 / P_MIN, N_FREQ)

    ls_potencia = LombScargle(t, mag, err).power(frecuencia)
    entropia = get_entropies(t, mag, 1.0 / frecuencia, L=L, K=K)  # unica conversion previa a periodo: entrada obligatoria de get_entropies

    ls_norm = normalizar(ls_potencia)
    entropia_norm = normalizar(entropia)
    bondad_entropia_norm = 1.0 - entropia_norm

    rango_ls = rankdata(ls_potencia) / len(ls_potencia)
    rango_ent = rankdata(-entropia) / len(entropia)

    curvas_indice = {
        "Lomb-Scargle": int(np.argmax(ls_norm)),
        "1 - entropia": int(np.argmax(bondad_entropia_norm)),
        "producto": int(np.argmax(ls_norm * bondad_entropia_norm)),
        "rank_product": int(np.argmax(rango_ls * rango_ent)),
    }

    filas = []
    for metodo, i_mejor in curvas_indice.items():
        periodo_encontrado = 1.0 / frecuencia[i_mejor]  # unica conversion a periodo del resultado
        error_relativo = abs(periodo_encontrado - periodo_catalogo) / periodo_catalogo
        multiplicador, periodo_alias, error_alias = mejor_alias(periodo_encontrado, periodo_catalogo)
        filas.append({
            "id": id_estrella, "metodo": metodo, "n_puntos": int(len(t)),
            "baseline_d": float(t.max() - t.min()), "periodo_catalogo": periodo_catalogo,
            "periodo_encontrado": periodo_encontrado, "error_relativo": error_relativo,
            "multiplicador_alias": multiplicador, "periodo_alias": periodo_alias,
            "error_relativo_alias": error_alias,
        })

    return filas, frecuencia, ls_norm, bondad_entropia_norm, curvas_indice


def graficar_ejemplo(id_estrella, periodo_catalogo, frecuencia, ls_norm, bondad_entropia_norm, curvas_indice):
    periodo = 1.0 / frecuencia
    producto = ls_norm * bondad_entropia_norm

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.3), sharex=True)
    axes[0].plot(periodo, ls_norm, lw=0.7, color=COLORES_METODO["Lomb-Scargle"])
    axes[1].plot(periodo, bondad_entropia_norm, lw=0.7, color=COLORES_METODO["1 - entropia"])
    axes[2].plot(periodo, producto, lw=0.7, color=COLORES_METODO["producto"])

    panel_de = {"Lomb-Scargle": axes[0], "1 - entropia": axes[1], "producto": axes[2], "rank_product": axes[2]}
    for ax in axes:
        ax.axvline(periodo_catalogo, color="black", ls="--", lw=1)
        ax.set_xscale("log")
    for metodo, i_mejor in curvas_indice.items():
        panel_de[metodo].axvline(periodo[i_mejor], color=COLORES_METODO[metodo], ls=":", lw=1,
                                  label=f"P {metodo} = {periodo[i_mejor]:.4f} d")
    for ax in axes:
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Periodo (d) -- eje solo para lectura, la busqueda fue en frecuencia")
    fig.suptitle(f"Binaria {id_estrella}  (P catalogo = {periodo_catalogo:.5f} d)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"ejemplo_{id_estrella}.png", dpi=150)
    plt.close(fig)


def graficar_resumen(cache):
    # --- boxplot de error relativo (con alias) por metodo -------------------
    metodos = ["Lomb-Scargle", "1 - entropia", "producto", "rank_product"]
    fig, ax = plt.subplots(figsize=(7, 5))
    datos = [cache.loc[cache["metodo"] == m, "error_relativo_alias"] for m in metodos]
    ax.boxplot(datos, showfliers=False)
    ax.set_xticks(range(1, len(metodos) + 1))
    ax.set_xticklabels(metodos, rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("Error relativo (con correccion de alias)")
    ax.set_title(f"Exactitud y precision por metodo -- {cache['id'].nunique()} binarias")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "resumen_error_por_metodo.png", dpi=150)
    plt.close(fig)

    # --- distribucion de multiplicadores por metodo -------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ancho = 0.2
    x = np.arange(len(ORDEN_MULTIPLICADORES))
    for i, metodo in enumerate(metodos):
        grupo = cache[cache["metodo"] == metodo]
        conteos = [np.mean(np.isclose(grupo["multiplicador_alias"], m)) for m in ORDEN_MULTIPLICADORES]
        ax.bar(x + (i - 1.5) * ancho, conteos, width=ancho, label=metodo, color=COLORES_METODO[metodo])
    ax.set_xticks(x)
    ax.set_xticklabels([ETIQUETA_MULTIPLICADOR[m] for m in ORDEN_MULTIPLICADORES])
    ax.set_ylabel("Fraccion de estrellas")
    ax.set_xlabel("Multiplicador necesario para igualar el periodo catalogado")
    ax.set_title(f"Distribucion de multiplicadores de alias -- {cache['id'].nunique()} binarias")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "distribucion_multiplicadores.png", dpi=150)
    plt.close(fig)

    # --- periodo encontrado (crudo) vs multiplicador necesario, por metodo --
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    rng = np.random.default_rng(RANDOM_SEED)
    for ax, metodo in zip(axes.flat, metodos):
        grupo = cache[cache["metodo"] == metodo]
        y_categoria = np.array([ORDEN_MULTIPLICADORES.index(m) for m in grupo["multiplicador_alias"]])
        jitter = rng.uniform(-0.15, 0.15, size=len(grupo))
        ax.scatter(grupo["periodo_encontrado"], y_categoria + jitter,
                   s=4, alpha=0.15, color=COLORES_METODO[metodo])
        ax.set_xscale("log")
        ax.set_yticks(range(len(ORDEN_MULTIPLICADORES)))
        ax.set_yticklabels([ETIQUETA_MULTIPLICADOR[m] for m in ORDEN_MULTIPLICADORES])
        ax.set_title(metodo, fontsize=10)
        ax.set_xlabel("Periodo encontrado, crudo (d)")
    fig.suptitle(f"Periodo encontrado vs. multiplicador de alias necesario -- {cache['id'].nunique()} binarias")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "periodo_vs_multiplicador.png", dpi=150)
    plt.close(fig)


def main():
    random.seed(RANDOM_SEED)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_CSV.exists():
        cache_previo = pd.read_csv(CACHE_CSV)
        procesados = set(cache_previo["id"])
    else:
        procesados = set()

    periodos_catalogo = cargar_catalogo()
    archivos = sorted((DATA_DIR / "binaries_phot").glob("*.dat"))
    ids_disponibles = [f.stem for f in archivos if f.stem in periodos_catalogo.index]

    resto = [i for i in ids_disponibles if i not in ESTRELLAS_FIJAS]
    n_aleatorias = min(N_MUESTRA - len(ESTRELLAS_FIJAS), len(resto))
    muestra = ESTRELLAS_FIJAS + random.sample(resto, n_aleatorias)
    muestra = [i for i in muestra if i not in procesados]

    print(f"{len(procesados)} estrellas ya en cache, {len(muestra)} por procesar de {N_MUESTRA} objetivo.")

    buffer_filas = []
    n_graficadas = 0

    def volcar_buffer():
        nonlocal buffer_filas
        if not buffer_filas:
            return
        df_nuevo = pd.DataFrame(buffer_filas, columns=COLUMNAS_CACHE)
        df_nuevo.to_csv(CACHE_CSV, mode="a", header=not CACHE_CSV.exists(), index=False)
        buffer_filas = []

    for n, id_estrella in enumerate(tqdm(muestra, desc="Binaria", unit="estrella"), start=1):
        path_dat = DATA_DIR / "binaries_phot" / f"{id_estrella}.dat"
        try:
            t, mag, err = np.loadtxt(path_dat, unpack=True)
        except (ValueError, OSError) as exc:
            warnings.warn(f"No se pudo leer {path_dat}: {exc}")
            continue
        if t.size < MIN_PUNTOS:
            continue

        periodo_catalogo = float(periodos_catalogo.loc[id_estrella])
        filas, frecuencia, ls_norm, bondad_entropia_norm, curvas_indice = analizar_estrella(
            id_estrella, periodo_catalogo, t, mag, err
        )
        buffer_filas.extend(filas)

        if n_graficadas < N_EJEMPLOS_PLOT:
            graficar_ejemplo(id_estrella, periodo_catalogo, frecuencia, ls_norm, bondad_entropia_norm, curvas_indice)
            n_graficadas += 1

        if n % GUARDAR_CADA == 0:
            volcar_buffer()

    volcar_buffer()

    cache = pd.read_csv(CACHE_CSV)
    graficar_resumen(cache)

    print(f"\nTotal en cache: {cache['id'].nunique()} binarias.")
    print("\nError relativo con alias, mediana por metodo:")
    print(cache.groupby("metodo")["error_relativo_alias"].median().sort_values().to_string(float_format=lambda x: f"{x:.5f}"))
    print(f"\nCSV: {CACHE_CSV}")
    print(f"Graficas: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
