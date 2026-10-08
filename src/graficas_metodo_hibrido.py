"""
Figuras del metodo hibrido (rank_product condicional + refinamiento de
subarmonicos por BIC) para la tesis. Lee el cache de
refinamiento_rank_product_enmascarado_bic.py (1000 binarias, semilla 2026) y
recalcula los periodogramas completos solo para las dos estrellas de ejemplo.

Figuras (en Refinamiento_Subarmonicos/graficas/ y copiadas a Tesis/img/):
  1. hibrido_ejemplo_combinacion.png: LS, entropia condicional y rank_product
     de una binaria en la que LS cae en un armonico alto y la combinacion no.
  2. hibrido_ejemplo_bic.png: Delta BIC vs. n y curvas plegadas en 2*P0 y en
     el n elegido, para una binaria en la que el BIC sube a n=6.
  3. hibrido_resumen_etapas.png: aciertos por etapa (directo, siempre x2,
     hibrido, techo x1-x6) para LS, entropia condicional y rank_product.
  4. hibrido_aciertos_por_periodo.png: aciertos del hibrido por rango de
     periodo catalogado, LS vs. rank_product condicional, con el techo.
  5. hibrido_matriz_confusion.png: n real vs. n elegido por la regla hibrida.
"""

import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.stats import rankdata

from graficas_periodograma import handle_punto, marcar_punto_borde, panel_curva_luz
from refinamiento_rank_product_enmascarado_bic import (
    CACHE_CSV, FRECUENCIA, N_FREQ, entropias_conjunta_y_fase,
)
from refinamiento_subarmonicos_bic import (
    BASE_DIR, N_MAX, OUT_DIR, PHOT_DIR, acierto, bic_fourier, n_hibrido,
)

GRAF_DIR = OUT_DIR / "graficas"
TESIS_IMG = BASE_DIR / "Tesis" / "img"
H0 = 4

# Paleta categorica de referencia, en orden fijo (azul, naranja, aguamarina)
# y tinta neutra para lo que no es una serie.
AZUL, NARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
GRIS, TINTA = "#9a9893", "#52514e"
COLOR_METODO = {"ls": AZUL, "ent_c": NARANJA, "rp_c": AGUA}
NOMBRE_METODO = {"ls": "Lomb–Scargle", "ent_c": "Entropía condicional",
                 "rp_c": "Producto de rangos"}

EJEMPLO_COMBINACION = "OGLE-LMC-ECL-11394"
F_MAX_EJEMPLO = 1.5  # d^-1, contiene f_cat, 2 f_cat y el maximo de LS
EJEMPLO_BIC = "OGLE-LMC-ECL-14782"


def guardar(fig, nombre):
    GRAF_DIR.mkdir(parents=True, exist_ok=True)
    ruta = GRAF_DIR / nombre
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(ruta, TESIS_IMG / nombre)


def cargar_estrella(id_estrella):
    t, mag, err = np.loadtxt(PHOT_DIR / f"{id_estrella}.dat", unpack=True)
    err = np.where(err > 0, err, np.median(err[err > 0]))
    return t, mag, err


def recuadro(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def figura_combinacion(cache):
    fila = cache.set_index("id").loc[EJEMPLO_COMBINACION]
    t, mag, err = cargar_estrella(EJEMPLO_COMBINACION)
    potencia = LombScargle(t, mag, err).power(FRECUENCIA)
    conjunta, fase = entropias_conjunta_y_fase(t, mag)
    condicional = conjunta - fase
    producto = (rankdata(potencia) / N_FREQ) * (rankdata(-condicional) / N_FREQ)

    f_cat = 1.0 / fila["periodo_catalogo"]
    f_ls = FRECUENCIA[np.argmax(potencia)]
    f_rp = FRECUENCIA[np.argmax(producto)]
    visible = FRECUENCIA <= F_MAX_EJEMPLO

    fig = plt.figure(figsize=(14, 15))
    rejilla = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.25], hspace=0.45, wspace=0.3)
    paneles = [
        (potencia, r"$P_{LS}(f)$", AZUL, f_ls, "(a) Lomb–Scargle"),
        (condicional, r"$\hat H(m\,|\,\varphi)$", NARANJA,
         FRECUENCIA[np.argmin(condicional)], "(b) Entropía condicional"),
        (producto, r"$1-R(f)$", AGUA, f_rp, "(c) Producto de rangos"),
    ]
    for i, (y, etiqueta, color, f_max, titulo) in enumerate(paneles):
        ax = fig.add_subplot(rejilla[i, :])
        if i == 2:
            # R(f) en [0, 1] satura visualmente con 1e5 frecuencias; 1 - R(f) en
            # escala logaritmica invertida conserva el orden y hace visible el maximo.
            ax.plot(FRECUENCIA[visible], 1.0 - y[visible] + 1.0 / N_FREQ, color=color, linewidth=0.6)
            ax.set_yscale("log")
            ax.invert_yaxis()
        else:
            ax.plot(FRECUENCIA[visible], y[visible], color=color, linewidth=0.6)
        marcar_punto_borde(ax, f_cat, "black")
        marcar_punto_borde(ax, 2 * f_cat, GRIS)
        marcar_punto_borde(ax, f_max, color)
        ax.set_xlim(0, F_MAX_EJEMPLO)
        ax.set_ylabel(etiqueta)
        ax.set_title(titulo, loc="left", fontsize=15, pad=16)
        if i == 2:
            ax.set_xlabel(r"Frecuencia (d$^{-1}$)")
        ax.legend(handles=[handle_punto("black", r"$f_{\mathrm{cat}}$"),
                           handle_punto(GRIS, r"$2f_{\mathrm{cat}}$"),
                           handle_punto(color, "óptimo")],
                  loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)

    ax_ls = fig.add_subplot(rejilla[3, 0])
    panel_curva_luz(ax_ls, t, mag, 2.0 / f_ls, AZUL,
                    rf"LS + híbrido: $2P_{{LS}}={2.0 / f_ls:.3f}$ d")
    ax_rp = fig.add_subplot(rejilla[3, 1])
    panel_curva_luz(ax_rp, t, mag, 2.0 / f_rp, AGUA,
                    rf"Combinado + híbrido: $2P_0={2.0 / f_rp:.3f}$ d")
    guardar(fig, "hibrido_ejemplo_combinacion.png")
    print(f"{EJEMPLO_COMBINACION}: P_cat={fila['periodo_catalogo']:.5f}, "
          f"P_LS={1 / f_ls:.5f}, P_ent_c={1 / FRECUENCIA[np.argmin(condicional)]:.5f}, P0={1 / f_rp:.5f}")


def figura_bic(cache):
    fila = cache.set_index("id").loc[EJEMPLO_BIC]
    t, mag, err = cargar_estrella(EJEMPLO_BIC)
    p0 = fila["periodo_rp_c"]
    bics = np.array([bic_fourier(t, mag, err, 1.0 / (n * p0), n * H0) for n in range(1, N_MAX + 1)])
    delta = bics - bics[1]
    n_elegido = int(np.argmin(np.where(np.arange(1, N_MAX + 1) >= 3, bics, np.inf)) + 1)
    if bics[n_elegido - 1] >= bics[1]:
        n_elegido = 2

    fig = plt.figure(figsize=(14, 9.5))
    rejilla = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(rejilla[0, :])
    colores = [GRIS if n == 1 else (AGUA if n == n_elegido else AZUL) for n in range(1, N_MAX + 1)]
    ax.bar(np.arange(1, N_MAX + 1), delta, width=0.6, color=colores)
    ax.axhline(0, color=TINTA, linewidth=1)
    for n, d in zip(range(1, N_MAX + 1), delta):
        ax.annotate(f"{d:+.0f}", (n, d), xytext=(0, 6 if d >= 0 else -6), textcoords="offset points",
                    ha="center", va="bottom" if d >= 0 else "top", fontsize=13, color=TINTA)
    ax.set_xticks(np.arange(1, N_MAX + 1), [rf"$n={n}$" for n in range(1, N_MAX + 1)])
    ax.set_ylabel(r"$\mathrm{BIC}_n-\mathrm{BIC}_2$")
    ax.set_title(r"(a) Diferencia de BIC respecto de $n=2$ ($H_0=4$)", loc="left", fontsize=15)
    margen = 0.18 * (delta.max() - delta.min())
    ax.set_ylim(delta.min() - margen, delta.max() + margen)
    recuadro(ax)

    ax_2 = fig.add_subplot(rejilla[1, 0])
    panel_curva_luz(ax_2, t, mag, 2 * p0, AZUL, rf"(b) $2P_0={2 * p0:.3f}$ d")
    ax_n = fig.add_subplot(rejilla[1, 1])
    panel_curva_luz(ax_n, t, mag, n_elegido * p0, AGUA, rf"(c) ${n_elegido}P_0={n_elegido * p0:.3f}$ d")
    guardar(fig, "hibrido_ejemplo_bic.png")
    print(f"{EJEMPLO_BIC}: P_cat={fila['periodo_catalogo']:.5f}, P0={p0:.5f}, n={n_elegido}, "
          f"delta BIC={np.round(delta, 1)}")


def tasas(cache):
    P = cache["periodo_catalogo"].to_numpy()
    salida = {}
    for m in COLOR_METODO:
        p0 = cache[f"periodo_{m}"].to_numpy()
        oraculo = np.zeros(len(cache), dtype=bool)
        for n in range(1, N_MAX + 1):
            oraculo |= acierto(n * p0, P)
        salida[m] = {
            "directo": acierto(p0, P),
            "siempre2": acierto(2 * p0, P),
            "hibrido": acierto(n_hibrido(cache, H0, f"bic_{m}") * p0, P),
            "oraculo": oraculo,
        }
    return salida


def figura_etapas(cache):
    r = tasas(cache)
    etapas = [("directo", r"Directo ($P_0$)"), ("siempre2", r"Siempre $2P_0$"),
              ("hibrido", "Regla híbrida\n(BIC)"), ("oraculo", "Techo\n$\\times1$–$\\times6$")]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ancho = 0.26
    x = np.arange(len(etapas))
    for j, m in enumerate(COLOR_METODO):
        valores = [100 * r[m][e].mean() for e, _ in etapas]
        barras = ax.bar(x + (j - 1) * ancho, valores, width=ancho - 0.03,
                        color=COLOR_METODO[m], label=NOMBRE_METODO[m])
        for b, v in zip(barras, valores):
            ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=11.5, color=TINTA)
    ax.axvspan(2.5, 3.5, color="#f1f0ec", zorder=0)
    ax.set_xticks(x, [e for _, e in etapas])
    ax.set_ylabel("Aciertos (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False)
    recuadro(ax)
    guardar(fig, "hibrido_resumen_etapas.png")


def figura_por_periodo(cache):
    r = tasas(cache)
    P = cache["periodo_catalogo"].to_numpy()
    bordes = [0, 1, 3, 10, 30, 100, 1e4]
    etiquetas = [r"$\leq 1$", "1–3", "3–10", "10–30", "30–100", r"$>100$"]
    rango = pd.cut(P, bordes, labels=False)
    conteo = np.bincount(rango, minlength=len(etiquetas))
    series = [("ls", "hibrido", AZUL, "Lomb–Scargle + regla híbrida"),
              ("rp_c", "hibrido", AGUA, "Producto de rangos + regla híbrida")]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(etiquetas))
    ancho = 0.36
    for j, (m, e, color, nombre) in enumerate(series):
        valores = [100 * r[m][e][rango == k].mean() for k in range(len(etiquetas))]
        ax.bar(x + (j - 0.5) * ancho, valores, width=ancho - 0.03, color=color, label=nombre)
    techo = [100 * r["rp_c"]["oraculo"][rango == k].mean() for k in range(len(etiquetas))]
    ax.scatter(x + 0.5 * ancho, techo, marker="_", s=900, linewidths=2.5, color=TINTA,
               label=r"Techo del producto de rangos ($\times1$–$\times6$)", zorder=4)
    ax.set_xticks(x, [f"{e}\n($N={c}$)" for e, c in zip(etiquetas, conteo)])
    ax.set_xlabel(r"Periodo catalogado $P_{\mathrm{cat}}$ (d)")
    ax.set_ylabel("Aciertos (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=False)
    recuadro(ax)
    guardar(fig, "hibrido_aciertos_por_periodo.png")


def figura_confusion(cache):
    P = cache["periodo_catalogo"].to_numpy()
    p0 = cache["periodo_rp_c"].to_numpy()
    n_real = np.zeros(len(cache), dtype=int)
    for n in range(N_MAX, 0, -1):
        n_real[acierto(n * p0, P)] = n
    n_eleg = n_hibrido(cache, H0, "bic_rp_c")
    filas = [1, 2, 3, 4, 5, 6, 0]
    matriz = np.array([[np.sum((n_real == a) & (n_eleg == b)) for b in range(1, N_MAX + 1)] for a in filas])
    fig, ax = plt.subplots(figsize=(9, 8))
    imagen = ax.imshow(np.log10(matriz + 1), cmap="Blues", aspect="auto")
    for i in range(len(filas)):
        for j in range(N_MAX):
            v = matriz[i, j]
            if v:
                oscuro = np.log10(v + 1) > 0.6 * np.log10(matriz.max() + 1)
                ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                        color="white" if oscuro else "black",
                        fontweight="bold" if filas[i] == j + 1 else "normal")
    ax.set_xticks(range(N_MAX), [rf"$\times{n}$" for n in range(1, N_MAX + 1)])
    ax.set_yticks(range(len(filas)), [rf"$\times{n}$" for n in filas[:-1]] + ["ninguno"])
    ax.set_xlabel(r"Múltiplo elegido $\hat n$")
    ax.set_ylabel(r"Múltiplo real $n^\ast$ ($P_{\mathrm{cat}}\approx n^\ast P_0$)")
    ax.axhline(5.5, color=TINTA, linewidth=1.2)
    barra = fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    barra.set_label(r"$\log_{10}(1+\mathrm{conteo})$")
    guardar(fig, "hibrido_matriz_confusion.png")


def main():
    warnings.filterwarnings("ignore")
    cache = pd.read_csv(CACHE_CSV)
    figura_etapas(cache)
    figura_por_periodo(cache)
    figura_confusion(cache)
    figura_bic(cache)
    figura_combinacion(cache)


if __name__ == "__main__":
    main()
