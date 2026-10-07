"""
Figuras en formato panoramico para la presentacion de avances. Reutiliza la
paleta, la muestra (cache de refinamiento_rank_product_enmascarado_bic.py) y
las rutinas de graficas_metodo_hibrido.py; no modifica las figuras de la tesis.

Figuras (en Presentacion_Avances/img/):
  1. pres_lomb_vs_entropia.png: aciertos directos de Lomb-Scargle y de la
     minima entropia por tipo de estrella (Tablas 4.1 y 4.2 de la tesis).
  2. pres_entropia_vs_condicional.png: entropia de Shannon y entropia
     condicional de una misma binaria, sin mascara.
  3. pres_combinacion.png: LS, entropia condicional y producto de rangos de
     OGLE-LMC-ECL-38031, con las curvas plegadas de cada metodo.
  4. pres_mcnemar.png: estrellas discordantes de las dos comparaciones
     pareadas del metodo hibrido.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.stats import rankdata

from graficas_metodo_hibrido import (
    AGUA, AZUL, EJEMPLO_COMBINACION, GRIS, NARANJA, TINTA, cargar_estrella, recuadro,
)
from graficas_periodograma import handle_punto, marcar_punto_borde, panel_curva_luz
from refinamiento_rank_product_enmascarado_bic import (
    CACHE_CSV, FRECUENCIA, N_FREQ, entropias_conjunta_y_fase,
)

PRES_IMG = Path(__file__).resolve().parent.parent / "Presentacion_Avances" / "img"

plt.rcParams.update({"font.size": 17, "axes.labelsize": 18, "xtick.labelsize": 16,
                     "ytick.labelsize": 16, "legend.fontsize": 16})


def guardar(fig, nombre):
    PRES_IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PRES_IMG / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figura_lomb_vs_entropia():
    # Tasas de aciertos directos (error relativo < 1 %) sobre 500 estrellas por tipo
    tipos = ["Cefeidas clásicas", "RR Lyrae", "Binarias eclipsantes"]
    ls = [98.2, 98.4, 3.8]
    ent = [20.6, 63.0, 14.6]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    x = np.arange(len(tipos))
    ancho = 0.36
    for j, (valores, color, nombre) in enumerate(
            [(ls, AZUL, "Lomb–Scargle"), (ent, NARANJA, "Mínima entropía")]):
        barras = ax.bar(x + (j - 0.5) * ancho, valores, width=ancho - 0.03, color=color, label=nombre)
        for b, v in zip(barras, valores):
            ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=16, color=TINTA)
    ax.set_xticks(x, tipos)
    ax.set_ylabel("Aciertos directos (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right", frameon=False)
    recuadro(ax)
    guardar(fig, "pres_lomb_vs_entropia.png")


def figura_entropia_vs_condicional():
    t, mag, err = cargar_estrella(EJEMPLO_COMBINACION)
    conjunta, fase = entropias_conjunta_y_fase(t, mag)
    condicional = conjunta - fase
    visible = FRECUENCIA <= 2.2
    cache = pd.read_csv(CACHE_CSV).set_index("id")
    f_cat = 1.0 / cache.loc[EJEMPLO_COMBINACION, "periodo_catalogo"]

    fig, ejes = plt.subplots(2, 1, figsize=(16, 7.2), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    paneles = [(conjunta, NARANJA, r"$\hat H(\varphi, m)$", "Entropía de Shannon"),
               (condicional, NARANJA, r"$\hat H(m\,|\,\varphi)$", "Entropía condicional")]
    for ax, (y, color, etiqueta, titulo) in zip(ejes, paneles):
        ax.plot(FRECUENCIA[visible], y[visible], color=color, linewidth=0.6)
        f_min = FRECUENCIA[visible][np.argmin(y[visible])]
        marcar_punto_borde(ax, f_cat, "black")
        marcar_punto_borde(ax, 2 * f_cat, GRIS)
        marcar_punto_borde(ax, f_min, color)
        ax.set_ylabel(etiqueta)
        ax.set_title(titulo, loc="left", fontsize=18, pad=14)
        ax.set_xlim(0, 2.2)
        ax.legend(handles=[handle_punto("black", r"$f_{\mathrm{cat}}$"),
                           handle_punto(GRIS, r"$2f_{\mathrm{cat}}$"),
                           handle_punto(color, "mínimo")],
                  loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
        print(f"{titulo}: minimo en f={f_min:.4f} d^-1 (P={1 / f_min:.4f} d)")
    ejes[1].set_xlabel(r"Frecuencia (d$^{-1}$)")
    guardar(fig, "pres_entropia_vs_condicional.png")


def figuras_esquema_metodos():
    """Esquemas ilustrativos (datos simulados) de los dos metodos: una curva
    plegada con un ajuste sinusoidal y la misma curva sobre una grilla L x K."""
    rng = np.random.default_rng(3)
    fase = rng.uniform(0, 1, 140)
    # Subida rapida y bajada lenta, en magnitud normalizada (0 = brillante)
    perfil = np.where(fase < 0.15, 1 - fase / 0.15, (fase - 0.15) / 0.85)
    y = np.clip(perfil + rng.normal(0, 0.05, fase.size), 0, 0.999)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.linspace(0, 1, 300)
    ax.scatter(fase, y, s=28, color=AZUL, alpha=0.75)
    ax.plot(x, 0.5 - 0.42 * np.cos(2 * np.pi * (x - 0.22)), color=TINTA, linewidth=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(1.05, -0.05)
    ax.set_xlabel(r"Fase $\varphi$")
    ax.set_ylabel("Magnitud normalizada")
    ax.spines[["top", "right"]].set_visible(False)
    guardar(fig, "pres_esquema_ls.png")

    L = K = 7
    conteo, _, _ = np.histogram2d(fase, y, bins=[L, K], range=[[0, 1], [0, 1]])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    # Celdas vacias en blanco: solo se colorean las ocupadas
    ax.pcolormesh(np.linspace(0, 1, L + 1), np.linspace(0, 1, K + 1),
                  np.ma.masked_equal(conteo.T, 0), cmap="Oranges",
                  vmin=-0.15 * conteo.max(), edgecolors=TINTA, linewidth=1.2)
    ax.set_facecolor("white")
    for borde in np.linspace(0, 1, L + 1):
        ax.axvline(borde, color=TINTA, linewidth=1.2)
    for borde in np.linspace(0, 1, K + 1):
        ax.axhline(borde, color=TINTA, linewidth=1.2)
    ax.scatter(fase, y, s=18, color="#141412", alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel(r"Fase $\varphi$")
    ax.set_ylabel("Magnitud normalizada")
    guardar(fig, "pres_esquema_entropia.png")


# Semana del semestre en que se presenta (08/10/2026)
SEMANA_ACTUAL = 9


def figura_cronograma():
    """Cronograma de la propuesta (semanas planeadas) con el estado de cada
    tarea a la semana actual."""
    tareas = [  # (nombre, semana inicial, semana final, estado, etiqueta)
        ("Revisión y redacción", 1, 16, "curso", "En curso"),
        ("Caracterizar Lomb–Scargle", 2, 4, "hecha", "Completa"),
        ("Caracterizar mínima entropía", 3, 6, "hecha", "Completa"),
        ("Diseñar método combinado", 5, 7, "hecha", "Completa (binarias)"),
        ("Generar curvas sintéticas", 6, 9, "atrasada", "Atrasada"),
        ("Validar con sintéticas", 8, 11, "atrasada", "Atrasada"),
        ("Aplicar a OGLE, VVV y TESS", 10, 13, "adelantada", "Adelantada: solo OGLE"),
        ("Análisis estadístico", 12, 14, "adelantada", "Adelantada"),
        ("Informe del 30%", 8, 8, "curso", "En curso"),
        ("Documento final", 14, 16, "pendiente", "Pendiente"),
    ]
    color = {"hecha": AGUA, "curso": AZUL, "adelantada": AZUL, "atrasada": NARANJA,
             "pendiente": "#d9d8d2"}
    fig, ax = plt.subplots(figsize=(16, 9))
    for i, (nombre, ini, fin, estado, etiqueta) in enumerate(tareas):
        y = len(tareas) - 1 - i
        ax.barh(y, fin - ini + 1, left=ini - 0.5, height=0.66, color=color[estado],
                edgecolor="white", linewidth=2)
        ax.annotate(etiqueta, (16.7, y), va="center", fontsize=24, color=TINTA,
                    annotation_clip=False)
    ax.axvline(SEMANA_ACTUAL, color="#141412", linewidth=2.5, linestyle="--")
    ax.annotate(f"Hoy (semana {SEMANA_ACTUAL})", (SEMANA_ACTUAL, len(tareas) - 0.35),
                xytext=(8, 0), textcoords="offset points", fontsize=24, color="#141412")
    ax.set_yticks(range(len(tareas)), [t[0] for t in tareas][::-1], fontsize=24)
    ax.set_xticks(range(1, 17), range(1, 17), fontsize=22)
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(-0.6, len(tareas) - 0.1)
    ax.set_xlabel("Semana del semestre", fontsize=24)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    guardar(fig, "pres_cronograma.png")


def figura_shannon():
    """Solo la entropia de Shannon sin mascara, con los multiplos simples de
    un dia senalados, para la diapositiva del problema de cadencia."""
    t, mag, err = cargar_estrella(EJEMPLO_COMBINACION)
    conjunta, _ = entropias_conjunta_y_fase(t, mag)
    cache = pd.read_csv(CACHE_CSV).set_index("id")
    f_cat = 1.0 / cache.loc[EJEMPLO_COMBINACION, "periodo_catalogo"]
    visible = FRECUENCIA <= 2.2
    fig, ax = plt.subplots(figsize=(16, 5.2))
    for f in (0.5, 1.0, 1.5, 2.0):
        ax.axvline(f, color="#e4e3df", linewidth=6, zorder=0)
    ax.plot(FRECUENCIA[visible], conjunta[visible], color=NARANJA, linewidth=0.6)
    marcar_punto_borde(ax, f_cat, "black")
    marcar_punto_borde(ax, FRECUENCIA[visible][np.argmin(conjunta[visible])], NARANJA)
    ax.set_xlim(0, 2.2)
    ax.set_xlabel(r"Frecuencia (d$^{-1}$)")
    ax.set_ylabel(r"$\hat H(\varphi, m)$")
    ax.legend(handles=[handle_punto("black", r"$f_{\mathrm{cat}}$"),
                       handle_punto(NARANJA, "mínimo global")],
              loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False)
    guardar(fig, "pres_shannon.png")


def figura_combinacion():
    cache = pd.read_csv(CACHE_CSV).set_index("id")
    t, mag, err = cargar_estrella(EJEMPLO_COMBINACION)
    potencia = LombScargle(t, mag, err).power(FRECUENCIA)
    conjunta, fase = entropias_conjunta_y_fase(t, mag)
    condicional = conjunta - fase
    producto = (rankdata(potencia) / N_FREQ) * (rankdata(-condicional) / N_FREQ)
    f_cat = 1.0 / cache.loc[EJEMPLO_COMBINACION, "periodo_catalogo"]
    f_ls = FRECUENCIA[np.argmax(potencia)]
    f_rp = FRECUENCIA[np.argmax(producto)]
    visible = FRECUENCIA <= 2.6

    fig = plt.figure(figsize=(20, 9))
    rejilla = fig.add_gridspec(3, 2, width_ratios=[2.1, 1], hspace=0.55, wspace=0.28)
    paneles = [(potencia, r"$P_{LS}$", AZUL, f_ls, "Lomb–Scargle"),
               (condicional, r"$\hat H(m\,|\,\varphi)$", NARANJA,
                FRECUENCIA[np.argmin(condicional)], "Entropía condicional"),
               (producto, r"$1-R$", AGUA, f_rp, "Producto de rangos")]
    for i, (y, etiqueta, color, f_max, titulo) in enumerate(paneles):
        ax = fig.add_subplot(rejilla[i, 0])
        if i == 2:
            ax.plot(FRECUENCIA[visible], 1.0 - y[visible] + 1.0 / N_FREQ, color=color, linewidth=0.6)
            ax.set_yscale("log")
            ax.invert_yaxis()
        else:
            ax.plot(FRECUENCIA[visible], y[visible], color=color, linewidth=0.6)
        marcar_punto_borde(ax, f_cat, "black")
        marcar_punto_borde(ax, 2 * f_cat, GRIS)
        marcar_punto_borde(ax, f_max, color)
        ax.set_xlim(0, 2.6)
        ax.set_ylabel(etiqueta)
        ax.set_title(titulo, loc="left", fontsize=18, pad=12)
        if i == 0:
            ax.legend(handles=[handle_punto("black", r"$f_{\mathrm{cat}}$"),
                               handle_punto(GRIS, r"$2f_{\mathrm{cat}}$"),
                               handle_punto(TINTA, "óptimo")],
                      loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=False)
        if i == 2:
            ax.set_xlabel(r"Frecuencia (d$^{-1}$)")
    sub = rejilla[:, 1].subgridspec(2, 1, hspace=0.45)
    ax_ls = fig.add_subplot(sub[0])
    panel_curva_luz(ax_ls, t, mag, 2.0 / f_ls, AZUL, rf"LS: $2P_{{LS}}={2.0 / f_ls:.3f}$ d")
    ax_rp = fig.add_subplot(sub[1])
    panel_curva_luz(ax_rp, t, mag, 2.0 / f_rp, AGUA, rf"Combinado: $2P_0={2.0 / f_rp:.3f}$ d")
    guardar(fig, "pres_combinacion.png")


def figura_mcnemar():
    # Estrellas discordantes de la Seccion 5.4 de la tesis
    comparaciones = ["vs. Lomb–Scargle\n+ regla híbrida", "vs. producto de rangos\n+ siempre $2P_0$"]
    solo_referencia = [16, 9]
    solo_hibrido = [47, 44]
    valores_p = [r"$p = 1.2\times10^{-4}$", r"$p = 1.2\times10^{-6}$"]
    fig, ax = plt.subplots(figsize=(13, 5.4))
    y = np.arange(len(comparaciones))[::-1]
    alto = 0.34
    ax.barh(y + alto / 2, solo_hibrido, height=alto - 0.03, color=AGUA, label="Solo acierta el método híbrido")
    ax.barh(y - alto / 2, solo_referencia, height=alto - 0.03, color=GRIS, label="Solo acierta la referencia")
    for yi, a, b, p in zip(y, solo_hibrido, solo_referencia, valores_p):
        ax.annotate(str(a), (a, yi + alto / 2), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=17, color=TINTA)
        ax.annotate(str(b), (b, yi - alto / 2), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=17, color=TINTA)
        ax.annotate(p, (60, yi), va="center", fontsize=18, color=TINTA)
    ax.set_yticks(y, comparaciones)
    ax.set_xlim(0, 78)
    ax.set_xlabel("Número de binarias discordantes")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False)
    guardar(fig, "pres_mcnemar.png")


def main():
    warnings.filterwarnings("ignore")
    figura_lomb_vs_entropia()
    figura_mcnemar()
    figura_entropia_vs_condicional()
    figura_combinacion()


if __name__ == "__main__":
    main()
