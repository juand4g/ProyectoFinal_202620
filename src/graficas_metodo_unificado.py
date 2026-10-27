"""
Figuras del metodo unificado (rank_product condicional + skewness corregida +
x2/BIC) para la tesis. Solo lee caches; no recalcula periodogramas.

Figuras (en Robustez_Sinteticas/graficas/ y copiadas a Tesis/img/):
  1. unificado_skew_distribucion.png: distribucion acumulada de la skewness
     corregida por grupo, en OGLE (6000 curvas) y en TESS (720 curvas), con el
     umbral de decision.
  2. unificado_sinteticas.png: acierto final vs. SNR en curvas sinteticas,
     para cada decision de rama (x2+BIC en la rama de binarias), con todos los
     puntos y con N = 100.
  3. unificado_tess.png: acierto por grupo en TESS (LS directo, P0 directo,
     rama perfecta, metodo completo).
"""

import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from prueba_tess import CACHE_CSV as CACHE_TESS, N_POR_GRUPO
from refinamiento_subarmonicos_bic import BASE_DIR, acierto, n_hibrido
from robustez_sinteticas import (
    CACHE_CSV as CACHE_SINTETICAS, CACHE_SKEW_OGLE, OUT_DIR, SNRS, UMBRAL_SKEW, evaluar,
)

warnings.filterwarnings("ignore")

GRAF_DIR = OUT_DIR / "graficas"
TESIS_IMG = BASE_DIR / "Tesis" / "img"
H0 = 4

# Misma paleta categorica que graficas_metodo_hibrido.py (orden fijo) y tinta
# neutra para referencias.
AZUL, NARANJA, AGUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
AZUL_CLARO, AZUL_OSCURO = "#86b6ef", "#184f95"
GRIS, TINTA, REJILLA = "#9a9893", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 13, "axes.edgecolor": TINTA, "axes.labelcolor": "#0b0b0b",
                     "xtick.color": TINTA, "ytick.color": TINTA})


def guardar(fig, nombre):
    GRAF_DIR.mkdir(parents=True, exist_ok=True)
    ruta = GRAF_DIR / nombre
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(ruta, TESIS_IMG / nombre)


def recuadro(ax, eje="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=eje, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)


X_MIN = -3.0


def ecdf(ax, valores, color, etiqueta, estilo="-"):
    """Distribucion acumulada; los NaN (varianza de senal <= 0) van a la rama
    de pulsantes, como un valor -inf: se dibujan en el borde izquierdo."""
    v = np.sort(np.where(np.isnan(valores), X_MIN, valores))
    y = np.arange(1, v.size + 1) / v.size
    ax.step(v, y, where="post", color=color, linewidth=2, linestyle=estilo, label=f"{etiqueta} ($n={v.size}$)")


def figura_skew():
    ogle = pd.read_csv(CACHE_SKEW_OGLE)
    tess = pd.read_csv(CACHE_TESS)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11.5), sharex=True)  # apilados: letras legibles a ancho de pagina
    for ax, series, titulo in [
        (ax1, [(ogle[ogle["clase"] == "Binaria"], AZUL, "Binarias", "-"),
               (ogle[ogle["clase"] == "Cefeida"], NARANJA, "Cefeidas", "--"),
               (ogle[ogle["clase"] == "RR Lyrae"], AGUA, "RR Lyrae", ":")], "(a) OGLE, banda $I$"),
        (ax2, [(tess[tess["grupo"] == "Binaria separada"], AZUL_OSCURO, "Binarias separadas", "-"),
               (tess[tess["grupo"] == "Binaria intermedia"], AZUL, "Binarias intermedias", "-"),
               (tess[tess["grupo"] == "Binaria contacto/ELL"], AZUL_CLARO, "Binarias de contacto/ELL", "-"),
               (tess[tess["clase"] == "Cefeida"], NARANJA, "Cefeidas", "--"),
               (tess[tess["clase"] == "RR Lyrae"], AGUA, "RR Lyrae", ":")], "(b) TESS, un sector"),
    ]:
        for d, color, etiqueta, estilo in series:
            ecdf(ax, d["skew_corregida"].to_numpy(), color, etiqueta, estilo)
        ax.axvline(UMBRAL_SKEW, color=TINTA, linewidth=1.2)
        ax.annotate(rf"$\gamma_\star={UMBRAL_SKEW}$", (UMBRAL_SKEW, 0.03), xytext=(6, 0),
                    textcoords="offset points", color=TINTA, fontsize=15)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(X_MIN, 1e3)
        marcas = [-1, 0, 1, 10, 100, 1000]
        ax.set_xticks(marcas, [r"$-1$", "0", "1", "10", "100", "1000"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.01)
        ax.set_ylabel("Fracción acumulada", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.set_title(titulo, loc="left", fontsize=17)
        recuadro(ax, "both")
    ax1.legend(loc="lower right", frameon=False, fontsize=14)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=14)
    ax2.set_xlabel(r"Coeficiente de asimetría corregido $\hat\gamma_{\mathrm{int}}$", fontsize=16)
    fig.tight_layout(h_pad=2.0)
    guardar(fig, "unificado_skew_distribucion.png")


def figura_sinteticas():
    cache = pd.read_csv(CACHE_SINTETICAS)
    df, decisiones, ok_selector, ok_p0 = evaluar(cache)
    es_bin = (df["clase"] == "Binaria").to_numpy()
    series = [("oraculo", "Rama perfecta", TINTA, "--", "s"),
              ("skew corr.>0.5", "Asimetría corregida", AGUA, "-", "o"),
              ("skew>0.5", "Asimetría sin corregir", NARANJA, "-", "^"),
              ("logistica>0.7", "Regresión logística", AZUL, "-", "D")]
    fig, ejes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
    for col, (n_obj, nombre_n) in enumerate([(0, "todos los puntos"), (100, r"$N=100$")]):
        for fila, (mascara_clase, nombre_clase) in enumerate([(es_bin, "Binarias"), (~es_bin, "Pulsantes")]):
            ax = ejes[fila, col]
            for clave, etiqueta, color, estilo, marcador in series:
                ok = np.where(decisiones[clave], ok_selector["x2+BIC"], ok_p0)
                y = [100 * ok[mascara_clase & (df["snr"] == s).to_numpy() & (df["n_objetivo"] == n_obj).to_numpy()].mean()
                     for s in SNRS]
                ax.plot(SNRS, y, color=color, linestyle=estilo, marker=marcador, markersize=8,
                        linewidth=2, label=etiqueta)
            ax.set_xscale("log")
            ax.set_xticks(SNRS, [str(s) for s in SNRS])
            ax.tick_params(labelbottom=True)  # sharex los oculta en la fila superior
            ax.minorticks_off()
            ax.set_ylim(0, 102)
            ax.set_title(f"({'abcd'[2 * fila + col]}) {nombre_clase}, {nombre_n}", loc="left", fontsize=15)
            recuadro(ax)
            if col == 0:
                ax.set_ylabel("Aciertos (%)")
            if fila == 1:
                ax.set_xlabel(r"Señal a ruido $A_{\mathrm{pp}}/\sigma$")
    ejes[0, 1].legend(loc="lower right", frameon=False, fontsize=12)
    fig.subplots_adjust(hspace=0.3, wspace=0.15)
    guardar(fig, "unificado_sinteticas.png")


def figura_tess():
    d = pd.read_csv(CACHE_TESS)
    d["n_hib"] = n_hibrido(d, H0, "bic_rp_c")
    ref = d["periodo_referencia"].to_numpy()
    p0 = d["periodo_rp_c"].to_numpy()
    es_bin = (d["clase"] == "Binaria").to_numpy()
    p_hib = d["n_hib"].to_numpy() * p0
    rama_bin = d["skew_corregida"].fillna(-np.inf).to_numpy() > UMBRAL_SKEW
    columnas = [("LS directo", acierto(d["periodo_ls"].to_numpy(), ref), AZUL, None),
                (r"$P_0$ directo", acierto(p0, ref), AGUA, None),
                ("Rama perfecta", acierto(np.where(es_bin, p_hib, p0), ref), "#d6d5d0", None),
                ("Método completo", acierto(np.where(rama_bin, p_hib, p0), ref), NARANJA, None)]
    grupos = list(N_POR_GRUPO)
    nombres = ["Binarias\nseparadas", "Binarias\nintermedias", "Binarias\ncontacto/ELL", "Cefeidas", "RR Lyrae"]
    x = np.arange(len(grupos))
    ancho = 0.2
    fig, ax = plt.subplots(figsize=(14, 5.5))
    for k, (etiqueta, ok, color, _) in enumerate(columnas):
        valores = [100 * ok[(d["grupo"] == g).to_numpy()].mean() for g in grupos]
        barras = ax.bar(x + (k - 1.5) * ancho, valores, width=ancho - 0.02, color=color, label=etiqueta)
        if etiqueta == "Método completo":
            for b, v in zip(barras, valores):
                ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom", fontsize=11, color=TINTA)
    ax.set_xticks(x, [f"{n}\n($n={(d['grupo'] == g).sum()}$)" for n, g in zip(nombres, grupos)])
    ax.set_ylabel("Aciertos (%)")
    ax.set_ylim(0, 105)
    recuadro(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    guardar(fig, "unificado_tess.png")


def main():
    figura_skew()
    figura_sinteticas()
    figura_tess()


if __name__ == "__main__":
    main()
