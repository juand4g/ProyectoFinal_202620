"""
Distribucion de periodos de referencia de la muestra TESS (cache_tess.csv,
producida por prueba_tess.py):
  1. Por grupo morfologico (binarias separadas, intermedias y de
     contacto/elipsoidales segun el parametro morfologico de Prsa et al. 2022;
     Cefeidas y RR Lyrae de VSX).
  2. Solo binarias de contacto/elipsoidales (morph >= 0.7): histograma del
     periodo y periodo frente al parametro morfologico.

Los periodos son los de catalogo (Prsa et al. 2022 y VSX), no los medidos.
La muestra se recorto a 0.2-10 d (binarias) y 1-10 d (Cefeidas), asi que la
forma de las distribuciones refleja tambien esos cortes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TESS_DIR = BASE_DIR / "TESS_Prueba"
CACHE_CSV = TESS_DIR / "cache_tess.csv"
FIG_DIR = TESS_DIR / "graficas"

BINARIA, CEFEIDA, RRLYRAE = "#2a78d6", "#eb6834", "#1baf7a"
GRUPOS = [
    ("Binaria separada", "Binarias separadas\n(morph < 0.4)", BINARIA),
    ("Binaria intermedia", "Binarias intermedias\n(0.4 ≤ morph < 0.7)", BINARIA),
    ("Binaria contacto/ELL", "Binarias de contacto /\nelipsoidales (morph ≥ 0.7)", BINARIA),
    ("Cefeida", "Cefeidas clásicas\n(DCEP, DCEPS)", CEFEIDA),
    ("RR Lyrae", "RR Lyrae\n(RRAB, RRC)", RRLYRAE),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def figura_por_morfologia(df):
    bins = np.logspace(np.log10(0.1), np.log10(12), 41)
    fig, axes = plt.subplots(len(GRUPOS), 1, figsize=(7, 8.5), sharex=True)
    for ax, (grupo, etiqueta, color) in zip(axes, GRUPOS):
        p = df.loc[df["grupo"] == grupo, "periodo_referencia"]
        ax.hist(p, bins=bins, color=color, edgecolor="white", linewidth=1)
        mediana = np.median(p)
        ax.axvline(mediana, color="#222222", lw=1, ls="--")
        ax.text(0.99, 0.92, f"N = {len(p)}\nmediana = {mediana:.2f} d", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#222222",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.text(0.01, 0.92, etiqueta, transform=ax.transAxes, ha="left", va="top", fontsize=9.5)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_ylabel("N")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
    axes[-1].set_xscale("log")
    axes[-1].set_xlim(bins[0], bins[-1])
    axes[-1].set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10], ["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    axes[-1].set_xlabel("Periodo de catálogo (días)")
    fig.suptitle("Muestra TESS: distribución de periodos por morfología", x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "periodos_por_morfologia.png", dpi=300)
    plt.close(fig)


def figura_contacto_elipsoidales(df):
    sub = df[df["grupo"] == "Binaria contacto/ELL"].copy()
    sub["morph"] = sub["subtipo"].str.replace("morph=", "").astype(float)
    p = sub["periodo_referencia"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.logspace(np.log10(0.2), np.log10(10), 31)
    ax1.hist(p, bins=bins, color=BINARIA, edgecolor="white", linewidth=1)
    mediana = np.median(p)
    ax1.axvline(mediana, color="#222222", lw=1, ls="--")
    ax1.text(0.98, 0.95, f"N = {len(p)}\nmediana = {mediana:.3f} d\n"
             f"P < 1 d: {(p < 1).sum()} ({100 * (p < 1).mean():.0f}%)",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.25)
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.set_xscale("log")
    ax1.set_xticks([0.2, 0.5, 1, 2, 5, 10], ["0.2", "0.5", "1", "2", "5", "10"])
    ax1.set_xlabel("Periodo de catálogo (días)")
    ax1.set_ylabel("N")
    ax1.grid(axis="y", color="#dddddd", lw=0.6)
    ax1.set_axisbelow(True)

    ax2.scatter(sub["morph"], p, s=22, color=BINARIA, edgecolor="white", linewidth=0.6)
    ax2.set_yscale("log")
    ax2.set_yticks([0.2, 0.5, 1, 2, 5, 10], ["0.2", "0.5", "1", "2", "5", "10"])
    ax2.set_xlabel("Parámetro morfológico")
    ax2.set_ylabel("Periodo de catálogo (días)")
    ax2.grid(color="#dddddd", lw=0.6)
    ax2.set_axisbelow(True)

    fig.suptitle("Muestra TESS: binarias de contacto / elipsoidales (morph ≥ 0.7)",
                 x=0.02, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "periodos_contacto_elipsoidales.png", dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CACHE_CSV)
    figura_por_morfologia(df)
    figura_contacto_elipsoidales(df)
    print(f"Figuras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
