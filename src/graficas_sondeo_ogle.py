"""
Genera las figuras (PNG) del sondeo de ogle_collection para uso en el
documento de la Tesis, a partir de las tablas CSV producidas por
sondeo_ogle_collection.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = PROJECT_DIR / "Sondeo_OGLE" / "tablas"
FIG_DIR = PROJECT_DIR / "Sondeo_OGLE" / "graficas"

BINARIA, CEFEIDA, RRLYRAE = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def figura_totales():
    df = pd.read_csv(TABLAS_DIR / "02_totales_por_tipo.csv")
    colores = {"Binaria": BINARIA, "Cefeida": CEFEIDA, "RR Lyrae": RRLYRAE}

    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = range(len(df))
    ax.barh(y, df["n_catalogo"], color=[colores[t] for t in df["tipo"]], height=0.55, label="Catalogo")
    ax.barh(y, df["n_con_fotometria"], color="white", edgecolor="#333333", height=0.22, hatch="", zorder=3)

    for i, row in df.iterrows():
        ax.text(row["n_catalogo"] + 700, i, f"{row['n_catalogo']:,}", va="center", fontsize=10)
        pct = 100 * row["n_con_fotometria"] / row["n_catalogo"]
        ax.text(row["n_con_fotometria"] / 2, i, f"{pct:.1f}% con fotometria", va="center", ha="center",
                fontsize=8.5, color="#222222")

    ax.set_yticks(y, df["tipo"])
    ax.set_xlabel("Numero de estrellas")
    ax.set_title("Estrellas variables catalogadas en ogle_collection\n(barra completa = catalogo; barra interior = con fotometria disponible)",
                 fontsize=11, loc="left")
    ax.set_xlim(0, df["n_catalogo"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "totales_por_tipo.png", dpi=300)
    plt.close(fig)


def figura_campo():
    df = pd.read_csv(TABLAS_DIR / "01_distribucion_por_campo.csv")
    cefeidas = df[df["tipo"] == "Cefeida"].sort_values("n_catalogo", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.barh(cefeidas["region"], cefeidas["n_catalogo"], color=CEFEIDA, height=0.55)
    for i, (_, row) in enumerate(cefeidas.iterrows()):
        ax.text(row["n_catalogo"] + 60, i, f"{row['n_catalogo']:,}", va="center", fontsize=10)

    ax.set_xlabel("Numero de Cefeidas clasicas")
    ax.set_title("Distribucion de Cefeidas clasicas", fontsize=12, loc="left")
    ax.set_xlim(0, cefeidas["n_catalogo"].max() * 1.2)

    nota = ("Binarias (40 204) y RR Lyrae (41 471): 100% en la Nube Mayor de Magallanes (LMC).\n"
            "Los campos GD y BLG son Cefeidas de disco galactico (poblacion I) vistas hacia el disco y el bulbo;\n"
            "el bulbo (poblacion vieja) no aloja Cefeidas clasicas.")
    fig.tight_layout(rect=(0, 0.24, 1, 1))
    fig.text(0.02, 0.17, nota, fontsize=8.3, color="#444444", va="top")
    fig.savefig(FIG_DIR / "distribucion_por_campo.png", dpi=300)
    plt.close(fig)


def figura_modos_cefeidas():
    df = pd.read_csv(TABLAS_DIR / "03_cefeidas_por_modo.csv").sort_values("n_catalogo", ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.barh(df["descripcion"], df["n_catalogo"], color=CEFEIDA, height=0.6)
    ax.set_xscale("log")
    for i, (_, row) in enumerate(df.iterrows()):
        pct = 100 * row["n_catalogo"] / df["n_catalogo"].sum()
        ax.text(row["n_catalogo"] * 1.15, i, f"{row['n_catalogo']:,}  ({pct:.1f}%)", va="center", fontsize=9,
                ha="left", clip_on=False)

    ax.set_xlabel("Numero de Cefeidas (escala logaritmica)")
    ax.set_title("Cefeidas clasicas por modo de pulsacion", fontsize=12, loc="left")
    ax.set_xlim(0.5, df["n_catalogo"].max() * 12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cefeidas_por_modo.png", dpi=300)
    plt.close(fig)


def figura_tipo_binarias():
    df = pd.read_csv(TABLAS_DIR / "04_binarias_por_tipo_morfologico.csv").sort_values("n_catalogo", ascending=True)
    etiquetas = {
        "NC": "No-contacto\n(separada o semi-separada)",
        "ELL": "Elipsoidal\n(sin eclipses claros)",
        "C": "Contacto",
    }

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh([etiquetas[t] for t in df["tipo_morfologico"]], df["n_catalogo"], color=BINARIA, height=0.55)
    for i, (_, row) in enumerate(df.iterrows()):
        pct = 100 * row["n_catalogo"] / df["n_catalogo"].sum()
        ax.text(row["n_catalogo"] + 500, i, f"{row['n_catalogo']:,}  ({pct:.1f}%)", va="center", fontsize=10)

    ax.set_xlabel("Numero de binarias")
    ax.set_title("Binarias eclipsantes por tipo morfologico", fontsize=12, loc="left")
    ax.set_xlim(0, df["n_catalogo"].max() * 1.35)

    nota = ("El catalogo OGLE no usa la nomenclatura Algol / Beta Lyrae / W Ursae Majoris (EA/EB/EW);\n"
            "clasifica por morfologia en NC, C y ELL. NC y C se aproximan a Algol/Beta Lyrae y W UMa,\n"
            "respectivamente, pero no son equivalentes uno a uno.")
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    fig.text(0.02, 0.13, nota, fontsize=8.3, color="#444444", va="top")
    fig.savefig(FIG_DIR / "binarias_por_tipo_morfologico.png", dpi=300)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figura_totales()
    figura_campo()
    figura_modos_cefeidas()
    figura_tipo_binarias()
    print(f"Figuras guardadas en {FIG_DIR}")


if __name__ == "__main__":
    main()
