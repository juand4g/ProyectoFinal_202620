"""
Genera las figuras (PNG) del sondeo de ogle_collection para el Capitulo 2 de
la tesis, a partir de las tablas CSV producidas por sondeo_ogle_collection.py.

Las etiquetas visibles se traducen aqui (con tildes y la nomenclatura de la
tesis: "sobretono", no "armonico"); las explicaciones van en los pies de
figura de main.tex, no dentro de la imagen.

Figuras (Sondeo_OGLE/graficas/, copiadas a Tesis/img/ con prefijo ogle_):
  totales_por_tipo.png, distribucion_por_campo.png, cefeidas_por_modo.png,
  binarias_por_tipo_morfologico.png
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

PROJECT_DIR = Path(__file__).resolve().parent.parent
TABLAS_DIR = PROJECT_DIR / "Sondeo_OGLE" / "tablas"
FIG_DIR = PROJECT_DIR / "Sondeo_OGLE" / "graficas"
TESIS_IMG = PROJECT_DIR / "Tesis" / "img"

BINARIA, CEFEIDA, RRLYRAE = "#2a78d6", "#eb6834", "#1baf7a"
TINTA = "#444444"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": TINTA,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

TIPO = {"Binaria": "Binarias eclipsantes", "Cefeida": "Cefeidas clásicas", "RR Lyrae": "RR Lyrae"}
REGION = {
    "LMC": "Nube Mayor de Magallanes",
    "SMC": "Nube Menor de Magallanes",
    "GD": "Disco de la Vía Láctea",
    "BLG": "Campos hacia el bulbo\nde la Vía Láctea",
}
MODO = {
    "F": "Fundamental (F)",
    "1": "Primer sobretono (1O)",
    "2": "Segundo sobretono (2O)",
    "F1": "Doble modo: F + 1O",
    "12": "Doble modo: 1O + 2O",
    "13": "Doble modo: 1O + 3O",
    "23": "Doble modo: 2O + 3O",
    "F12": "Triple modo: F + 1O + 2O",
    "123": "Triple modo: 1O + 2O + 3O",
}
MORFOLOGIA = {
    "NC": "No-contacto (NC)",
    "C": "Contacto (C)",
    "ELL": "Elipsoidal (ELL)",
}


def miles(n):
    """Separador de miles con espacio fino, como en el texto en espanol."""
    return f"{int(n):,}".replace(",", " ")


EJE_MILES = FuncFormatter(lambda x, _: miles(x))


def guardar(fig, nombre, nombre_tesis=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIG_DIR / nombre
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    shutil.copy(ruta, TESIS_IMG / (nombre_tesis or f"ogle_{nombre}"))


def figura_totales():
    df = pd.read_csv(TABLAS_DIR / "02_totales_por_tipo.csv")
    colores = {"Binaria": BINARIA, "Cefeida": CEFEIDA, "RR Lyrae": RRLYRAE}
    fig, ax = plt.subplots(figsize=(7, 3.0))
    y = range(len(df))
    ax.barh(y, df["n_catalogo"], color=[colores[t] for t in df["tipo"]], height=0.55, label="En el catálogo")
    ax.barh(y, df["n_con_fotometria"], color="white", edgecolor="#333333", height=0.22, zorder=3,
            label="Con fotometría disponible")
    for i, (_, fila) in enumerate(df.iterrows()):
        pct = 100 * fila["n_con_fotometria"] / fila["n_catalogo"]
        ax.text(fila["n_catalogo"] + 600, i, f"{miles(fila['n_catalogo'])}  ({pct:.1f} % con fotometría)",
                va="center", fontsize=10)
    ax.set_yticks(list(y), [TIPO[t] for t in df["tipo"]])
    ax.set_xlabel("Número de estrellas")
    ax.xaxis.set_major_formatter(EJE_MILES)
    ax.set_xlim(0, df["n_catalogo"].max() * 1.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#9a9893", label="En el catálogo (color según el tipo)"),
                       Patch(facecolor="white", edgecolor="#333333", label="Con fotometría disponible")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9)
    guardar(fig, "totales_por_tipo.png")


def figura_campo():
    df = pd.read_csv(TABLAS_DIR / "01_distribucion_por_campo.csv")
    cef = df[df["tipo"] == "Cefeida"].sort_values("n_catalogo", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh([REGION[c] for c in cef["campo"]], cef["n_catalogo"], color=CEFEIDA, height=0.55)
    for i, (_, fila) in enumerate(cef.iterrows()):
        ax.text(fila["n_catalogo"] + 60, i, miles(fila["n_catalogo"]), va="center", fontsize=10)
    ax.set_xlabel("Número de Cefeidas clásicas")
    ax.xaxis.set_major_formatter(EJE_MILES)
    ax.set_xlim(0, cef["n_catalogo"].max() * 1.2)
    guardar(fig, "distribucion_por_campo.png")


def figura_modos_cefeidas():
    df = pd.read_csv(TABLAS_DIR / "03_cefeidas_por_modo.csv")
    df["modo"] = df["modo"].astype(str)
    df = df.sort_values("n_catalogo", ascending=True)
    total = df["n_catalogo"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh([MODO[m] for m in df["modo"]], df["n_catalogo"], color=CEFEIDA, height=0.6)
    ax.set_xscale("log")
    for i, (_, fila) in enumerate(df.iterrows()):
        pct = 100 * fila["n_catalogo"] / total
        etiqueta = f"{pct:.1f} %" if pct >= 0.1 else "< 0.1 %"
        ax.text(fila["n_catalogo"] * 1.15, i, f"{miles(fila['n_catalogo'])}  ({etiqueta})", va="center",
                fontsize=9, ha="left", clip_on=False)
    ax.set_xlabel("Número de Cefeidas clásicas (escala logarítmica)")
    ax.set_xlim(0.5, df["n_catalogo"].max() * 12)
    guardar(fig, "cefeidas_por_modo.png")


def figura_tipo_binarias():
    df = pd.read_csv(TABLAS_DIR / "04_binarias_por_tipo_morfologico.csv").sort_values("n_catalogo", ascending=True)
    total = df["n_catalogo"].sum()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh([MORFOLOGIA[t] for t in df["tipo_morfologico"]], df["n_catalogo"], color=BINARIA, height=0.55)
    for i, (_, fila) in enumerate(df.iterrows()):
        pct = 100 * fila["n_catalogo"] / total
        ax.text(fila["n_catalogo"] + 500, i, f"{miles(fila['n_catalogo'])}  ({pct:.1f} %)", va="center",
                fontsize=10)
    ax.set_xlabel("Número de binarias")
    ax.xaxis.set_major_formatter(EJE_MILES)
    ax.set_xlim(0, df["n_catalogo"].max() * 1.35)
    guardar(fig, "binarias_por_tipo_morfologico.png", "ogle_binarias_por_tipo.png")  # nombre que usa main.tex


def main():
    figura_totales()
    figura_campo()
    figura_modos_cefeidas()
    figura_tipo_binarias()
    print("Figuras guardadas en", FIG_DIR, "y copiadas a", TESIS_IMG)


if __name__ == "__main__":
    main()
