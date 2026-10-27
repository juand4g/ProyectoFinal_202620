"""
Curvas de luz de TESS plegadas con el periodo del catalogo (Prsa et al. 2022
para las binarias, VSX para las pulsantes) y con el periodo del metodo
unificado. Solo lee TESS_Prueba/cache_tess.csv y las curvas de
TESS_Prueba/fotometria/; no recalcula periodogramas.

Mismo estilo que las curvas de luz de OGLE (panel_curva_luz de
graficas_periodograma.py): dos paneles por renglon, "Curva real" en verde con
el periodo de catalogo y "Curva encontrada" en rojo con el periodo del metodo.

Seis ejemplos, elegidos con criterios fijos y, dentro de cada criterio, la
estrella cuya senal a ruido es la mas cercana a la mediana del grupo:
  curvas_tess_aciertos.png: binaria intermedia, Cefeida, RR Lyrae.
  curvas_tess_fallos.png:
    - binaria separada enviada a la rama de binarias con P0 ya correcto
      (la regla hibrida lo multiplica);
    - Cefeida enviada a la rama de binarias;
    - binaria de contacto/ELL enviada a la rama de pulsantes (P0 = P/2).

Figuras en TESS_Prueba/graficas/, copiadas a Tesis/img/.
"""

import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graficas_periodograma import panel_curva_luz
from prueba_tess import CACHE_CSV, MUESTRA_CSV, OUT_DIR, archivo
from refinamiento_subarmonicos_bic import BASE_DIR, acierto, n_hibrido
from robustez_sinteticas import UMBRAL_SKEW

warnings.filterwarnings("ignore")

GRAF_DIR = OUT_DIR / "graficas"
TESIS_IMG = BASE_DIR / "Tesis" / "img"
H0 = 4
TIPO_VSX = {"DCEP": "Cefeida clásica fundamental (DCEP)", "DCEPS": "Cefeida clásica de primer sobretono (DCEPS)",
            "RRAB": "RR Lyrae de tipo ab (RRAB)", "RRC": "RR Lyrae de tipo c (RRC)"}
GRUPO_BIN = {"Binaria separada": "Binaria eclipsante separada", "Binaria intermedia": "Binaria eclipsante intermedia",
             "Binaria contacto/ELL": "Binaria de contacto o elipsoidal"}


def cargar():
    d = pd.read_csv(CACHE_CSV)
    d = d[d["nombre"].isin(set(pd.read_csv(MUESTRA_CSV)["nombre"]))].reset_index(drop=True)
    d["n_hib"] = n_hibrido(d, H0, "bic_rp_c")
    d["rama_bin"] = d["skew_corregida"].fillna(-np.inf) > UMBRAL_SKEW
    d["p_final"] = np.where(d["rama_bin"], d["n_hib"] * d["periodo_rp_c"], d["periodo_rp_c"])
    ref = d["periodo_referencia"].to_numpy()
    d["ok"] = acierto(d["p_final"].to_numpy(), ref)
    d["p0_ok"] = acierto(d["periodo_rp_c"].to_numpy(), ref)
    d["p0_mitad"] = acierto(d["periodo_rp_c"].to_numpy(), ref / 2)
    return d


def representativa(d, mascara):
    """Estrella con la senal a ruido mas cercana a la mediana de su grupo."""
    c = d[mascara]
    mediana = d.loc[d["grupo"].isin(c["grupo"].unique()), "snr"].median()
    return c.loc[(np.log(c["snr"]) - np.log(mediana)).abs().idxmin()]


def ejemplos(d):
    g = d["grupo"]
    aciertos = [representativa(d, (g == "Binaria intermedia") & d["ok"]),
                representativa(d, (g == "Cefeida") & d["ok"]),
                representativa(d, (g == "RR Lyrae") & d["ok"])]
    fallos = [representativa(d, (g == "Binaria separada") & d["rama_bin"] & ~d["ok"] & d["p0_ok"]),
              representativa(d, (g == "Cefeida") & d["rama_bin"] & ~d["ok"]),
              representativa(d, (g == "Binaria contacto/ELL") & ~d["rama_bin"] & ~d["ok"] & d["p0_mitad"])]
    return aciertos, fallos


def descripcion(fila):
    if fila["clase"] == "Binaria":
        morf = float(str(fila["subtipo"]).split("=")[1])
        return f"{GRUPO_BIN[fila['grupo']]}, morfología {morf:.2f}", "Prša et al. (2022)"
    return TIPO_VSX.get(str(fila["subtipo"]).strip(), fila["subtipo"]), "VSX"


def razon(p, ref):
    """Relacion entre el periodo encontrado y el de catalogo (n o 1/n)."""
    r = p / ref
    if abs(r - 1) < 0.01:
        return ""
    if r > 1 and abs(r - round(r)) / r < 0.01:
        return rf" $({round(r)}P_{{\mathrm{{cat}}}})$"
    if r < 1 and abs(1 / r - round(1 / r)) * r < 0.01:
        return rf" $(P_{{\mathrm{{cat}}}}/{round(1 / r)})$"
    return rf" $({r:.2f}P_{{\mathrm{{cat}}}})$"


def figura(estrellas, letras, nombre):
    # proporcion ancha: ocupa \textwidth sin pasar de ~0.6 \textheight
    fig = plt.figure(figsize=(12, 3.3 * len(estrellas)), constrained_layout=True)
    for subfig, f, letra in zip(fig.subfigures(len(estrellas), 1, hspace=0.06), estrellas, letras):
        t, mag, _ = np.loadtxt(archivo(f["nombre"]), unpack=True)
        ax_cat, ax_enc = subfig.subplots(1, 2, sharey=True)
        panel_curva_luz(ax_cat, t, mag, f["periodo_referencia"], color="green",
                        titulo=f"Curva real: {f['periodo_referencia']:.4f} d")
        panel_curva_luz(ax_enc, t, mag, f["p_final"], color="red",
                        titulo=f"Curva encontrada: {f['p_final']:.4f} d"
                        + razon(f["p_final"], f["periodo_referencia"]))
        ax_cat.set_ylabel("Magnitud relativa")
        ax_enc.set_ylabel("")
        ax_enc.yaxis.set_inverted(True)
        tipo, catalogo = descripcion(f)
        subfig.suptitle(f"({letra}) {f['nombre']}: {tipo}\nTipo y periodo de catálogo: {catalogo}",
                        fontsize=15, x=0.02, ha="left")
    GRAF_DIR.mkdir(parents=True, exist_ok=True)
    ruta = GRAF_DIR / nombre
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    shutil.copy(ruta, TESIS_IMG / nombre)


def main():
    d = cargar()
    aciertos, fallos = ejemplos(d)
    for f in aciertos + fallos:
        print(f"{f['nombre']:28s} {f['grupo']:22s} {f['subtipo']:12s} P_cat={f['periodo_referencia']:.4f} "
              f"P0={f['periodo_rp_c']:.4f} final={f['p_final']:.4f} n={f['n_hib']} rama_bin={f['rama_bin']} "
              f"skew={f['skew_corregida']:.2f} snr={f['snr']:.0f} ok={f['ok']} ciclos={f['baseline_d'] / f['periodo_referencia']:.1f}")
    figura(aciertos, "abc", "curvas_tess_aciertos.png")
    figura(fallos, "abc", "curvas_tess_fallos.png")


if __name__ == "__main__":
    main()
