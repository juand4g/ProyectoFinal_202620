"""
Ejemplo de construccion de las curvas sinteticas (robustez_sinteticas.py):
para una plantilla de binaria y una de Cefeida, la curva original de OGLE de
la que se extrae la plantilla y cuatro de sus curvas sinteticas (SNR 50 y 4,
con todos los puntos de la cadencia y con N = 100). Las curvas se regeneran
con curva_sintetica() y las especificaciones guardadas en el cache, de modo
que son exactamente las analizadas en la prueba.

Plantilla de cada clase: la de la configuracion "todos los puntos" cuyo
numero de puntos es el mas cercano a la mediana de la clase.

Figuras (Robustez_Sinteticas/graficas/, copiadas a Tesis/img/):
  sinteticas_ejemplo_binaria.png, sinteticas_ejemplo_cefeida.png
"""

import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import graficas_periodograma  # noqa: F401  (tamanos de fuente de las demas curvas de luz)
from graficas_metodo_unificado import AZUL, TINTA
from robustez_sinteticas import CACHE_CSV, OUT_DIR, curva_sintetica, evaluar_plantilla, leer, plantilla
from refinamiento_subarmonicos_bic import BASE_DIR

warnings.filterwarnings("ignore")

GRAF_DIR = OUT_DIR / "graficas"
TESIS_IMG = BASE_DIR / "Tesis" / "img"
SNR_BAJO_RUIDO, SNR_ALTO_RUIDO = 50, 4
GRIS = "#9a9893"


def elegir(cache, clase):
    todos = cache[(cache["clase"] == clase) & (cache["n_objetivo"] == 0) & (cache["snr"] == SNR_BAJO_RUIDO)]
    mediana = todos["n_puntos"].median()
    return int(todos.loc[(todos["n_puntos"] - mediana).abs().idxmin(), "id_plantilla"])


def puntos(ax, fase, mag, color):
    for d in (0.0, 1.0):
        ax.scatter(fase + d, mag, s=8, alpha=0.6, color=color, linewidths=0)


def molde(ax, c, v, nivel):
    x = np.linspace(0, 2, 800)
    ax.plot(x, nivel + evaluar_plantilla(c, v, x % 1.0), color=TINTA, linewidth=1.3)


def ajustar(ax, titulo):
    ax.set_xlim(0, 2)
    ax.invert_yaxis()
    ax.set_title(titulo, fontsize=14, loc="left")
    ax.set_xlabel("Fase")
    ax.set_ylabel("Magnitud")


def figura(cache, clase, nombre):
    id_p = elegir(cache, clase)
    specs = cache[cache["id_plantilla"] == id_p]
    s0 = specs.iloc[0]
    t_p, mag_p, _ = leer(s0["plantilla"])
    c, v = plantilla(t_p, mag_p, s0["periodo_plantilla"])
    nivel_p = np.median(mag_p)

    fig = plt.figure(figsize=(12, 13), constrained_layout=True)
    rejilla = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    ax0 = fig.add_subplot(rejilla[0, :])
    puntos(ax0, ((t_p - t_p.min()) / s0["periodo_plantilla"]) % 1.0, mag_p, GRIS)
    molde(ax0, c, v, nivel_p)
    # escala de la plantilla con un margen: un punto atipico aislado aplastaria el panel
    margen = 0.25 * (v.max() - v.min())
    lim = (nivel_p + v.max() + margen, nivel_p + v.min() - margen)
    fuera = int(np.sum((mag_p > lim[0]) | (mag_p < lim[1])))
    ajustar(ax0, f"(a) Curva original de OGLE: {Path(s0['plantilla']).stem}, "
                 f"$P={s0['periodo_plantilla']:.4f}$ d, $N={t_p.size}$; la línea es la plantilla")
    ax0.set_ylim(lim)

    paneles = [(1, 0, SNR_BAJO_RUIDO, 0, "b"), (1, 1, SNR_BAJO_RUIDO, 100, "c"),
               (2, 0, SNR_ALTO_RUIDO, 0, "d"), (2, 1, SNR_ALTO_RUIDO, 100, "e")]
    ejes = []
    for fila, col, snr, n_obj, letra in paneles:
        spec = specs[(specs["snr"] == snr) & (specs["n_objetivo"] == n_obj)].iloc[0]
        t, mag, _ = curva_sintetica(spec)
        fase = ((t - t.min()) / spec["periodo_verdadero"] + spec["fase0"]) % 1.0
        ax = fig.add_subplot(rejilla[fila, col])
        puntos(ax, fase, mag, AZUL)
        molde(ax, c, v, 15.0)
        ruido = "poco ruido" if snr == SNR_BAJO_RUIDO else "mucho ruido"
        cuantos = "todos los puntos" if n_obj == 0 else "submuestra"
        ajustar(ax, f"({letra}) {ruido} ($\mathrm{{SNR}}={snr}$), {cuantos} ($N={t.size}$)")
        ejes.append(ax)
    # misma escala vertical en las cuatro sinteticas para comparar el ruido
    lo = min(a.get_ylim()[1] for a in ejes)
    hi = max(a.get_ylim()[0] for a in ejes)
    for a in ejes:
        a.set_ylim(hi, lo)
    fig.suptitle(f"Periodo verdadero de las curvas sintéticas: {s0['periodo_verdadero']:.4f} d "
                 f"(cadencia prestada de {Path(s0['cadencia']).stem})", fontsize=14)

    GRAF_DIR.mkdir(parents=True, exist_ok=True)
    ruta = GRAF_DIR / nombre
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    shutil.copy(ruta, TESIS_IMG / nombre)
    print(f"{clase}: plantilla {Path(s0['plantilla']).stem} (subtipo {s0['subtipo']}), "
          f"P_plantilla={s0['periodo_plantilla']:.4f}, P_v={s0['periodo_verdadero']:.4f}, "
          f"fuera de escala en (a): {fuera}, cadencia {Path(s0['cadencia']).stem}, N todos={specs[specs['n_objetivo'] == 0]['n_puntos'].iloc[0]}")


def main():
    cache = pd.read_csv(CACHE_CSV)
    figura(cache, "Binaria", "sinteticas_ejemplo_binaria.png")
    figura(cache, "Cefeida", "sinteticas_ejemplo_cefeida.png")


if __name__ == "__main__":
    main()
