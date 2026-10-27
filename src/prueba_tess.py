"""
Prueba del metodo unificado sobre curvas de luz de TESS (otra cadencia,
otro instrumento, otra banda; sin ningun reentrenamiento ni ajuste).

Metodo (el de trabajo desde el 07/10/2026; ver bitacora, exp. 14):
  1. P0 = rank_product condicional (misma grilla: 0.1-100 d, 100 000
     frecuencias, L=10, K=7), funcion periodos_base.
  2. Decision de rama: skewness corregida por ruido > 0.5 -> binaria.
  3. Binaria: x2 + BIC (H0=4). Pulsante: P0 directo.

Muestra (estrellas brillantes, para limitar la mezcla de luz en los
pixeles de 21" de TESS):
  - Binarias: catalogo de binarias eclipsantes de TESS (Prsa et al. 2022,
    ApJS 258, 16; VizieR J/ApJS/258/16), 8 <= Tmag <= 12, 0.2 <= P <= 10 d,
    estratificadas por el parametro morfologico (0 = separada, 1 =
    contacto/elipsoidal) en tres rangos.
  - Cefeidas clasicas (DCEP, DCEPS) y RR Lyrae (RRAB, RRC): VSX (VizieR
    B/vsx/vsx), magnitud maxima en V, tipo sin incertidumbre, 1 <= P <= 10 d
    para Cefeidas.
Curva: UN sector por estrella (el primero disponible), preferencia SPOC
2 min > TESS-SPOC > QLP; flujo PDCSAP o equivalente, sin NaN y con la
mascara de calidad por defecto de lightkurve; agrupado en bins de 30 min
(flujo medio; error = sqrt(sum dF_i^2) / n, corregido el 26/10/2026: el
error por defecto de lightkurve.bin es la media cuadratica de los errores,
~sqrt(n) veces mayor que el de la media); convertido a magnitud relativa
m = -2.5 log10(F / mediana(F)), error 1.0857 * dF / F. Cada curva se guarda en TESS_Prueba/fotometria/ (formato
OGLE: t, mag, err) para no volver a descargarla.

Con un solo sector (~27 d) la resolucion en frecuencia es ~1/27 d^-1: para
P >~ 3 d el error de 1% puede quedar por debajo de lo que el sector permite
medir. Por eso se reporta acierto al 1% (criterio OGLE) y al 5%.
"""

import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from refinamiento_rank_product_bic import N_PROCESOS
from refinamiento_rank_product_enmascarado_bic import periodos_base
from refinamiento_subarmonicos_bic import N_MAX, RANDOM_SEED, bic_fourier, n_hibrido
from robustez_sinteticas import UMBRAL_SKEW, skew_corregida

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "TESS_Prueba"
FOTO_DIR = OUT_DIR / "fotometria"
MUESTRA_CSV = OUT_DIR / "muestra_tess.csv"
INTENTOS_CSV = OUT_DIR / "intentos_descarga.csv"
CACHE_CSV = OUT_DIR / "cache_tess.csv"
RESUMEN_TXT = OUT_DIR / "resumen_tess.txt"

# Ampliada el 09/10/2026 (antes 40/40/40/60/60); las curvas originales se conservan
N_POR_GRUPO = {"Binaria separada": 120, "Binaria intermedia": 120, "Binaria contacto/ELL": 120,
               "Cefeida": 180, "RR Lyrae": 180}
CORTES_MORFOLOGIA = [(0.0, 0.4, "Binaria separada"), (0.4, 0.7, "Binaria intermedia"),
                     (0.7, 1.01, "Binaria contacto/ELL")]
CANDIDATOS_POR_GRUPO = 4  # se piden 4x candidatos: no todos tienen datos de TESS
BIN_MIN = 30.0
H0 = 4
# Sin QLP: su flujo por defecto (KSPSAP/DET) esta filtrado y borra variabilidad
# lenta; su SAP_FLUX trae tramos con sistematicas fuertes (luz dispersa) y sus
# errores son ~5x la dispersion real punto a punto (bitacora, exp. 15).
AUTORES = ["SPOC", "TESS-SPOC"]
UMBRAL_REF_DUDOSA = 0.1  # potencia LS en P_ref (o P_ref/2, 2 P_ref) < 0.1 x maximo
RADIO_ARCSEC = 15
N_HILOS_DESCARGA = 6
MIN_PUNTOS = 100

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ muestra
def consultar_catalogos(rng=None, excluir=(), objetivo=None):
    """Candidatos por grupo. `objetivo` = candidatos a extraer por grupo
    (por defecto CANDIDATOS_POR_GRUPO x N_POR_GRUPO); `excluir` = nombres ya
    presentes, para ampliar la lista sin repetir."""
    from astroquery.vizier import Vizier
    rng = rng or random.Random(RANDOM_SEED)
    objetivo = objetivo or {g: n * CANDIDATOS_POR_GRUPO for g, n in N_POR_GRUPO.items()}
    excluir = set(excluir)
    filas = []

    v = Vizier(columns=["TIC", "RAJ2000", "DEJ2000", "Tmag", "Per", "Morph"], row_limit=-1)
    ebs = v.query_constraints(catalog="J/ApJS/258/16/tess-ebs", Tmag="8..12", Per="0.2..10")[0].to_pandas()
    ebs = ebs.dropna(subset=["Morph", "Per"])
    for lo, hi, grupo in CORTES_MORFOLOGIA:
        g = ebs[(ebs["Morph"] >= lo) & (ebs["Morph"] < hi)]
        g = g[~g["TIC"].map(lambda x: f"TIC {int(x)}").isin(excluir)]
        idx = rng.sample(list(g.index), min(len(g), objetivo[grupo]))
        for _, r in g.loc[idx].iterrows():
            filas.append({"grupo": grupo, "clase": "Binaria", "subtipo": f"morph={r['Morph']:.2f}",
                          "nombre": f"TIC {int(r['TIC'])}", "ra": r["RAJ2000"], "dec": r["DEJ2000"],
                          "periodo_referencia": float(r["Per"]), "catalogo": "Prsa+2022"})

    v = Vizier(columns=["Name", "RAJ2000", "DEJ2000", "Type", "max", "n_max", "Period"], row_limit=-1)
    consultas = [("Cefeida", ["DCEP", "DCEPS"], "7.5..12", "1..10"),
                 ("RR Lyrae", ["RRAB", "RRC"], "9..12.5", "0.2..1")]
    for clase, tipos, mag, periodo in consultas:
        tablas = [v.query_constraints(catalog="B/vsx/vsx", Type=tp, max=mag, Period=periodo)[0].to_pandas()
                  for tp in tipos]
        g = pd.concat(tablas, ignore_index=True)
        g = g[g["n_max"].astype(str).str.strip() == "V"].dropna(subset=["Period"])
        g = g[~g["Name"].astype(str).str.strip().isin(excluir)]
        idx = rng.sample(list(g.index), min(len(g), objetivo[clase]))
        for _, r in g.loc[idx].iterrows():
            filas.append({"grupo": clase, "clase": clase, "subtipo": str(r["Type"]).strip(),
                          "nombre": str(r["Name"]).strip(), "ra": r["RAJ2000"], "dec": r["DEJ2000"],
                          "periodo_referencia": float(r["Period"]), "catalogo": "VSX"})
    return pd.DataFrame(filas)


def archivo(nombre):
    return FOTO_DIR / (nombre.replace(" ", "_").replace("/", "-") + ".dat")


def descargar(fila, producto=None):
    """Descarga un sector y lo guarda en formato OGLE. Devuelve metadatos o None.
    `producto` = (autor, sector, exptime) fuerza ese producto en lugar de elegirlo
    (para regenerar las curvas de la muestra ya congelada)."""
    import astropy.units as u
    import lightkurve as lk
    from astropy.coordinates import SkyCoord

    warnings.filterwarnings("ignore")
    destino = archivo(fila["nombre"])
    try:
        objetivo = (fila["nombre"] if fila["nombre"].startswith("TIC")
                    else SkyCoord(fila["ra"], fila["dec"], unit="deg"))
        busqueda = lk.search_lightcurve(objetivo, mission="TESS", author=AUTORES,
                                        radius=RADIO_ARCSEC * u.arcsec)
        if len(busqueda) == 0:
            return None
        tabla = busqueda.table.to_pandas()
        tabla["prioridad"] = tabla["author"].map({a: k for k, a in enumerate(AUTORES)})
        tabla["dist"] = tabla.get("distance", 0.0)
        if producto is not None:
            autor, sector, exptime = producto
            tabla = tabla[(tabla["author"] == autor) & (tabla["mission"] == sector)
                          & np.isclose(tabla["exptime"].astype(float), exptime)]
            if tabla.empty:
                return None
        elegido = tabla.sort_values(["dist", "prioridad", "mission"]).index[0]
        lc = busqueda[int(elegido)].download()
        if lc is None or not guardar_curva(lc, destino):
            return None
        return {"autor": tabla.loc[elegido, "author"], "sector": tabla.loc[elegido, "mission"],
                "exptime": tabla.loc[elegido, "exptime"], "distancia_arcsec": tabla.loc[elegido, "dist"],
                "crowdsap": lc.meta.get("CROWDSAP", np.nan)}
    except Exception as e:  # red, archivos corruptos, productos sin flujo...
        log(f"  {fila['nombre']}: {type(e).__name__}: {str(e)[:80]}")
        return None


def guardar_curva(lc, destino):
    """Limpia, agrupa en bins de BIN_MIN y guarda la curva en formato OGLE.
    Devuelve False si quedan menos de MIN_PUNTOS puntos validos."""
    lc = lc.remove_nans()
    lc = lc[lc.flux.value > 0]
    # lightkurve asigna a cada bin la media cuadratica de los errores
    # individuales (no el error de la media): se divide por sqrt(n) del bin.
    plano = lambda x: np.asarray(np.ma.filled(np.ma.asarray(x), np.nan), dtype=float)
    t_orig, df_orig = plano(lc.time.value), plano(lc.flux_err.value)
    lc = lc.bin(time_bin_size=BIN_MIN / 1440.0)
    f, t = plano(lc.flux.value), plano(lc.time.value)
    bordes = np.append(t - BIN_MIN / 2880.0, t[-1] + BIN_MIN / 2880.0)
    n_bin = np.histogram(t_orig[np.isfinite(df_orig)], bins=bordes)[0]
    df = plano(lc.flux_err.value) / np.sqrt(np.where(n_bin > 0, n_bin, np.nan))
    mag = -2.5 * np.log10(f / np.nanmedian(f))
    err = 1.0857 * df / f
    bueno = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if bueno.sum() < MIN_PUNTOS:
        return False
    np.savetxt(destino, np.column_stack([t[bueno], mag[bueno], err[bueno]]), fmt="%.6f")
    return True


def log(mensaje):
    """lightkurve/astroquery redirigen sys.stdout desde cada hilo y, al
    restaurarlo en paralelo, pueden dejarlo cerrado: se escribe a la consola
    original del proceso."""
    sys.stdout = sys.__stdout__
    sys.__stdout__.write(mensaje + "\n")
    sys.__stdout__.flush()


def construir_muestra():
    """Recorre los candidatos de cada grupo hasta juntar N_POR_GRUPO curvas.
    Cada intento (exitoso o no) queda en INTENTOS_CSV para poder retomar."""
    FOTO_DIR.mkdir(parents=True, exist_ok=True)
    if MUESTRA_CSV.exists():
        muestra = pd.read_csv(MUESTRA_CSV)
        if all((muestra["grupo"] == g).sum() >= n for g, n in N_POR_GRUPO.items()):
            return muestra
    ruta_candidatos = OUT_DIR / "candidatos_tess.csv"
    if ruta_candidatos.exists():
        candidatos = pd.read_csv(ruta_candidatos)
        faltan = {g: n * CANDIDATOS_POR_GRUPO - int((candidatos["grupo"] == g).sum())
                  for g, n in N_POR_GRUPO.items()}
        if any(v > 0 for v in faltan.values()):
            # ampliacion: candidatos nuevos al final de la lista, con otra semilla,
            # para que la muestra original siga siendo el prefijo de cada grupo
            nuevos = consultar_catalogos(random.Random(RANDOM_SEED + 1), set(candidatos["nombre"]),
                                         {g: max(v, 0) for g, v in faltan.items()})
            candidatos = pd.concat([candidatos, nuevos], ignore_index=True)
            candidatos.to_csv(ruta_candidatos, index=False)
            log("Candidatos tras la ampliacion: " + str(candidatos.groupby("grupo").size().to_dict()))
    else:
        candidatos = consultar_catalogos()
        candidatos.to_csv(ruta_candidatos, index=False)
    intentos = pd.read_csv(INTENTOS_CSV) if INTENTOS_CSV.exists() else pd.DataFrame(columns=["nombre", "exito"])
    elegidas = []
    for grupo, n in N_POR_GRUPO.items():
        g = candidatos[candidatos["grupo"] == grupo].to_dict("records")
        previos = intentos.set_index("nombre")
        conseguidas = [previos.loc[f["nombre"]].to_dict() | f for f in g
                       if f["nombre"] in previos.index and bool(previos.at[f["nombre"], "exito"])]
        pendientes = [f for f in g if f["nombre"] not in previos.index]
        with ThreadPoolExecutor(N_HILOS_DESCARGA) as pool:
            for k in range(0, len(pendientes), N_HILOS_DESCARGA):
                if len(conseguidas) >= n:
                    break
                lote = pendientes[k:k + N_HILOS_DESCARGA]
                nuevos = []
                for fila, meta in zip(lote, pool.map(descargar, lote)):
                    nuevos.append({"nombre": fila["nombre"], "exito": meta is not None} | (meta or {}))
                    if meta is not None:
                        conseguidas.append(fila | meta)
                intentos = pd.concat([intentos, pd.DataFrame(nuevos)], ignore_index=True)
                intentos.to_csv(INTENTOS_CSV, index=False)
                log(f"{grupo}: {len(conseguidas)}/{n} (pendientes revisados {k + len(lote)}/{len(pendientes)})")
        elegidas += [{c: f[c] for c in f if c != "exito"} for f in conseguidas[:n]]
    muestra = pd.DataFrame(elegidas)
    muestra.to_csv(MUESTRA_CSV, index=False)
    return muestra


def regenerar_fotometria(muestra):
    """Vuelve a escribir las curvas de la muestra congelada con el mismo producto
    (autor, sector, exptime) registrado en MUESTRA_CSV. Usada el 26/10/2026 para
    corregir el error de los bins de 30 min (antes sin dividir por sqrt(n))."""
    filas = muestra.to_dict("records")
    productos = [(f["autor"], f["sector"], float(f["exptime"])) for f in filas]
    fallidas = []
    with ThreadPoolExecutor(N_HILOS_DESCARGA) as pool:
        for fila, meta in zip(filas, pool.map(descargar, filas, productos)):
            if meta is None:
                fallidas.append(fila["nombre"])
    log(f"Curvas regeneradas: {len(filas) - len(fallidas)}/{len(filas)}; fallidas: {fallidas}")
    return fallidas


def indice_cache_lightkurve():
    """Productos SPOC y TESS-SPOC presentes en la cache local de lightkurve:
    autor, sector, exptime (s), TIC y coordenadas, leidos del encabezado."""
    from astropy.io import fits
    base = Path.home() / ".lightkurve" / "cache" / "mastDownload"
    filas = []
    for ruta in list(base.glob("TESS/*/*_lc.fits")) + list(base.glob("HLSP/hlsp_tess-spoc*/*.fits")):
        h0, h1 = fits.getheader(ruta, 0), fits.getheader(ruta, 1)
        filas.append({"ruta": str(ruta), "autor": "SPOC" if ruta.parent.parent.name == "TESS" else "TESS-SPOC",
                      "sector_num": int(h0["SECTOR"]), "exptime": round(h1["TIMEDEL"] * 86400.0),
                      "tic": int(h0["TICID"]), "ra": h0["RA_OBJ"], "dec": h0["DEC_OBJ"]})
    return pd.DataFrame(filas)


def regenerar_desde_cache(muestra):
    """Como regenerar_fotometria, pero leyendo los FITS de la cache local de
    lightkurve (sin red). Binarias: por TIC; pulsantes: el producto mas cercano
    dentro de RADIO_ARCSEC. En ambos casos con el autor, sector y exptime de
    MUESTRA_CSV. Devuelve las filas sin archivo en la cache."""
    import lightkurve as lk
    indice = indice_cache_lightkurve()
    faltan = []
    for fila in muestra.to_dict("records"):
        sector = int(str(fila["sector"]).split()[-1])
        c = indice[(indice["autor"] == fila["autor"]) & (indice["sector_num"] == sector)
                   & (indice["exptime"] == round(float(fila["exptime"])))]
        if fila["nombre"].startswith("TIC"):
            c = c[c["tic"] == int(fila["nombre"].split()[1])]
        else:
            dist = 3600.0 * np.hypot((c["ra"] - fila["ra"]) * np.cos(np.radians(fila["dec"])), c["dec"] - fila["dec"])
            c = c.assign(dist=dist).sort_values("dist")
            c = c[c["dist"] <= RADIO_ARCSEC]
        if c.empty or not guardar_curva(lk.read(c["ruta"].iloc[0]), archivo(fila["nombre"])):
            faltan.append(fila)
    log(f"Curvas regeneradas desde la cache local: {len(muestra) - len(faltan)}/{len(muestra)}")
    return pd.DataFrame(faltan)


# ------------------------------------------------------------------ analisis
def analizar(fila):
    warnings.filterwarnings("ignore")
    t, mag, err = np.loadtxt(archivo(fila["nombre"]), unpack=True)
    out = dict(fila) | {"n_puntos": t.size, "baseline_d": t.max() - t.min()}
    for metodo, p in periodos_base(t, mag, err).items():
        out[f"periodo_{metodo}"] = p
    p0 = out["periodo_rp_c"]
    for n in range(1, N_MAX + 1):
        out[f"bic_rp_c_h{H0}_n{n}"] = (bic_fourier(t, mag, err, 1.0 / (n * p0), n * H0)
                                       if n * p0 <= out["baseline_d"] else np.nan)
    from scipy.stats import skew
    out["skew_mag"] = float(skew(mag))
    out["skew_corregida"] = skew_corregida(mag, err)
    out["snr"] = (np.percentile(mag, 98) - np.percentile(mag, 2)) / np.median(err)
    # Diagnostico de la referencia: potencia LS en P_ref, P_ref/2 y 2 P_ref
    # relativa al maximo del periodograma (misma grilla del metodo).
    from astropy.timeseries import LombScargle
    from refinamiento_rank_product_enmascarado_bic import FRECUENCIA
    ls = LombScargle(t, mag, err)
    maximo = ls.power(FRECUENCIA).max()
    ref = fila["periodo_referencia"]
    out["potencia_ref_relativa"] = float(ls.power(1.0 / np.array([ref, ref / 2, 2 * ref])).max() / maximo)
    return out


def ok(p, ref, tol):
    return np.abs(p - ref) / ref < tol


def resumen(d):
    d = d.copy()
    d["n_hib"] = n_hibrido(d, H0, "bic_rp_c")
    ref = d["periodo_referencia"].to_numpy()
    p0 = d["periodo_rp_c"].to_numpy()
    es_bin = (d["clase"] == "Binaria").to_numpy()
    ruta_bin = d["skew_corregida"].fillna(-np.inf).to_numpy() > UMBRAL_SKEW
    p_hib = d["n_hib"].to_numpy() * p0
    p_final = np.where(ruta_bin, p_hib, p0)
    p_oraculo = np.where(es_bin, p_hib, p0)

    lineas = []
    w = lineas.append
    w("PRUEBA DEL METODO UNIFICADO EN TESS (rank_product condicional + skew corregida + x2/BIC)")
    w("=" * 78)
    w(f"{len(d)} estrellas, un sector cada una, bins de {BIN_MIN:.0f} min. Productos: "
      f"{d['autor'].value_counts().to_dict()}. N mediano {d['n_puntos'].median():.0f}, "
      f"linea base mediana {d['baseline_d'].median():.1f} d.")
    w("Referencias: binarias Prsa et al. (2022); pulsantes VSX. Sin reentrenar ni ajustar nada.")
    w("")
    for tol in [0.01, 0.05]:
        w(f"Acierto con error < {tol:.0%}")
        w(f"  {'Grupo':22s}{'n':>4}{'LS dir.':>9}{'P0 dir.':>9}{'oraculo':>9}{'FINAL':>8}"
          f"{'->rama bin.':>12}{'P0~P/2':>8}{'P0~2P':>7}")
        for grupo in list(N_POR_GRUPO) + ["Todas las binarias"]:
            m = es_bin if grupo == "Todas las binarias" else (d["grupo"] == grupo).to_numpy()
            if not m.any():
                continue
            w(f"  {grupo:22s}{m.sum():4d}{ok(d['periodo_ls'].to_numpy(), ref, tol)[m].mean():9.1%}"
              f"{ok(p0, ref, tol)[m].mean():9.1%}{ok(p_oraculo, ref, tol)[m].mean():9.1%}"
              f"{ok(p_final, ref, tol)[m].mean():8.1%}{ruta_bin[m].mean():12.1%}"
              f"{ok(p0, ref / 2, tol)[m].mean():8.1%}{ok(p0, 2 * ref, tol)[m].mean():7.1%}")
        w("")
    dudosa = d["potencia_ref_relativa"].to_numpy() < UMBRAL_REF_DUDOSA
    w(f"Referencia dudosa (potencia LS en P_ref, P_ref/2 o 2 P_ref < {UMBRAL_REF_DUDOSA:.0%} del maximo): "
      + ", ".join(f"{g} {dudosa[(d['grupo'] == g).to_numpy()].sum()}" for g in N_POR_GRUPO))
    w("Acierto con error < 1% EXCLUYENDO referencias dudosas:")
    for grupo in list(N_POR_GRUPO) + ["Todas las binarias"]:
        m = (es_bin if grupo == "Todas las binarias" else (d["grupo"] == grupo).to_numpy()) & ~dudosa
        if m.any():
            w(f"  {grupo:22s}{m.sum():4d}{ok(d['periodo_ls'].to_numpy(), ref, 0.01)[m].mean():9.1%}"
              f"{ok(p0, ref, 0.01)[m].mean():9.1%}{ok(p_oraculo, ref, 0.01)[m].mean():9.1%}"
              f"{ok(p_final, ref, 0.01)[m].mean():8.1%}{ruta_bin[m].mean():12.1%}")
    w("")
    w("Columnas: LS dir. = Lomb-Scargle directo; P0 dir. = rank_product condicional directo;")
    w("oraculo = rama elegida con la clase real; FINAL = metodo completo; ->rama bin. = fraccion")
    w("enviada a la rama de binarias por la skewness corregida.")
    w("")
    w("Skewness corregida (mediana) por grupo: " + ", ".join(
        f"{g} {d.loc[d['grupo'] == g, 'skew_corregida'].median():.2f}" for g in N_POR_GRUPO))
    w("SNR (amplitud 2-98% / error mediano) mediana por grupo: " + ", ".join(
        f"{g} {d.loc[d['grupo'] == g, 'snr'].median():.0f}" for g in N_POR_GRUPO))
    if "subtipo" in d:
        w("")
        w("Por subtipo de pulsante (FINAL, error < 1% / 5%):")
        for (clase, sub), g in d[~es_bin].groupby(["clase", "subtipo"]):
            i = g.index
            w(f"  {clase:9s} {sub:6s} n={len(g):3d}   {ok(p_final[i], ref[i], 0.01).mean():6.1%} / "
              f"{ok(p_final[i], ref[i], 0.05).mean():6.1%}")

    texto = "\n".join(lineas)
    RESUMEN_TXT.write_text(texto, encoding="utf-8")
    print(texto)


def main():
    muestra = construir_muestra()
    if "--regenerar" in sys.argv:
        faltan = regenerar_desde_cache(muestra)
        if len(faltan) and regenerar_fotometria(faltan):
            return
    print(muestra.groupby("grupo").size().to_dict(), flush=True)
    cache = pd.read_csv(CACHE_CSV) if CACHE_CSV.exists() else pd.DataFrame(columns=["nombre"])
    pendientes = [f for f in muestra.to_dict("records") if f["nombre"] not in set(cache["nombre"])]
    filas = []
    with ProcessPoolExecutor(max_workers=N_PROCESOS) as pool:
        futuros = [pool.submit(analizar, f) for f in pendientes]
        for k, futuro in enumerate(tqdm(as_completed(futuros), total=len(futuros), desc="TESS", unit="estrella")):
            filas.append(futuro.result())
            if (k + 1) % 25 == 0:
                cache = pd.concat([cache, pd.DataFrame(filas)], ignore_index=True)
                cache.to_csv(CACHE_CSV, index=False)
                filas = []
    if filas:
        cache = pd.concat([cache, pd.DataFrame(filas)], ignore_index=True)
        cache.to_csv(CACHE_CSV, index=False)
    resumen(cache[cache["nombre"].isin(set(muestra["nombre"]))].reset_index(drop=True))


if __name__ == "__main__":
    main()
