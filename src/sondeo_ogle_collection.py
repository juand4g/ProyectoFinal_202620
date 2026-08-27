"""
Sondeo descriptivo de la coleccion OGLE (ogle_collection): distribucion de
las estrellas variables disponibles por tipo, campo galactico, modo de
pulsacion (Cefeidas) y tipo morfologico (Binarias).

Genera tablas en CSV dentro de Sondeo_OGLE/tablas/. Se puede volver a
ejecutar cuando cambien los datos de ogle_collection.
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "ogle_collection"
OUT_DIR = PROJECT_DIR / "Sondeo_OGLE" / "tablas"

CAMPOS = {
    "LMC": "Nube Mayor de Magallanes",
    "SMC": "Nube Menor de Magallanes",
    "GD": "Disco galactico",
    "BLG": "Disco galactico (campo hacia el bulbo)",
}

MODOS_CEFEIDAS = {
    "F": "Fundamental",
    "1": "Primer armonico (1O)",
    "2": "Segundo armonico (2O)",
    "F1": "Doble modo: Fundamental + 1er armonico",
    "12": "Doble modo: 1er + 2do armonico",
    "F12": "Triple modo: Fundamental + 1er + 2do armonico",
    "13": "Doble modo: 1er + 3er armonico",
    "23": "Doble modo: 2do + 3er armonico",
    "123": "Triple modo: 1er + 2do + 3er armonico",
}

TIPOS_BINARIAS = {
    "NC": "No-contacto (separada o semi-separada)",
    "C": "Contacto",
    "ELL": "Elipsoidal (sin eclipses claros)",
}


def cargar(nombre_txt, carpeta_phot):
    df = pd.read_csv(DATA_DIR / nombre_txt, sep="\t", comment="#")
    df["campo"] = df["ID"].str.split("-").str[1]
    ids_con_foto = {p.stem for p in (DATA_DIR / carpeta_phot).glob("*.dat")}
    df["con_fotometria"] = df["ID"].isin(ids_con_foto)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    binarias = cargar("binaries.txt", "binaries_phot")
    cefeidas = cargar("classical_cefeids.txt", "classical_cefeids_phot")
    rrlyrae = cargar("rrlyrae.txt", "rrlyrae_phot")

    # --- Tabla 1: tipo de variable x campo galactico ------------------------
    filas = []
    for tipo, df in [("Binaria", binarias), ("Cefeida", cefeidas), ("RR Lyrae", rrlyrae)]:
        for campo, grupo in df.groupby("campo"):
            filas.append({
                "tipo": tipo,
                "campo": campo,
                "region": CAMPOS.get(campo, campo),
                "n_catalogo": len(grupo),
                "n_con_fotometria": int(grupo["con_fotometria"].sum()),
            })
    tabla_campo = pd.DataFrame(filas).sort_values(["tipo", "campo"])
    tabla_campo.to_csv(OUT_DIR / "01_distribucion_por_campo.csv", index=False)

    # --- Tabla 2: totales por tipo de variable ------------------------------
    tabla_totales = pd.DataFrame([
        {"tipo": "Binaria", "n_catalogo": len(binarias), "n_con_fotometria": int(binarias["con_fotometria"].sum())},
        {"tipo": "Cefeida", "n_catalogo": len(cefeidas), "n_con_fotometria": int(cefeidas["con_fotometria"].sum())},
        {"tipo": "RR Lyrae", "n_catalogo": len(rrlyrae), "n_con_fotometria": int(rrlyrae["con_fotometria"].sum())},
    ])
    tabla_totales.to_csv(OUT_DIR / "02_totales_por_tipo.csv", index=False)

    # --- Tabla 3: Cefeidas por modo de pulsacion ----------------------------
    tabla_modos = (
        cefeidas.groupby("Mode")
        .agg(n_catalogo=("ID", "count"), n_con_fotometria=("con_fotometria", "sum"))
        .reset_index()
        .rename(columns={"Mode": "modo"})
    )
    tabla_modos["descripcion"] = tabla_modos["modo"].map(MODOS_CEFEIDAS)
    tabla_modos = tabla_modos.sort_values("n_catalogo", ascending=False)
    tabla_modos.to_csv(OUT_DIR / "03_cefeidas_por_modo.csv", index=False)

    # --- Tabla 4: Binarias por tipo morfologico -----------------------------
    tabla_tipos_bin = (
        binarias.groupby("Type")
        .agg(n_catalogo=("ID", "count"), n_con_fotometria=("con_fotometria", "sum"))
        .reset_index()
        .rename(columns={"Type": "tipo_morfologico"})
    )
    tabla_tipos_bin["descripcion"] = tabla_tipos_bin["tipo_morfologico"].map(TIPOS_BINARIAS)
    tabla_tipos_bin = tabla_tipos_bin.sort_values("n_catalogo", ascending=False)
    tabla_tipos_bin.to_csv(OUT_DIR / "04_binarias_por_tipo_morfologico.csv", index=False)

    # --- Tabla 5: Cefeidas por campo x modo (detalle cruzado) --------------
    tabla_cruzada = (
        cefeidas.groupby(["campo", "Mode"])
        .size()
        .reset_index(name="n_catalogo")
        .rename(columns={"Mode": "modo"})
    )
    tabla_cruzada.to_csv(OUT_DIR / "05_cefeidas_campo_x_modo.csv", index=False)

    for nombre, tabla in [
        ("Distribucion por campo galactico", tabla_campo),
        ("Totales por tipo", tabla_totales),
        ("Cefeidas por modo", tabla_modos),
        ("Binarias por tipo morfologico", tabla_tipos_bin),
        ("Cefeidas: campo x modo", tabla_cruzada),
    ]:
        print(f"\n{nombre}")
        print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
