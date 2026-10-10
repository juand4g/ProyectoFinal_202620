#!/usr/bin/env bash
# Corrida nocturna del 09/10/2026: amplia x3 todas las muestras de la tesis
# (Cap. 4: 500 -> 1500 por tipo; Cap. 5: 1000 -> 3000 binarias; Cap. 6:
# 500 -> 1500 pulsantes por tipo; Cap. 7: 240 -> 720 plantillas sinteticas y
# 240 -> 720 estrellas de TESS). Todos los scripts son append-safe: reutilizan
# lo que ya esta en cache y solo calculan las estrellas nuevas.
#
# Tres carriles en paralelo:
#   A (pesado, ProcessPool de N_PROCESOS): cadena del metodo unificado.
#   B (un proceso): caracterizacion individual del Cap. 4 y experimentos ligeros.
#   C (red): descargas de TESS, para que esten listas cuando A llegue a TESS.
# Uso: bash src/corrida_ampliacion.sh   (desde la raiz del proyecto)

cd "$(dirname "$0")"
LOG=../corrida_ampliacion_log.txt
export PYTHONIOENCODING=utf-8

paso() {  # paso <carril> <nombre> <comando...>
    local carril=$1 nombre=$2; shift 2
    echo "$(date '+%d/%m/%Y %H:%M:%S') [$carril] INICIO $nombre" >> "$LOG"
    "$@" > "../log_ampliacion_${nombre}.txt" 2>&1
    echo "$(date '+%d/%m/%Y %H:%M:%S') [$carril] FIN $nombre (codigo $?)" >> "$LOG"
}

carril_a() {
    paso A hibrido_binarias python refinamiento_rank_product_enmascarado_bic.py
    paso A etapa1_pulsantes python etapa1_cefeidas_rrlyrae.py
    paso A selector_indicadores python selector_multiplo_binarias.py
    paso A decision_forma python decision_binaria_pulsante.py
    paso A sinteticas python robustez_sinteticas.py
    while [ ! -f ../TESS_Prueba/descargas_listas.txt ]; do sleep 60; done  # espera al carril C
    paso A tess python prueba_tess.py
    paso A entropia_sin_mascara python refinamiento_rank_product_bic.py
}

carril_b() {
    paso B caracterizacion_ls python caracterizacion_lomb_scargle.py
    paso B caracterizacion_entropia python caracterizacion_entropia.py
    paso B barrido_L python barrido_L_entropia_binarias.py
    paso B ls_bic_libre python refinamiento_subarmonicos_bic.py
}

carril_c() {
    paso C descargas_tess python -c "import prueba_tess; prueba_tess.construir_muestra()"
    touch ../TESS_Prueba/descargas_listas.txt
}

rm -f ../TESS_Prueba/descargas_listas.txt
echo "$(date '+%d/%m/%Y %H:%M:%S') corrida de ampliacion iniciada" >> "$LOG"
carril_a & carril_b & carril_c &
wait
echo "$(date '+%d/%m/%Y %H:%M:%S') corrida de ampliacion terminada" >> "$LOG"
