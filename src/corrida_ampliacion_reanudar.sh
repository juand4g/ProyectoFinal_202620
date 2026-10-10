#!/usr/bin/env bash
# Reanudacion del 10/10/2026: el paso de sinteticas se detuvo (excepcion en el
# pool sin traza). Espera a que TESS termine, detiene el exp. 9 que lanza la
# corrida principal, corre las sinteticas y despues el exp. 9.
cd "$(dirname "$0")"
LOG=../corrida_ampliacion_log.txt
export PYTHONIOENCODING=utf-8
until grep -q "INICIO entropia_sin_mascara" "$LOG"; do sleep 30; done
sleep 5
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'refinamiento_rank_product_bic|multiprocessing-fork' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
echo "$(date '+%d/%m/%Y %H:%M:%S') [R] exp. 9 pospuesto; INICIO sinteticas (reintento)" >> "$LOG"
python robustez_sinteticas.py > ../log_ampliacion_sinteticas_reintento.txt 2>&1
echo "$(date '+%d/%m/%Y %H:%M:%S') [R] FIN sinteticas (reintento, codigo $?)" >> "$LOG"
python refinamiento_rank_product_bic.py > ../log_ampliacion_entropia_sin_mascara.txt 2>&1
echo "$(date '+%d/%m/%Y %H:%M:%S') [R] FIN entropia_sin_mascara (codigo $?)" >> "$LOG"
