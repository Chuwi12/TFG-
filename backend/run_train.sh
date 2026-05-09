#!/bin/bash

# Comprobar si se ha pasado un argumento
if [ -z "$1" ]; then
  echo "Uso: ./run_train.sh <numero_de_ciclos>"
  exit 1
fi

ciclos=$1

for ((i = 1; i <= ciclos; i++)); do
  # Ejecutamos python (el .env se encargará de la GPU automáticamente)
  python train.py
done
