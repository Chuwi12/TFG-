#!/usr/bin/fish

# Comprobar si se ha pasado un argumento
if test -z "$argv[1]"
    echo "Uso: ./run_train.fish <numero_de_ciclos>"
    exit 1
end

set ciclos $argv[1]

for i in (seq $ciclos)
    echo "Ciclo $i"
    # Ejecutamos python
    python train.py
end
