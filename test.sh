#!/bin/sh
run=true
cooldown=5
times_executed=0

while [ "$run" = true ]
do
    python ./src/cbdgen-framework.py --classes 3 --distribution classf --features 4 --instances 150 --ngen 20000 --filename $times_executed

    echo "Start Cooldown"
    for i in $(seq 0 $cooldown)
    do
        echo "Cooldown: $(expr 5 - $i) mins"
        sleep 1m
    done
    echo "Cooldown has finished"
    
    let times_executed++
done