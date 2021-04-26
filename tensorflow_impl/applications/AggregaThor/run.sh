#!/bin/bash

dir=config

if [[ ! -e $dir ]]; then
    echo "Creating $dir folder" 1>&2
    mkdir $dir
else
    echo "$dir folder already exists, removing previous configuration" 1>&2
    rm $dir/*
fi

python3 config_generator.py

for filename in config/*; do
    xterm -hold -e python3 trainer.py --config $filename --log True --max_iter 1000 --batch_size 126 & 
done