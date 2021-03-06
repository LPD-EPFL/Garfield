#!/bin/bash

# rm -r config/*
# uniq $OAR_NODEFILE | python3 config_generator.py


# for filename in config/*; do
#     IP=$(echo $filename | cut -c8- | tr -d '\n')
    
#     oarsh $IP python3 Garfield_TF/applications/LEARN/trainer.py \
#         --config_w Garfield_TF/applications/LEARN/config/$IP/TF_CONFIG_W \
# 	    --config_ps Garfield_TF/applications/LEARN/config/$IP/TF_CONFIG_PS \
#        	--log True         \
#         --max_iter 2001     \
#         --batch_size 64    \
#         --dataset cifar10  \
#     	--nbbyzwrks 3      \
#         --model Cifarnet &
# done



common="cd Garfield/tensorflow_impl/applications/LEARN/"
common="$common && python3 trainer.py  \
            --log True                 \
            --max_iter 1000            \
            --batch_size 64"


while read p; do
  ssh ${p%:*} $common --config_w config/${p}/TF_CONFIG_W --config_ps config/${p}/TF_CONFIG_PS < /dev/tty &
done < nodes