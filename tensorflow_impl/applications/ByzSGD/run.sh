#!/bin/bash

# rm config/*
# uniq $OAR_NODEFILE | python3 config_generator.py


# for filename in config/*; do
#     IP=$(echo $filename | cut -c18- | tr -d '\n')
    
#     oarsh $IP python3 Garfield_TF/applications/MSMW/trainer.py \
#         --config Garfield_TF/applications/MSMW/$filename \
#         --log True         \
#         --max_iter 2000    \
#         --batch_size 64    \
#         --dataset cifar10  \
#         --nbbyzwrks 3      \
#         --model VGG &
# done

common="cd Garfield/tensorflow_impl/applications/ByzSGD/"
common="$common && python3 trainer.py  \
            --log True                 \
            --max_iter 1000            \
            --batch_size 64"


while read p; do
  ssh ${p%:*} $common --config config/${p} < /dev/tty &
done < nodes