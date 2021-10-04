#!/bin/bash

pwd=`pwd`
common="cd $pwd"
common="$common && /usr/local/bin/python3 trainer.py --log True --max_iter 1000 --batch_size 126"

#/usr/local/bin/python3
#for filename in config/*; do
#    echo "$filename"
#    IP=$(echo $filename | cut -c18- | tr -d '\n')
#    #echo "ssh -p 9092 root@$IP $common --config $filename < /dev/tty &"
#    ssh -p 9092 root@$IP "$common --config $filename" < /dev/tty &
#done

while read p; do
  ssh ${p%:*} $common --config config/${p} < /dev/tty &
done < nodes

#ssh -p 9092 root@127.0.0.1 $common --config config/TF_CONFIG-127.0.0.1 < /dev/tty &
#ssh -p 9093 root@127.0.0.1 $common --config config/TF_CONFIG_127.0.0.1 < /dev/tty &


#if [[ ! -e $dir ]]; then
#    echo "Creating $dir folder" 1>&2
#    mkdir $dir
#else
#     echo "$dir folder already exists, removing previous configuration" 1>&2
#     rm $dir/*
# fi

# python3 config_generator.py

# for filename in config/*; do
#     xterm -hold -e python3 trainer.py --config $filename --log True --max_iter 1000 --batch_size 126 & 
# done
