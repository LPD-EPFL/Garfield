#!/bin/bash

nodes_names=nodes
num_nodes=10000			#The upper bound on the number of nodes in any deployment
iter=52			#100000
dataset='cifar10'
model=$1		#resnet18
optimizer='sgd'
batch=100			#250
loss='cross-entropy'
lr=0.2				#0.1
master=''
nnodes=0
while read p; do
    	nnodes=$((nnodes+1))
	if [ $nnodes -eq  1 ]
  	then
     		master=$p
  	fi
        if [ $num_nodes -eq $nnodes ]
        then
                break
        fi
done < $nodes_names

pwd=`pwd`
#CUDAHOSTCXX=/usr/bin/gcc-5
common="python3 $pwd/trainer.py --master $master --num_iter $iter --dataset $dataset --model $model --batch $batch --loss $loss"
common="$common --optimizer $optimizer --opt_args '{\"lr\":\"$lr\",\"momentum\":\"0.9\",\"weight_decay\":\"0.0005\"}' --num_nodes $nnodes"
i=0
while read p; do
        cmd="$common --rank $i"
        ssh $p "$cmd" < /dev/tty &
        echo "running $cmd on $p"
        i=$((i+1))
        if [ $i -eq $nnodes ]
        then
                break
        fi
done < $nodes_names
