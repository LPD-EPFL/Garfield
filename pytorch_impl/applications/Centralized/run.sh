#!/bin/bash

node=$1
iter=100000
dataset='cifar10'
model='resnet50'
batch=250
loss='cross-entropy'
lr=0.2

pwd=`pwd`

common="CUDAHOSTCXX=/usr/bin/gcc-5 python3 $pwd/trainer.py --num_iter $iter --dataset $dataset --model $model --batch $batch --loss $loss"
common="$common --optimizer sgd --opt_args '{\"lr\":\"$lr\",\"momentum\":\"0.9\",\"weight_decay\":\"0.0005\"}'"
cmd="$common"
ssh $node "$cmd" < /dev/tty &
echo "running $cmd on $node"
