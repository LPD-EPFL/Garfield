#!/bin/bash

ps_names="servers"
worker_names="workers"
num_wrk=100				#Upper limit on the number of workers
fw=0			#1
fps=0			#1
epochs=100
dataset='cifar10'
model='resnet50'
batch=250
loss='cross-entropy'
lr=0.2
momentum=0.9
wd=0.0005
backend='nccl'
gar='average'	#average
mar='vanilla'	#crash

master=''
num_ps=0
while read p; do
    	num_ps=$((num_ps+1))
	if [ $num_ps -eq  1 ]
  	then
     		master=$p
  	fi
done < $ps_names

num_workers=0
while read p; do
	num_workers=$((num_workers+1))
	if [ $num_workers -eq $num_wrk ]
  	then
     		break
  	fi
done < $worker_names

pwd=`pwd`

common="CUDAHOSTCXX=/usr/bin/gcc-5 python3 $pwd/trainer.py --master $master --epochs $epochs --dataset $dataset --model $model --batch $batch --loss $loss --lr $lr"
common="$common --momentum $momentum --wd $wd --num_ps $num_ps --num_workers $num_workers --fw $fw --fps $fps --aggregator $gar --mar $mar"
common="$common --backend $backend --log True" # --bench True" #--log True

i=0
while read p; do
	cmd="$common --rank $i"
	ssh $p "$cmd" < /dev/tty &
        echo "running $cmd on $p"
	i=$((i+1))
done < $ps_names

count_workers=0
while read p; do
        cmd="$common --rank $i"
        ssh $p "$cmd" < /dev/tty &
        echo "running $cmd on $p"
        i=$((i+1))
	count_workers=$((count_workers+1))
        if [ $count_workers -eq $num_wrk ]
        then
                break
        fi
done < $worker_names
