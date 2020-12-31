#!/bin/bash

if [ $# != 2 ]
  then
    echo "You should pass 2 arguments to this script: num_nodes and tensor_size respectively"
    exit
fi

nodes_names="nodes"
num_nodes=$1			#The upper bound on the number of nodes in any deployment
iter=10			#100000
master=''
nnodes=0
d=$2
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
common="python3 $pwd/rpc_bench.py --master $master --num_iter $iter --d $d --n $nnodes"
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
