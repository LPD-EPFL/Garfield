# coding: utf-8
###
 # @file   rpc_bench.py
 # @author Arsany Guirguis  <arsany.guirguis@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright (c) 2020 Arsany Guirguis.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in all
 # copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 # @section DESCRIPTION
 # Benchmarking RPC calls in the Garfield++ framework.
###

#!/usr/bin/env python

import os
import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_async, remote
from time import time, sleep
import argparse
import sys

import garfieldpp
from garfieldpp.worker import Worker
from garfieldpp.server import Server
from garfieldpp.tools import get_bytes_com,convert_to_gbit, adjust_learning_rate

import aggregators
from math import log2, ceil

#First, parse the inputs
parser = argparse.ArgumentParser(description="Benchmarking RPC calls in the Garfield++ framework", formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument("--master",
    type=str,
    default="",
    help="Master node in the deployment. This node takes rank 0, usually the first node.")
parser.add_argument("--rank",
    type=int,
    default=0,
    help="Rank of a process in a distributed setup.")
parser.add_argument("--d",
    type=int,
    default=100000,
    help="Simulated model size")
parser.add_argument("--n",
    type=int,
    default=1,
    help="Number of nodes in the deployment.")
parser.add_argument("--num_iter",
    type=int,
    default=10,
    help="Number of RPC calls to do (for statistical variance).")


FLAGS = parser.parse_args(sys.argv[1:])

master = FLAGS.master
assert len(master) > 0
rank = FLAGS.rank
assert rank >= 0
n = FLAGS.n
assert n >= 1
world_size = n
d = FLAGS.d
assert d >= 1
num_iter = FLAGS.num_iter
assert num_iter > 0

from timeit import timeit
dev = 'cuda'
os.environ['MASTER_ADDR'] = master
os.environ['MASTER_PORT'] = '27800'

#Basically, each node has one PS object and one worker object
rpc.init_rpc('node:{}'.format(rank), rank=rank, world_size=world_size)
#initialize a worker here...the worker is created first because the server relies on the worker creation
Worker(rank, world_size, n, 32, 'cnn', 'mnist', 'nll')
#Initialize a parameter server
ps = Server(rank, world_size, n, n, 0, 0,  'node:', 'node:', 32, 'cnn', 'mnist', 'sgd', **{'lr':0.1})
sleep(5)			#works as a synchronization step
ps.model = torch.rand(d).to(dev)
#Training loop
ranks = [i for i in range(world_size)]
#ranks = [0]
if rank in ranks:
    for i in range(num_iter):
        ts = time()
        if dev == 'cuda':
            ps.model.to('cpu:0')
        ta = time()
        models = ps.get_fake_models()
        tf = time()
        ps.model.to(dev)
        print(" ** d={} n={} actual_transfer_time={} total_time={} **".format(d,n,tf-ta,time()-ts))

rpc.shutdown()
print("End of test")
