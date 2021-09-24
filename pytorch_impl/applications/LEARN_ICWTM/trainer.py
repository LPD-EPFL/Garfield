# coding: utf-8
###
 # @file   trainer.py
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
 #
 # LEARN implementation: a Byzantine-resilient decentralized ML system
 # based on "Collaborative Learning as an Agreement Problem"
 # Note that here all nodes are equa: no PS and no workers
 # Basically each node has PS object and worker object
 # original paper: https://arxiv.org/abs/2008.00742
###

#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote
from time import time, sleep
import argparse
import sys
import json
import threading

import garfieldpp
from garfieldpp.worker import Worker
from garfieldpp.server import Server
from garfieldpp.tools import get_bytes_com,convert_to_gbit, adjust_learning_rate

import aggregators
from math import log2, ceil

CIFAR_NUM_SAMPLES = 50000
#First, parse the inputs
parser = argparse.ArgumentParser(description="LEARN implementation using Garfield library", formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument("--master",
    type=str,
    default="",
    help="Master node in the deployment. This node takes rank 0, usually the first node.")
parser.add_argument("--rank",
    type=int,
    default=0,
    help="Rank of a process in the distributed setup.")
parser.add_argument("--dataset",
    type=str,
    default="mnist",
    help="Dataset to be used, e.g., mnist, cifar10.")
parser.add_argument("--batch",
    type=int,
    default=32,
    help="Minibatch size to be employed by each node.")
parser.add_argument("--num_nodes",
    type=int,
    default=1,
    help="Number of nodes in the deployment.")
parser.add_argument("--f",
    type=int,
    default=0,
    help="Number of declared Byzantine workers.")
parser.add_argument("--model",
    type=str,
    default='convnet',
    help="Model to be trained, e.g., convnet, cifarnet, resnet.")
parser.add_argument("--loss",
    type=str,
    default='nll',
    help="Loss function to optimize.")
parser.add_argument("--optimizer",
    type=str,
    default='sgd',
    help="Optimizer to use.")
parser.add_argument("--opt_args",
    type=json.loads,
    default={'lr':'0.1'},
    help="Optimizer arguments; passed in dict format, e.g., '{\"lr\":\"0.1\"}'")
parser.add_argument("--num_iter",
    type=int,
    default=5000,
    help="Number of training iterations to execute.")
parser.add_argument("--gar",
    type=str,
    default='average',
    help="Aggregation rule for aggregating gradients.")
parser.add_argument('--acc_freq',
    type=int,
    default=100,
    help="The frequency of computing accuracy while training.")
parser.add_argument('--non_iid',
    type=bool,
    default=True,
    help="If True, the algorithm with non-iid data will be executed; otherwise, iid data will be assumed. Note: this flag does not itslef change the distriibution of data over workers.")
parser.add_argument('--bench',
    type=bool,
    default=False,
    help="If True, time elapsed in each step is printed.")
parser.add_argument('--log',
    type=bool,
    default=False,
    help="If True, accumulated loss at each iteration is printed.")


FLAGS = parser.parse_args(sys.argv[1:])

master = FLAGS.master
assert len(master) > 0

rank = FLAGS.rank
assert rank >= 0

n = FLAGS.num_nodes
assert n >= 1
world_size = n

f = FLAGS.f
assert f*2 < n

dataset = FLAGS.dataset
assert len(dataset) > 0
batch = FLAGS.batch
assert batch >= 1
model = FLAGS.model
assert len(model) > 0
loss = FLAGS.loss
assert len(loss) > 0
optimizer = FLAGS.optimizer
assert len(optimizer) > 0
opt_args = FLAGS.opt_args
for k in opt_args:
    opt_args[k] = float(opt_args[k])
assert opt_args['lr']

num_iter = FLAGS.num_iter
assert num_iter > 0

gar = FLAGS.gar
assert len(gar) > 0

acc_freq = FLAGS.acc_freq
assert(acc_freq > 10)
non_iid=FLAGS.non_iid
bench = FLAGS.bench
if bench:
  from timeit import timeit
else:
  timeit = None
log = FLAGS.log

#os.environ['CUDA_VISIBLE_DEVICES'] = str((rank%2))

print("**** SETUP AT NODE {} ***".format(rank))
print("Number of nodes: ", n)
print("Number of declared Byzantine nodes: ", f)
print("GAR: ", gar)
print("Dataset: ", dataset)
print("Model: ", model)
print("Batch size: ", batch)
print("Loss function: ", loss)
print("Optimizer: ", optimizer)
print("Optimizer Args", opt_args)
print("Assume Non-iid data? ", non_iid)
print("Benchmarking? ", bench)
print("Logging loss at each iteration?", log)
print("------------------------------------")
sys.stdout.flush()

lr = opt_args['lr']
#initiating the GAR
gar = aggregators.gars.get(gar)
assert gar is not None

os.environ['MASTER_ADDR'] = master
os.environ['MASTER_PORT'] = '29700'
torch.manual_seed(1234)					#For reproducibility
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(1234)                                    #For reproducibility
if bench:
  torch.backends.cudnn.benchmark=True

def avg_agree(ps, gar, aggr_grad, lnum_iter, num_wait_ps):
  """Execute the average agreement protocol as in the paper
     Basically, exchange and aggregate gradients for log2(t) time
     Args
        ps		the local server object
        gar		GAR used for aggregation
        aggr_grad	the initial aggregated gradient
        lnum_iter	the number of iterations to be done; should be log2(t)
        num_wait_ps	the number of servers that should be waited for
  """
  for _ in range(lnum_iter):
    ps.latest_aggr_grad = aggr_grad
    aggr_grads = ps.get_aggr_grads(num_wait_ps)
    for _ in range(num_wait_ps*2):			#simulate reliable and FIFO broadcasts in the best case
      ps.latest_aggr_grad = aggr_grad
      aggr_grads = ps.get_aggr_grads(num_wait_ps)
    aggr_grad = gar(gradients=aggr_grads, f=f)
  return aggr_grad

#No branching here! All nodes are created equal: no PS and no workers
#Basically, each node has one PS object and one worker object
rpc.init_rpc('node:{}'.format(rank), rank=rank, world_size=world_size)
#rpc._set_rpc_timeout(100000)
#initialize a worker here...the worker is created first because the server relies on the worker creation
Worker(rank, world_size, n, batch, model, dataset, loss)
#Initialize a parameter server
ps = Server(rank, world_size, n, n, f, f,  'node:', 'node:', batch, model, dataset, optimizer, **opt_args)
sleep(30)                        #works as a synchronization step
scheduler = torch.optim.lr_scheduler.MultiStepLR(ps.optimizer, milestones=[150, 250, 350], gamma=0.1)		#This line shows sophisticated stuff that can be done out of the Garfield++ library
start_time = time()
iter_per_epoch = CIFAR_NUM_SAMPLES//(n * batch)		#this value records how many iteration per sample
print("One EPOCH consists of {} iterations".format(iter_per_epoch))
sys.stdout.flush()
#Training loop
for i in range(num_iter):
  print("iter: {} time {}".format(i,time()-start_time))
  sys.stdout.flush()
  if i%(iter_per_epoch*30) == 0 and i!=0:			#One hack for better convergence with Cifar10
    lr*=0.2
    adjust_learning_rate(ps.optimizer, lr)
  #training loop goes here
  def train_step():
    if bench:
      bytes_rec = get_bytes_com()			#record number of bytes sent before the training step to work as a checkpoint
    with torch.autograd.profiler.profile(enabled=bench) as prof:
      gradients = ps.get_gradients(i, n-f)     #get_gradients(iter_num, num_wait_wrk)
      #Aggregating gradients once is good for IID data; for non-IID, one should execute this log2(i)
      aggr_grad = gar(gradients=gradients, f=f)
      if i > 0:
        aggr_grad = avg_agree(ps, gar, aggr_grad, ceil(log2(i+1)) if non_iid else 1, n-f)
      else:
        ps.latest_aggr_grad = aggr_grad
      ps.update_model(aggr_grad)
      #Then, communicate models, aggregate, and write the new model
      for _ in range(2*(n-f)):		#simulate reliable and FIFO broadcasts
        models = ps.get_models(n-f)
      aggr_models = gar(gradients=models,f=f)
      ps.write_model(aggr_models)
      ps.model.to('cpu:0')
      if bench:
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        bytes_train = get_bytes_com()
        print("Consumed bandwidth in this iteration: {} Gbits".format(convert_to_gbit(bytes_train-bytes_rec)))
#        print("Memory allocated to GPU {} Memory cached on GPU {}".format(torch.cuda.memory_allocated(0), torch.cuda.memory_cached(0)))
        sys.stdout.flush()
  if timeit is not None:
    res = timeit(train_step,number=1)
    print("Training step {} takes {} seconds".format(i,res))
    sys.stdout.flush()
  else:
    train_step()

  if i%iter_per_epoch == 0:
    def test_step():
      acc = ps.compute_accuracy()
      num_epochs = i/iter_per_epoch
      print("Epoch: {} Accuracy: {} Time: {}".format(num_epochs,acc,time()-start_time))
      sys.stdout.flush()
    if timeit is not None:
      res = timeit(test_step,number=1)
      print("Test step takes {} seconds".format(res))
    else:
      test_step()		#Though threading is a good idea, applying it here messes the use of CPU with GPU
#      if model.startswith('resnet') and i!=0:
#        scheduler.step()
#      threading.Thread(target=test_step).start()

rpc.shutdown()
