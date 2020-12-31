# coding: utf-8
###
 # @file   trainer.py
 # @author Arsany Guirguis  <arsany.guirguis@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright (c) 2019 Arsany Guirguis.
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
 # Distributed SGD implementation. Deploy this file on both workers and parameter servers
###

#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import argparse
import sys
import time
import numpy as np

from math import ceil
from torch.autograd import Variable

from datasets import DatasetManager
from models import *
import aggregators

real_byzantine=False
def reduce_gradients(model, rank, device, index):
    """ Gradient averaging. This function should be applied only by the parameter server
	Args
	model	the used model to gather its gradients
	rank	rank of the current worker
	device	device which the model resides on
    """
    #Initialize few lists for benchmarking
    if bench:
      zero=[]
      aggr=[]
      gather=[]
      all_gather=[]
      flat=[]
      reshape=[]
      broad=[]
    # Communicate the gradients for each layer in the model layers
    for idx, param in enumerate(model.parameters()):
        #Parameter server always has the lowest rank numbers
        if rank < num_ps:		#PS....does not compute gradients
            zero_t = time.time()
            param.grad = torch.zeros(param.size()).to(device)
            if bench:
              zero.append(time.time() - zero_t)
        if rank < fps and real_byzantine:		#simulating Byzantine servers behavior
            param.data = torch.rand(param.size()).to(device)
        if real_byzantine and (rank < fw+num_ps and rank >= num_ps):	#simulating Byzantine workers behavior
            param.grad.data = torch.rand(param.size()).to(device)
        #If the vanilla PyTorch deployment is required, use the most effective aggregation method: reduce with sum then average
        if aggregator == 'vanilla':
          aggr_t = time.time()
          dist.reduce(param.grad.data, dst=0, op=dist.ReduceOp.SUM)
          param.grad.data /= float(num_workers)
          if bench and rank < num_ps:		#Aggregation happens only at PS
            aggr.append(time.time() - aggr_t)
        else:					#Using one of the aggregation rules (including average) is requested
          gather_list = []
          if rank < num_ps:			#PS branch....aggregation happens here
            zero_t = time.time()
            gather_list = [torch.zeros(param.size()).to(device) for _ in range(num_workers+1)]
            if bench:
              zero.append(time.time() - zero_t)
          if fps == 0 and mar != 'crash':				#This is only aggregathor then...no Byzantine PS
            gather_t = time.time()
            #Gather and then apply the requested aggregator later
            dist.gather(param.grad.data, gather_list, dst=0)
            if bench:
              gather.append(time.time() - gather_t)
          else:					#PS are also Byzantine....multiple PS should gather the gradients now
            gather_t = time.time()
            if rank < num_ps:		#This is a parameter server branch
              #As a PS, I gather all gradients from all worekers from the specific group from all_workers_ps group
              for i,worker_group in enumerate(all_ps_worker):
                dist.broadcast(gather_list[i+1], src=i+num_ps, group=worker_group)	#The (+1) in the first argument os to make the code compatible with the rest of the code based on "gather"
            else:
              # Each of these groups has all workers with one of the PS.....loop on all of them (as a worker here) and execute gather to send gradient to one PS
              dist.broadcast(param.grad.data, src=rank, group=all_ps_worker[rank-num_ps])
          if bench and rank < num_ps:
            gather.append(time.time() - gather_t)
          if rank < num_ps:
            if aggregator != "average":				#Average aggragthor does not require this reshaping so, skip it for averaging only
              flat_t = time.time()
              #flattening the gradients to be available for aggregators
              for i, l in enumerate(gather_list):
                gather_list[i].data = torch.reshape(l,(-1,)) #tools.flatten(l)
              if bench:
                flat.append(time.time() - flat_t)
            aggr_t = time.time()
            #aggregate gradients with some aggregation rule
#            print("1) gather list on cuda? ", gather_list[1].is_cuda)
            param.grad.data = gar(gradients=gather_list[1:], f=fw) #gar.aggregate(gather_list[1:])		#aggregate only workers gradients, excluding PS ZEROS
            if bench:
              aggr.append(time.time() - aggr_t)

        if rank < num_ps:						#If this is a PS, probably it requires some reshaping back
          reshape_t = time.time()
          param.grad.data = torch.reshape(param.grad.data, param.size())
          if bench:
            reshape.append(time.time()-reshape_t)

        #Now, broadcast the aggregated gradient....only if no Byzantine PS are there (vanilla or aggregathor)
        broad_t = time.time()
        if fps == 0 or mar == 'crash' or mar == 'vanilla':	#So, no need for multiple PS....the trusted PS can broadcast directly
          dist.broadcast(param.grad.data, src=0)
          if bench:
            broad.append(time.time() - broad_t)
        else:
          gather_t = time.time()
          #According to GuanYu, PS communicate the gradient between each other and get the median before sending to workers
          if rank < num_ps:						#PS branch
            gather_list = [torch.zeros(param.size()).to(device) for _ in range(num_ps)]
            #Use all gahter between the PS group....everybody sends to everybody
            dist.all_gather(gather_list, param.grad, group=all_ps)
#            dist.all_reduce(param.grad.data,op=dist.ReduceOp.SUM,group=all_ps)
#            param.grad.data /= num_ps
            if bench:
              all_gather.append(time.time() - gather_t)
            reshape_t = time.time()
            #Again, reshaping as aggregators do not accept shaped inputs (only flat inputs are accepted)
            for i, l in enumerate(gather_list):
              gather_list[i].data = torch.reshape(l,(-1,))
            if bench:
              reshape.append(time.time() - reshape_t)
            aggr_t = time.time()
            #Aggregate (theoretically models but practically gradients) with the model aggregation rule
#            print("2) gather list on cuda? ", gather_list[1].is_cuda)
            param.grad.data = mar(gradients=gather_list, f=fps) #mar.aggregate(gather_list)
            if bench:
              aggr.append(time.time() - aggr_t)
            #Reshape again
            reshape_t = time.time()
            param.grad.data = torch.reshape(param.grad.data, param.size())
            if bench:
              reshape.append(time.time() - reshape_t)

          gather_list=[]
          #Now workers should gather from all servers....
          if rank < num_ps:				#PS branch
            #Same logic as before but reversed a bit.....now, groups are each worker with all PS
            dist.broadcast(param.grad.data, src=rank, group=all_workers_ps[rank])
          else:
            gather_t = time.time()
            gather_list = [torch.zeros(param.size()).to(device) for _ in range(num_ps)]
            #As a worker, gather the new (theoreticall model, practically gradient) from all PS
            for i,group in enumerate(all_workers_ps):
              dist.broadcast(gather_list[i], src=i, group=group)
            if bench:
              gather.append(time.time() - gather_t)
            #Reshaping before aggregation
            reshape_t = time.time()
            for i, l in enumerate(gather_list):
              gather_list[i].data = torch.reshape(l,(-1,))
            if bench:
              reshape.append(time.time() - reshape_t)
            aggr_t = time.time()
            #Aggregation with the models aggregation rule
#            print("3) gather list on cuda? ", gather_list[1].is_cuda)
            param.grad.data = mar(gradients=gather_list, f=fps) #mar.aggregate(gather_list)
            if bench:
              aggr.append(time.time() - aggr_t)
            #Reshaping back tghe data
            param.grad.data = torch.reshape(param.grad.data, param.size())

    if bench and rank < num_ps:
      print("Time STATS for RANK {} at iteration {}".format(rank, index))
      print("Time for creating zeros: ", np.sum(zero))
      print("Time for aggregation with ", aggregator, " : ", np.sum(aggr))
      print("Time for gathering gradients: ", np.sum(gather))
      print("Time for gathering models: ", np.sum(all_gather))
      print("Time for flattening: ", np.sum(flat))
      print("Time for reshaping: ", np.sum(reshape))
      print("Time for broadcast: ", np.sum(broad))
      print("===============================================================")
      sys.stdout.flush()

def select_model(model, device):
    """ Select model to train
	Args
	model	model name required to be trained
	device	device to put model on (cuda or cpu)
    """
    if model == 'convnet':
      model = Net()
    elif model == 'cifarnet':
      model = Cifarnet()
    elif model == 'resnet50':
      model = torchvision.models.resnet50()
      #model = models.ResNet50()
    elif model == 'resnet152':
      model = torchvision.models.resnet152()
    elif model == 'inception':
      model = torchvision.models.inception_v3()
    elif model == 'vgg':
      model = torchvision.models.vgg19()
    else:
      print("The specified model is undefined")
      raise

    model = model.to(device)
    if device.type == "cuda":
      model = torch.nn.DataParallel(model)

    return model

def select_loss(loss_fn):
  """ Select loss function to optimize with
	Args
	loss_fn		Name of the loss function to optimize against
  """
  if loss_fn == 'nll':
    return nn.NLLLoss()
  elif loss_fn == 'cross-entropy':
    return nn.CrossEntropyLoss()
  else:
    print("The selected loss function is undefined")
    raise

def run(rank, size):
    """ Distributed Synchronous SGD main function
	Args
	rank	Rank of the current process
	size	Total size of the world (num_workers + num_servers)
    """

    # Preparing hyper-parameters
    torch.manual_seed(1234)
    manager = DatasetManager(dataset, minibatch, num_workers, size, rank)
    train_set, bsz = manager.get_train_set()
    test_set = manager.get_test_set()
    if torch.cuda.device_count() > 0: # and rank >= num_ps:
      device = torch.device("cuda") #(rank-num_ps)%torch.cuda.device_count()))
    else:
      device = torch.device("cpu:0")
      print("CPU WARNING =====================================================================")

    print("Rank {} -> Device {}".format(rank, device))
    model = select_model(model_n, device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd) #for Cifar10, 0.001 and 0.9, MNIST: 0.01 and 0.5
    if model_n == 'resnet50':
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    loss_fn = select_loss(loss_fn_n)
    g_l = [i for i in range(size)]
    world = dist.new_group(g_l)
    init_groups()
    #If PS are Byzantine, some subgroups of the world are required.....will be initialized as follows...
    print("-------------------------------- Rank {} have already done init groups...".format(rank))
    sys.stdout.flush()

    start_time = time.time()
    # Training loop
    print("One epoch has how many iterations: ", len(train_set))
    for epoch in range(epochs):
        epoch_loss = 0.0
        if model_n == 'resnet50':
          scheduler.step()
        model.train()
        for index, (data, target) in enumerate(train_set):
            if log:
              print("Rank {} Starting iteration {}".format(rank, index))
            train_time = time.time()
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            if rank >= num_ps:
                loss = loss_fn(output, target)
                loss.backward()
                epoch_loss += loss.item()
            if bench:
              print("Rank {} Train time {} ".format(rank, time.time() - train_time))
            if log:
              print("Rank {} Loop iteration {} Loss {}".format(rank,index, epoch_loss))
              sys.stdout.flush()
            reduce_time = time.time()
            reduce_gradients(model,rank, device, index)
            if bench:
              print("Rank {}, reduce time {} ".format(rank, time.time() - reduce_time))
            dist.barrier(world)
            optimizer.step()

        # Testing
        if rank < num_ps:
          test_time = time.time()
          acc = get_accuracy(model, test_set, device)
          print('Rank ', rank, ' epoch: ', epoch, ' acc: ', acc, "time: ", time.time() - start_time)
          print("Rank {}, test time {} ".format(rank, time.time() - test_time))
        else:
          print('Rank ', rank, 'epoch: ', epoch, 'loss: ', epoch_loss, "time: ", time.time() - start_time)
        sys.stdout.flush()


def get_accuracy(model, test_set, device):
    """ Calculate accuracy at some training step
	Args
	model		current state of the model
	test_set	testing set for model evaluation
	device		Device to place data on
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
      for idx, (inputs, targets) in enumerate(test_set):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _ , predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return (correct * 100 / total)


def init_groups():
  """ Creating communication groups at servers and workers. Since init groups should be executed the same way
	for all machines in the group, all machines will store groups of itself and other machines as well.
  """
  global all_ps
  global all_workers_ps
  if fps > 0 or mar == 'crash':            #Now we need to tolerate Byzantine PS -> Do GuanYu (for now)
    #Create groups of communication
    all_ps = dist.new_group([i for i in range(num_ps)])
    #Creating groups of all workers with each of the PS....useful in collecting gradients from workers on the PS side
    for ps in range(num_ps):
      g = [i+num_ps for i in range(num_workers+1)]
      g[-1] = ps
      all_workers_ps[ps] = dist.new_group(g)
    #Create groups of all PS with one worker...useful in collecting aggregated gradients by workers in GuanYu
    for worker in range(num_workers):
      g = [i for i in range(num_ps+1)]
      g[-1] = worker+num_ps
      all_ps_worker[worker] = dist.new_group(g)

def init_processes(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment.
	Args
	rank	Rank of the current process
	size	Number of participating machines (num_workers + num_ps)
	fn	Pointer to the main training function
	backend	Communication abstraction for communication among running machines (gloo or nccl)
    """
    os.environ['MASTER_ADDR'] = master
    os.environ['MASTER_PORT'] = '29500'
#    os.environ['NCCL_DEBUG'] = 'INFO'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


sys.stdout.flush()
parser = argparse.ArgumentParser(description="Distributed SGD playground", formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument("--master",
    type=str,
    default="",
    help="Master node in the deployment. This node takes rank 0, usually the first PS.")
parser.add_argument("--rank",
    type=int,
    default=0,
    help="Rank of a process in a distributed setup.")
parser.add_argument("--dataset",
    type=str,
    default="mnist",
    help="Dataset to be used, e.g., mnist, cifar10,...")
parser.add_argument("--batch",
    type=int,
    default=128,
    help="Minibatch size to be employed by each worker.")
parser.add_argument("--num_ps",
    type=int,
    default=1,
    help="Number of parameter servers in the deployment.")
parser.add_argument("--num_workers",
    type=int,
    default=1,
    help="Number of workers in the deployment.")
parser.add_argument("--fw",
    type=int,
    default=0,
    help="Number of declared Byzantine workers.")
parser.add_argument("--fps",
    type=int,
    default=0,
    help="Number of declared Byzantine parameter servers.")
parser.add_argument("--model",
    type=str,
    default='convnet',
    help="Model to be trained, e.g., convnet, cifarnet, resnet,...")
parser.add_argument("--loss",
    type=str,
    default='nll',
    help="Loss function to optimize against.")
parser.add_argument("--lr",
    type=float,
    default=0.01,
    help="Initial learning rate.")
parser.add_argument("--momentum",
    type=float,
    default=0.5,
    help="Momentum of learning.")
parser.add_argument("--wd",
    type=float,
    default=0,
    help="Learning weight decay.")
parser.add_argument("--epochs",
    type=int,
    default=10,
    help="Number of training epochs to execute.")
parser.add_argument("--aggregator",
    type=str,
    default='vanilla',
    help="Aggregation rule for aggregating gradients at the parameter server side. Put 'vanilla' for the native averaging.")
parser.add_argument("--mar",
    type=str,
    default='vanilla',
    help="Aggregation rule for aggregating models at both sides: parameter servers and workers. Put 'vanilla' for the native averaging.")
parser.add_argument("--backend",
    type=str,
    default='gloo',
    help="Backend for communication. This should be 'gloo' or 'nccl'.")
parser.add_argument('--bench',
    type=bool,
    default=False,
    help='If True, time elapsed in each step is printed.')
parser.add_argument('--log',
    type=bool,
    default=False,
    help='If True, accumulated loss at each iteration is printed.')


FLAGS = parser.parse_args(sys.argv[1:])

master = FLAGS.master
assert len(master) > 0
dataset = FLAGS.dataset
minibatch = FLAGS.batch
num_workers = FLAGS.num_workers
fw = FLAGS.fw
fps = FLAGS.fps
num_ps = FLAGS.num_ps
size = num_workers + num_ps
model_n = FLAGS.model
rank = FLAGS.rank
loss_fn_n = FLAGS.loss
lr = FLAGS.lr
momentum = FLAGS.momentum
wd = FLAGS.wd
epochs = FLAGS.epochs
aggregator = FLAGS.aggregator
mar = FLAGS.mar
bench = FLAGS.bench
log = FLAGS.log
backend = FLAGS.backend

os.environ['CUDA_VISIBLE_DEVICES'] = str((rank%2))

print("**** SETUP OF THIS EXPERIMENT ***")
print("Number of workers: ", num_workers)
print("Number of servers: ", num_ps)
print("Number of declared Byzantine workers: ", fw)
print("Number of declared Byzantine parameter servers: ", fps)
print("Algorithm: ", aggregator)
if mar != 'vanilla' and mar != 'crash':
  print("Model aggregator: ", mar)
print("Dataset: ", dataset)
print("Model: ", model_n)
print("Batch size: ", minibatch)
print("Loss function: ", loss_fn_n)
print("Learning rate args: lr {} momentum {} wd {}".format(lr,momentum,wd))
print("Benchmarking? ", bench)
print("Logging loss at each iteration?", log)
print("Using {} as a backend for communication".format(backend))
print("Printing this message from Machine number ", rank)
print("------------------------------------")
sys.stdout.flush()

#Initializations....
if aggregator != 'vanilla' and aggregator != 'average':
  assert fw > 0 and fw < num_workers

if mar != 'vanilla' and mar != 'average' and mar != 'crash':
  assert fps > 0 and fps < num_ps

if aggregator != 'vanilla':
  gar = aggregators.gars.get(aggregator) #instantiate(aggregator, f=fw)

if mar != 'vanilla' and mar != 'crash':
  mar = aggregators.gars.get(mar) #instantiate(mar, f=fps)
else:	#Does not have a meaning to run vanilla or crash-resilient algorithms with Byzantine nodes!
  assert fps == 0

if fps == 0 and aggregator != "vanilla" and backend != "gloo":
    print("WARNING: SWITCH TO GLOO BACKEND. AGGREGATHOR WORKS ONLY WITH GLOO AS NCCL DOES NOT SUPPORT GATHER.")
    backend = 'gloo'

all_ps=None
all_workers_ps=[None for _ in range(num_ps)]
all_ps_worker=[None for _ in range(num_workers)]
init_processes(rank, size, run,backend)
