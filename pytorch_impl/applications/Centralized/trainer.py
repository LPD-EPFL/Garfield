# coding: utf-8
###
 # @file   cent_trainer.py
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
 # Centralized training. This should work as a baseline to our distributed implementation whatsoever.
###

#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
import argparse
import sys
import json
import threading

import garfieldpp
from garfieldpp.server import Server
from garfieldpp.worker import Worker
from garfieldpp import tools
from garfieldpp.datasets import DatasetManager
from garfieldpp.tools import adjust_learning_rate
import aggregators

CIFAR_NUM_SAMPLES = 50000
#First, parse the inputs
parser = argparse.ArgumentParser(description="Centralized Training", formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument("--dataset",
    type=str,
    default="mnist",
    help="Dataset to be used, e.g., mnist, cifar10,...")
parser.add_argument("--batch",
    type=int,
    default=32,
    help="Minibatch size to be employed by each worker.")
parser.add_argument("--model",
    type=str,
    default='convnet',
    help="Model to be trained, e.g., convnet, cifarnet, resnet,...")
parser.add_argument("--loss",
    type=str,
    default='nll',
    help="Loss function to optimize against.")
parser.add_argument("--optimizer",
    type=str,
    default='sgd',
    help="Optimizer to use.")
parser.add_argument("--opt_args",
    type=json.loads,
    default={'lr':0.1},
    help="Optimizer arguments; passed in dict format, e.g., '{\"lr\":\"0.1\"}'")
parser.add_argument("--num_iter",
    type=int,
    default=100,
    help="Number of training iterations to execute.")
parser.add_argument("--acc_freq",
    type=int,
    default=100,
    help="The frequency of computing accuracy while training.")
parser.add_argument('--bench',
    type=bool,
    default=False,
    help="If True, time elapsed in each step is printed.")
parser.add_argument('--log',
    type=bool,
    default=False,
    help="If True, accumulated loss at each iteration is printed.")


FLAGS = parser.parse_args(sys.argv[1:])

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

acc_freq = FLAGS.acc_freq
assert(acc_freq > 10)
bench = FLAGS.bench
if bench:
  from timeit import timeit
else:
  timeit = None
log = FLAGS.log

print("**** SETUP OF THIS CENTRAL DEPLOYMENT ***")
print("Dataset: ", dataset)
print("Model: ", model)
print("Batch size: ", batch)
print("Loss function: ", loss)
print("Optimizer: ", optimizer)
print("Optimizer Args", opt_args)
print("Benchmarking? ", bench)
print("Logging loss at each iteration?", log)
print("------------------------------------")
sys.stdout.flush()

lr = opt_args['lr']

torch.manual_seed(1234)					#For reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
if bench:
  torch.backends.cudnn.benchmark=True

ps = Server(0, 1, 1,1, 0, 0, ' ', ' ', batch, model, dataset, optimizer, **opt_args)
wrk = Worker(1, 2, 1, batch, model, dataset, loss)
gar = aggregators.gars.get('average')

num_train_samples = len(wrk.train_set)
iter_per_epoch = CIFAR_NUM_SAMPLES//batch
assert num_train_samples == iter_per_epoch
scheduler = torch.optim.lr_scheduler.MultiStepLR(ps.optimizer, milestones=[25, 50], gamma=0.1)		#This line shows sophisticated stuff that can be done out of the Garfield++ library
start_time = time()
print("One EPOCH consists of {} iterations".format(iter_per_epoch))
sys.stdout.flush()
#acc_freq = iter_per_epoch
if torch.cuda.device_count() > 0:
  device = torch.device("cuda")
else:
  device = torch.device("cpu:0")
scheduler.step()
for i in range(num_iter):
#  if i%iter_per_epoch == 0 and model == 'resnet50':
#    if i%(iter_per_epoch*30) == 0:			#One hack for better convergence with Cifar10
#      lr*=0.1
#      adjust_learning_rate(ps.optimizer, lr)
    #training loop goes here
  ps.model.train()
  ps.optimizer.zero_grad()
  r, grad, l = wrk.compute_gradients(i, ps.model)
  grad = grad.to(device)
  grad = gar(gradients=[grad], f=0)
  ps.model = ps.model.to('cpu')
  ps.update_model(grad)
  if i%num_train_samples == 0 and i != 0:
    acc = ps.compute_accuracy()
    num_epochs = i/num_train_samples
    print("Epoch: {} Accuracy: {} Time: {}".format(num_epochs,acc,time()-start_time))
    sys.stdout.flush()
    scheduler.step()
