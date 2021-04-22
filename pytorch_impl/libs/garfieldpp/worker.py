# coding: utf-8
###
 # @file   worker.py
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
 # Worker class.
###

#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import garfieldpp.tools as tools
from garfieldpp.datasets import DatasetManager
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote
from time import sleep, time
import sys

class Worker:
    """ Byzantine-resilient worker """
    def __init__(self, rank, world_size, num_workers, batch_size, model, dataset, loss):
        """ Constructor of worker Object
        Args
        rank           unique ID of this worker node in the deployment
        world_size     total number of nodes in the deployment
        num_workers    total number of workers in the deployment
        batch_size     size of the batch to be used for training
        model          the name of the NN model to be used   FIXME: not used?
        dataset        the name of the dataset to be used for training
        loss           the name of the loss function to be applied
        """
        if torch.cuda.device_count() > 0:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu:0")
            print("Using CPU at rank {}".format(rank))
        self.rank = rank
        self.batch_size = batch_size
        self.loss = tools.select_loss(loss)
        manager = DatasetManager(dataset, batch_size, num_workers, world_size, rank)
        self.train_set = manager.get_train_set()        #train_set is a list of pairs: (data, target)
        self.num_train_samples = len(self.train_set)
        tools.worker_instance = self

    def compute_gradients(self, iter_num, model):
        """ compute gradients using the submitted model and a local batch size
        Args
        iter_num     the number of current iteration; this determines the local batch to be used for training
        model        the model state using which training should happen
        """
        with torch.autograd.profiler.profile(enabled=False) as prof:
            #First, fetch the correct batch from the training set, using iter_num
            model = model.to(self.device)
            model.train()
            data, target = self.train_set[iter_num%self.num_train_samples]
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = self.loss(output, target)
            loss.backward()
            #Now, we need to extract the full gradient from the model parameters
            grad = [torch.reshape(param.grad, (-1,)) for param in model.parameters()]
            grad_cat = torch.cat(grad).to("cpu")
#        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return self.rank, grad_cat, loss.item()
