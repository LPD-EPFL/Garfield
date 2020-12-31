# coding: utf-8
###
 # @file   byzServer.py
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
 # Byzantine Parameter Server class; implements some attacks of the parameter server
###

#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import garfieldpp.tools as tools
from garfieldpp.tools import _call_method, _remote_method_sync, _remote_method_async, get_worker, get_server
from garfieldpp.datasets import DatasetManager
import torch.distributed.rpc as rpc
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote
from garfieldpp.worker import Worker
from garfieldpp.server import Server
from garfieldpp.byzWorker import ByzWorker
from time import sleep,time
import copy
import sys
import threading
from multiprocessing.dummy import Pool as ThreadPool

class ByzServer(Server):
    """ Byzantine-resilient parameter server """
    def __init__(self, rank, world_size, num_workers, num_ps, byz_wrk, byz_ps, wrk_base_name, ps_base_name, batch, model, dataset, optimizer, attack,  *args, **kwargs):
        """ Constructor of server Object
        Args
        rank           unique ID of this worker node in the deployment
        world_size     total number of nodes in the deployment
        num_workers    number of workers in the deployment
        num_ps	       number of servers in the deployment
        byz_wrk        number of (possible) Byzantine workers in the deployment
        byz_ps         number of (possible) Byzantine servers in the deployment
        wrk_base_name  the naming convention of workers; used to get the rrefs of remote workers
        ps_base_name   the naming convention of servers; used to get the rrefs of remote servers
        batch	       the batch size per worker; used to build the computation graph
        model          the name of the NN model to be used
        dataset        the name of the dataset to be used for training
        optimizer      the name of the optimizaer used by the server
        attack         name of the attack to be applied by this server
        args, kwargs   additional arguments to be passed to the optimizaer constructor
        """
        super().__init__(rank, world_size, num_workers, num_ps, byz_wrk, byz_ps, wrk_base_name, ps_base_name, batch, model, dataset, optimizer, *args, **kwargs)
        attacks = {'random':self.random_attack,
                    'reverse':self.reverse_attack,
                    'drop':self.partial_drop_attack}
        assert attack in attacks, "The requested attack is not implemeneted; available attacks are:"+str(attacks.keys())
        self.attack = attacks[attack]

    def get_model(self):
        """ return the current model
        Args
        """
        return self.attack()

    def random_attack(self):
        """ return a random model with the same size of the employed model
        Args
        """
        model = super().get_model()
        return torch.rand(model.size())

    def reverse_attack(self):
        """ return the model, yet in the opposite direction and amplified
        Args
        """
        model = super().get_model()
        return model*-100

    def partial_drop_attack(self):
        """ return the model but with some missing coordinates (replaced by zeros)
        Args
        """
        model = super().get_model()
        p=0.3                   #percent of the values that should be replaced by zeros
        mask = torch.rand(len(model)) > 1-p
        model.masked_fill(mask, 0)
        return model
