# coding: utf-8
###
 # @file   server.py
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
 # Parameter Server class.
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
from garfieldpp.byzWorker import ByzWorker
from time import sleep,time
import copy
import sys
import threading
from multiprocessing.dummy import Pool as ThreadPool

class Server:
    """ Byzantine-resilient parameter server """
    def __init__(self, rank, world_size, num_workers, num_ps, byz_wrk, byz_ps, wrk_base_name, ps_base_name, batch, model, dataset, optimizer,  *args, **kwargs):
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
        args, kwargs   additional arguments to be passed to the optimizaer constructor
        """
        if torch.cuda.device_count() > 0:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu:0")
            print("Using CPU at rank {}".format(rank))
        self.rank = rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.byz_wrk = byz_wrk
        self.byz_ps = byz_ps
        if world_size > 1:
            self.workers_types, self.workers_rref = self.get_rrefs(wrk_base_name,0,num_workers, True)
        self.num_ps = num_ps
        self.model = tools.select_model(model, torch.device("cpu:0"))	#We should always put the model on CPU because RPC is not supported on GPUs
        manager = DatasetManager(dataset, batch*num_workers, 1, 2, 1)			#The parameters actually are dummy
        self.test_set = manager.get_test_set()
        self.train_set = manager.get_train_set()
        self.optimizer = tools.select_optimizer(self.model, optimizer,  *args, **kwargs)
#        self.grads = []						#placeholder to carry gradients in each iteration
        tools.server_instance = self
        if self.num_ps > 1:			#This should be done after the server announces itself; otherwise, we'd fall in deadlock
            self.ps_types, self.ps_rref = self.get_rrefs(ps_base_name,0, num_ps, False)
#        self.pool_wrk = ThreadPool()                            #No need to specify the number of concurrent processes; try to use max anyway
        self.latest_aggr_grad = None

    def get_rrefs(self, base_name, base_id, num_nodes, worker=True):
        """ get rrefs of remote machines (workers or servers)
        Args
        base_name     template name of deployed workers
        base_id       the lowest rank of a node in the deployment
        num_nodes     the number of nodes of which the server should fetch references
        """
        rrefs = [remote(base_name+str(node_id), get_worker if worker else get_server) for node_id in range(base_id, base_id+num_nodes)]
        types = [type(rref.to_here()) for rref in rrefs]
        return types, rrefs

    def get_gradients(self, iter_num, num_wait_wrk=-1):
        """ ask workers to compute gradients and return them
        Args
        iter_num     the number of the current iteration, to be passed to workers
        num_wait_wrk number of workers to wait for their response (useful in asynchronous deployments)
        """
        if num_wait_wrk < 0:
            num_wait_wrk = self.num_workers - self.byz_wrk
        self.model.train()
        self.optimizer.zero_grad()
        #Fast path
        if num_wait_wrk == self.num_workers:
            def get_grad(fut):
                return fut.wait()[1].to(self.device)
            pool_wrk = ThreadPool()
            futs = [_remote_method_async(wrk_type.compute_gradients, wrk_rref, iter_num, self.model) for wrk_rref, wrk_type in zip(self.workers_rref, self.workers_types)]
            build_th = threading.Thread(target=self.build_graph, args=(iter_num,))
            build_th.start()
            grads = pool_wrk.map(get_grad, futs)
            pool_wrk.terminate()
            pool_wrk.join()
            del pool_wrk
        else:       #The call should be done asynchronously, and we should only consider the fastest workers responding
            grads=[None for i in range(self.num_workers)]           #placeholder for possible received gradients
            def assign_grad_async(fut):
                """ callback function that is called when some gradient is received asynchronously
                Args
                fut     the future object on which the callback is called
                """
                id, gradient, loss = fut.wait()
                grads[id] = gradient.to(self.device)
            for id, (wrk_rref, wrk_type) in enumerate(zip(self.workers_rref,self.workers_types)):
                fut = _remote_method_async(wrk_type.compute_gradients, wrk_rref, iter_num, self.model)
                #fut.then allows to do something after the future object returns a result
                #x here is the future object itself; result of waiting it should return a grad from that worker
                fut.then(assign_grad_async)
            #busy-wait for the replies
            build_th = threading.Thread(target=self.build_graph, args=(iter_num,))
            build_th.start()
            while self.num_workers - grads.count(None) < num_wait_wrk:
                sleep(1)
            #now, I'm sure I have at least "num_wait_wrk" replies at least
            #let's remove all "None" values
            grads = [grad for grad in grads if grad is not None]
#            del grads				#empty it for the next iteration
        #make sure that the graph is built (regardless of synchrony or not)
        build_th.join()
        return grads

    def get_models(self, num_wait_ps=-1):
        """ ask servers to get their latest models
        Args
        num_wait_ps number of servers to wait for their response (useful in asynchronous deployments)
        """
        if num_wait_ps < 0:
            num_wait_ps = self.num_ps - self.byz_ps
        if num_wait_ps == self.num_ps:			#FAST PATH: synchronous
            futs = [_remote_method_async(ps_type.get_model, ps_rref) for ps_rref, ps_type in zip(self.ps_rref, self.ps_types)]
            models = [fut.wait()[1].to(self.device) for fut in futs]
        else:						#Asynchronous path
            models_ph=[None for i in range(self.num_ps)]           #placeholder for possible received models
            def assign_model_async(fut):
                id, mod = fut.wait()
                models_ph[id] = mod.to(self.device)
            for id, (ps_rref, ps_type) in enumerate(zip(self.ps_rref,self.ps_types)):
                fut = _remote_method_async(ps_type.get_model, ps_rref)
                #fut.then allows to do something after the future object returns a result
                #x here is the future object itself; result of waiting it should return a model from that server
                fut.then(assign_model_async)
            while self.num_ps - models_ph.count(None) < num_wait_ps:
                sleep(1)
            models = [mod for mod in models_ph if mod is not None]
        return models

    def build_graph(self, iter_num):
        """ Prepares the computation graph for the update step
        Args
        iter_num     the iteration number
        """
        data, _ = self.train_set[iter_num%len(self.train_set)]
        data, self.model = data.to(self.device), self.model.to(self.device)
        self.model(data)
        self.model = self.model.to('cpu')

    def get_model(self):
        """ return the current model
        Args
        """
        return self.rank, torch.cat([torch.reshape(param.data.to("cpu"), (-1,)) for param in self.model.parameters()]).to('cpu:0')

    def get_latest_aggr_grad(self):
        """ return the latest aggregated gradient at a server
        Args
        """
        while self.latest_aggr_grad is None:		#useful only with initialization
            sleep(1)
        return self.rank, self.latest_aggr_grad.to('cpu:0')

    def get_aggr_grads(self, num_wait_ps=-1):
        """ ask servers to get their latest aggregated gradients
        Args
        num_wait_ps number of servers to wait for their response (useful in asynchronous deployments)
        """
        if num_wait_ps < 0:
            num_wait_ps = self.num_ps - self.byz_ps
        if num_wait_ps == self.num_ps:                  #FAST PATH: synchronous
            futs = [_remote_method_async(ps_type.get_latest_aggr_grad, ps_rref) for ps_rref, ps_type in zip(self.ps_rref, self.ps_types)]
            aggr_grads = [fut.wait()[1].to(self.device) for fut in futs]
        else:                                           #Asynchronous path
            aggr_grads_ph=[None for i in range(self.num_ps)]           #placeholder for possible received aggregated gradients
            def assign_grads_async(fut):
                id, aggr_grad = fut.wait()
                aggr_grads_ph[id] = aggr_grad.to(self.device)
            for id, (ps_rref, ps_type) in enumerate(zip(self.ps_rref,self.ps_types)):
                fut = _remote_method_async(ps_type.get_latest_aggr_grad, ps_rref)
                #fut.then allows to do something after the future object returns a result
                #x here is the future object itself; result of waiting it should return a model from that server
                fut.then(assign_grads_async)
            while self.num_ps - aggr_grads_ph.count(None) < num_wait_ps:
                sleep(1)
            aggr_grads = [ag for ag in aggr_grads_ph if ag is not None]
        return aggr_grads

    def compute_accuracy(self):
        """ compute the accuracy of the current model, based on the test set
        Args
        """
        correct = 0
        total = 0
        model_cpy = copy.deepcopy(self.model).cuda()
        model_cpy.eval()
        with torch.no_grad():
          for idx, (inputs, targets) in enumerate(self.test_set):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model_cpy(inputs)
            _ , predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        del model_cpy
        return (correct * 100 / total)

    def update_model(self, grad):
        """ update the local model using the updating gradient
        Args
        grad         the updating gradient
        """
        grad = grad.to(torch.device("cpu:0"))
        cur_pos=0
        for param in self.model.parameters():
            param.grad = torch.reshape(torch.narrow(grad,0,cur_pos,param.nelement()), param.size()).detach()
            cur_pos = cur_pos+param.nelement()
        self.optimizer.step()

    def write_model(self, model):
        """ replace the local model with the one given
        Args
        model         the new model (flattened) to replace the old one
        """
        cur_pos=0
        for param in self.model.parameters():
            param.data = torch.reshape(torch.narrow(model,0,cur_pos,param.nelement()), param.size())
            cur_pos = cur_pos+param.nelement()

#The next two methods are only used for benchmarking RPC calls
    def get_fake_models(self):
        futs = [_remote_method_async(ps_type.get_fake_model, ps_rref) for ps_rref, ps_type in zip(self.ps_rref, self.ps_types)]
        models = [fut.wait().to(self.device) for fut in futs]

    def get_fake_model(self):
        return self.model.to('cpu:0')
