# coding: utf-8
###
 # @file   byzWorker.py
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
 # Byzantine worker class: inherits the typical worker class and applies some set of attacks.
###

#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from garfieldpp.worker import Worker
import garfieldpp.tools as tools
from time import sleep, time
import sys

class ByzWorker(Worker):
    """ Byzantine worker """
    def __init__(self, rank, world_size, num_workers, batch_size, model, dataset, loss, attack, fw=1):
        """ Constructor of Byzanrtine worker Object
        Args
        world_size     total number of nodes in the deployment
        num_workers    total number of workers in the deployment
        rank           unique ID of this worker node in the deployment
        batch_size     size of the batch to be used for training
        model          the name of the NN model to be used
        dataset        the name of the dataset to be used for training
        loss           the name of the loss function to be applied
        attack	       name of the attack to be applied by this worker
        fw             the number of Byzantine workers; used to simulate sophisticated attacks
        """
        super().__init__(rank, world_size, num_workers, batch_size, model, dataset, loss)
        self.fw = fw
        attacks = {'random':self.random_attack,
			'reverse':self.reverse_attack,
			'drop':self.partial_drop_attack,
			'lie':self.little_is_enough_attack,
			'empire':self.fall_empires_attack}
        assert attack in attacks, "The requested attack is not implemeneted; available attacks are:"+str(attacks.keys())
        self.attack = attacks[attack]

    def compute_gradients(self, iter_num, model):
        """ compute a Byzantine gradient
        Args
        model        the model state using which training should happen
        iter_num     the number of current iteration; this determines the local batch to be used for training
        """
        return self.attack(iter_num, model)

    def random_attack(self, iter_num, model):
        """ return a random gradient with the same size of the submitted one
        Args
        model        the model state using which training should happen
        iter_num     the number of current iteration; this determines the local batch to be used for training
        """
        rank, grad, loss = super().compute_gradients(iter_num, model)
        return rank, torch.rand(grad.size()), loss

    def reverse_attack(self, iter_num, model):
        """ return the gradient, yet in the opposite direction and amplified
        Args
        model        the model state using which training should happen
        iter_num     the number of current iteration; this determines the local batch to be used for training
        """
        rank, grad, loss = super().compute_gradients(iter_num, model)
        return rank, grad*-100, loss

    def partial_drop_attack(self, iter_num, model):
        """ return the gradient but with some missing coordinates (replaced by zeros)
        Args
        model        the model state using which training should happen
        iter_num     the number of current iteration; this determines the local batch to be used for training
        """
        rank, grad, loss = super().compute_gradients(iter_num, model)
        p=0.3			#percent of the values that should be replaced by zeros
        mask = torch.rand(len(grad)) > 1-p
        grad.masked_fill(mask, 0)
        return rank, grad, loss

    def little_is_enough_attack(self, iter_num, model):
        """ return a Byzantine gradient based on the little is enough attack
        Args
        model        the model state using which training should happen
        iter_num     the number of current iteration; this determines the local batch to be used for training
        """
        #First, calculate fw true gradients; this simulates the cooperation of fw Byzantine workers
        rank, grad, loss = super().compute_gradients(iter_num, model)
        est_grads = [super().compute_gradients(iter_num+i+1, model)[1] for i in range(self.fw-1)]
        est_grads.append(grad)
        #Stack these gradients together and calcualte their mean and standard deviation
        est_grads = torch.stack(est_grads)
        mu = torch.mean(est_grads,axis=0)
        sigma = torch.std(est_grads,axis=0)
        #Now, apply the rule of the attack to generate the Byzantine gradient
        z = 1.035                      #Pre-calculated value for z_{max} from z-table, based on n=20, f=8 (and hence, s=3)
        grad = mu + z*sigma
        return rank, grad, loss

    def fall_empires_attack(self, iter_num, model):
        """ return a Byzantine gradient based on the fall of empires attack
        Args
        model        the model state using which training should happen
        iter_num     the number of current iteration; this determines the local batch to be used for training
        """
        rank, grad, loss = super().compute_gradients(iter_num, model)
        #First, calculate fw true gradients; this simulates the cooperation of fw Byzantine workers
        rank, grad, loss = super().compute_gradients(iter_num, model)
        est_grads = [super().compute_gradients(iter_num+i+1, model)[1] for i in range(self.fw-1)]
        est_grads.append(grad)
        #Stack these gradients together and calcualte their mean and standard deviation
        est_grads = torch.stack(est_grads)
        mu = torch.mean(est_grads,axis=0)
        eps = 10		#The value of epsilon is purely empirical and relies on the GAR used too
        grad = -eps*mu
        return rank, grad, loss

