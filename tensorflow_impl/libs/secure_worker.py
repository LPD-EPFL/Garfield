# coding: utf-8
###
# @file   worker.py
# @author  Anton Ragot <anton.ragot@epfl.ch>, Jérémy Plassmann <jeremy.plassmann@epfl.ch>
#
# @section LICENSE
#
# MIT License
#
# Copyright (c) 2020 Distributed Computing Laboratory, EPFL
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
###

# !/usr/bin/env python

import time

import tensorflow as tf
import numpy as np

from .server import Server
# from .seucre_server import Secure_server

from .new_server import NewServer
from .worker import Worker
from .grpc_secure_woker_servicer import SecureMessageExchangeServicerWorker
from . import tools


class SecureWorker(Worker):
    """ Worker node used to calculate the gradient of a model. """

    def __init__(self, network=None, log=False, dataset="mnist", model="Simple", batch_size=128, nb_byz_worker=0):
        """ Create a Worker node.

            args:
                - network:    State of the cluster
                - log:        Boolean indicating whether to log or not
                - asyncr:     Boolean
                - is_secure:  Boolean
                - servier:    Define a servicer for grpc servicer

        """
        super().__init__(network, log, dataset, model, batch_size, nb_byz_worker, is_secure= True, servicer= SecureMessageExchangeServicerWorker)


    def commit_gradients(self, grads):
        """ Make the gradients available for each server.
        
            Args:
              - grads: Computed gradients for each server
        """
        self.service.partial_gradients_history_model_server.append(grads[0])
        self.service.partial_gradients_history_worker_server.append(grads[1])
