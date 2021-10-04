# coding: utf-8
###
 # @file   ps.py
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

#!/usr/bin/env python

import time

import numpy as np
from tensorflow.keras.optimizers import Adam

from . import garfield_pb2
from . import tools
from .seucre_server import Secure_server 
from .server import Server
from .new_server import NewServer
from .grpc_message_exchange_servicer import MessageExchangeServicer

class PS(NewServer):
    """ Parameter Server node, handles the updates of the parameter of the model. """

    def __init__(self, network=None, log=False, dataset="mnist", model="Small", batch_size=128, nb_byz_worker= 0, is_secure = False , servicer = MessageExchangeServicer):
        """ Create a Parameter Server node.

            args:
                - network:   State of the cluster
                - log:      Boolean indicating whether to log or not
                - asyncr:   Boolean

        """
        super().__init__(network, log, dataset, model, batch_size, nb_byz_worker, is_secure , servicer)

        self.optimizer = Adam(lr=1e-3)



    def get_gradients(self, iter):
        """ Get gradients from the workers at a specific iteration.
        
            args:
                - iter: integer
            Returns:
                Gradients from the different PS.   
        """

        gradients = []

        for i, connection in enumerate(self.worker_connections):
            counter = 0
            read = False
            while not read:
                try: 
                    response = connection.GetGradient(garfield_pb2.Request(iter=iter,
                                                                        job="ps",
                                                                        req_id=self.task_id))
                    serialized_gradient = response.gradients
                    gradient = np.frombuffer(serialized_gradient, dtype=np.float32)
                    gradients.append(gradient)
                    read = True
                except Exception as e:
                    print("Trying to connect to Worker node ", i)
                    time.sleep(5)
                    counter+=1
                    if counter > 10:			#any reasonable large enough number
                        exit(0)
        return gradients

    def upate_model(self, gradient):
        """ Update the model with the aggregated gradients. 

            Args:
                - gradient: gradient to update the model.
            Returns:
                Model after update.        
        """
        reshape_gradient = tools.reshape_weights(self.model, gradient)
        self.optimizer.apply_gradients(zip(reshape_gradient, self.model.trainable_variables))
        return tools.flatten_weights(self.model.trainable_variables)

    def commit_model(self, model):
        """ Make the model available on the network. 

            Args:
                - model: model to commit
        """
        self.service.model_wieghts_history.append(model)
