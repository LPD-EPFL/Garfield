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

import pickle
import time
from typing import final

import numpy as np
from tensorflow.keras.optimizers import Adam


from . import garfield_pb2
from . import tools
from .seucre_server import Secure_server 
from .server import Server
from .new_server import NewServer
from .grpc_secure_workerServer_servicer import SecureMessageExchangeServicerWorkerServer

class WorkerServer(NewServer):
    """ Parameter Server node, handles the updates of the parameter of the model. """

    def __init__(self, network=None, log=False, dataset="mnist", model="Small", batch_size=128, nb_byz_worker= 0):
        """ Create a Parameter Server node.

            args:
                - network:   State of the cluster
                - log:      Boolean indicating whether to log or not
                - asyncr:   Boolean

        """
        super().__init__(network, log, dataset, model, batch_size, nb_byz_worker, is_secure = True , servicer = SecureMessageExchangeServicerWorkerServer)



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
                    time.sleep(5)
                    counter+=1
                    if counter > 10:			#any reasonable large enough number
                        exit(0)
        return gradients

    def compute_final_pairwise_distances(self, partial_difference , iter):

        counter = 0
        read = False
        model_server_address = self.network.get_other_ps()
        model_server_connection = [self.ps_connections_dicts[host] for host in model_server_address]
        # print("this should be model server address" , model_server_address)


        while not read:
            # print("enter the compute_final_pairwise_distances")
            try:
                for i, connection in enumerate(model_server_connection):
                    # print("the connection" , connection)
                    response = connection.GetGradient(garfield_pb2.Request(iter=iter,
                                                                job="ps",
                                                                req_id=self.task_id))
                                                                
                    # print("in worker server, succefully get the partial discount")
                    serialized_model_server_difference = response.gradients
                    model_server_gradient = pickle.loads(serialized_model_server_difference)
                    # print("in worker server and in compute the final pair wise distances, the gradient from the model server" , model_server_gradient)
                    read = True
            except Exception as e:
                time.sleep(5)
                counter += 1
                if counter > 10:
                    exit(0)

        eculidean_distances = np.square(model_server_gradient + partial_difference)
        # print("the eculidean distances in the worker server" , eculidean_distances)
        return eculidean_distances

    def commit_semi_gradient(self, partial_final_gradient):
        self.service.partial_final_gradient.append(partial_final_gradient)


    def commit_model(self , iter):

        counter = 0
        read = False

        model_server_address = self.network.get_other_ps()
        model_server_connection = [self.ps_connections_dicts[host] for host in model_server_address]

        for i, connection in enumerate(model_server_connection):

            counter = 0
            read = False

            while not read:
                try:

                    response = connection.GetModel(garfield_pb2.Request(iter=iter,
                                                                job="worker",
                                                                req_id=self.task_id))

                    serialized_model = response.model
                    model = np.frombuffer(serialized_model, dtype=np.float32)
                    read = True

                except Exception as e:
                    print("Trying to connect to PS node ", i)
                    time.sleep(5)
                    counter+=1
                    if counter > 10:			#any reasonable large enough number
                        exit(0)  
        
        self.service.model_wieghts_history.append(model)




