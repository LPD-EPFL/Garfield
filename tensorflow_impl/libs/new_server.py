# coding: utf-8
###
 # @file   server.py
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
from concurrent import futures

import grpc
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from . import tools as tools

from .dataset import DatasetManager
from .model import ModelManager
from . import garfield_pb2_grpc
from . import garfield_pb2
from . import grpc_message_exchange_servicer



class NewServer:
    """ Superclass defining a server entity. """

    def __init__(self, network=None, log=False, dataset="mnist", model="Small", batch_size=128, nb_byz_worker=0, is_secure = False , servicer = grpc_message_exchange_servicer.MessageExchangeServicer):
        self.log = log

        self.network = network
        self.nb_byz_worker = nb_byz_worker
        self.batch_size = batch_size

        dsm = DatasetManager(network, dataset, self.batch_size)
        self.train_data, self.test_data = dsm.data_train, dsm.data_test

        devices = tf.config.list_physical_devices('gpu')

        if len(devices) == 1:
            '''
            tf.distribute.OneDeviceStrategy is a strategy to place all variables and computation on a single specified device.
            '''
            self.strategy = tf.distribute.OneDeviceStrategy()
        elif len(devices) > 1:
            '''
            tf.distribute.MirroredStrategy supports synchronous distributed training on multiple GPUs on one machine.
            It creates one replica per GPU device. Each variable in the model is mirrored across all the replicas.
            Together, these variables form a single conceptual variable called MirroredVariable.
            These variables are kept in sync with each other by applying identical updates.
            '''
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = tf.distribute.get_strategy()

        with self.strategy.scope():
            mdm = ModelManager(model=model, info=dsm.ds_info)
            self.model = mdm.model

            self.loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        ps_hosts = network.get_all_ps()
        worker_hosts = network.get_all_other_worker()
        self.port = network.get_my_port()
        self.task_id = network.get_task_index()

        self.m = tf.keras.metrics.Accuracy()
        if is_secure:
            #load credentials
            self.root_certificate = tools.load_credential_from_file("../rsrcs/credentials/root.crt")
            self.private_key = tools.load_credential_from_file("../rsrcs/credentials/127.0.0.1:"+ self.port + ".pem")
            self.certificate = tools.load_credential_from_file("../rsrcs/credentials/127.0.0.1:"+ self.port + ".crt")

            #set a secure connection
            self.ps_connections_dicts = {host : tools.set_secure_connetion(host, self.root_certificate, self.private_key , self.certificate)  for host in ps_hosts}
            self.ps_connections = [self.ps_connections_dicts[host] for host in self.ps_connections_dicts]

            self.worker_connections = [tools.set_secure_connetion(host , self.root_certificate , self.private_key , self.certificate) for host in worker_hosts]

            #define a secure grpc server
            self.service = servicer(tools.flatten_weights(self.model.trainable_variables))
            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=30),
                options=[
                    ('grpc.max_send_message_length', 500 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 500 * 1024 * 1024)
                ])

            garfield_pb2_grpc.add_MessageExchangeServicer_to_server(self.service, self.server)

            server_credentials = grpc.ssl_server_credentials(
                [(self.private_key, self.certificate)],
                root_certificates = self.root_certificate,
                require_client_auth = True)

            self.server.add_secure_port('localhost:' + str(self.port) , server_credentials)
        else:
            #set a insecure connection 
            self.ps_connections = [tools.set_connection(host) for host in ps_hosts]
            self.worker_connections = [tools.set_connection(host) for host in worker_hosts]

            #define a insecure grpc server
            self.service = servicer(tools.flatten_weights(self.model.trainable_variables))

            self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=30), options=[
                ('grpc.max_send_message_length', 500 * 1024 * 1024),
                ('grpc.max_receive_message_length', 500 * 1024 * 1024)
            ])
            garfield_pb2_grpc.add_MessageExchangeServicer_to_server(self.service, self.server)
            self.server.add_insecure_port('localhost:' + str(self.port))



        self.aggregated_weights = None

        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)



    
    def start(self):
        """ Starts the gRPC server. """

        self.server.start()
        if self.log:
            print("Starting on port: " + str(self.port))

    def stop(self, grace):
        """ Stops the gRPC server. """

        self.server.stop(grace)
        if self.log:
            print("Stopping server")

    def get_models(self, iter):
        """ Get all the models of the parameter servers. 
        
            args:
                - iter: int the iteration of the training

            returns:
                A list of model
        """
        models = []
        for i, connection in enumerate(self.ps_connections):
            counter = 0
            read = False
            while not read:
                try:
                    response = connection.GetModel(garfield_pb2.Request(iter=iter,
                                                                job="worker",
                                                                req_id=self.task_id))

                    serialized_model = response.model
                    model = np.frombuffer(serialized_model, dtype=np.float32)
                    models.append(model)
                    read = True
                except Exception as e:
                    print("Trying to connect to PS node ", i)
                    time.sleep(5)
                    counter+=1
                    if counter > 10:			#any reasonable large enough number
                        exit(0)

        return models

    def write_model(self, model):
        """ Build a Keras model from flatten weights. """
        """
            tools.reshape_weights(model , flatten_weight):
                add layers based on second argument

        """
        for l, weights in zip(self.model.trainable_variables, tools.reshape_weights(self.model,model)):
            l.assign(weights.reshape(l.shape))


    def compute_accuracy(self):
        """ Compute the accuracy of the model on the test set and print it. """
        predictions = []
        true_val = []
        for X, y in self.test_data:
            preds = self.model(X)
            predictions = predictions + [float(tf.argmax(p).numpy()) for p in preds]
            true_val.extend(y)

        self.m.reset_states()
        self.m.update_state(y_pred=predictions, y_true=true_val)
        return self.m.result().numpy() * 100

