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
from .new_server import NewServer
from .grpc_message_exchange_servicer import MessageExchangeServicer
from . import tools


class Worker(NewServer):
    """ Worker node used to calculate the gradient of a model. """

    def __init__(self, network=None, log=False, dataset="mnist", model="Simple", batch_size=128, nb_byz_worker=0, is_secure = False , servicer = MessageExchangeServicer):
        """ Create a Worker node.

            args:
                - network:  State of the cluster
                - log:      Boolean indicating whether to log or not
                - asyncr:   Boolean

        """
        super().__init__(network, log, dataset, model, batch_size, nb_byz_worker, is_secure , servicer)

    def compute_gradients(self, iter):
        """ Compute gradients.zz

            Args:
                - iter: iteration of the training
            Returns:
                Gradient of the model based on the data of a specific iteration.
        """

        X, y = self.train_data[iter % len(self.train_data)]
        dataset = tf.data.Dataset.from_tensors((X, y))
        dist_dataset = self.strategy.experimental_distribute_dataset(dataset)

        with self.strategy.scope():
            @tf.function
            def train_step(inputs):
                features, labels = inputs

                with tf.GradientTape() as tape:
                    preds = self.model(features, training=True)

                    loss = tf.reduce_sum(self.loss_fn(labels, preds)) * (1. / self.batch_size)

                grads = tape.gradient(loss, self.model.trainable_variables)
                flattened = tools.flatten_weights(grads)

                return loss, flattened

            @tf.function
            def distributed_train_step(dist_inputs):
                per_replica_losses, per_replica_grad = self.strategy.run(train_step, args=(dist_inputs,))
                return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), \
                       self.strategy.experimental_local_results(value=per_replica_grad)

            tf.config.run_functions_eagerly(True)
            losses = []
            grads = []
            for dist_inputs in dist_dataset:
                l, g = distributed_train_step(dist_inputs)
                losses.append(l)
                grads.append(g)

            return np.mean(losses), np.mean(grads, axis=0)

    def commit_gradients(self, grads):
        """ Make the gradients available to the other nodes on the network.
        
            Args:
                - grads: Computed gradient.
        """

        self.service.gradients_history.append(grads)
