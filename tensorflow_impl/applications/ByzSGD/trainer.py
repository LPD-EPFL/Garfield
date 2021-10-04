# coding: utf-8
###
 # @file   trainer.py
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

import argparse
import os
import sys


from libs.worker import Worker
from libs.ps import PS
from libs.byz_worker import ByzWorker
from libs import tools

from rsrcs.aggregator_tf.aggregator import Aggregator_tf

from rsrcs.network import Network

# Allowing visualization of the log while the process is running over ssh
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


FLAGS = None


def main():
    n = Network(FLAGS.config)

    if n.get_task_type() == 'worker':
        if n.get_my_attack() != 'None':
            w = ByzWorker(n, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
        else:
            w = Worker(n, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
        w.start()
        model_aggregator = Aggregator_tf(n.get_model_strategy(), len(n.get_all_workers()), FLAGS.nbbyzwrks, FLAGS.native)

        for iter in range(FLAGS.max_iter):
            """
                - get_models is a function from parents class of worker : server
                - get_models getting models from each server's connection(which is a garfield_pb2_grpc.MessageExchangeStub object) by calling GetModel Function
                connection.GetModel(garfield_pb2.Request(iter=iter,job="worker",req_id=self.task_id))
                - GetModel is a attribute(a little bit confusing for me) that calling unary_unary and unary_unary is calling handler which called 
                GetModel from service attribute at server
            """
            models = w.get_models(iter)

            """
                - aggregate models and gradient are done by the same class
                - need more time to explore ...
            """
            aggregated_model = model_aggregator.aggregate(models)
            """
                - each server(either server or worker) has a attribute named model
                - model is a tenserflow model and assigned by Model manager class
                - set the model to aggregated model
            """
            w.write_model(aggregated_model)
            """
                - basicly compute the gradients
            """
            loss, grads = w.compute_gradients(iter)

            """
                - each server(mean servers and workers) has a attribute named service. service is a MessageExchangeServicer.
                - MessageExchangeServicer has a attribute named gradient_history, which gradient_history[i] is a gradient at iteration i.
                - commit_gradient add gradient to the gradient_history

            """
            w.commit_gradients(grads)

        w.wait_until_termination()

    elif n.get_task_type() == 'ps':
        p = PS(n, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
        p.start()

        model_aggregator = Aggregator_tf(n.get_model_strategy(), len(n.get_all_workers()), FLAGS.nbbyzwrks, FLAGS.native)
        gradient_aggregator = Aggregator_tf(n.get_gradient_strategy(), len(n.get_all_workers()), FLAGS.nbbyzwrks, FLAGS.native)

        accuracy = 0
        accuracies = {}
        for iter in range(FLAGS.max_iter):
            """
                - same a above
            """
            models = p.get_models(iter)
            """
                - same as above
            """
            aggregate_model = model_aggregator.aggregate(models)
            """
                -same as above
            """
            p.write_model(aggregate_model)
            """
                - same as get_model
            """
            gradients = p.get_gradients(iter)

            aggregated_gradient = gradient_aggregator.aggregate(gradients)

            
            model = p.upate_model(aggregated_gradient)
            p.commit_model(model)
            
            tools.training_progression(FLAGS.max_iter, iter, accuracy)
            if iter%200 == 0:
                accuracy = p.compute_accuracy()

            
        p.wait_until_termination()
    else:
        print("Unknown task type, please check TF_CONFIG file")
        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Flags for defining current Node
    parser.add_argument('--config',
                        type=str,
                        default="TF_CONFIG",
                        help='Config file location.')
    parser.add_argument('--log',
                        type=bool,
                        default=False,
                        help='Add flag to print intermediary steps.')
    parser.add_argument('--max_iter',
                        type=int,
                        default="2000",
                        help='Maximum number of epoch')
    parser.add_argument('--dataset',
                        type=str,
                        default="mnist",
                        help='Choose the dataset to use')
    parser.add_argument('--model',
                        type=str,
                        default="Small",
                        help='Choose the model to use')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Set the batch size')
    parser.add_argument('--nbbyzwrks',
                        type=int,
                        default=0,
                        help='Set the number of byzantine workers (necessary for Krum aggregation)')
    parser.add_argument('--native',
                        type=bool,
                        default=False,
                        help='Choose to use the native aggregators.')

    FLAGS, unparsed = parser.parse_known_args()
    main()
