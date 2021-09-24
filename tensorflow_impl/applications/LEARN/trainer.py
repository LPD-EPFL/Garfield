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

from rsrcs.network import Network
from libs.worker import Worker
from libs.ps import PS
from libs.ByzWorker import ByzWorker
from libs.byz_worker import tools

from rsrcs.aggregator_tf.aggregator import Aggregator_tf

import time
import os
import sys


# Allowing visualization of the log while the process is running over ssh
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


FLAGS = None


def main():
    n_ps = Network(FLAGS.config_ps)
    n_w = Network(FLAGS.config_w)
        
    p = PS(n_ps, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
    p.start()

    if n_w.get_my_attack() != 'None':
        w = ByzWorker(n_w, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
    else:
        w = Worker(n_w, FLAGS.log, FLAGS.dataset, FLAGS.model, FLAGS.batch_size, FLAGS.nbbyzwrks)
    
    w.start()
        
    model_aggregator = Aggregator_tf(n_ps.get_model_strategy(), len(n_w.get_all_workers()), FLAGS.nbbyzwrks)
    gradient_aggregator = Aggregator_tf(n_ps.get_gradient_strategy(), len(n_ps.get_all_workers()), FLAGS.nbbyzwrks)

    accuracy = 0
    for iter in range(FLAGS.max_iter):
        models = w.get_models(iter)
        aggregated_model = model_aggregator.aggregate(models)
        w.write_model(aggregated_model)
        p.write_model(aggregated_model)
        loss, grads = w.compute_gradients(iter)
        w.commit_gradients(grads)
            
        gradients = p.get_gradients(iter)
        aggregated_gradient = gradient_aggregator.aggregate(gradients)
        model = p.upate_model(aggregated_gradient)
        p.commit_model(model)

        tools.training_progression(FLAGS.max_iter, iter, accuracy)
        if iter % 200 == 0:
            accuracy = p.compute_accuracy()

    p.wait_until_termination()
    w.wait_until_termination()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining current Node
    parser.add_argument('--config_w',
                        type=str,
                        default="TF_CONFIG",
                        help='Config file location.')
    parser.add_argument('--config_ps',
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

    FLAGS, unparsed = parser.parse_known_args()
    main()
