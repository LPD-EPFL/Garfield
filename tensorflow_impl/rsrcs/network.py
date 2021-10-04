# coding: utf-8
###
 # @file   network.py
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

import json


class Network:

    def __init__(self, tf_location):
        with open(tf_location) as json_file:
            self._data = json.load(json_file)
            self._ps = []
            self._worker = []

            self._ps = self._data['cluster']['ps']
            print(self._ps)

            self._worker = self._data['cluster']['worker']
            print(self._worker)

    def get_task_type(self):
        return self._data['task']['type']

    def get_task_index(self):
        return self._data['task']['index']

    def get_model_strategy(self):
        return self._data['task']['strategy_model']

    def get_gradient_strategy(self):
        return self._data['task']['strategy_gradient']

    def get_my_attack(self):
        return self._data['task']['attack']

    def get_all_ps(self):
        """
            Return all PSs.
            If my server is a PS as well, it is excluded from the list
        """
        return self._ps.copy()

    def get_other_ps(self):
        if self.get_task_type() != "worker":
            return self._ps.copy()[1:]
        else:
            return self._ps.copy()

    def get_other_workers(self):
        if self.get_task_type() == "worker":
            return self._ps.copy()[1:]
        else:
            return self._ps.copy()


    def get_all_other_worker(self):
        """
            Return all workers.
            If my server is a worker as well, it is excluded from the list
        """
        return self._worker.copy()

    def get_all_workers(self):
        return self._worker

    def get_my_node(self):
        index = self._data['task']['index']
        if self.get_task_type() == "worker":
            return self._worker[index]
        else:
            return self._ps[index]

    def get_my_port(self):
        return self.get_my_node().split(':')[1]
