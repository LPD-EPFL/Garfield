# coding: utf-8
###
 # @file   byz_ps.py
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

from .ps import PS
from .attacker import Attacker
from . import tools

class ByzPS(PS):
    """ Class defining a byzantine parameter server. """

    def __init__(self, network=None, log=False, dataset="mnist", max_episods=20, batch_size=128, nb_byz_worker=0):
        """ Create a Byzantine parameter server node.

            args:
                - network:  State of the cluster
                - log:      Boolean indicating whether to log or not
                - asyncr:   Boolean

        """
        super().__init__(network, log, dataset, max_episods, batch_size, nb_byz_worker)
        self.attacker = Attacker(network.get_my_attack())

    def commit_model(self, **kwargs):
        """ Make the modification of the model available on the network. 

            Args:
                - kwargs:   Arguments needed for the attack
        """
        attacked_parameters = self.attacker.attack(tools.flatten_weights(self.model.trainable_variables), kwargs)
        self.service.model_wieghts_history.append(attacked_parameters)
