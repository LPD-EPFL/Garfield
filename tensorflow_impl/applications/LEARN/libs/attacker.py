# coding: utf-8
###
 # @file   attacker.py
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

import numpy as np

class Attacker:
    """ Class defining the attack used. """

    def __init__(self, attack):
        _possible_attacks = {
            'Random': self.random_attack,
            'Reverse': self.reverse_attack,
            'PartialDrop': self.partial_drop_attack,
            'LittleIsEnough': self.little_is_enough_attack,
            'FallEmpires': self.fall_empires_attack
        }
        self.attack_strategy = _possible_attacks[attack]

    def attack(self, **kwargs):
        """ Compute the attack.
            
            Args:
                - gradients: numpy array
            Returns:
                gradients: numpy array
        """
        return self.attack_strategy(**kwargs)

    def random_attack(self, gradient):
        """ Return a random gradient with the same size of the submitted one.
            Args:
                - gradients        numpy array
            Returns:
                Random gradient
        """
        return np.random.random(gradient.shape).astype(np.float32)

    def reverse_attack(self, gradient, coeff=100):
        """ Return the gradient, yet in the opposite direction and amplified.
            Args:
                - gradients        numpy array
                - coeff            float number representing the amplification
            Returns:
                numpy array
        """

        return gradient*coeff*(-1.)

    def partial_drop_attack(self, gradient, probability):
        """ return the gradient but with some missing coordinates (replaced by zeros)
            Args
                - gradient        numpy array
                - probability     float number representing the percent of the values that should be replaced by zeros
            Returns:
                numpy array
        """
        
        mask = np.random.rand(gradient.shape) > 1-probability
        return np.ma.array(gradient, mask=mask).filled(fill_value=0)


    ### Should be available only for Byzantine Workers:

    def little_is_enough_attack(self, gradient, byz_gradients):
        """ return a Byzantine gradient based on the little is enough attack
            Args:
                - gradient        numpy array
                - byz_gradients   list of numpy array
        """

        #First, calculate fw true gradients; this simulates the cooperation of fw Byzantine workers
        grad = gradient
        est_grads = byz_gradients
        est_grads.append(grad)
        #Stack these gradients together and calcualte their mean and standard deviation
        est_grads = np.stack(est_grads)
        mu = np.mean(est_grads,axis=0)
        sigma = np.std(est_grads,axis=0)
        #Now, apply the rule of the attack to generate the Byzantine gradient
        z = 1.035                      #Pre-calculated value for z_{max} from z-table, based on n=20, f=8 (and hence, s=3)
        grad = mu + z*sigma
        return grad

    def fall_empires_attack(self, gradient, byz_gradients):
        """ return a Byzantine gradient based on the fall of empires attack
            Args:
                - gradient        numpy array
                - byz_gradients   list of numpy array
        """
        #First, calculate fw true gradients; this simulates the cooperation of fw Byzantine workers
        grad = gradient
        est_grads = byz_gradients
        est_grads.append(grad)
        #Stack these gradients together and calcualte their mean and standard deviation
        est_grads = np.stack(est_grads)
        mu = np.mean(est_grads,axis=0)
        eps = 10		#The value of epsilon is purely empirical and relies on the GAR used too
        grad = -eps*mu
        return grad