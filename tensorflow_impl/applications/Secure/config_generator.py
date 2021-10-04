# coding: utf-8
###
 # @file   config_generator.py
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
import sys


def main():
    
    ipport_workers = []
    ipport_ps = []
    
    print("How many workers ?")
    nb_workers = int(input())

    if nb_workers < 1:
        print("There should be at least one worker!")
        exit(0)

    ipports = ["127.0.0.1:" + str(5000 + id) for id in range(nb_workers + 1)]
    ipport_ps = [ipports[0]]
    ipport_workers = ipports[1:nb_workers+1]
    cluster = {
        "worker": ipport_workers,
        "ps": ipport_ps
    }

    print("Let's configure the workers !")

    for i, ipport in enumerate(ipport_workers):
        print(f"Worker {str(i)} - {ipport} :")
        
        print("Strategy (Average):")
        strategy_g = input()
        if strategy_g == "":
            strategy_g = "Average"

        print("Attack (None):")
        attack = input()
        if attack == "":
            attack = "None"

        task = {"type": "worker",
                "index": i, 
                "strategy_model": "Average", # Fixed to average as it will not be used
                "strategy_gradient": strategy_g,
                "attack": attack
               }

        f = open("config/TF_CONFIG_W" + str(i), "w+")
        f.write(json.dumps({
            "cluster": cluster,
            "task": task
        }))
        f.close

    print("Let's configure the parameter server !")

    
    for i, ipport in enumerate(ipport_ps):
        print(f"PS {str(i)} - {ipport} :")
    

        print("Strategy (Average):")
        strategy_g = input()
        if strategy_g == "":
            strategy_g = "Average"

        print("Attack (None):")
        attack = input()
        if attack == "":
            attack = "None"

        task = {"type": "ps",
                "index": i, 
                "strategy_model": "Average", # Fixed to average as it will not be used
                "strategy_gradient": strategy_g,
                "attack": attack
               }

        f = open("config/TF_CONFIG_PS" + str(i), "w+")
        f.write(json.dumps({
            "cluster": cluster,
            "task": task
        }))
        f.close

    
if __name__ == "__main__":
    main()