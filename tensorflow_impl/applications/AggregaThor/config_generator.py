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

    ipports = [ip[:-1] + ":5000" for ip in sys.stdin]
    sys.stdin = open("/dev/tty")

    ipport_workers = []
    ipport_ps = []
    tasks_workers = []
    tasks_ps = []

    print("How many workers ?")
    nb_workers = int(input())

    if nb_workers > len(ipports):
        print("There are more workers than available nodes.")
        exit(0)

    
   
    nb_ps = 1

    if nb_ps > len(ipports) - nb_workers:
        print("There are more ps than available nodes.")
        exit(0)

    ipport_workers = ipports[:nb_workers]
    ipport_ps = ipports[nb_workers:nb_workers+nb_ps]
    cluster = {
        "worker": ipport_workers,
        "ps": ipport_ps
    }

    print("Let's configure the workers !")

    for i, ipport in enumerate(ipport_workers):
        print(f"Worker {str(i)} - {ipport} :")
        
        print("Strategy (Average):")
        strategy = input()
        if strategy == "":
            strategy = "Average"

        print("Attack (None):")
        attack = input()
        if attack == "":
            attack = "None"
        task = {"type": "worker", "index": i, "strategy": strategy, "attack": attack}

        f = open("config/TF_CONFIG_" + ipport.split(':')[0], "w")
        f.write(json.dumps({
            "cluster": cluster,
            "task": task
        }))
        f.close

    print("Let's configure the parameter servers !")

    
    for i, ipport in enumerate(ipport_ps):
        print(f"PS {str(i)} - {ipport} :")
        
        print("Strategy (Average):")
        strategy = input()
        if strategy == "":
            strategy = "Average"

        print("Attack (None):")
        attack = input()
        if attack == "":
            attack = "None"
        task = {"type": "ps", "index": i, "strategy": strategy, "attack": attack}

        f = open("config/TF_CONFIG_" + ipport.split(':')[0], "w")
        f.write(json.dumps({
            "cluster": cluster,
            "task": task
        }))
        f.close

    
if __name__ == "__main__":
    main()
