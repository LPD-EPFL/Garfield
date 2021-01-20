# coding: utf-8
###
 # @file   config_generator_learn.py
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
import os, os.path
import errno

# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')


def main():

    ipports = [(ip[:-1] + ":5000", ip[:-1] + ":6000") for ip in sys.stdin]
    sys.stdin = open("/dev/tty")

    #ipport_workers, ipport_ps = zip(*ipports)

    tasks_workers = []
    tasks_ps = []


    
    print("How many servers ?")
    nb_servers = int(input())
    if nb_servers > len(ipports):
        print("There are more workers than available nodes.")
        exit(0)

    ipport_workers = ipports[:nb_servers]
    ipport_ps = ipports[:nb_servers]
    cluster = {
            "worker": ipport_workers,
            "ps": ipport_ps
        }
    print("Let's configure the workers !")

    for i, ipport in enumerate(ipport_workers):
        print(f"Worker {str(i)} - {ipport} :")
        
        print("Strategy (average):")
        strategy = input()
        if strategy == "":
            strategy = "Median"

        print("Attack (None):")
        attack = input()
        if attack == "":
            attack = "None"
        task = {"type": "worker", "index": i, "strategy": strategy, "attack": attack}


        with safe_open_w("config/" + ipport.split(':')[0] + "/TF_CONFIG_W") as f:
            f.write(json.dumps({
                "cluster": cluster,
                "task": task
            }))
        
    print("Let's configure the parameter servers !")

    
    for i, ipport in enumerate(ipport_ps):
        print(f"PS {str(i)} - {ipport} :")
        
        print("Strategy (average):")
        strategy = input()
        if strategy == "":
            strategy = "Average"

        print("Attack (None):")
        attack = input()
        if attack == "":
            attack = "None"
        task = {"type": "ps", "index": i, "strategy": strategy, "attack": attack}

        with safe_open_w("config/" + ipport.split(':')[0] + "/TF_CONFIG_PS") as f:
            f.write(json.dumps({
                "cluster": cluster,
                "task": task
            }))
        
    
if __name__ == "__main__":
    main()