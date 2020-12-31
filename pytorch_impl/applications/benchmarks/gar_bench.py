# coding: utf-8
###
 # @file   gar_bench.py
 # @author Arsany Guirguis  <arsany.guirguis@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright (c) 2020 Arsany Guirguis.
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
 #
 # @section DESCRIPTION
 # Benchmarking GARs in the Garfield++ library
###

#!/usr/bin/env python

import os
import torch
from time import time
import sys
import aggregators

gars = aggregators.gars
ds = [pow(10,i) for i in range(5)]
ns = [pow(2,i) for i in range(10)]
fs = [pow(2,i) for i in range(10)]
dev='cuda'

def bench_n(d,f):
  print("Benchmarking GARs with increasing n")
  for k in gars:
    for n in ns:
      grads = [torch.rand(d).to(dev) for i in range(n)]
      if n < 2*f+1 or (n < 2*f+3 and k.find('krum') != -1) or ((n < 4*f+3 or n > 23) and k.find('bulyan') != -1):
        continue
      print("** n={} f={} d={} GAR={} **".format(n,f,d,k))
      gar = gars.get(k)
      t = time()
      gar(gradients=grads, f=f)
      print("time: ", time()-t)
      del grads

def bench_d(n,f):
  print("Benchmarking GARs with increasing d")
  for k in gars:
    for d in ds:
      grads = [torch.rand(d).to(dev) for i in range(n)]
      if n < 2*f+1 or (n < 2*f+3 and k.find('krum') != -1) or (n < 4*f+3 and k.find('bulyan') != -1):
        continue
      print("** n={} f={} d={} GAR={} **".format(n,f,d,k))
      gar = gars.get(k)
      t = time()
      gar(gradients=grads, f=f)
      print("time: ", time()-t)
      del grads

def bench_f(n,d):
  print("Benchmarking GARs with increasing f")
  grads = [torch.rand(d).to(dev) for i in range(n)]
  for k in gars:
    for f in fs:
      if n < 2*f+1 or (n < 2*f+3 and k.find('krum') != -1) or (n < 4*f+3 and k.find('bulyan') != -1):
        break
      print("** n={} f={} d={} GAR={} **".format(n,f,d,k))
      gar = gars.get(k)
      t = time()
      gar(gradients=grads, f=f)
      print("time: ", time()-t)

bench_n(100000,1) #bench_n(d,f)
#bench_d(23,5) #bench_d(n,f)
#bench_f(23,100000) #bench_f(n,d)
