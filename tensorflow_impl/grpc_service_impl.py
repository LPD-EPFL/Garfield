# coding: utf-8
###
 # @file   grpc_service_impl.py
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
 #
 # Garfield implementation. The gRPC functions interface
###

#!/usr/bin/env python
import all_pb2, all_pb2_grpc
import grpc
import time
import sys
import random
import numpy as np

VERY_BIG_NUM = -1000000
class TrainMessageExchangeServicer(all_pb2_grpc.TrainMessageExchangeServicer):

  def __init__(self, num_workers, smart, index):
    self.index = index
    self.smart = smart

    self.mod = all_pb2.Model()
    self.mod.init = False

    self.pbuf_grad = all_pb2.Gradients()
    self.pbuf_grad.iter = VERY_BIG_NUM

    self.batches = [[] for i in range(num_workers)]

  def GetUnifiedModel(self, request, context):
    while not self.mod.init:
      time.sleep(0.05)
    return self.mod

  def GetGradients(self, request, context):
    iter_num = request.iter
    #This last clause indicate a PS to PS connection (uid of a PS is always < 0)
    while (self.smart and self.pbuf_grad.iter < iter_num) or ((not self.smart) and self.pbuf_grad.iter < iter_num) or ((request.req_id < 0 and self.index < 0) and (iter_num >= self.pbuf_grad.iter)):
      time.sleep(0.05)
    return self.pbuf_grad
