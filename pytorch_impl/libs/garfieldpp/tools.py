# coding: utf-8
###
 # @file   tools.py
 # @author Arsany Guirguis  <arsany.guirguis@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright (c) 2019 Arsany Guirguis.
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
 # Misc functions that are useful for training.
###

#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import psutil
from garfieldpp.models import *
import torch.optim as optim
import torchvision
from torch.distributed.rpc import rpc_sync, rpc_async
from time import sleep
import sys

def select_loss(loss_fn):
    """ Select loss function to optimize with
    Args
    loss_fn        Name of the loss function to optimize against
    """
    losses = {'nll': nn.NLLLoss, 'cross-entropy':nn.CrossEntropyLoss}
    if loss_fn in losses.keys():
        return losses[loss_fn]()
    else:
        print("The selected loss function is undefined, available losses are: ", losses.keys())
        raise
  
def select_model(model, device):
    """ Select model to train
    Args
    model    model name required to be trained
    device    device to put model on (cuda or cpu)
    """
    models = {'convnet':Net,
		'cifarnet':Cifarnet,
		'cnn': CNNet,
		'resnet18':ResNet18, #torchvision.models.resnet18,
                'resnet34':torchvision.models.resnet34,
                'resnet50':torchvision.models.resnet50,
                'resnet152':torchvision.models.resnet152,
		'inception':torchvision.models.inception_v3,
		'vgg16':torchvision.models.vgg16,
		'vgg19':torchvision.models.vgg19,
		'preactresnet18': PreActResNet18,
		'googlenet': GoogLeNet,
		'densenet121': DenseNet121,
		'resnext29': ResNeXt29_2x64d,
		'mobilenet': MobileNet,
		'mobilenetv2': MobileNetV2,
		'dpn92': DPN92,
		'shufflenetg2': ShuffleNetG2,
		'senet18': SENet18,
		'efficientnetb0': EfficientNetB0,
		'regnetx200': RegNetX_200MF}
    if model in models.keys():
        model = models[model]()
    else:
        print("The specified model is undefined, available models are: ", models.keys())
        raise

    model = model.to(device)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)

    return model

def select_optimizer(model, optimizer, *args, **kwargs):
    """ Select optimizer to use
    Args
    model        the model to optimize
    optimizer    optimizer name required to be initialized
    device       device to put model on (cuda or cpu)
    """
    optimizers = {'sgd': optim.SGD,
		'adam': optim.Adam,
		'adamw':optim.AdamW,
		'rmsprop': optim.RMSprop,
		'adagrad': optim.Adagrad}
    if optimizer in optimizers.keys():
        return optimizers[optimizer](model.parameters(),  *args, **kwargs)
    else:
        print("The selected optimizer is undefined, the available optimizers are: ", optimizers.keys())
        raise

def _call_method(method, rref, *args, **kwargs):
    """ call a local method
    Args
    method        Name of the method to call
    rref          remote reference that should point to this process
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method_sync(method, rref, *args, **kwargs):
    """ call a remote method synchronously
    Args
    method        Name of the method to call
    rref          remote reference that should be called
    """
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def _remote_method_async(method, rref, *args, **kwargs):
    """ call a remote method asynchronously
    Args
    method        Name of the method to call
    rref          remote reference that should be called
    """
    args = [method, rref] + list(args)
    return rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)

def get_bytes_com():
    """ get the number of bytes sent and received
    Args
    """
    return psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

def convert_to_gbit(value):
    """ convert bytes to Gbits
    Args
    value		Value in bytes to be converted to Gbits
    """
    return value/1024./1024./1024.*8

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate
    Args
    optimizer		The optimizer whose the learning rate is needed to be adjusted
    lr			The new learning rate to be set
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

global server_instance, worker_instance
server_instance, worker_instance = None, None

def get_server():
    """
    Global method to return the instance of this created server
    """
    global server_instance
    while server_instance is None:
        sleep(1)
    return server_instance

def get_worker():
    """
    Global method to return the instance of this created worker
    """
    global worker_instance
    while worker_instance is None:
        sleep(1)
    return worker_instance

