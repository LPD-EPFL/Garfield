# coding: utf-8
###
 # @file   train_dist.py
 # @author Arsany Guirguis  <arsany.guirguis@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright (c) 2019 Arsany Guirguis.
 #
 # @section DESCRIPTION
 #
 # Distributed training in PyTorch.
###

#!/usr/bin/env python

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import sys

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

class Cifarnet(nn.Module):
    def __init__(self):
        super(Cifarnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_test_set():
    """ Returning test set """
#    dataset = datasets.MNIST(
#        './data',
#        train=False,
#        download=True,
#        transform=transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize((0.1307, ), (0.3081, ))
#        ]))
    dataset = datasets.CIFAR10(
           './data',
           train=True,
           download=True,
           transform=transforms.Compose(
             [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    test_set = torch.utils.data.DataLoader(dataset)
    return test_set

def partition_dataset():
    """ Partitioning CIFAR10 """
#    dataset = datasets.MNIST(
#        './data',
#        train=True,
#        download=True,
#        transform=transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize((0.1307, ), (0.3081, ))
#        ]))
    dataset = datasets.CIFAR10(
           './data',
           train=True,
           download=True,
           transform=transforms.Compose(
             [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


def average_gradients(model, rank):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        #Using reduce instead of all_reduce because only the parameter server gathers gradients from all but not itself
        #Parameter server always has rank 0
        if rank == 0:
            param.grad = torch.zeros(param.size())
        dist.reduce(param.grad.data, 0, op=dist.ReduceOp.SUM)
        #list=[]
        #dist.gather(param.grad.data,list,0,group=group)
        # size - 1 because gradients are collected from workers only
        param.grad.data /= (size - 1)


def run(rank, size):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    test_set = get_test_set()
    if torch.cuda.device_count() > 0:
      device = torch.device("cuda:{}".format(rank))
    else:
      device = torch.device("cpu:0")
#    model = Net().to(device)
    model = Cifarnet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #for Cifar10, 0.001 and 0.9, MNIST: 0.01 and 0.5
    #This group should contain all numbers except 0
    group = dist.new_group([1])
    world = dist.new_group([0,1])

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            if rank != 0:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
#                loss = F.nll_loss(output, target)
                loss = criterion(output, target)
#                epoch_loss += loss.data.item()
                epoch_loss += loss.item()
                loss.backward()
            average_gradients(model,rank)
            optimizer.step()
        dist.barrier(world)
        #Getting accuracy...
        correct = 0
        total = 1
        if rank == 0:
          total = total - 1
          model.eval()
          with torch.no_grad():
            for inputs, targets in test_set:
              inputs, targets = inputs.to(device), targets.to(device)
              outputs = model(inputs)
              _ , predicted = outputs.max(1)
              total += targets.size(0)
              correct += predicted.eq(targets).sum().item()
        print('Rank ',
              dist.get_rank(), ', epoch ', epoch, ' loss: ',
              epoch_loss / num_batches, 'acc: ', (correct/total))


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'chifflot-7'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
#    size = 2
#    processes = []
#    for rank in range(size):
#        p = Process(target=init_processes, args=(rank, size, run))
#        p.start()
#        processes.append(p)

#    for p in processes:
#        p.join()
    parser = argparse.ArgumentParser(description="Distributed SGD playground", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rank",
    type=int,
    default="0",
    help="Rank of a process in a distributed setup.")

    FLAGS = parser.parse_args(sys.argv[1:])

    init_processes(FLAGS.rank, 2, run)
