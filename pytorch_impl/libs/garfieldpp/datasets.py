# coding: utf-8
###
 # @file   datasets.py
 # @author Arsany Guirguis  <arsany.guirguis@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright (c) 2019-2020 Arsany Guirguis.
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
 # Datasets management and partitioning.
###

#!/usr/bin/env python

import pathlib
import torch
from random import Random
from torchvision import datasets, transforms

datasets_list = ['mnist', 'cifar10']
MNIST = datasets_list.index('mnist')
CIFAR10 = datasets_list.index('cifar10')

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
       """ Constructor of Partiotion Object
           Args
           data		dataset needs to be partitioned
           index	indices of datapoints that are returned
        """
       self.data = data
       self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """ Fetching a datapoint given some index
	    Args
            index	index of the datapoint to be fetched
        """
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        """ Constructor of dataPartitioner object
	    Args
	    data	dataset to be partitioned
	    size	Array of fractions of each partition. Its contents should sum to 1
	    seed	seed of random generator for shuffling the data
	"""
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
        """ Fetch some partition in the dataset
	    Args
	    partition	index of the partition to be fetched from the dataset
	"""
        return Partition(self.data, self.partitions[partition])

class DatasetManager(object):
    """ Manages training and test sets"""

    def __init__(self, dataset, minibatch, num_workers, size, rank):
        """ Constrctor of DatasetManager Object
	    Args
		dataset		dataset name to be used
		batch		minibatch size to be employed by each worker
		num_workers	number of works employed in the setup
		rabk		rank of the current worker
	"""
        if dataset not in datasets_list:
            print("Existing datasets are: ", datasets_list)
            raise
        self.dataset = datasets_list.index(dataset)
        self.batch = minibatch * num_workers
        self.num_workers = num_workers
        self.num_ps = size - num_workers
        self.rank = rank

    def fetch_dataset(self, dataset, train=True):
        """ Fetch train or test set of some dataset
		Args
		dataset		dataset index from the global "datasets" array
		train		boolean to determine whether to fetch train or test set
	"""
        homedir = str(pathlib.Path.home())
        if dataset == MNIST:
            return datasets.MNIST(
              homedir+'/data',
              train=train,
              download=train,
              transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307, ), (0.3081, ))
              ]))

        if dataset == CIFAR10:
            if train:
              transforms_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
#                transforms.Resize(299),		#only use with inception
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
#		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),])
              return datasets.CIFAR10(
               homedir+'/data',
               train=True,
               download=True,
               transform=transforms_train)
            else:
              transforms_test = transforms.Compose([
#                transforms.Resize((299,299)),			#only use with inception
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
#		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
              return datasets.CIFAR10(
                homedir+'/data',
                train=False,
                download=False,
                transform=transforms_test)

#            return datasets.CIFAR10(
#               homedir+'/data',
#               train=train,
#               download=train,
#               transform=transforms.Compose(
#                  [transforms.ToTensor(),
#                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    def get_train_set(self):
        """ Fetch my partition of the train set"""
        train_set = self.fetch_dataset(self.dataset, train=True)
        size = self.num_workers
        bsz = int(self.batch / float(size))
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_set, partition_sizes)
        partition = partition.use(self.rank - self.num_ps)
        print("Using batch size = ", bsz)
        train_set = torch.utils.data.DataLoader(
            partition, batch_size=bsz, shuffle=False, pin_memory=True, num_workers=2)
        return [sample for sample in train_set]

    def get_test_set(self):
        """ Fetch test set, which is global, i.e., same for all entities in the deployment"""
        test_set = self.fetch_dataset(self.dataset, train=False)
        test_set = torch.utils.data.DataLoader(test_set, batch_size=100, #len(test_set),
		 pin_memory=True, shuffle=False, num_workers=2)
        return test_set
