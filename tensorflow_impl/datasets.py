# coding: utf-8
###
 # @file   datasets.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
###

import numpy
import random

import tensorflow as tf

# ---------------------------------------------------------------------------- #

# Trivial graph data set wrapper class
class DataSet:
    def __init__(self, dataset):
        """ Data set constructor.
        Args:
            dataset (Numpy) dataset to use
        """
        if type(dataset) == NumpySet:
            raise NotImplementedError("DataSet from-numpy constructor not implemented")
        else:
            self.__dataset = dataset

    def iter(self):
        """ Get a new iterator on the batches.
        Returns:
            Iterator on tuples (inputs, labels)
        """
        return self.__dataset.make_initializable_iterator()

    def get(self):
        """ Get a uniformly sampled batch.
        Returns:
            Uniformly sampled minibatch tuple (inputs, labels)
        """
        return tf.contrib.data.get_single_element(self.__dataset)

    def shuffle(self, buffer_size=64, buffer_size_tensor=None, **kwargs):
        """ Return a (shallow) copy of this data set, uniformly shuffled.
        Args:
            buffer_size        Constant size of the sampling buffer, ignored if '..._tensor' is set (optional, default to 64)
            buffer_size_tensor Tensor with the size of the sampling buffer (optional, None to use 'buffer_size')
            ...                Forwarded to 'shuffle' optional arguments
        Returns:
            A (shallow) copy uniformly shuffled
        """
        if buffer_size_tensor is None:
            buffer_size_tensor = tf.constant(buffer_size, dtype=tf.int64)
        return DataSet(self.__dataset.shuffle(buffer_size_tensor, **kwargs))

    def cut(self, btstart, btstop, btsize, btsize_tensor=None):
        """ Return a (shallow) copy of this data set, cut into batches.
        Args:
            btstart       Start index (included)
            btstop        Stop index (excluded)
            btsize        Constant batch size (ignored if 'btsize_tensor' is set)
            btsize_tensor Tensor than defines the batch size (optional, None to use 'btsize')
        Returns:
            A (shallow) copy cut into batches
        """
        if btsize_tensor is None:
            btsize_tensor = tf.constant(btsize, dtype=tf.int64)
        return DataSet(self.__dataset.range(btstart, btstop).batch(btsize_tensor))

# Simple (numpy) data set class
class NumpySet:
    # Minibatch iterator class
    class Iterator:
        def __init__(self, batches):
            """ Iterator constructor.
            Args:
                batches List of tuples (inputs, labels)
            """
            batches = batches[:]    # Shallow copy
            random.shuffle(batches) # Uniform, random shuffle
            self.__batches = batches
            self.__current = 0
            self.__limit   = len(batches)

        def __iter__(self):
            return self

        def __next__(self):
            """ Get the next minibatch.
            Returns:
                Tuple (inputs, labels)
            """
            if self.__current >= self.__limit:
                raise StopIteration()
            selected = self.__batches[self.__current]
            self.__current += 1
            return selected

    def __init__(self, *args):
        """ Dataset basic constructor.
        Args:
            inputs  Input tensor
            labels  Label tensor
            batches List of mini-batches (optional)
        -or-
            Graph dataset constructor.
        Args:
            dataset Graph dataset to extract
        """
        nbargs = len(args)
        if nbargs >= 2 and nbargs <= 3: # Basic constructor
            self.__inputs  = args[0]
            self.__labels  = args[1]
            self.__batches = [(args[0], args[1])] if nbargs == 2 else args[2]
        else: # Graph constructor
            raise NotImplementedError("NumpySet from-graph constructor not implemented")

    def __iter__(self):
        """ Get a new iterator on the batches.
        Returns:
            Iterator on tuples (inputs, labels)
        """
        return NumpySet.Iterator(self.__batches)

    def __len__(self):
        """ Get the number of batches.
        Returns:
            Number of batches
        """
        return len(self.__batches)

    def get(self):
        """ Get a uniformly sampled random minibatch.
        Returns:
            Uniformly sampled (inputs, labels)
        """
        if len(self.__batches) == 0:
            raise RuntimeError("Data set is empty")
        return self.__batches[random.randint(0, len(self.__batches) - 1)]

    def shuffle(self):
        """ Return a (shallow) copy of this data set, uniformly shuffled.
        Returns:
            A (shallow) copy uniformly shuffled
        """
        indexing = [i for i in range(self.__inputs.shape[0])]
        random.shuffle(indexing)
        inputs = numpy.take(self.__inputs, indexing, axis=0)
        labels = numpy.take(self.__labels, indexing, axis=0)
        return NumpySet(inputs, labels)

    def cut(self, btstart, btstop, btsize):
        """ Return a (shallow) copy of this data set, cut into batches.
        Args:
            btstart Start index (included)
            btstop  Stop index (excluded)
            btsize  Size of a batches
        Returns:
            A (shallow) copy cut into batches
        """
        length = self.__inputs.shape[0]
        assert btstart >= 0 and btstop >= btstart
        assert btsize > 0 #and (btstop - btstart) % btsize == 0		#The commented condition restricts having only a subset of possible batch sizes...removing it gives us more freedom
        assert btstop <= length
        batches = list()
        for i in range(btstart, btstop, btsize):
            if i + btsize > btstop:
                break
            j = i + btsize
            batches.append((self.__inputs[i:j], self.__labels[i:j]))
        return NumpySet(self.__inputs[btstart:btstop], self.__labels[btstart:btstop], batches)

    @staticmethod
    def from_keras(loader, post=None):
        """ Keras dataset constructor.
        Args:
            loader Loader function
            post   Post-processing function (inputs, labels) -> (inputs, labels) (optional)
        """
        train, test = loader()
        inputs = numpy.concatenate((train[0], test[0]), axis=0)
        labels = numpy.concatenate((train[1], test[1]), axis=0)
        if post is not None:
            inputs, labels = post(inputs, labels)
        return NumpySet(inputs, labels)

# ---------------------------------------------------------------------------- #

def load_mnist():
    """ Load the MNIST dataset.
    """
    def post(inputs, labels):
        return numpy.reshape(inputs, (inputs.shape[0], inputs.shape[1] * inputs.shape[2])).astype(numpy.float32) / 255, labels.astype(numpy.int32)
    return NumpySet.from_keras(tf.keras.datasets.mnist.load_data, post=post)

def load_cifar10():
    """ Load the CIFAR-10 dataset.
    """
    def post(inputs, labels):
        return inputs.astype(numpy.float32) / 255, labels.flatten().astype(numpy.int32)
    return NumpySet.from_keras(tf.keras.datasets.cifar10.load_data, post=post)

def load_cifar100():
    """ Load the CIFAR-100 dataset.
    """
    def post(inputs, labels):
        return inputs.astype(numpy.float32) / 255, labels.flatten().astype(numpy.int32)
    return NumpySet.from_keras(tf.keras.datasets.cifar100.load_data, post=post)
