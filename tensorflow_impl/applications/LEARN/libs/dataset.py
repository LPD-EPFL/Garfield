# coding: utf-8
###
 # @file   dataset.py
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

import tensorflow_datasets as tfds
import tensorflow as tf


class DatasetManager:

    def __init__(self, network, dataset, batch_size=128):
        split_low, split_high = self.get_data_partition(network)[network.get_task_index()]
        print(split_low, split_high)
        (ds_train, ds_test), ds_info = tfds.load(
            dataset,
            split=['train[{}%:{}%]'.format(split_low, split_high), 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        self.data_train, self.data_test = self.process_data(ds_train, ds_test, ds_info, batch_size)
        self.ds_info = ds_info

    def normalize_img(self, image, label):
        return tf.cast(image, tf.float32) / 255.0, tf.cast(label, tf.float32)

    def process_data(self, ds_train, ds_test, ds_info, batch_size):
        ds_train = ds_train.map(
            self.normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        ds_train = list(tfds.as_numpy(ds_train))

        ds_test = ds_test.map(
            self.normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.batch(600)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        ds_test = list(tfds.as_numpy(ds_test))

        return ds_train, ds_test

    def get_data_partition(self, network):
        if network.get_task_type() == "ps":
            return {network.get_task_index(): (0, 100)}
        number_worker = len(network.get_all_workers())
        partition_size = int(100 / number_worker)

        partition = {}
        starting = 0

        for worker_task in range(number_worker):
            partition[worker_task] = (starting, min(100, starting + partition_size))
            starting = starting + partition_size

        return partition
