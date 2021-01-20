# coding: utf-8
###
 # @file   model.py
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

import tensorflow as tf
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3

from tensorflow.keras.applications import VGG16, ResNet152V2

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation


class ModelManager:

    def __init__(self, model, info):
        self.info = info

        possible_model = {
            "Small": self.getSmallModel(),
            "MobileNetV2": MobileNetV2(),
            "Resnet50": ResNet50(weights=None, classes=self.info.features['label'].num_classes),
            "Resnet200": ResNet152V2(weights=None, include_top=False),
            "VGG": VGG16(weights=None, include_top=False, classes=self.info.features['label'].num_classes),
            "DenseNet": DenseNet121(),
            "Cifarnet": self.cifarnet(),
            "Inception": InceptionV3(),
            "CNN": self.getCNN()
        }

        assert model in possible_model.keys(), "Selected model not available"
        self.model = possible_model[model]

    def getSmallModel(self):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=self.info.features['image'].shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.info.features['label'].num_classes, activation='softmax')
        ])

        return model

    def getCNN(self):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(128, (3, 3), input_shape=self.info.features['image'].shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Flatten())

        model.add(Dense(64))
        model.add(Dense(self.info.features['label'].num_classes, activation='softmax'))

        return model


    def cifarnet(self):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=self.info.features['image'].shape))
        model.add(MaxPooling2D((3, 3)))

        model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((3, 3)))
        
        model.add(Flatten())
        model.add(Dense(384, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(192, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.info.features['label'].num_classes, activation='softmax'))
        return model