# coding: utf-8
###
 # @file   tools.py
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

import numpy as np
import tensorflow as tf
import os
import jwt
import sys
import grpc
from . import garfield_pb2_grpc

def bytes_to_chunks(s, size=4194290):
    """ Take bytes as argument and split them in diffenrent chunks
    
        args: 
            - s: bytes
        
        returns: list of bytes
    """
    return [s[i:i+size] for i in range(0, len(s), size)]



def load_credential_from_file(filepath):
    real_path = os.path.join(os.path.dirname(__file__), filepath)
    with open(real_path, 'rb') as f:
        return f.read()


def model_to_bytes(model):
    """ Serialize the model parameters into a stream of bytes.
    
        args: 
            - model: keras model
        
        returns: list of bytes ready to be sent as a stream
    """
    parameters = model.get_weights()
    chunks = [p.tobytes() for p in parameters]
    return chunks

def weights_to_bytes(weights):
    """ Serialize the model parameters into a bytes.
    
        args: 
            - weights: list of numpy array
        
        returns: 
            list of bytes representing layer's weights
    """
    serialized_weights = [p.numpy().tobytes() for p in weights]
    return serialized_weights

def bytes_to_weights(bytes_list):
    """ Deserialize the model parameters from bytes.
    
        args: 
            - bytes: list of bytes
        
        returns: 
            list of numpy array containing the layer's flatten weights
    """
    flatten_weights = [np.frombuffer(b, dtype=np.float32) for b in bytes_list]
    return flatten_weights

def tensor_to_bytes(tensor):
    """ Serialize a TensorFlow Tensor into bytes.

        args: 
            - Tensor
        returns:
            bytes
    """
    return tf.io.serialize_tensor(tensor).numpy()

def bytes_to_model(model, serialized_weights):
    """ Take the list of bytes of the parameters to recreate a Keras model.
    
        args: 
            - model: Keras model
            - serialized_parameters: list of bytes
              
        returns: Keras model
    """
    model_shape = [p.shape for p in model.get_weights()]

    weights = [np.frombuffer(w, dtype=np.float32).reshape(shape) for w, shape in zip(serialized_weights, model_shape)]
    model.set_weights(weights)
    return model

def flatten_weights(model):
    """ Flatten the model.

        args:
            - model: Keras model trainable variables
        
        returns:
            numpy array of all the weights.
    """
    return np.concatenate([l.numpy().reshape((-1,)) for l in model])

def reshape_weights(model, flatten_weights):
    """ Reshape the model from flatten numpy array.

        args:
            - model: Keras model
            - flatten_weights: numpy array
        
        returns:
            list of numpy array.
    """
    model_weights = []
    i=0
    for var in model.trainable_variables:
        shape = var.shape
        len_ = 1
        for j in shape:
            len_ *= j

        layer = np.array(flatten_weights[i:i+len_]).reshape(shape)
        model_weights.append(layer)
        i += len_
    return model_weights

def set_connection(host):
    """ Set the connection of a specific server.

        args:
            - hosts: string of the form "ip:port"

        returns:
            Stub
    """
    channel = grpc.insecure_channel('localhost' + host[-5:], options=[('grpc.max_send_message_length', -1)
                                    , ('grpc.max_receive_message_length', -1)])
    stub = garfield_pb2_grpc.MessageExchangeStub(channel)
    
    return stub

class AuthGateway(grpc.AuthMetadataPlugin):
    """
        class grpc.AuthMetadataPlugin[source]
       
        A specification for custom authentication.

        __call__(context, callback)[source]
        Implements authentication by passing metadata to a callback.

        This method will be invoked asynchronously in a separate thread.Parameters
        context – An AuthMetadataContext providing information on the RPC that the plugin is being called to authenticate.

        callback – An AuthMetadataPluginCallback to be invoked either synchronously or asynchronously.
    """
    def __init__(self , private_key , role):
        self.private_key = private_key
        self.role = role


    def __call__(self, context, callback):

        """Implements authentication by passing metadata to a callback.

        Implementations of this method must not block.

        Args:
          context: An AuthMetadataContext providing information on the RPC that
            the plugin is being called to authenticate.
          callback: An AuthMetadataPluginCallback to be invoked either
            synchronously or asynchronously.
        """
        print(context)
        encoded = jwt.encode({"request": context.method_name}, (self.private_key).decode("utf-8") , algorithm="RS256")
        # decoded = jwt.decode(encoded, (_credentials.CLIENT_CERTIFICATE_PUBLIC_KEY).decode("utf-8"), algorithms="RS256")
        # print(decoded)
        # decoded = jwt.decode(encoded, _credentials.CLIENT_CERTIFICATE_PUBLIC_KEY, algorithms=["RS256"])

        signature = context.method_name[::-1]
        callback((("role" , self.role) , ("locked_role", encoded),), None)


def set_secure_connetion(host , root_certificate , private_key , certificate_chain):
    """
        Set the secure connection of specific server.

        args:
            - hosts:  string of the form "ip:port"
            - role :  role and access of the hosts
        
        returns:
            - messageExchangeStub
        
    """
    
    channel_credential = grpc.ssl_channel_credentials(
        root_certificates = root_certificate,
        private_key = private_key,
        certificate_chain = certificate_chain
    )
    

    channel = grpc.secure_channel('localhost' + host[-5:], channel_credential)
    stub = garfield_pb2_grpc.MessageExchangeStub(channel)
    return stub


def training_progression(MAX, iter, accuracy):
    char = chr(0x2588)
    i = int(iter/(MAX-1)*20)
    sys.stdout.write(f"\rTraining |{char * i}" + f"." * (20 - i) + f" | iter: {str(iter)}/{str(MAX-1)} Accuracy: {accuracy:.2f}%")
    sys.stdout.flush()