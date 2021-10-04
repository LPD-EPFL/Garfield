# coding: utf-8
###
 # @file   grpc_message_exchange_servicer.py
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

import time
import grpc
from . import garfield_pb2
from . import garfield_pb2_grpc
from . import tools
from OpenSSL import crypto

class SecureMessageExchangeServicerWorker(garfield_pb2_grpc.MessageExchangeServicer):

    def __init__(self, model_weights):
        """
            args: 
                - model_weights: 
                - workers : list of workers' port
                - model_server : model server's port
                - worker_server : worker server's port
        """

        self.model_wieghts_history = [model_weights]
        self.partial_gradients_history_model_server = []
        self.partial_gradients_history_worker_server = []




    def GetModel(self, request, context):
        """Get the model weights of a specific iteration stored on the server."""
        iter = request.iter
        job = request.job
        req_id = request.req_id

        while iter >= len(self.model_wieghts_history):
            time.sleep(0.001)
        
        serialized_model = self.model_wieghts_history[iter].tobytes()
        #serialized_model = tools.weights_to_bytes(self.model_wieghts_history[iter])
        return garfield_pb2.Model(model=serialized_model,
                                  init=True,
                                  iter=iter)
        

    def SendModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetGradient(self, request, context):
        """Get the graidents of a specific iteration stored on the server."""
        iter = request.iter
        job = request.job
        req_id = request.req_id

        # check the premission
        auth_data = context.auth_context()
        certificate = crypto.load_certificate(crypto.FILETYPE_PEM, auth_data['x509_pem_cert'][0])
        entrant = certificate.get_subject().organizationName
        # print("This is entrant" , entrant)
        if entrant == "worker":
            # print("in the servicer , I am worker")
            serialized_gradients = bytes("You don't have the premission to access this data", 'utf-8')

        elif entrant == "model server":
            # print("in the servicer , I am model server")
            while iter >= len(self.partial_gradients_history_model_server):
                time.sleep(0.001)
            serialized_gradients = self.partial_gradients_history_model_server[iter].tobytes()
        elif entrant == "worker server":
            # print("in the servicer , I am worker server")
            while iter >= len(self.partial_gradients_history_worker_server):
                time.sleep(0.001)
            serialized_gradients = self.partial_gradients_history_worker_server[iter].tobytes()            
        else:
            serialized_gradients = bytes("Could not authenticate the request", 'utf-8')

        return garfield_pb2.Gradients(gradients=serialized_gradients,
                                      iter=iter)

    def SendGradient(self, request, context):
        """Missing associated documentation comment in .proto file."""
        
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

        