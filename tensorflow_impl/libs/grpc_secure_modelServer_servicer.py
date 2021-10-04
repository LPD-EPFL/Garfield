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
import numpy as np
import sys
from . import garfield_pb2
from . import garfield_pb2_grpc
from . import tools
import pickle
from OpenSSL import crypto

class SecureMessageExchangeServicerModelServer(garfield_pb2_grpc.MessageExchangeServicer):

    def __init__(self, model_weights):
        # print("SecureMessageExchangeServicerModelServer activated")
        """
            args: 
                - model_weights: 
                - workers : list of workers' port
                - model_server : model server's port
                - worker_server : worker server's port
        """

        self.model_wieghts_history = [model_weights]
        self.partial_gradient_different = []




    def GetModel(self, request, context):
        """Get the model weights of a specific iteration stored on the server."""
        # print("in the SecureMessageExchangeServicerModelServer , get Model")

        iter = request.iter
        job = request.job
        req_id = request.req_id
        # print("iteration" , iter)
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
        # print("in the SecureMessageExchangeServicerModelServer , get gradeint")
        """Get the graidents of a specific iteration stored on the server."""
        iter = request.iter
        job = request.job
        req_id = request.req_id

        # check the premission
        auth_data = context.auth_context()
        certificate = crypto.load_certificate(crypto.FILETYPE_PEM, auth_data['x509_pem_cert'][0])
        entrant = certificate.get_subject().organizationName

        # print("this is enterrrrrrrant" , entrant)
        serialized_gradients = bytes("unathurized member is trying to connect" , 'utf-8')
        if entrant == "worker":
            serialized_gradients = bytes("You don't have the premission to access this data", 'utf-8')


        elif entrant == "worker server":
            # print("at least getting here, while the size of partial gradient is " , len(self.partial_gradient_different))
            # print("this is id of list in the model server servicer" , id(self.partial_gradient_different), "and the iteration number is:" , iter) 
            while iter >= len(self.partial_gradient_different):
                time.sleep(0.001)
            # print("did I pass this fucking stupid while")
            # print("the size is" , sys.getsizeof(self.partial_gradient_different[iter]))
            # print("the length of array" , self.partial_gradient_different[iter].shape)
            
            serialized_gradients = pickle.dumps(self.partial_gradient_different[iter])
            # print("in model server servicer, the distances is " , self.partial_gradient_different[iter]) 
            # print("in model server and the type is :" , self.partial_gradient_different[iter].dtype.name)  
            # print("in model server servicer, desrialized the disntaces" , np.frombuffer(serialized_gradients, dtype=np.float64))       
            # print("finallllyyy I can get it")

            
        
        return garfield_pb2.Gradients(gradients=serialized_gradients,
                                      iter=iter)


    def SendGradient(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')
