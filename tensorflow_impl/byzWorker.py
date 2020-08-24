# coding: utf-8
###
 # @file   byzWorker.py
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
 # Garfield implementation. The worker implementation
###

#!/usr/bin/env python

import numpy as np
import argparse
import os
import pathlib
import signal
import sys
import time
import time
import random
import math
import all_pb2, all_pb2_grpc
import grpc
from grpc_service_impl_slim import TrainMessageExchangeServicer
from concurrent import futures

import collections
import builtins as __builtin__

import tensorflow as tf
from tensorflow.python.client import timeline

root = pathlib.Path(__file__).resolve().parent
sys.path.append(str(root / "models"))

import aggregators
import native
import tools
import models
import experiments
import helper

def println(str):
  print(str)
  sys.stdout.flush()

device='gpu'
lengths = {"cnnet" : 1756426, "mnist" : 79510, "slim-cifarnet-cifar10" : 1756426, "slim-vgg_a-cifar10" : 128807306, "slim-inception_v1-cifar10" : 5602874, "slim-inception_v3-cifar10" : 5602874, "slim-resnet_v2_50-cifar10" : 23539850, "slim-resnet_v2_200-cifar10": 62697610}
MAX_MESSAGE_LENGTH = 600000000
# Argument parsing
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--ps_hosts',
type=str,
default="127.0.0.1:4523",
help='Comma-separated list of hostname:port pairs')
parser.add_argument('--worker_hosts',
type=str,
default="127.0.0.1:4524",
help='Comma-separated list of hostname:port pairs')
parser.add_argument('--log_dir',
type=str,
default='',
help='Directory where to write event logs and checkpoint.')
parser.add_argument('--job_name',
type=str,
default='',
help="One of 'ps', 'worker'")
parser.add_argument('--task_index',
type=int,
default=0,
help="Index of task. e.g., if 2 workers: \n1st worker-> task_index 0 & 2nd worker-> task_index 1")
parser.add_argument('--max_steps',
type=int,
default=3000,
help='Maximum number of steps to run.')
parser.add_argument('--eval_steps',
type=float,
default=10,
help='Save model checkpoints frequency.')
parser.add_argument('--log_frequency_steps',
type=int,
default=10,
help='How often to log results to the console.')
parser.add_argument('--cpus',
type=int,
default=0,
help='Number of CPU per worker (does not affect the chief). 0 for every available logical core.')
parser.add_argument('--rule',
type=str,
default="krum",
help='Rule to choose (one of "krum", "kardam", ...).')
parser.add_argument('--rule_m',
type=int,
default=0,
help='Parameter for some rules (currently only for "krum"); optional, 0 for default.')
parser.add_argument('--experiment',
type=str,
default="slim-cifarnet-cifar10",
help='Problem and associated, fixed model to choose (one of "mnist", "cifar10", ...).')
parser.add_argument('--batch',
type=int,
default=128,
help='Batch size to use.')
parser.add_argument('--rate',
type=float,
default=1e-3,
help='(Base) learning rate to use.')
parser.add_argument('--nbbyzwrk',
type=int,
default=0,
help='(Declared) number of byzantine workers.')
parser.add_argument('--nbbyzps',
type=int,
default=0,
help='(Declared) number of byzantine PSes.')
parser.add_argument('--vanilla',
type=bool,
default=False,
help='If True, the vanilla, non-byzantine case is executed. This is the baseline that we should compare with.')
parser.add_argument('--async',
type=bool,
default=False,
help='If True, the network is assumed to be asynchronous. Thus, only qourum is waited before proceeding.')
parser.add_argument('--smart',
type=bool,
default=False,
help='If True, the smart byzPS algorithm runs.')
parser.add_argument('--log',
type=bool,
default=False,
help='If True, detailed steps are printed')
parser.add_argument('--bench',
type=bool,
default=False,
help='If True, time elapsed in each step is printed.')
parser.add_argument('--time_save',
type=bool,
default=False,
help='If True, a BW aggressive but time-saver algorithm is applied.')
parser.add_argument('--less_grad',
type=bool,
default=False,
help='If True, PS collects only 2f+3 grads instead of asking all workers to send.')
parser.add_argument('--l',
type=int,
default=1,
help='Lipschtiz constant l. This should depend on the loss function used. For now, I will keep it to 1.')

FLAGS = parser.parse_args()

grad_length = lengths[FLAGS.experiment]

T = int(1/(3*FLAGS.l*FLAGS.rate))

print("T: ", T)
print("Using smart? ", FLAGS.smart)
print("Using batch: ", FLAGS.batch)
# User home for slim (suppose you downloaded the associated files)
homedir = pathlib.Path.home()

# Parsing command line arguments
ps_hosts = FLAGS.ps_hosts.split(",")
ps_addrs = [addr[:addr.index(":")] for addr in ps_hosts]
ps_ports = [int(addr[addr.index(":")+1:]) for addr in ps_hosts]
num_ps = len(ps_hosts)

worker_hosts = FLAGS.worker_hosts.split(",")
worker_addrs = [addr[:addr.index(":")] for addr in worker_hosts]
worker_ports = [int(addr[addr.index(":")+1:]) for addr in worker_hosts]
num_workers = len(worker_hosts)

num_byzwrks = FLAGS.nbbyzwrk
num_byzps = FLAGS.nbbyzps
if FLAGS.job_name == "ps":
  uid = -1*(FLAGS.task_index+1)
else:
  uid = FLAGS.task_index+1

#grpc service class
service = TrainMessageExchangeServicer(num_workers, FLAGS.smart, uid)
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1000))
    all_pb2_grpc.add_TrainMessageExchangeServicer_to_server(
        service, server)
    server.add_insecure_port(worker_hosts[int(FLAGS.task_index)])
    print("gRPC server is waiting now at WORKER index ", FLAGS.task_index)
    sys.stdout.flush()
    return server
server = serve()
server.start()

#Method convert tensor to a protocol buffer (serialization)
def tensor_to_protobuf(tensor):
  return tensor.tobytes()

#Read gradients from PSes
quorum = (2*num_byzps+3) if FLAGS.async else 1
grads = []
#Special for the synchronous algorithm algorithm
next_ps = random.randint(0, num_ps-1)
def read_gradients(quorum_smart=-1):
  global grads
  if FLAGS.log:
    println("worker: iteration num: "+str(iteration_num))
  quorum = (2*num_byzps+3) if FLAGS.async else 1
  if iteration_num < 0:
    return [tf.placeholder(dtype=tf.float32, shape=(grad_length)) for _ in range(quorum)]
  grads = []
  ii = random.randint(0,len(connections)-1)
  if FLAGS.vanilla:
    ii=0
  elif FLAGS.smart:
    ii = (next_ps+1)%num_ps
  request.iter = iteration_num
  while True:
    s = connections[ii]
    try:
      proto_grad = s.GetGradients(request, timeout=60)
    except grpc.RpcError as e:
      println(e)
      exit(0)
    grad = np.frombuffer(proto_grad.gradients, np.float32)
    grads.append(grad)
    ii = (ii+1)%len(connections)
    if FLAGS.smart and quorum_smart == -1:
      return grads
    if len(grads) >= quorum and quorum_smart == -1:
      break
    if len(grads) == quorum_smart:
      break
  return grads

request = all_pb2.Request()
request.req_id = FLAGS.task_index
empty = all_pb2.Empty()
connections = []
for i in range(num_ps):
  channel = grpc.insecure_channel(ps_hosts[i], options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
  stub = all_pb2_grpc.TrainMessageExchangeStub(channel)
  connections.append(stub)

iteration_num=-1
graph = tf.get_default_graph()
with tf.device('/'+device+':'+str(FLAGS.task_index%2)):
  last_grad = tf.constant(0)		#any unused initialization...this should hold the latest computed gradient each iteration
  optimizer = tf.train.RMSPropOptimizer(FLAGS.rate)
  experiment = experiments.instantiate(FLAGS.experiment, ['batch-size:'+str(FLAGS.batch), 'eval-batch-size:1000'])

  #The 'Experiment' interface related definitions
  loss_tn = experiment.losses('/cpu:0', ['/'+device+':'+str(FLAGS.task_index%2)], trace=False)
  grad_vars = optimizer.compute_gradients(loss_tn[0])
  gradient_tn, flatmap = helper.flatten(grad_vars)

  #Initializing aggregators.....
  #Krum
  q_size = num_ps
  if FLAGS.async:
    q_size = (2*num_byzps) + 3
  krum_op = aggregators.instantiate("krum", q_size, num_byzps, None)

  #1) Read latest gradients from PSes...
  if FLAGS.async:
    grads_ps = [tf.placeholder(dtype=tf.float32, shape=(grad_length)) for _ in range(quorum)]
    last_grad = krum_op.aggregate(grads_ps)
    apply_op = optimizer.apply_gradients(helper.inflate(last_grad, helper.mapflat(flatmap)))
  elif FLAGS.smart:
    grads_ps = [tf.placeholder(dtype=tf.float32, shape=(grad_length)) for _ in range(num_ps)]
    last_grad = krum_op.aggregate(grads_ps)
    apply_op_gather = optimizer.apply_gradients(helper.inflate(last_grad, helper.mapflat(flatmap)))
    update_grad = tf.placeholder(dtype=tf.float32, shape=(grad_length))
    apply_op = optimizer.apply_gradients(helper.inflate(update_grad, helper.mapflat(flatmap)))

  else:
    #2) Applying the read gradient so that I can have the updated version of the model
    update_grad = tf.placeholder(dtype=tf.float32, shape=(grad_length))
    apply_op = optimizer.apply_gradients(helper.inflate(update_grad, helper.mapflat(flatmap)))

def getUnifiedModel(sess):
  model = connections[0].GetUnifiedModel(request)
  model = np.frombuffer(model.model, np.float32)
  model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  cur = 0
  if FLAGS.log:
    println("Worker will try to write the same model of PS, received model is of length "+str(len(model)))
  for var in model_variables:
    println("Fetching new variable...")
    next = cur + tf.size(var)
    next = sess.run(next)
    assign = tf.assign(var, tf.reshape(model[cur:next],tf.shape(var)))
    sess.run(assign)
    cur = next
  println("Worker is done with writing the new model....now worker has the same model like the PS, model size: "+str(len(model)))

#This is to read the model if you want
model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
mod_res = []
for var in model_variables:
  mod_res.append(tf.reshape(var, (-1,)))
mod_res = tf.concat(mod_res, 0)
def read_model(sess):
  return sess.run(mod_res)

config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(graph=graph, config=config)
with sess.as_default():
  sess.run(tf.global_variables_initializer())
  coord = tf.train.Coordinator()
  tf.train.start_queue_runners(sess=sess, coord=coord) # If there is any queue in the graph

  #Variables required for "synchronous" algorithm
  if FLAGS.smart:
    past_model = []
    cur_model = []
    past_grad = []
    cur_grad = []
    lipschitzes = []
  #Let's pull a unified model for now before start training....pull it from the first PS
  if FLAGS.log:
    println("Getting unified model at the worker side..")
  try:
    getUnifiedModel(sess)
  except Exception as e:
    println("Printing exception of getting unified model....")
    println(e)

  if FLAGS.log:
    model_np = read_model(sess)
    println("Model on Worker: ")
    println(model_np)

  if FLAGS.smart:
    cur_model = read_model(sess)

  #Training loop
  while iteration_num < FLAGS.max_steps:
    if FLAGS.bench:
      iter_t = time.time()
    iteration_num+=1
    if iteration_num > 0:
      try:
        if FLAGS.bench:
          read_t = time.time()
        up_grad = read_gradients(num_ps if FLAGS.smart and iteration_num%T == 0 else -1)
        #If the synchronous algorithm is used, it's important to keep the grad for comparison
        if FLAGS.smart:
          if iteration_num%T != 0:
            cur_grad = up_grad[0]
          else:				#This is the gather step now!
            feed_dict = dict([(grads_ps[i], up_grad[i]) for i in range(len(grads_ps))])
            cur_grad, _ = sess.run([last_grad, apply_op_gather], feed_dict=feed_dict)
        if FLAGS.bench:
          println("[Worker "+str(FLAGS.task_index)+"] time to read gradients: "+str(time.time() - read_t))
          apply_t = time.time()
        if FLAGS.async:
          feed_dict = dict([(grads_ps[i], up_grad[i]) for i in range(len(grads_ps))])
          sess.run([last_grad, apply_op], feed_dict=feed_dict)
        else:
          if not (FLAGS.smart and iteration_num%T == 0):
            sess.run(apply_op, feed_dict={update_grad: up_grad[0]})					#Trigger applying the old gradient to get the new model
        if FLAGS.bench:
          println("[Worker "+str(FLAGS.task_index)+"] time to apply gradient: " + str(time.time() - apply_t))
      except Exception as e:
        println("Problem with either reading or applyging gradients")
        println(e)
        exit(0)
    #Ok...I got the new model -> now to the backpropagation step...
    if FLAGS.bench:
      backprop_t = time.time()
    grad = sess.run(gradient_tn)                          #Trigger computing the new gradient
    if FLAGS.bench:
      println("[Worker "+str(FLAGS.task_index)+"] time to calculate gradients: " + str(time.time() - backprop_t))

    #If the synchronous algorithm is used, make sure that the sent model is legit first....
    if FLAGS.smart and len(past_grad) != 0:
      if FLAGS.log:
        println("Applying filters....")
      #Apply filters of the synchronous algorithm
      cur_model = read_model(sess)
      local_model = cur_model
      if FLAGS.bench:
        kardam_t = time.time()
      try:
        lipschitz = np.linalg.norm(grad - past_grad)/np.linalg.norm(local_model - past_model)
      except Exception as e:
        println(e)
        exit(0)
      if FLAGS.log:
        println("lipschitz value: " + str(lipschitz))
      lipschitzes.append(lipschitz)
      n_f_percentile = np.percentile(lipschitzes, 100*(num_ps-num_byzps)/num_ps)
      if FLAGS.bench:
        println("Kardam takes (in sec)"+str(time.time() - kardam_t))
      if FLAGS.log:
        println("Percentile value: " + str(n_f_percentile))
      if len(lipschitzes) == 100:		#This trimming is crucial in performance
        lipschitzes = lipschitzes[-50:]
      if num_byzwrks != 0:		#This test is only valid if f_w>0 (otherwise does not make sense)
        models_diff = np.linalg.norm(local_model - cur_model)
        reference = FLAGS.rate * np.linalg.norm(grad) * (((3*T + 2)*(num_workers-num_byzwrks))/(4*num_byzwrks) + 2*((iteration_num - 1)%T))
        if FLAGS.log:
          print("models diff %.2f reference %.2f" % (models_diff, reference))

      np_t = time.time()
    if FLAGS.log:
      print("One worker calculated the gradients: ", grad)
      sys.stdout.flush()
    #Last step: convert tensor to protobuf so that it can be available when the PS asks for it
    service.pbuf_grad.gradients = tensor_to_protobuf(grad)			#convert grad tensor to a protocol buffer
    service.pbuf_grad.iter = iteration_num
    if FLAGS.bench:
      println("[Worker "+str(FLAGS.task_index)+"] time to serialize to bytes: "+str(time.time() - np_t))
      println("[Worker "+str(FLAGS.task_index)+"] time of one complete iteration: "+ str(time.time() - iter_t))
    if FLAGS.smart:
      past_model = cur_model
      past_grad = grad
      model.update(past_gradients)              #local_model is \theta_{t+1}^l
      local_model = model.read()

server.stop(0)
