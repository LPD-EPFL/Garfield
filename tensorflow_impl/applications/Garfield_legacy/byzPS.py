# coding: utf-8
###
 # @file   byzPS.py
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
 # Garfield implementation. The parameter server implementation
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
from grpc_service_impl import TrainMessageExchangeServicer
from concurrent import futures
import threading
from multiprocessing.dummy import Pool as ThreadPool

import collections
import builtins as __builtin__

import tensorflow as tf

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
lengths = {"cnnet" : 1756426, "mnist" : 79510, "slim-cifarnet-cifar10" : 1756426, "slim-vgg_a-cifar10" : 128807306, "slim-inception_v1-cifar10" : 5602874, "slim-inception_v3-cifar10" : 24353332, "slim-resnet_v2_50-cifar10" : 23539850, "slim-resnet_v2_200-cifar10": 62697610}
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
parser.add_argument('--asyncr',
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
help='If True, detailed steps are printed.')
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
T = int(1/(3*FLAGS.l*FLAGS.rate))
grad_length = lengths[FLAGS.experiment]

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

#Initializing the service class which holds all information may be pulled by the client (workers for PS and vice versa)
service = TrainMessageExchangeServicer(num_workers, FLAGS.smart, uid)
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1000))
    all_pb2_grpc.add_TrainMessageExchangeServicer_to_server(
        service, server)
    server.add_insecure_port(ps_hosts[int(FLAGS.task_index)])
    print("gRPC server now will start at PS index ", FLAGS.task_index)
    sys.stdout.flush()
    return server

server = serve()
server.start()
time.sleep(10)

#Connect to all other PSes
def connect_PS():
  for i in range(num_ps):
    if i != FLAGS.task_index:
      channel = grpc.insecure_channel(ps_hosts[i], options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
      stub = all_pb2_grpc.TrainMessageExchangeStub(channel)
      ps_conns.append(stub)

#Connect to all workers
def connect_workers():
  for i in range(num_workers):
    channel = grpc.insecure_channel(worker_hosts[i], options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    stub = all_pb2_grpc.TrainMessageExchangeStub(channel)
    worker_conns.append(stub)

#Read gradient from given connection
def get_grad(conn):
  global request
  read = False
  counter = 0		#So that we do not loop forever
  while not read:
    try:
      proto_grad = conn.GetGradients(request)
      grad = np.frombuffer(proto_grad.gradients, np.float32)
      read = True
    except Exception as e:
      print("PS read_gradients exception: ")
      counter+=1
      if counter > 10:			#any reasonable large enough number
        exit(0)
  return grad

quorum = (2*num_byzwrks + 3) if FLAGS.asyncr else num_workers
quorum_ps = (2*num_byzps + 3) if FLAGS.asyncr else num_ps
grads = [tf.placeholder(dtype=tf.float32, shape=(grad_length)) for _ in range(quorum)]
grads_ps = [tf.placeholder(dtype=tf.float32, shape=(grad_length)) for _ in range(quorum_ps)]
grads_nt = []
grads_ps_nt = []
#Create pools for parallelizing reading gradients
pool_ps = ThreadPool(quorum_ps)
pool_wk = ThreadPool(quorum)
#Read gradients from workers (or PSes, in case of the asyncr protocol)
def read_gradients(PS=False):
  if FLAGS.log:
    println("[PS] trying to read gradients")
  global grads_nt
  global grads_ps_nt
  if PS:
    grads_ps_nt = []
  else:
    grads_nt = []

# Read gradients in a parallel fashion
  if PS:
    if FLAGS.log:
      println("[PS] Preparing to read from how many PS? " + str(quorum_ps))
    grads_ps_nt = pool_ps.map(get_grad, ps_conns)
  else:
    if FLAGS.log:
      println("[PS] Preparing to read from how many Worker? " + str(quorum))
    grads_nt = pool_wk.map(get_grad, worker_conns)

  if FLAGS.log:
    println("At PS, done reading gradients")

request = all_pb2.Request()
request.req_id = uid
ps_conns = []
connect_PS()
empty = all_pb2.Empty()
worker_conns = []
connect_workers()

iteration_num=-1
graph = tf.get_default_graph()

#Building the graph....
with tf.device('/'+device+':'+str(FLAGS.task_index%2)):
  optimizer = tf.train.RMSPropOptimizer(FLAGS.rate) #tf.train.AdadeltaOptimizer() #tf.train.AdamOptimizer() #tf.train.RMSPropOptimizer(FLAGS.rate) #tf.train.GradientDescentOptimizer(FLAGS.rate)
  experiment = experiments.instantiate(FLAGS.experiment, ['batch-size:'+str(FLAGS.batch), 'eval-batch-size:1'])
  with tf.device('/cpu:*'):
    acc_tn = experiment.accuracy('/cpu:0', ['/'+device+':'+str(FLAGS.task_index%2)], trace=False)
  #Initializing aggregators.....
  #Krum
  q_size = num_workers
  if FLAGS.asyncr:
    q_size = (2*num_byzwrks) + 3
  krum_op = aggregators.instantiate("krum", q_size, num_byzwrks, None)
  #Average
  avg_op = aggregators.instantiate("average", num_workers, num_byzwrks, None)
  #Median
  med_op = aggregators.instantiate("median", quorum_ps, num_byzps, None)
  #Bulyan
  bul_op = aggregators.instantiate("bulyan", q_size, num_byzwrks, None)

  #The 'Experiment' interface related definitions
  loss_tn = experiment.losses('/cpu:0', ['/'+device+':'+str(FLAGS.task_index%2)], trace=False)
  grad_vars = optimizer.compute_gradients(loss_tn[0])
  gradient_tn, flatmap = helper.flatten(grad_vars)

  #2) Read the gradients for the new iteration and #3) The logic of aggregation
  grads = [tf.placeholder(dtype=tf.float32, shape=(grad_length)) for _ in range(quorum)]
  if FLAGS.smart:
    last_grad = krum_op.aggregate(grads)		#This list is defined above
  elif FLAGS.asyncr:
    last_grad = krum_op.aggregate(grads)			#Apply the strongest GAR
  elif FLAGS.vanilla:
    last_grad = avg_op.aggregate(grads)                #This list is defined above
  if FLAGS.asyncr:		#Then more steps are required before applying this aggregated gradients
    grads_ps = [tf.placeholder(dtype=tf.float32, shape=(grad_length)) for _ in range(quorum_ps)]
    med_grad = med_op.aggregate(grads_ps)
    apply_op = optimizer.apply_gradients(helper.inflate(med_grad, helper.mapflat(flatmap)))
  elif FLAGS.smart:
    grads_ps = [tf.placeholder(dtype=tf.float32, shape=(grad_length)) for _ in range(num_ps)]
    med_grad = med_op.aggregate(grads_ps)
    apply_op_gather = optimizer.apply_gradients(helper.inflate(med_grad, helper.mapflat(flatmap)))
    apply_op = optimizer.apply_gradients(helper.inflate(last_grad, helper.mapflat(flatmap)))
  else:
    #4) finally, apply this aggregation
    apply_op = optimizer.apply_gradients(helper.inflate(last_grad, helper.mapflat(flatmap)))

#Define get accuracy function separately to run it in a different thread
def get_acc(sess,train_t, cur_t, iter_n):
  acc = sess.run(acc_tn)
  for key, val in acc.items():
    println(key+":"+str(val)+" elapsed_time: "+str(cur_t - train_t)+" iteration: "+str(iter_n))
  println("Throughput: " + str(iter_n/(cur_t - train_t)) + "steps/sec")

#get unified model in the beginning so that all PSes are on the same page....
def getUnifiedModel(sess):
  model = ps_conns[0].GetUnifiedModel(request)
  model = np.frombuffer(model.model, np.float32)
  model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  cur = 0
  if FLAGS.log:
    println("PS will try to write the same model of PS, received model is of length "+str(len(model)))
  for var in model_variables:
    println("Fetching new variable...")
    next = cur + tf.size(var)
    next = sess.run(next)
    assign = tf.assign(var, tf.reshape(model[cur:next],tf.shape(var)))
    sess.run(assign)
    cur = next
  println("PS is done with writing the new model....now this PS has the same model like the master PS, model size: "+str(len(model)))
  return model

#Read the current state of the model
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

  #First, read the model and distribute it to everybody so that all of the machines are on the same page
  if FLAGS.task_index == 0:
    model_np = read_model(sess)
    service.mod.model = model_np.tobytes()
    service.mod.init = True
  else:
    model_np = getUnifiedModel(sess)

  if FLAGS.log:
    println(model_np)
    println("The model is of size: "+str(len(model_np)))

  train_t = time.time()
  while iteration_num < FLAGS.max_steps:
    if FLAGS.bench:
      iter_t = time.time()
    iteration_num+=1
    if FLAGS.log:
      println("PS iteration_num: "+str(iteration_num))
    #Get the accuracy from time to time
    if iteration_num%FLAGS.eval_steps == 1:
      if FLAGS.bench:
        acc_t = time.time()
      acc_th = threading.Thread(target=get_acc, args=(sess,train_t,time.time(),iteration_num,))
      acc_th.start()

      if FLAGS.bench:
        println("[PS "+str(FLAGS.task_index)+"] Time elapsed to get accuracy: " + str(time.time() - acc_t))
    if iteration_num > 0:
      #1) Prepare gradients from the last iteartion
      if FLAGS.log:
        println("PS prepares gradient for the next iteration")
      if FLAGS.bench:
        np_t = time.time()
      service.pbuf_grad.gradients = aggr_res.tobytes()			#convert grad tensor to a protocol buffer
      service.pbuf_grad.iter = iteration_num
      if FLAGS.bench:
        println("[PS "+str(FLAGS.task_index)+"] time to convert gradient to bytes: " + str(time.time() - np_t))
    if FLAGS.bench:
      read_t = time.time()
    request.iter = iteration_num
    read_gradients()
    if FLAGS.bench:
      println("[PS "+str(FLAGS.task_index)+"] time to read gradients is: " + str(time.time() - read_t))
      feed_t = time.time()
    #Now, feed values to input of aggregate: grads (from grads_nt numpy array)
    if FLAGS.log:
      println("Number of collected grads: " + str(len(grads_nt)))
    try:
      feed_dict = dict([(grads[i], grads_nt[i]) for i in range(len(grads))])
    except Exception as e:
      println(e)
      exit(0)
    if FLAGS.log:
      println("[PS] done feed dict....")
      for k,v in feed_dict.items():
        println(str(len(v)))
    if FLAGS.bench:
      println("[PS "+str(FLAGS.task_index)+"] time to create feed_dict is: " + str(time.time() - feed_t))
      run_t = time.time()
    try:
      if FLAGS.asyncr:
        aggr_res = sess.run(last_grad, feed_dict=feed_dict)
      else:
        aggr_res,_ = sess.run([last_grad, apply_op], feed_dict=feed_dict)
      if FLAGS.bench:
        println("[PS "+str(FLAGS.task_index)+"] time to aggregate and apply gradients: " + str(time.time() - run_t))
    except Exception as e:
      println("problems with running last_grad and apply_op")
      println(e)
      exit(0)
    if FLAGS.log:
      println("[PS] Aggregation at PS...done!")
    #If the asyncr protocol, one more round of exchanging gradients between PSes is required
    if FLAGS.asyncr or (FLAGS.smart and iteration_num%T == 0):
      if FLAGS.log:
        println("[PS] Starting the additional asyncronous block")
      service.pbuf_grad.gradients = aggr_res.tobytes()
      service.pbuf_grad.iter = iteration_num+0.5
      if FLAGS.bench:
        read_ps = time.time()
      read_gradients(PS=True)
      if FLAGS.bench:
        println("[PS "+str(FLAGS.task_index)+"] Time to read from other PSes: " + str(time.time()-read_ps))
      if FLAGS.log:
        println("[PS] Read gradients from other PSes")
      #Add myself to the list
      grads_ps_nt.append(aggr_res)
      #Feed the tensor and do the median
      feed_dict = dict([(grads_ps[i], grads_ps_nt[i]) for i in range(len(grads_ps))])

      if FLAGS.log:
        println("[PS] Done feeding to tensor")
      if FLAGS.bench:
        final_up = time.time()
      if FLAGS.asyncr:
        aggr_res,_ = sess.run([med_grad, apply_op], feed_dict=feed_dict)
      else:
        aggr_res,_ = sess.run([med_grad, apply_op_gather], feed_dict=feed_dict)
      if FLAGS.bench:
        print("[PS "+str(FLAGS.task_index)+"] Time to aggregate PS responses and apply them: " + str(time.time()-final_up))
      if FLAGS.log:
        println("[PS] Done the additional asyncronous block")

    if FLAGS.bench:
      println("[PS "+str(FLAGS.task_index)+"] time of one complete iteration: " + str(time.time() - iter_t))
time.sleep(10)
server.stop(0)
