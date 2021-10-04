import random
from sys import displayhook
import numpy as np

def divide_gradeint(gradient):
    rand = random.random()
    # print("random numebr" , rand)
    return np.array(((gradient + rand) / 2 , (gradient - rand) / 2) , np.float32)


def compute_distances(grads):

    distances = np.zeros((len(grads) , len(grads)) , np.float32)

    for i,a in enumerate(grads):
        for j,b in enumerate(grads):
            distances[i][j] = np.linalg.norm(a-b)

    return distances

def multi_krum_aggregator(distances, nbworkers, nbbyzwrks, number_of_gradient):
    # print("the distances that multi krum aggregathor received" , distances)
    nbselected = nbworkers - nbbyzwrks 
    scores = []
    for row in distances:
        # sum over n - f - 2 closest gradient to current gradient
        scores.append(sum(np.sort(row)[0:nbselected]))
    # print("multi_krum_aggregator" , scores)

    idx = np.argpartition(scores, number_of_gradient - 1)
    # print("multi_krum_aggregator argpartition", idx)
    gar_weight = np.zeros(nbworkers , np.float32)
    np.put(gar_weight , idx, 1)
    # print("finaly and hopefully this is gar weight" , gar_weight)
    return gar_weight


def compute_weighted_gradient(partial_gradient , aggregation_weight):
    # print("in tools, partial gradients" , type(partial_gradient) , len(partial_gradient) , partial_gradient)
    weighted_gradient = np.zeros(len(partial_gradient[0]))
    for weight , gradient in zip(aggregation_weight , partial_gradient):
        # print("this is weight and this is gradient, in model server" , weight , gradient , final_gradient)
        if weight == 1:
            weighted_gradient += gradient
    # print("this issss theeee enddddd" , weighted_gradient)
    return weighted_gradient
    