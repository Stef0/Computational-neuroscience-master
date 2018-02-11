# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:00:35 2017

@author: Stefano
"""


import numpy as np
import math
import matplotlib.pyplot as plt
import chainer
from chainer import report, optimizers, Link, Chain
import chainer.functions as F
import chainer.links as L
from chainer.datasets import TupleDataset
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


def create_data(n=2):

    X = np.random.rand(n,1).astype('float32')
    X = X
    T = np.sum(np.hstack((X[0:-1],X[1:])),axis=1)
    T = np.hstack([0, T[0:]]).astype('float32')
    T = T.reshape([n,1])

    return TupleDataset(X, T)

data=create_data()

class MyRegressor(chainer.Chain):
    def __init__(self, predictor):
        super(MyRegressor, self).__init__(predictor=predictor)

    def __call__(self, x, y):
        # This chain just computes the mean absolute and squared
        # errors between the prediction and y.
        pred = self.predictor(x)
        abs_error = np.absolute(pred - y) 
        loss = F.mean_squared_error(pred, y)

        # Report the mean absolute and squared errors.
        report({'abs_error': abs_error, 'squared_error': loss}, self)

        return loss

    

class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__()
        with self.init_scope():
            self.mid = L.Linear(None, 100)  # the first LSTM layer
            self.out = L.Linear(100, 1)  # the feed-forward output layer


    def __call__(self, cur_word):
        # Given the current word ID, predict the next word.
        h = self.mid(cur_word)
        y = self.out(h)
        return y

rnn = RNN()
model = MyRegressor(rnn)
accfun=F.accuracy
optimizer = optimizers.SGD()
optimizer.setup(model)


def compute_loss(data):
    loss = 0
    for inputt, outputt in zip(data._datasets[0], data._datasets[1]):
        inputt=data._datasets[0].reshape([1,2]).astype('float32')
        outputt=data._datasets[1][1].reshape([1,1]).astype('float32')
        loss = model(inputt, outputt)
    return loss


# "epoch" iteration here
epochs=100

for i in range(1, epochs):
    
    data=create_data()
    model.cleargrads()
    loss = compute_loss(data)
    loss.backward()
    optimizer.update()
    expected=data._datasets[1][(data._datasets[1]).size-1] # takes last value of t
    lastinput=data._datasets[0] #takes last value of x
    predicted=rnn(lastinput.reshape([1,2]))
    accuracy = predicted-expected # we want accuracy close to zero because percentages are difficult to calculate URRDURR
    print (loss,accuracy) 
