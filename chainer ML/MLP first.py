# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 21:37:33 2017

@author: Stefano

oh oh oh to touch
happyness
but it burns

"""

import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import TupleDataset


def get_mnist(n_train=100, n_test=100, n_dim=1, with_label=True, classes = None):
    """

    :param n_train: nr of training examples per class
    :param n_test: nr of test examples per class
    :param n_dim: 1 or 3 (for convolutional input)
    :param with_label: whether or not to also provide labels
    :param classes: if not None, then it selects only those classes, e.g. [0, 1]
    :return:
    """

    train_data, test_data = chainer.datasets.get_mnist(ndim=n_dim, withlabel=with_label)

    if not classes:
        classes = np.arange(10)
    n_classes = len(classes)

    if with_label:

        for d in range(2):

            if d==0:
                data = train_data._datasets[0]
                labels = train_data._datasets[1]
                n = n_train
            else:
                data = test_data._datasets[0]
                labels = test_data._datasets[1]
                n = n_test

            for i in range(n_classes):
                lidx = np.where(labels == classes[i])[0][:n]
                if i==0:
                    idx = lidx
                else:
                    idx = np.hstack([idx,lidx])

            L = np.concatenate([i*np.ones(n) for i in np.arange(n_classes)]).astype('int32')

            if d==0:
                train_data = TupleDataset(data[idx],L)
            else:
                test_data = TupleDataset(data[idx],L)

    else:

        tmp1, tmp2 = chainer.datasets.get_mnist(ndim=n_dim,withlabel=True)

        for d in range(2):

            if d == 0:
                data = train_data
                labels = tmp1._datasets[1]
                n = n_train
            else:
                data = test_data
                labels = tmp2._datasets[1]
                n = n_test

            for i in range(n_classes):
                lidx = np.where(labels == classes[i])[0][:n]
                if i == 0:
                    idx = lidx
                else:
                    idx = np.hstack([idx, lidx])

            if d == 0:
                train_data = data[idx]
            else:
                test_data = data[idx]

    return train_data, test_data



class MLP(Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
        # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_out)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        y = self.l2(h1)
        return y



class RandomIterator(object):
    """
    Generates random subsets of data
    """

    def __init__(self, data, batch_size=32):
        """

        Args:
            data (TupleDataset):
            batch_size (int):

        Returns:
            list of batches consisting of (input, output) pairs
        """

        self.data = data

        self.batch_size = batch_size
        self.n_batches = len(self.data) // batch_size

    def __iter__(self):

        self.idx = -1
        self._order = np.random.permutation(len(self.data))[:(self.n_batches * self.batch_size)]

        return self

    def next(self):

        self.idx += 1

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        # handles unlabeled and labeled data
        if isinstance(self.data, np.ndarray):
            print 'yes'
            return self.data[self._order[i:(i + self.batch_size)]],self._order
        else:
            return list(self.data[self._order[i:(i + self.batch_size)]]),self._order


class Classifier(Link):

    """A simple classifier model.
    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.
    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.
    """

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=F.softmax_cross_entropy,
                 accfun=F.accuracy):
        super(Classifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.
        It also computes accuracy and stores it to the attribute.
        Args:
            args (list of ~chainer.Variable): Input minibatch.
        The all elements of ``args`` but last one are features and
        the last element corresponds to ground truth labels.
        It feeds features to the predictor and compare the result
        with ground truth labels.
        Returns:
            ~chainer.Variable: Loss value.
        """

        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            report({'accuracy': self.accuracy}, self)
        return self.loss , self.accuracy

train,test=get_mnist(100,100)

our_net=MLP(10,10)
optimizer=optimizers.SGD()
optimizer.setup(our_net)
our_classifier=Classifier(our_net)

max_epo=20

result=np.zeros((32,1),dtype=float)
result2=np.zeros((max_epo,1),dtype=float)

acc=np.zeros((max_epo,1),dtype=float)
lossData=np.zeros((max_epo,1),dtype=float)
accData=np.zeros((max_epo,1),dtype=float)

counter2=0 

for epochs in range(0,max_epo):

    traindata=np.array(test._datasets[0])
    testlabels=np.array(test._datasets[1])
    our_net.cleargrads()
    loss_ , acc_ =our_classifier(traindata,testlabels)  
    result[counter2]=loss_.data
    accData[counter2]=acc_.data
    loss_.backward()
    optimizer.update()
    counter2=counter2+1
    
"""    
    tmpdata=RandomIterator(train._datasets)
    counter = -1
    for mainloop in range(0,31):
        counter=counter+1
        ind_currdata=range(mainloop*tmpdata.batch_size,mainloop*tmpdata.batch_size+tmpdata.batch_size)
        currdata=tmpdata.data[0][ind_currdata,:]
        currlabels=tmpdata.data[1][ind_currdata]
        our_net.cleargrads()
        loss_ , acc_ =our_classifier(currdata,currlabels)
        result[counter]=loss_.data
        loss_.backward()
        optimizer.update()
        
        lossData[counter2]=loss_.data
        accData[counter2]=acc_.data
        counter2=counter2+1
        
    result2[epochs]=np.mean(result)


plt.plot(result2[:])
"""
plt.plot(result)

finaldata=np.array(test._datasets[0])
finallabels=np.array(test._datasets[1])
loss_ , acc_final =our_classifier(finaldata,finallabels)
