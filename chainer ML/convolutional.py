# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:10:17 2017

@author: Stefano
"""
'''Create a network consisting of a convolutional layer, a max pooling layer and one
 fully connected layer. For the convolutional layers, use 5 output channels, a kernel
 size of 5, stride of 1 and padding of 0.
'''

import numpy as np
import math
import matplotlib.pyplot as plt
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import TupleDataset

# for the classifier
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter




class LeNet5(Chain):
    def __init__(self):
        super(LeNet5, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=1, out_channels=5, ksize=5, stride=1)
            self.fc4 = L.Linear(None, 10)
            self.fc5 = L.Linear(10, 10)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.fc4(h))
        if chainer.config.train:
            return self.fc5(h)
        return F.softmax(self.fc5(h))


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



# Custom iterator
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

# Classifier
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
        return self.loss

# Defining the parameters for the model
n_train = 500
n_test = 500
n_units = 10 # number of hidden units
n_out = 10 # number of classes
n_data = n_train * n_out # total data
n_epochs = 20# number of epochs
batch_size = 32# size of one batch
n_batches = int(math.ceil(n_data / float(batch_size)))

train_set, test_set = get_mnist()

# Creating the model
ConvNeuralNet = LeNet5()

# Creating the optimizer
optimizer = optimizers.SGD() # we use the Stochastic Gradient Descent (SGD)
optimizer.setup(ConvNeuralNet)

# setting weight decay regularization
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0004))

# Creating the classifier
classifier = Classifier(ConvNeuralNet)

# Arrays to store the training and test loss for each epoch for the plotting
loss_train = np.zeros((n_epochs, 1), dtype = float)
loss_test = np.zeros((n_epochs, 1), dtype = float)
acc = np.zeros((n_epochs, 1), dtype = float)


for epoch in range(n_epochs):
    DATA=RandomIterator(train_set)
    for current_batch in DATA:
        
        # Clearing the gradients
        ConvNeuralNet.cleargrads()
        
        # Computing the loss
        a = np.reshape(current_batch[0][0],[32,1,28,28])
        #loss_ = F.softmax_cross_entropy(ConvNeuralNet(a), current_batch[0][1])
        loss_ = classifier(a, current_batch[0][1])

        loss_.backward()
        
        
        
        optimizer.update()

        loss_train[epoch] += loss_.data * current_batch[0][1].shape[0] # Multiply for the length of the current batch to
                                                                    # get the right mean                                                          

    loss_train[epoch] = float(loss_train[epoch]) / n_data # Divide for total number of data to get the mean
    
    DATA=RandomIterator(test_set)
 
    # Computing the loss and accuracy in the test set
    for current_batch in DATA:

        # Computing the loss
        a = np.reshape(current_batch[0][0],[32,1,28,28])
        #loss_ = F.softmax_cross_entropy(ConvNeuralNet(a), current_batch[0][1])

        loss_ = classifier(a, current_batch[0][1])
        loss_test[epoch] += loss_.data * current_batch[0][1].shape[0]

        # Computing the accuracy
        #acc[epoch]=chainer.functions.accuracy(,)
        acc[epoch] += classifier.accuracy.data * current_batch[0][1].shape[0]

    loss_test[epoch] = float(loss_test[epoch]) / current_batch[1].size # Divide for total number of data to get the mean
    acc[epoch] = float(acc[epoch]) / current_batch[1].size # Divide for total number of data to get the mean
    acc[epoch] = classifier.accuracy.data
    
    #Printing the accuracy for each epoch
    print('The accuracy in epoch ' + str(epoch) + ' is ' + str(acc[epoch]))

# Plotting the train and test loss over epochs


'''

plt.plot(np.arange(0, n_epochs), loss_train[:], 'b')
plt.plot(loss_test[:], 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Decrease in training and test loss over epochs')
plt.legend(['train', 'test'])
'''
plt.figure(1)

plt.subplot(211) # Plotting the train and test loss over epochs
plt.plot(np.arange(0, n_epochs), loss_train[:], 'b')
plt.plot(loss_test[:], 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Decrease in training and test loss over epochs')
plt.legend(['train', 'test'])

plt.subplot(212) # Plotting the accuracy over epochs
plt.plot(acc[:], 'r')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
'''
finaldata=np.array(test_set._datasets[0])
finallabels=np.array(test_set._datasets[1])
loss_ =classifier(finaldata,finallabels)    
classifier.accuracy.data    
'''    
    
