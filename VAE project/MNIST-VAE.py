from __future__ import print_function
import numpy as np
from scipy.stats import norm
import six
import chainer
from chainer import serializers
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.vae import gaussian_kl_divergence
from chainer.datasets import TupleDataset
import matplotlib.pyplot as plt

# iterator for the training
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
            return self.data[self._order[i:(i + self.batch_size)]],self._order
        else:
            return list(self.data[self._order[i:(i + self.batch_size)]]),self._order



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




class VAE(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent = 20):
        super(VAE, self).__init__()
        with self.init_scope():
            
            self.n_h = 500
            
            # encoder
            self.enc_l1 = L.Linear(n_in, self.n_h)
            self.enc_b1 = L.BatchNormalization(self.n_h)
            self.enc_l2_mu = L.Linear(self.n_h, n_latent)
            self.enc_b2_mu = L.BatchNormalization(n_latent)
            self.enc_l2_ln_var = L.Linear(self.n_h, n_latent)
            self.enc_b2_ln_var = L.BatchNormalization(n_latent)
            
            # decoder
            self.dec_l1 = L.Linear(n_latent, self.n_h)
            self.dec_b1 = L.BatchNormalization(self.n_h)
            self.dec_l2 = L.Linear(self.n_h, n_in)
            self.dec_b2 = L.BatchNormalization(n_in)

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        #h1 = self.enc_b1(F.tanh(self.enc_l1(x)))
        h1 = F.tanh(self.enc_l1(x))
        mu = self.enc_l2_mu(h1)
        ln_var = self.enc_l2_ln_var(h1)
        #return self.enc_b2_mu(mu), self.enc_b2_ln_var(ln_var)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        #h1 = self.dec_b1(F.tanh(self.dec_l1(z)))
        h1 = F.tanh(self.dec_l1(z))
        h2 = self.dec_l2(h1)
        if sigmoid:
            #return F.softmax(self.dec_b2(h2))
            return F.sigmoid(h2)
        else:
            #return self.dec_b2(h2)
            return h2

    def get_loss_func(self):
        
        def lf(x):
            # getting the mu and ln_var of the prior of z with the encoder
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)
            # creating the latent variable z by sampling from the encoder output
            z = F.gaussian(mu, ln_var)
            # computing the reconstruction loss
            self.rec_loss = F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / batchsize
            #self.rec_loss = F.sigmoid_cross_entropy(x, self.decode(z, sigmoid=False))
            # computing the KL divergence
            self.KL_loss = gaussian_kl_divergence(mu, ln_var) / batchsize
            # computing the total loss
            self.loss = self.rec_loss + self.KL_loss
            # returning the losses separately
            return [self.rec_loss, self.loss]
        return lf




# defining the parameters
n_epochs = 150
input_size = 28

# getting the data
train_set, test_set = get_mnist(n_train = 1600, n_test = 400, n_dim=1)

# creating the model
model = VAE(input_size * input_size, n_latent=2)

# defining the classifier
classifier = model.get_loss_func()

# creating the vectors to store the training and test loss over epochs
loss_train = np.zeros((n_epochs*4), dtype = float)
loss_test = np.zeros((n_epochs*4), dtype = float)
rec_loss_train = np.zeros((n_epochs*4), dtype = float)
rec_loss_test = np.zeros((n_epochs*4), dtype = float)

# creating template for displaying a 2D manifold of the digits
n = 15  # figure with 15x15 digits
figure = np.zeros((input_size * n, input_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


# creating the optimizer
optimizer = chainer.optimizers.Adam(0.00005) #predef adam alpha=0.001 rms lr=0.01
optimizer.setup(model)
# setting weight decay regularization
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

# looping over epochs
for epoch in six.moves.range(50, 60):
    # defining the iterator for the training set
    DATA = RandomIterator(train_set)
    
    # looping over batches
    for current_batch in DATA:

        # clearing the gradients
        model.cleargrads()

        # computing the reconstruction training loss and the total training loss
        loss_ = classifier(current_batch[0][0])
        
        # doing the backward step
        loss_[1].backward()

        # updating the optimizer
        optimizer.update()
        
        # saving the reconstruction train loss and the total train loss
        rec_loss_train[epoch] += loss_[0].data
        loss_train[epoch] += loss_[1].data
        
    # computing the reconstruction test loss and the total test loss
    loss_ = classifier(test_set._datasets[0])
    
    # normalizing the train loss
    rec_loss_train[epoch] = rec_loss_train[epoch] / DATA.idx
    loss_train[epoch] = loss_train[epoch] / DATA.idx
    # saving the reconstruction test loss and the total test loss
    rec_loss_test[epoch] = loss_[0].data
    loss_test[epoch] = loss_[1].data

    # Printing the training and test total loss for each epoch
    print('In epoch ' + str(epoch + 1) + ' - train loss: ' + str(loss_train[epoch]) + ', test loss: ' + str(loss_test[epoch]))
    
    
    # creating latent variables z from the input test images using the encoder
    x = np.reshape(np.array(test_set._datasets[0]),[-1, 1, input_size, input_size])
    mu, var = model.encode(x)
    z = F.gaussian(mu, var)
    
    # if n_latent = 2
    # displaying a scatterplot of the latent variables corresponding to the test images
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(np.array(z.data[:, 0]), np.array(z.data[:, 1]), c = np.array(test_set._datasets[1]))
    plt.xlim(-7.5,7.5)
    plt.ylim(-7.5,7.5)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.colorbar()
    plt.text(2.5, 8, 'epoch ' + str(epoch + 1), fontsize=16)
    plt.show()
    
    # if n_latent = 2
    # producing latent variables z in (grid_x)x(grid_y) by applying the inverse of
    # a Gaussian (p(z) is assumed to be Gaussian) to linearly spaced values
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            # sampling an image from the latent variable using the decoder
            x_decoded = model.decode(z_sample.astype('float32'))
            # reshaping the image and saving it 
            digit = x_decoded._data[0].reshape(input_size, input_size)
            figure[i * input_size: (i + 1) * input_size, j * input_size: (j + 1) * input_size] = digit

    fig = plt.figure(figsize = (10, 10))
    plt.imshow(figure, cmap = 'Greys_r')
    plt.show()
    
    # if n_latent != 2
    # displaying a 2D manifold of the digits
    # creating a figure with 2x5 digits
    n1 = 2 
    n2 = 5
    figure = np.zeros((input_size * n1, input_size * n2))
    for digit in six.moves.range(10):
      # using the latent variables samples generated by the encoder
      x_decoded = model.decode(np.array(z.data[digit * 1600 : digit * 1600 + 1, :]))
      img = x_decoded.reshape(input_size, input_size)
      figure[(digit % 2) * input_size: ((digit % 2) + 1) * input_size, (digit % 5) * input_size: ((digit % 5) + 1) * input_size] = img.data

    plt.figure(figsize = (10, 10))
    plt.imshow(figure, cmap = 'Greys_r')
    plt.show()


# saving the model
#serializers.save_hdf5('', model)
