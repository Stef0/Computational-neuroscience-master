from __future__ import print_function
import numpy as np
import six
import glob
import gc
import midi_manipulation
from tqdm import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.vae import gaussian_kl_divergence
from chainer import serializers

import matplotlib.pyplot as plt


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



class VAE(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent = 20):
        super(VAE, self).__init__()
        with self.init_scope():
            self.n_h1 = 1000
            self.n_h2 = 500
            self.n_h3 = 100
            # encoder
            self.le1 = L.Linear(n_in, self.n_h1)
            self.le2 = L.Linear(self.n_h1, self.n_h2)
            self.le3 = L.Linear(self.n_h2, self.n_h3)
            self.le_mu = L.Linear(self.n_h3, n_latent)
            self.le_ln_var = L.Linear(self.n_h3, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, self.n_h3)
            self.ld2 = L.Linear(self.n_h3, self.n_h2)
            self.ld3 = L.Linear(self.n_h2, n_in)

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.relu(self.le1(x))
        h1 = F.relu(self.le2(h1))
        h1 = F.relu(self.le3(h1))
        mu = self.le_mu(h1)
        ln_var = self.le_ln_var(h1)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.relu(self.ld1(z))
        h1 = F.relu(self.ld2(h1))
        h2 = self.ld3(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
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


genres = {'ebmajor', 'cminor'}
songs = []

samplespersong = 200
timesteps = 100
freqs = 156
batchsize=64
n_epochs= 100

# Prepare the dataset
# the midi files have to be in a folder 'data/*', where * is the genre (or class) name
# where data in is the same folder of the script
for genre in genres:
    songsgenre = []
    files = glob.glob('data/' + genre + '/*.mid')
    for f in tqdm(files):
        song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
        #song = np.reshape(song, [1, -1])
        songsgenre.append(song)
    songs.append(songsgenre)


genre=0
n_traingenre = int(samplespersong * len(songs[genre]) * 4 / 5)
n_testgenre = int(samplespersong * len(songs[genre]) * 1 / 5)
#train = np.zeros((len(genres) * n_traingenre, args.freqs * args.timesteps))
#test = np.zeros((len(genres) * n_testgenre, args.freqs * args.timesteps))
train = np.zeros((len(genres) * n_traingenre, 1, timesteps, freqs))
test = np.zeros((len(genres) * n_testgenre, 1, timesteps, freqs))
for genre in range(len(genres)):
    #DATAgenre = np.zeros((args.samplespersong * len(songs[genre]), args.freqs * args.timesteps))
    DATAgenre = np.zeros((samplespersong * len(songs[genre]), timesteps, freqs))
    for song in range(len(songs[genre])):
        gc.collect()
        #overlap = (songs[genre][song].size - args.freqs * args.timesteps) / args.samplespersong
        overlap = (songs[genre][song].shape[0] - timesteps) / samplespersong
        #app = [songs[genre][song][0][i * overlap : i * overlap + args.timesteps * args.freqs] for i in range(args.samplespersong)]
        app = [songs[genre][song][i * overlap : i * overlap + timesteps] for i in range(samplespersong)]
        app = np.array(app)
        #DATAgenre[song * args.samplespersong : (song + 1) * args.samplespersong, :] = app #[random.sample(range(len(app)), min - n_freqs * n_timesteps)]
        DATAgenre[song * samplespersong : (song + 1) * samplespersong, :, :] = app #[random.sample(range(len(app)), min - n_freqs * n_timesteps)]
    DATAgenre = np.random.permutation(DATAgenre) # shuffle along first index
    train[genre * n_traingenre : (genre + 1) * n_traingenre, 0, :, :] = DATAgenre[0 : n_traingenre, :, :]
    test[genre * n_testgenre : (genre + 1) * n_testgenre, 0, :, :] = DATAgenre[n_traingenre : n_traingenre + n_testgenre, :, :]



# initializing the variables to store the loss over epochs
loss_train = np.zeros((n_epochs), dtype = float)
loss_test = np.zeros((n_epochs), dtype = float)
rec_loss_train = np.zeros((n_epochs), dtype = float)
rec_loss_test = np.zeros((n_epochs), dtype = float)

# Creating the model
model = VAE(timesteps * freqs)

# creating the classifier
classifier = model.get_loss_func()

# creating the optimizer
optimizer = chainer.optimizers.Adam(alpha=0.00001)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))


for epoch in six.moves.range(55, 60):

    DATA = RandomIterator(train.astype('float32'))
    for current_batch in DATA:

        # Clearing the gradients
        model.cleargrads()

        # Computing the loss
        loss_ = classifier(current_batch[0].reshape(-1, timesteps * freqs).astype('float32'))
        #loss_ = classifier(current_batch[0])

        loss_[1].backward()

        optimizer.update()

        loss_train[epoch] += loss_[1].data 
        rec_loss_train[epoch] += loss_[0].data 
        

    loss_ = classifier(test.reshape(-1, timesteps * freqs).astype('float32'))
    #loss_ = classifier(test.astype('float32'))
    
    # normalizing the train loss
    rec_loss_train[epoch] = rec_loss_train[epoch] / DATA.idx
    loss_train[epoch] = loss_train[epoch] / DATA.idx
    rec_loss_test[epoch] = loss_[0].data
    loss_test[epoch] = loss_[1].data

    # Printing the accuracy for each epoch
    print('In epoch ' + str(epoch + 1) + ' - train loss: ' + str(loss_train[epoch]) + ', test loss: ' + str(loss_test[epoch]))


  
# plotting the loss
fig = plt.figure(figsize = (15, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_train[11:60], label='train loss',  c='xkcd:tomato')
plt.plot(loss_test[11:60], label='test loss', c='xkcd:crimson')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(prop={'size': 16})
plt.subplot(1, 2, 2)
plt.plot(loss_train[11:60] - rec_loss_train[11:60], label='train KL divergence', c='xkcd:tomato')
plt.plot(loss_test[11:60] - rec_loss_test[11:60], label='test KL divergence', c='xkcd:crimson')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(prop={'size': 16})


# generate song
# from test sample:
sample = 5
x = model(test[5, :, :, :].reshape([1, timesteps*freqs]).astype('float32'))
# or from randomly sampled z
n_examples = 1
z = chainer.Variable(
    np.random.normal(0, 1, (n_examples, 20)).astype(np.float32))
# z=np.float32(np.reshape(song[160:260, :], [1, 15600]))
x = model.decode(z)
# convert to midi
generated_song = np.reshape(x.data, [timesteps, freqs])
generated_song = (generated_song>0.2)*1
#save generated midi
midi_manipulation.noteStateMatrixToMidi(generated_song)

