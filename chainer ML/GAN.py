import matplotlib.pyplot as plt
import numpy as np
import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L



def get_mnist(n_train=1000, n_test=100, n_dim=1, with_label=False, classes = [0]):
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
                train_data = chainer.datasets.TupleDataset(data[idx],L)
            else:
                test_data = chainer.datasets.TupleDataset(data[idx],L)

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

class discriminator(chainer.Chain):
    def __init__(self):
        super(discriminator, self).__init__()
        with self.init_scope():
            self.l1 = L.Convolution2D(784, 784, 1, pad=1)
            self.l2 = L.Linear(3 * 3 * 784, 1)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        return self.l2(h)

class generator(chainer.Chain):
    def __init__(self,):
        super(generator, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(100, 3*3*784)
            self.l2 = L.BatchNormalization(3 * 3 * 784)
            self.l3 = L.Deconvolution2D(784, 784, 1,pad=1)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = self.l2(h)
        h = F.reshape(h,(x.data.shape[0], 784, 3, 3))
        h = F.sigmoid(self.l3(h))
        return h


train_data, test_data = get_mnist(n_train=1000,n_test=100,with_label=False,classes= [0])

num_train_it = 2000

batchsize = 10

dis = discriminator()
gen = generator()

optimizer4gen = optimizers.MomentumSGD()
optimizer4gen.setup(gen)

optimizer4dis = optimizers.MomentumSGD()
optimizer4dis.setup(dis)

losscoll_dis = []
losscoll_gen = []


for trainit in range(num_train_it):
    
    z = chainer.Variable(np.random.uniform(-1, 1, (batchsize, 100)).astype('float32'))

    gendata = gen(z)

    class_of_fake = dis(gendata)

    Loss_gen = F.sigmoid_cross_entropy(class_of_fake, chainer.Variable(np.ones(batchsize, dtype=np.int32).reshape(10,1)))
    
    origdata = train_data[np.random.randint(0,999,10L)].reshape(batchsize,784,1,1)

    class_of_true = dis(chainer.Variable(origdata))

    Loss_dis = F.sigmoid_cross_entropy(class_of_fake, chainer.Variable(np.zeros(batchsize, dtype=np.int32).reshape(10,1))) + F.sigmoid_cross_entropy(class_of_true, chainer.Variable(np.ones(batchsize, dtype=np.int32).reshape(10,1)))

    gen.cleargrads()
    Loss_gen.backward()
    optimizer4gen.update()

    dis.cleargrads()
    Loss_dis.backward()
    optimizer4dis.update()

    losscoll_gen.append(Loss_gen.data)
    losscoll_dis.append(Loss_dis.data)
    
    print('iter ' + str(trainit) + '  : GEN = ' + str(Loss_gen.data) + ' DIS = ' + str(Loss_dis.data))




plt.figure()     
plt.plot(losscoll_dis)
plt.figure() 
plt.plot(losscoll_gen)  


z = chainer.Variable(np.random.uniform(0, 1, (len(test_data), 100)).astype('float32'))
imm = gen(z)
plt.figure()
plt.imshow(imm.data[1].reshape(28,28))

plt.figure()
plt.imshow(imm.data[5].reshape(28,28))

plt.figure()
plt.imshow(imm.data[9].reshape(28,28))


#the discriminator in generated image
z = chainer.Variable(np.random.uniform(0, 1, (1, 100)).astype('float32'))
dis(gen(z))

#the descriminator in true image
origdata = train_data[np.random.randint(0,999,1L)].reshape(1,784,1,1)
dis(chainer.Variable(origdata))
 





