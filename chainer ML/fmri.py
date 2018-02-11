import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as mat
from scipy.stats.stats import pearsonr   


# constants
lambda_const = 10e-06
sigma_const = 10e-03 * np.identity(3092)

data = scipy.io.loadmat('69dataset.mat')

Xim = data['X']
Yim = data['Y']
Xim_prior = data['prior']
Xim = np.array(Xim, dtype=float)
Xim_prior = np.array(Xim_prior, dtype=float)

X = (Xim - np.mean(Xim,0)) / np.std(Xim,0)
Y = (Yim - np.mean(Yim,0)) / np.std(Yim,0)
X_prior = (Xim_prior - np.mean(Xim_prior,0)) / np.std(Xim_prior,0)


X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0
X_prior[np.isnan(X_prior)] = 0

X_train = np.concatenate([X[0:40], X[50:90]])
X_test = np.concatenate([X[40:50], X[90:100]])
Y_train = np.concatenate([Y[0:40], Y[50:90]])
Y_test = np.concatenate([Y[40:50], Y[90:100]])

# point 1)

B = np.dot(np.linalg.inv(np.dot(Y_train.transpose(), Y_train) + lambda_const * np.identity(3092)), np.dot(Y_train.transpose(), X_train))
X_pred = [np.dot(B.transpose(),Y_test[i]) for i in range(0,20)]

X_pred = X_pred * np.std(Xim,0) + np.mean(Xim,0)

# Code to plot the predictd test image i of a six if i < 50 and of a nine if i >=50



for i in range(0,20):
    image_i = np.array(X_test[i])
    image_i = image_i.reshape((28, 28)) # Reshape the array into 28 x 28 array (2-dimensional array)
    plt.title('Real Image ' + str(i))
    plt.imshow(image_i.transpose())
    plt.show()

# Code to plot image i of a six if i < 50 and of a nine if i >=50

for i in range(0,20):
    image_i = np.array(X_pred[i])
    image_i = image_i.reshape((28, 28)) # Reshape the array into 28 x 28 array (2-dimensional array)
    plt.title('Predicted Image ' + str(i))
    plt.imshow(image_i.transpose())
    plt.show()

# point 2)

Bltest=np.linalg.inv(X_train.transpose().dot(X_train) + lambda_const+np.identity(784)).dot(X_train.transpose()).dot(Y_train)

Bl = np.dot(np.linalg.inv(np.dot(X_train.transpose(), X_train) + lambda_const * np.identity(784)), np.dot(X_train.transpose(), Y_train))
n = X_prior.shape[0]

sigma_prior = X_prior.transpose().dot(X_prior) / float(n - 1) + lambda_const * np.identity(784)


# Code to plot sigma_prior

plt.title('Sigma_prior')
plt.imshow(sigma_prior,)
plt.show()

# point 3)

muy = np.linalg.inv(np.linalg.inv(sigma_prior) + Bl.dot(np.linalg.inv(sigma_const)).dot(Bl.transpose())).dot(Bl).dot(np.linalg.inv(sigma_const))

X_p = [muy.dot(Y_test[i]) for i in range(0,20)]

X_p= X_p * np.std(Xim,0) + np.mean(Xim,0)

mu_post = np.dot(np.linalg.inv(np.linalg.inv(sigma_prior) + np.dot(Bl, np.dot(np.linalg.inv(sigma_const), Bl.transpose()))), np.dot(Bl, np.dot(np.linalg.inv(sigma_const), Y_test.transpose())))
X_pred2 = mu_post.transpose()
X_pred2 = X_pred2 * np.std(Xim,0) + np.mean(Xim,0)

# Code to plot X_pred2
for i in range(0,20):
    plt.title('X_pred'+ str(i))
    plt.imshow(X_pred2[i].reshape((28, 28)).transpose())
    plt.show()


X_test = X_test * np.std(Xim,0) + np.mean(Xim,0)

corr_bayes = np.array([pearsonr(np.array(X_p[i]),np.array(X_test[i])) for i in range(0,20)])[:,0]
corr_pred1 = np.array([pearsonr(np.array(X_pred[i]),np.array(X_test[i])) for i in range(0,20)])[:,0]
[corr_bayes[i]> corr_pred1[i] for i in range(0,20)]

np.sum(corr_pred1)
np.sum(corr_bayes)

