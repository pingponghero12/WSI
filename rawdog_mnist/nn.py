import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# train_images.shape = (60000, 28, 28)

# ON_ma.shape = (784, 60000)

train_images = train_images.reshape(60000, 784).T.astype('float32') / 255

def init_parameters():
    W1 = np.random.randn(32, 784)
    b1 = np.zeros((32, 1))

    W2 = np.random.randn(10, 32)
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2

def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    return x>0

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

def forward_propagation(W1, b1, W2, b2, X):
    L1u = W1.dot(X)+b1
    #print(W1.shape, " ", X.shape, " ", b1.shape)
    # (32, 784)   (784, 2000)   (32, 1)
    

    # (32x784) @ (6000, 28, 28)
    # (32x784) @ (784, 60000)

    L1 = ReLU(L1u)

    L2u = W2.dot(L1)+b2
    L2 = softmax(L2u)

    return L1u, L1, L2u, L2

def back_propagation(W1, b1, W2, b2, L1, L2, X, Y):
    m = Y.size

    CL2 = L2 - Y
    dW2 = 1/m * CL2.dot(L1.T)
    db2 = 1/m * np.sum(CL2, axis=1, keepdims=True)

    CL1 = W2.T.dot(CL2) * dReLU(L1)
    dW1 = 1/m * CL1.dot(X.T)
    db1 = 1/m * np.sum(CL1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1

    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2

    return W1, b1, W2, b2

def init_y(train_labels):
    y = np.zeros(shape=(10, train_labels.size), dtype='float32')
    for i in range(train_labels.size):
        y[train_labels[i]][i] = 1
    return y

def get_predictions(L2):
    return np.argmax(L2, 0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

def gradient_descent(X, Y, itr, alpha):
    W1, b1, W2, b2 = init_parameters()

    for i in range(itr):
        L1u, L1, L2u, L2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_propagation(W1, b1, W2, b2, L1, L2, X, Y)
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i%50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(L2), np.argmax(Y, 0)))

    return W1, b1, W2, b2

print(train_images[:, :1000].shape)

W1, b1, W2, b2 = gradient_descent(train_images[:, :2000], init_y(train_labels[:2000]), 2000, 0.35)
