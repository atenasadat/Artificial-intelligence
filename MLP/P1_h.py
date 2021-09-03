from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM


def draw_plot(loss, x_train, y_train, x_predict, y_predict):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(loss)
    ax1.set_title('loss')
    ax2.plot(x_train, y_train)
    ax2.set_title('main function')
    ax3.plot(x_predict, y_predict)
    ax3.set_title('predicted function')
    ax1.set(xlabel='Epoch Number', ylabel="Loss Magnitude")
    plt.show()


def train_function(x):
    return np.sin(x)


x_train = np.arange(-10*np.pi, 10*np.pi, 0.1)
y_train = train_function(x_train)
model = tf.keras.Sequential()
model.add(LSTM(10, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer='adam', loss='mse')


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# define input sequence
xaxis = np.arange(-50 * np.pi, 50 * np.pi, 0.1)
train_seq = train_function(xaxis)
n_steps = 20
X, y = split_sequence(train_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print("X.shape = {}".format(X.shape))
print("y.shape = {}".format(y.shape))

history = model.fit(X, y, epochs=20, verbose=1)


test_xaxis = np.arange(0, 10*np.pi, 0.1)

calc_y = train_function(test_xaxis)
# start with initial n values, rest will be predicted
test_y = calc_y[:n_steps]
results = []
for i in range( len(test_xaxis) - n_steps ):
    net_input = test_y[i : i + n_steps]
    net_input = net_input.reshape((1, n_steps, n_features))
    y = model.predict(net_input, verbose=0)
    test_y = np.append(test_y, y)

draw_plot(history.history['loss'], test_xaxis, calc_y, test_xaxis[n_steps:], test_y[n_steps:])
