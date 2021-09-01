from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.keras.layers import Dense, Activation
from mpl_toolkits import mplot3d



# ___________ FIRST ___________
def func1(x , y):
    return 24*np.sin(x) - 5*y
out_rands =[]
x_rand = sorted([random.uniform(-100, 100) for i in range(200)])
y_rand = sorted([random.uniform(-100, 100) for i in range(200)])
x_tr = []

input1 = np.array(x_rand, dtype=float)
input2 = np.array(y_rand, dtype=float)
for i in range(len(x_rand)):
    out_rands.append(func1(x_rand[i] , y_rand[i]))
    x_tr.append([x_rand[i] , y_rand[i]])
x_tr = np.array(x_tr, dtype=float)
out2 = np.array(out_rands, dtype=float)


## create model
model = tf.keras.Sequential([
                Dense(40, input_dim=2),
                Activation('relu'),
                Dense(12),
                Activation('relu'),
                Dense(1)
            ])
model.compile(optimizer='adam', loss='mse')

traina = model.fit( x_tr, out2, epochs=1000, verbose=True)

x_rand_pred =sorted([random.uniform(300, 500) for i in range(200)])
y_rand_pred = sorted([random.uniform(300, 500) for i in range(200)])
x_test = []
z_test =[]
for i in range(len(input1)):
    x_test.append([x_rand_pred[i] ,y_rand_pred[i]])
    z_test.append(func1(x_rand_pred[i] , y_rand_pred[i]))
x_test = np.array(x_test, dtype=float)
z_test = np.array(z_test, dtype=float)

z_predict = model.predict(x_test)

print(z_predict)
ax = plt.axes(projection='3d')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('OUT')
ax.scatter3D(x_rand_pred, y_rand_pred,z_test)
ax.scatter3D(x_rand_pred, y_rand_pred,z_predict)
plt.show()



# ___________ SECOND ___________
def func2(x , y):
    return 12*x  - 5*y
out_rands =[]
# x_rands = sorted([[random.uniform(-100, 100) for i in range(100)] for j in range(100)])
x_rand = [random.uniform(-100, 100) for i in range(500)]
y_rand = [random.uniform(-100, 100) for i in range(500)]

input1 = np.array(x_rand, dtype=float)
input2 = np.array(y_rand, dtype=float)
for i in range(len(x_rand)):
    out_rands.append(func2(x_rand[i] , y_rand[i]))

out2 = np.array(out_rands, dtype=float)

model = tf.keras.Sequential()
model.add(Dense(500, input_dim=2, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
x_tr = []

for i in range(len(input1)):
    x_tr.append([x_rand[i] ,y_rand[i]])
x_tr = np.array(x_tr, dtype=float)

traina = model.fit( x_tr, out2, epochs=2000, verbose=True)

x_rand_pred = [random.uniform(300, 400) for i in range(500)]
y_rand_pred = [random.uniform(300, 400) for i in range(500)]
x_test = []
z_test =[]
for i in range(len(input1)):
    x_test.append([x_rand_pred[i] ,y_rand_pred[i]])
    z_test.append(func2(x_rand_pred[i] , y_rand_pred[i]))
x_test = np.array(x_test, dtype=float)
z_predict = model.predict(x_test)

print(z_predict)
ax = plt.axes(projection='3d')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('OUT')
ax.scatter3D(x_rand_pred, y_rand_pred,z_test)
ax.scatter3D(x_rand_pred, y_rand_pred,z_predict)
plt.show()
# # # ___________ THIRD ___________
# #

def func3(x , y):
    return x*x  + 12.5*y
out_rands =[]
# x_rands = sorted([[random.uniform(-100, 100) for i in range(100)] for j in range(100)])
x_rand = [random.uniform(-100, 100) for i in range(700)]
y_rand = [random.uniform(-100, 100) for i in range(700)]

input1 = np.array(x_rand, dtype=float)
input2 = np.array(y_rand, dtype=float)
for i in range(len(x_rand)):
    out_rands.append(func3(x_rand[i] , y_rand[i]))

out2 = np.array(out_rands, dtype=float)

model = tf.keras.Sequential()
model.add(Dense(500, input_dim=2, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
x_tr = []

for i in range(len(input1)):
    x_tr.append([x_rand[i] ,y_rand[i]])
x_tr = np.array(x_tr, dtype=float)

traina = model.fit( x_tr, out2, epochs=2000, verbose=True)

x_rand_pred = [random.uniform(300, 400) for i in range(700)]
y_rand_pred = [random.uniform(300, 400) for i in range(700)]
x_test = []
z_test =[]
for i in range(len(input1)):
    x_test.append([x_rand_pred[i] ,y_rand_pred[i]])
    z_test.append(func3(x_rand_pred[i] , y_rand_pred[i]))
x_test = np.array(x_test, dtype=float)
z_predict = model.predict(x_test)

print(z_predict)
ax = plt.axes(projection='3d')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('OUT')
ax.scatter3D(x_rand_pred, y_rand_pred,z_test)
ax.scatter3D(x_rand_pred, y_rand_pred,z_predict)
plt.show()
