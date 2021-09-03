
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import Dense, Activation



# ___________ GUI ___________
import tkinter as tk


fields = 'Loss', 'Optimizer ',' first_func_epoches', 'sec_func_epoches','third_func_epoches'
attributes =[]
def fetch(entries):
    for entry in entries:
        fields = entry[0]
        text  = entry[1].get()
        attributes.append(text)
    P1()

def makeform(root, fields):
    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=15, text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries

def P1():
            Loss=attributes[0]
            Optimizer = attributes[1]

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


            # ___________ FIRST ___________
            rands = [random.uniform(-100, 100) for i in range(100)]
            out_rands = [2. * x + 21 for x in rands]
            in1 = np.array(rands, dtype=float)
            out1 = np.array(out_rands, dtype=float)

            # Create the model

            ly = tf.keras.layers.Dense(units=1, input_shape=[1])
            model = tf.keras.Sequential([ly])
            # model = tf.keras.Sequential()
            # model.add(Dense(1000, input_dim=1, activation="relu"))
            # model.add(Dense(500, activation="relu"))
            # model.add(Dense(100, activation="relu"))
            # model.add(Dense(1))

            # Compile the model
            model.compile(loss=Loss, optimizer=tf.keras.optimizers.Adam(0.1))

            # Fit data to model
            traina = model.fit(in1, out1, epochs=int(attributes[2]), verbose=False)

            # Generate predictions
            x_predict = np.array(range(-200, 200))
            y_predict = model.predict(x_predict)

            # Draw Plot
            draw_plot(traina.history['loss'], in1, out1, x_predict, y_predict)
            print("These are the layer variables: {}".format(model.get_weights()))

            # ___________ SECOND ___________
            rands = sorted([random.uniform(-100, 100) for i in range(100)])
            out_rands = [64.3 * x - 112 for x in rands]
            in2 = np.array(rands, dtype=float)
            out2 = np.array(out_rands, dtype=float)

            ly = tf.keras.layers.Dense(units=1, input_shape=[1])
            model = tf.keras.Sequential([ly])
            model.compile(loss=Loss, optimizer=tf.keras.optimizers.Adam(0.1))
            traina = model.fit(in2, out2, epochs=int(attributes[3]), verbose=False)

            x_predict = np.array(range(-100, 100))
            y_predict = model.predict(x_predict)
            draw_plot(traina.history['loss'], in2, out2, x_predict, y_predict)
            print("These are the layer variables: {}".format(ly.get_weights()))

            # ___________ THIRD ___________


            def train_function(x):
                return np.sin(x)


            x_train = np.arange(-40 * np.pi, 40 * np.pi, 0.1)
            y_train = train_function(x_train)
            model = tf.keras.Sequential([
                Dense(40, input_shape=(1,)),
                Activation('relu'),
                Dense(12),
                Activation('relu'),
                Dense(1)
            ])
            # model.compile(optimizer='adam', loss='mse')

            model.compile(optimizer=attributes[1], loss=Loss)
            traina = model.fit(x_train, y_train, epochs=int(attributes[4]), verbose=False)

            x_predict = np.arange(0, 5 * np.pi, 0.1)
            y_predict = model.predict(x_predict)
            draw_plot(traina.history['loss'], x_train[:300], y_train[:300], x_predict, y_predict)
            print("These are the layer variables: {}".format(model.get_weights()))


if __name__ == '__main__':
    root = tk.Tk()
    root.title("part5_learning mnist")
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))
    b1 = tk.Button(root, text='Show',
                  command=(lambda e=ents: fetch(e)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root, text='Quit', command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()



