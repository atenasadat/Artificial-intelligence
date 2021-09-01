import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np




# ___________ GUI ___________
import tkinter as tk


fields = 'epochsnumber', 'batch_size ',' validation_split', 'test_images_no','max_norm_value','noise_factor'
attributes =[]
def fetch(entries):
    for entry in entries:
        fields = entry[0]
        text  = entry[1].get()
        attributes.append(text)
    P6()

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




def P6():

            # configuration
            img_width, img_height = 28, 28
            batch_size = int(attributes[1])#150
            epochsnumber = int(attributes[0])#1
            validation_split = float(attributes[2])#0.2
            verbosity = 1
            max_norm_value = float(attributes[4])#2.0
            factor = attributes[5]#1
            number_of_visualizations = int(attributes[3])#10

            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            if K.image_data_format() == 'channels_first':
                x_train = x_train.reshape(x_train.shape[0], 1, img_width, img_height)
                x_test = x_test.reshape(x_test.shape[0], 1, img_width, img_height)
                input_shape = (1, img_width, img_height)
            else:
                x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
                x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)
                input_shape = (img_width, img_height, 1)


            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')

            ############ Normalize data between 0 and 1

            x_train = x_train / 255
            x_test = x_test / 255

            ############ Adding noise to images
            original = x_train
            original_test = x_test
            noise = np.random.normal(0, 1, original.shape)
            noise_test = np.random.normal(0, 1, original_test.shape)
            noisy_input = original + factor * noise
            noisy_input_test = original_test + factor * noise_test


            model = Sequential()
            model.add(Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
            model.add(Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
            model.add(Conv2DTranspose(32, kernel_size=(3,3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
            model.add(Conv2DTranspose(64, kernel_size=(3,3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
            model.add(Conv2D(1, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))
            model.summary()


            #Compile and fit data
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.fit(noisy_input,
                      original,
                      epochs=epochsnumber,
                      batch_size=batch_size,
                      validation_split=validation_split)

            #Generate denoised images
            samples = noisy_input_test[:number_of_visualizations]
            targets = y_test[:number_of_visualizations]
            denoised_images = model.predict(samples)

            for i in range(0, number_of_visualizations):

              noisy_img = noisy_input_test[i][:, :, 0]
              original_img  = original_test[i][:, :, 0]
              denoised_img = denoised_images[i][:, :, 0]
              input_class = targets[i]

              fig, axes = plt.subplots(1, 3)
              fig.set_size_inches(8, 3.5)


              axes[1].imshow(noisy_img)
              axes[1].set_title('Noisy image')
              axes[0].imshow(original_img)
              axes[0].set_title('original image')
              axes[2].imshow(denoised_img)
              axes[2].set_title('Denoised image')
              fig.suptitle(f'MNIST target = {input_class}')
              plt.show()







if __name__ == '__main__':
    root = tk.Tk()
    root.title("part6_Denoising mnist")
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))
    b1 = tk.Button(root, text='Show',
                  command=(lambda e=ents: fetch(e)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root, text='Quit', command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()



