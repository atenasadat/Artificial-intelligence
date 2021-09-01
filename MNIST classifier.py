
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import time



# ___________ GUI ___________
import tkinter as tk


fields = 'Solver', 'Learning_rate_init ',' momentum', 'learning_rate'
attributes =[]
def fetch(entries):
    for entry in entries:
        fields = entry[0]
        text  = entry[1].get()
        attributes.append(text)
    P5()

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




def P5():

        # load data
        print("fetching data set")
        print(len(attributes))
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


        # map pixels to 0 or 1
        X = X / 255.
        print(len(X))
        # K_split=train/test split (total: 70000)
        K_split = 60000
        X_train, X_test = X[:K_split], X[K_split:]
        y_train, y_test = y[:K_split], y[K_split:]


        # docs: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html


        # different params for creating MLP classifier
        params = [

            # Test set score: 0.979200
            # Test set loss: 0.003509
            # Iterations: 99
            {
                'solver': attributes[0],
                'learning_rate': attributes[3],
                'momentum':float(attributes[2]),
                'learning_rate_init': float(attributes[1])
             }
            # ,
            #
            # # Test set score: 0.980800
            # # Test set loss: 0.000790
            # # Iterations: 32
            # {
            #     'solver': 'sgd',
            #     'learning_rate': 'constant',
            #     'momentum': .9,
            #     'learning_rate_init': 0.2
            # },
            #
            # # Test set score: 0.926600
            # # Test set loss: 0.262210
            # # Iterations: 32
            # {
            #     'solver': 'sgd',
            #     'learning_rate': 'invscaling',
            #     'momentum': 0,
            #     'learning_rate_init': 0.2
            # },
            #
            # # Test set score: 0.968800
            # # Test set loss: 0.086138
            # # Iterations: 60
            # {
            #     'solver': 'sgd',
            #     'learning_rate': 'invscaling',
            #     'momentum': .9,
            #     'learning_rate_init': 0.2
            # },
            #
            # # Test set score: 0.936300
            # # Test set loss: 0.165384
            # # Iterations: 116
            # {
            #     'hidden_layer_sizes': (10,),
            #     'solver': 'adam',
            #     'learning_rate_init': 0.01
            # },
            #
            # # Test set score: 0.970500
            # # Test set loss: 0.037968
            # # Iterations: 29
            # {
            #     'hidden_layer_sizes': (100,),
            #     'solver': 'adam',
            #     'learning_rate_init': 0.01
            # },
            #
            # # Test set score: 0.976200
            # # Test set loss: 0.038021
            # # Iterations: 30
            # {
            #     'hidden_layer_sizes': (100, 100),
            #     'solver': 'adam',
            #     'learning_rate_init': 0.01,
            # }
        ]

        for param in params:
            t1  = time.time()
            mlp = MLPClassifier(verbose=True, **param)
            mlp.fit(X_train, y_train)
            t2  = time.time()
            print("------------ time :" , t2-t1)
            print("------------ Prams:", param)
            print("------------ Test set score: %f" % mlp.score(X_test, y_test))
            print("------------ Test set loss: %f\n" % mlp.loss_)



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



