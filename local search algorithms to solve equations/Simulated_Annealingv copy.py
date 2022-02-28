import numpy as np
import copy
import time
import random
from random import randint
import warnings
def cost(A, B, X):
    lst = np.dot(A, X)
    dif = [0 for x in range(len(B))]
    for i in range(len(B)):
        dif[i] = (B[i] - lst[i]) * (B[i] - lst[i])
    return sum(dif)

def simulatedAnnealing(current , A, B , step):

    cnt =0
    a = 1
    T = 10000
    while cnt < 100000 and T >0 :

        T =  step * a

        next_neighbour = findrandomneighbour(step , current)
        next_cost = cost(A , B , next_neighbour)
        current_cost = cost(A , B , current)
        delta_energy = next_cost - current_cost

        warnings.filterwarnings("ignore")
        if delta_energy < 0 or (np.exp(-delta_energy/ T) >= random.random()):
           current = next_neighbour.copy()

        cnt += 1
        a *= 0.9
    return current

def findrandomneighbour(step , X):
    temp_X = X.copy()

    randindex = randint(0,2*len(X)-1)
    if randindex % 2 != 0:
        randindex /= 2

        temp_X[int(randindex)] += step

    else:
        randindex /= 2
        temp_X[int(randindex)] -= step

    # randindex = randint(0 , len(X) -1)
    # rand2 = randint(0,2)
    # if rand2 > 1:
    #  temp_X[int(randindex)] += step
    # else:
    #     temp_X[int(randindex)] -= step



    return temp_X

#main
myArray = []
textFile = open("/Users/atena/Desktop/new_example.txt", "r")
lines = textFile.readlines()
B = []
A = []
values = []
results = []
error=[]
#start the program

st = time.time()

start, end, step = input().split()
ts = time.time()
for line in lines:
    myArray.append(line.strip().split(","))

for i in range(len(myArray)):
    B.append(myArray[i][-1])
    A.append(myArray[i][0:-1])

X = [0 for x in range(len(A[0]))]

for i in range(len(X)):
    X[i] = random.uniform(float(start), float(end))

for i in range(len(A)):
    A[i] = [float(k) for k in A[i]]
B = [float(k) for k in B]



res = simulatedAnnealing(X , A , B , float(step))
et = time.time()

for i in range(len(B)):
        error.append((B[i] - np.dot(A,res)[i]))


print("Final x: " , res)
print("Final cost: " , cost(A,B,res))
print("Errors :" , error)
print("The Time: " , et-st)
