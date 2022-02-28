import copy
import numpy as np
import random
import time

def cost(A, B, X):

    lst = np.dot(A, X)
    dif = [0 for x in range(len(B))]

    for i in range(len(B)):
        dif[i] = (B[i] - lst[i]) * (B[i] - lst[i])

    return sum(dif)


def findnextneighbour(A, B, step, current):
    temp_curr = copy.copy(current)
    cnt = 0
    current_cost = cost(A, B, current)
    while cnt < 10000:
        for i in range(len(current)):

            temp_curr[i] -= step
            left_cost = cost(A, B, temp_curr)

            temp_curr[i] += (2 * step)
            right_cost = cost(A, B, temp_curr)

            if right_cost > current_cost and current_cost < left_cost:
                temp_curr[i] = temp_curr[i] - step


            elif left_cost < right_cost:

                temp_curr[i] = temp_curr[i] - 2 * step
                current_cost = left_cost

            elif right_cost < left_cost:

                current_cost = right_cost

        cnt = cnt + 1
        # print("cost ", current_cost)
    # print(np.dot(A, temp_curr))
    return temp_curr

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


start, end, step = input().split()
st = time.time()
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


res = findnextneighbour(A, B, float(step), X)
et = time.time()
#end time

# print("Random x :", X)
print("Final x: " , res)
print("Final cost: " , cost(A , B , res))
err = np.dot(A  , res)

for i in range(len(B)):
        error.append((B[i] - np.dot(A , res)[i]))

print("Errors :" , error)
ts2 = time.time()

print("The time: " , et -st )
