from random import random, randint, seed
from statistics import mean
import math
import cmath
import random
import sys
import copy
import time

POP_SIZE = 1000
res = []
points = []
outs = []

class newNode:
    def __init__(self, data):
        self.data = data
        self.left = self.right = None

def insertLevelOrder(arr, root, i, n):

    if i < n:
        temp = newNode(arr[i])
        root = temp

        root.left = insertLevelOrder(arr, root.left, 2 * i + 1, n)
        root.right = insertLevelOrder(arr, root.right, 2 * i + 2, n)
    return root

def inOrder(root):
    if root != None:
        inOrder(root.left)
        res.append(root.data)
        inOrder(root.right)
    return res

###another fitness

# def fitness(points , outs , chromosome):
#     dif = []
#     for i in range(len(points)):
#         dif.append((calculate(chromosome , points[i]) - outs[i] ))
#     error = mean(dif)
#     return 1 / 1+error

def fitness(points , outs , chromosome):
    dif = []
    for i in range(len(points)):
        dif.append(pow(calculate(chromosome , points[i]) - outs[i] , 2 ))
    error = math.sqrt(sum(dif))
    return 1 / 1+error

def find_height(n):
     n += 1
     h=0
     while n!=1:
         n /=2
         h +=1
     return h

def op(a,operation,b ):
    if operation =='add':
        return a+b
    elif operation == 'sub':
        return a-b
    elif operation == 'mul':
        return a*b
    elif operation == 'power':
        if cmath.phase(pow(a  , b)) > sys.maxsize:
            return 0
        return pow(a , b)

    elif operation == 'divide':
        if b == 0:
            return 1
        return a/b
    elif operation == 'sin':
        return a * math.sin(b)
    elif operation == 'cos':
        return a * math.cos(b)

def calculate(lst,val):
        if len(lst) == 3:
            if lst[0]=='x' :
                lst[0]= val
            if lst[2] =='x':
                lst[2] = val
            return op(lst[0] ,lst[1], lst[2] )

        size = len(lst) -1
        size = size // 2

        if lst[size] == 'add':
           return calculate(lst[0:size] , val) + calculate(lst[size + 1:] , val)

        elif lst[size] == 'sub':
            return calculate(lst[0:size], val) - calculate(lst[size + 1:] ,val)

        elif lst[size] == 'mul':
            return calculate(lst[0:size] , val) * calculate(lst[size + 1:], val)

        elif lst[size] == 'divide':
            if calculate(lst[size + 1:], val) == 0:
                 return 1
            return calculate(lst[0:size] , val) / calculate(lst[size + 1:], val)

        elif lst[size] == 'power':
           return pow(calculate(lst[0:size], val),calculate(lst[size + 1:], val))

        elif lst[size] == 'sin':
            return calculate(lst[0:size], val) * math.sin(calculate(lst[size+1:] , val))

        elif lst[size] == 'cos':
            return calculate(lst[0:size] , val) * math.cos(calculate(lst[size+1:] , val))

def randomize_chromosome(size):

    init=[0 for x in range(size)]
    functions=['add' , 'sub' , 'mul', 'sin','cos' , 'divide' ]
    terminals = ['x' , -1 , -2 , 3 , 4  ]
    height = find_height(size)
    leafidx = size - pow(2 , height-1)
    i=0

    while i != size:
        if i <leafidx:
            init[i] = functions[randint(0 , len(functions) - 1)]
        else:
            init[i] = terminals[randint(0 , len(terminals) -1)]
        i +=1
    return init

def selection(pop, fit):

    next_pop = random.choices(pop , weights = fit ,k = POP_SIZE)
    return next_pop

def mutation(chromosome):

    functions=['add' , 'sub' , 'mul','sin' , 'cos' ,'divide']
    terminals = ['x' , 8 , -2 , 3 , 4 ]

    idx = randint(0 , len(chromosome)-1)
    t = idx % 2
    if t == 0:
         chromosome[idx] = terminals[randint(0 , len(terminals)-1)]
    else:
        chromosome[idx] = functions[randint(0 , len(functions)-1)]
    return chromosome

def cross_over(first_pop , sec_pop):
    child_pop =[]
    tem_pop = copy.copy(first_pop)

    for i in range(len(first_pop)):
        rand = random.random()
        if rand<0.43:
            child_pop.append(first_pop[i])
        else:
            child_pop.append(sec_pop[i])
    if random.random() < 0.1:
            child_pop = mutation(tem_pop)
    return child_pop




# ---- main ---- #

population = []
pop_ =[]
i = 0
for x in range(0 , 10):
    points.append(random.uniform(-30 , 80))
for x in points:
    outs.append(x*x - 4*x)
n = int(input())
# n in length of a chromosome(binary tree)
for _ in range(POP_SIZE):
    pop_.append(inOrder(insertLevelOrder(randomize_chromosome(n),None ,0 ,n)))

while i != (POP_SIZE*n):
    population.append(pop_[0][i:i+n])
    i += n

found = False

new_generation= []
generation = 0
fitness_cout =0
print("x :" , points)
print("y :" , outs , '\n')

# GP algorithme
st = time.time()
while not found:

        fit = []

        for i in range(POP_SIZE):
                    fitness_cout +=1
                    fit.append(fitness(points , outs , population[i]))
        for x in fit:
                    ans=[]
                    if abs(x - 1) < pow(10 , -6):
                        found = True
                        et = time.time()
                        ans.append(x)
                        print("final chromosome" ,population[fit.index(max(ans))])
                        print("fitness :" ,fitness(points , outs ,population[fit.index(max(ans))]))
                        print("generation :" , generation)
                        print("fitness_cout :" , fitness_cout)
                        print("times :" , et-st)

                        break

        first_parent_pop = selection(population ,fit)
        sec_parent_pop = selection(population , fit)

        for i in range(POP_SIZE):
                    child = cross_over(first_parent_pop[i] , sec_parent_pop[i])
                    new_generation.append(child)

        population = new_generation.copy()
        new_generation = []
        generation += 1


