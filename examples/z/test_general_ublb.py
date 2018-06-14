import numpy as np
import math
import random
 #test some functionalities
n=4
dimension = 2
lb=0
ub=5
agents = np.random.uniform(lb, ub, (n, dimension))
print(agents)
agents = np.random.randint(lb, ub, (n, dimension))
print(agents)
lb = [0,0]
ub = [5,10]
r = random.choice([(lb[i],ub[i]) for i in range(dimension)])
agents = np.random.uniform(*r, (n, dimension))
print(agents)

r = random.choice([(lb[i],ub[i]) for i in range(dimension)])
agents = np.random.random_integers(*r, (n, dimension))
print(agents)

