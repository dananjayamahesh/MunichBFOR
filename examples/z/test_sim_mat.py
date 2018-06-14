import numpy as np
import math

lb = -1
ub =  1
dimension = 3
n = 10
agents = np.random.uniform(lb, ub, (n, dimension))        
print(agents)
print(agents[0])
print(agents.shape)

c,r = agents.shape
sim_mat = np.zeros((10,10))

print(sim_mat)

for i in range(c):
	for j in range(i,c):
		sim_mat[i,j]= math.sqrt(sum((agents[i]-agents[j])**2))

print(sim_mat)
#self.sim_mat = []