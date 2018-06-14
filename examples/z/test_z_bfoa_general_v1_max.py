import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt
import numpy as np
import time
# from SwarmPackagePy import revenue_optimization_function as revf

# Scenario:
# For checking date = 12/Apr/2018
# M = 3
# R = 2
# lrm = 3 for all r,m, Prmk
# T = 2 (2-Now, 1-then and 0-On the day, -1- Day after), t =2,1,0,-1
# J(-1) = 0

M = 3
R = 2
lrm = 3  # for all r,m, Prmk
T = 2
T = T + 1

Dimention = M * T

D = np.zeros((R, T, M, lrm))
print('Demand ', D)

P = np.zeros((R, T, M, lrm))
print('Price Tag ', D)

for r in range(R):
	for t in range(T):
		for m in range(M):
			for k in range(lrm):
				D[r][t][m][k] = 50 * (r + t + m + k)

for r in range(R):
	for t in range(T):
		for m in range(M):
			for k in range(lrm):
				P[r][t][m][k] = 5000 / D[r][t][m][k]

print('Demand ', D)
print('Price Tag ', D)

Nr = [(i + 1) * 10 for i in range(R)]
print(Nr)  # Room

OptAlloc = []
OptTags = []
MaxRev = 0

start_time = time.time()
# Brute FOrce Attack
for a in range(Nr[0]):
	for b in range(Nr[0]):
		for c in range(Nr[0]):
			for d in range(Nr[0]):
				for e in range(Nr[0]):
					for f in range(Nr[0]):
						for g in range(Nr[0]):
							for h in range(Nr[0]):
								for i in range(Nr[0]):
									alloc = [a, b, c, d, e, f, g, h, i]
									print('Brute Force Element ', alloc)
									sum_alloc = total = sum(alloc)
									if (sum_alloc <= Nr[0]):
									    profits = [0.0 for n in range(9)]
									    tags = [0 for o in range(9)]
									    for j in range(9):
									    	r = 0
									    	t = int(j / T)
									    	m = int(j % T)
									    	price_tags = P[r][t][m]
									    	demand_prediction = D[r][t][m]
									    	l = len(price_tags)
									    	max_rev_tag = [0.0 in range(l)]
									    	alloc_tmp = alloc[j]
									    	profit_tag = alloc_tmp * price_tags # profit_tag = profit_tag - demand_prediction[j]
									    	max_val = 0
									    	max_tag = 0
									    	for p in range(l):
									    		if(profit_tag[p]>max_val and alloc_tmp<=demand_prediction[p]):
									    			max_val = profit_tag[p]
									    			max_tag = p
									    	profits[j] = max_val
									    	tags[j]=max_tag
									    revenue = sum(profits)
									    if(revenue > MaxRev):
									    	MaxRev = revenue
									    	OptAlloc = alloc
									    	OptTags = tags
print("Optimal Allocation")
print(OptAlloc, MaxRev, OptTags)

end_time = time.time()
print('Time: ',end_time-start_time)

# r = 20
r = Nr[0]
r=20

lamda = 100
f = tf.gaussian_multimodal4_positive_max
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal4_positive_max

revf = SwarmPackagePy.revenue_optimization_function(M, R , lrm ,T)
# f = SwarmPackagePy.revenue_optimization_function.__rev_function
# revf.rev_function()
# f = revf.rev_function

dim =2

lb = [-r for i in range(dim)]
ub = [ r for i in range(dim)]

# alh = SwarmPackagePy.z_bfoa_general_v1_max(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'none', 'false', 'discrete','max')
alh = SwarmPackagePy.z_bfoa_general_v1_max(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'none', 'false', 'discrete','max')


fits = alh._get_jfits()
plt.plot(fits, 'b', label='J-fit')

jcclist = alh._get_jcclist()
plt.plot(jcclist,'r', label='J-cc')

jarlist = alh._get_jarlist()
plt.plot(jarlist,'g', label='J-ar')

jlist = alh._get_jlist()
plt.plot(jlist,'y', label='J')

jblist = alh._get_jblist()
plt.plot(jblist,'p', label='J-best')

plt.legend()
plt.show()

# plt.pause(0.001)
# plt.subplot(2, 1, 2)
steps = alh._get_csteps()
# print(steps)
plt.plot(steps)
plt.show()


animation(alh.get_agents(), f, -r, r,dim)
animation3D(alh.get_agents(), f, -r, r,dim)
