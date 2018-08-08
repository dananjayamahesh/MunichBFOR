import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D,test_function_shape
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

#import os
#import system
##########################WITHOUT BRUTE FORCE########################################
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

Nr = [(i + 1) * 400 for i in range(R)]

print("Available Capacity ",Nr)  # Room

OptAlloc = []
OptTags = []
MaxRev = 0
'''
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
'''
# r = 20
r = Nr[0]
r = 20
r =100
#dim =2

dim = M*T
x = np.random.uniform(0,r, (1,dim))
print('Random X ',x)
lamda = 100
f = tf.gaussian_multimodal4_positive_max
n =100 #50
#r = 20
r =400
lamda = 100
f = tf.gaussian_diff_multimodal4_positive_max

N = 400
#D = 20 #Divisions

M = 2
R = 2
lrm = 4  # for all r,m, Prmk
T = 19
T = T + 1
Nr = [75, 20]
r = max(Nr)
D = 800
Dimention = M*T
dim = Dimention

revf = SwarmPackagePy.revenue_optimization_function(M, R , lrm ,T)
#filename = '../../data/data1.csv'
filename = '/home/mahesh/paraqum/repos/SwarmPackagePy/data/data2.csv' #ALl numbers

#revf.read_paramemeters_file(filename)
#read_paramemeters_file(self, filename, Nr, M_=3, R_=2, lrm_=3, T_=2, D = 20 ):
revf.read_paramemeters_file(filename, Nr, 2, 2, 4, 20, 800) #TTotal
print("self.N",revf.N)
print('Press ENTER to continue!')
input()
#No Need
#Set Parameter Commented
#revf.set_parameters(N, M, R , lrm ,T)

#raw_input("Press Enter to continue ...")
#pause()
print('Press ENTER to continue!')
input()                                 

# f = SwarmPackagePy.revenue_optimization_function.__rev_function

#tmp = revf.rev_function(x)
#print(tmp)
f = revf.rev_function

#f = revf.gaussian_diff_multimodal4_positive_max

#test_function_shape(f, -r, r,dim)

#dim =2

lb = [0 for i in range(dim)]
ub = [ r for i in range(dim)]

# alh = SwarmPackagePy.z_bfoa_general_v1_max(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'none', 'false', 'discrete','max')
#alh = SwarmPackagePy.z_bfoa_general_v1_max(100, f, lb, ub, dim, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'none', 'false', 'discrete','max')

#alh = SwarmPackagePy.z_bfoa_general_v1_max(100, f, lb, ub, dim, 100, 8, 16, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'none', 'false', 'discrete','max')
alh = SwarmPackagePy.z_bfoa_general_v1_max(100, f, lb, ub, dim, 100, 16, 16, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'none', 'false', 'discrete','max')

print('Maximum Revenue : ',revf.MaxRev)
print('Optimum Allocation : ',revf.OptAlloc)
#print('Max Revenue : ',revf.OptAlloc)

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


animation(alh.get_agents(), f, 0, r,dim)
animation3D(alh.get_agents(), f, 0, r,dim)
