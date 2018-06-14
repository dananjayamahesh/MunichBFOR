import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import multiniche_benchmark as mbtf
from SwarmPackagePy import animation, animation3D, animation1D
import matplotlib.pyplot as plt
import numpy as np
from math import *

 #(self, n, function, lb, ub, dimension, iteration, Nre=16, Ned=4, Nc=2, Ns=12, C=0.1, Ped=0.25, Da=0.1, Wa=0.2, Hr=0.1, Wr=10, lamda=400, L=0.03, arga='none', argj='none', arged='false'):

#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f1_sphere_function, -100, 100, 3, 100, 8, 4, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,10 )
#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f5_griewank_function, -10, 10, 30, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
#VERYGOOD alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f3_ackley_function, -32, 32, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100 )
r = 1
#lamda = 100 #FOR MULTINICHE GOOD
lamda = 10
f = tf.F1

alh = SwarmPackagePy.z_bfoa_multiniche_sharing(50, f, -r, r, 1, 100, 8, 8, 12, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, lamda, 0.03, 'adaptive1', 'none', 'false')


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

#plt.pause(0.001)
#plt.subplot(2, 1, 2)
steps = alh._get_csteps()
#print(steps)
plt.plot(steps)
plt.show()

#animation(alh.get_agents(), f, -r, r)
#animation3D(alh.get_agents(), f, -r, r)
def fun(x):
	return 1 - sin(5*pi*x)**6

def fun2(x):
	return exp(-2*(0.693)*(((x-0.1)/(0.8))**2))*(sin(5*pi*x)**6)

#plt.close()
x = np.arange(-1, 1, 0.01)
#y = [[tf.F1(i)] for i in x]
y= [fun2(i) for i in x]
plt.plot(x, y)
plt.show()

#animation1D(alh.get_agents(), f, -r, r)

