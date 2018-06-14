import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt


 #(self, n, function, lb, ub, dimension, iteration, Nre=16, Ned=4, Nc=2, Ns=12, C=0.1, Ped=0.25, Da=0.1, Wa=0.2, Hr=0.1, Wr=10, lamda=400, L=0.03, arga='none', argj='none', arged='false'):

#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f1_sphere_function, -100, 100, 3, 100, 8, 4, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,10 )
#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f5_griewank_function, -10, 10, 30, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
#VERYGOOD alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f3_ackley_function, -32, 32, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100 )
r = 20
lamda = 100
f = tf.gaussian_multimodal4_positive
#f = tf.gaussian_multimodal_positive #32 #lamda=50
#f = tf.f3_ackley_function #lamda=100 
#f = tf.f2_rosenbrock_function #lamda=400 not enough.

#HARSHA alh = SwarmPackagePy.z_ibfoa_jun_li(100, f, -r, r, 30, 100, 6, 8, 10, 5, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100,0.03)
#1 alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 6, 4, 16, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100,0.03, 'adaptive1', 'swarm1', 'false')
#ORIGINAL alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 16, 4, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400,0.03, 'adaptive1', 'swarm1', 'false')
#alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 4, 4, 16, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'swarm1', 'false')

#ORIGINAL alh = SwarmPackagePy.z_bfoa(50, f, -r, r, 30, 100, 16, 4, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'swarm1', 'false')
#ORIGINAL 
#BEST 1 alh = SwarmPackagePy.z_bfoa(50, f, -r, r, 30, 100, 16, 4, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'adaptive1', 'swarm1', 'false')

#alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 16, 4, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'swarm1', 'true')
#BEST 2 
#alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 8, 8, 12, 12, 1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'adaptive1', 'swarm1', 'true')

#HARSHA BEST3 
#alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 8, 8, 12, 12, 1, 0.25, 0.5, 0.2, 0.1, 10, 200, 0.1, 'adaptive1', 'swarm1', 'true')




#3 BEST FOR DEMO MULTI-SWARM - LAST ONE

#alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 8, 8, 12, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'adaptive1', 'swarm1', 'true')
#4 alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 8, 8, 12, 12, 0.9, 0.25, 0.1, 0.2, 0.1, 10, 400, 0.03, 'adaptive1', 'swarm1', 'true')


#HARSHA BEST2 : Without Improvment to Elimintion-Dispersal
#alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 8, 8, 12, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'adaptive1', 'swarm1', 'false')
#BEST 
#alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 8, 8, 12, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 400, 0.03, 'adaptive1', 'swarm1', 'false')

#HARSHA BEST1 DEMO1 Org alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 30, 100, 8, 8, 12, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 400, 0.03, 'adaptive1', 'swarm1', 'false')


#GOOD 1 alh = SwarmPackagePy.z_bfoa(50, f, -r, r, 30, 100, 16, 4, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'adaptive1', 'swarm1', 'false')
#NOT GOOD alh = SwarmPackagePy.z_bfoa(50, f, -r, r, 30, 100, 16, 4, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'adaptive1', 'swarm1', 'true')


#Different Heuristics for the Reproductions, chem, jcc, jdelta, and weighted_jdelta
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
#alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false','std','jcc')
#alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm3', 'false','std','jdelta')
#alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.1, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false','none','chem')
alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.1, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false','std','chem')

#TEST CASES

#################################################
#7- SUS only model is quite working for the F2 as well - Whaaaaat?
"""
n =100 #50
r = 1
lamda = 500 #500 is BEST
argrep = 'susonly'
f = tf.F2_var1
dim =1
alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false','susonly')
"""

#################################################
#6-WORKING SUS only model for F1
"""
n =100 #50
r = 1
lamda = 500 #500 is BEST
argrep = 'susonly'
f = tf.F1_var1
dim =1
alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false','susonly')
"""
#################################################
"""
#5_ SUS ONLY MODEL with 4multimodal-WORKING -OBSERVATION is IF NICHES ARE SIMILAR, SUS ONly MODEL WORKS   
n =100 #50
r = 20
lamda = 100
argrep = 'susonly'
f = tf.gaussian_multimodal4_positive
dim =2
alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false','susonly')
"""

#################################################
#4_Ooops FAILED WITH ONLY SUS BFOS finds Multi Niches - BEST SO FAR
"""
n =100 #50
r = 20
lamda = 100
argrep = 'susonly'
f = tf.gaussian_diff_multimodal4_positive
dim =2
alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false','susonly')
"""
#################################################
"""
#3_WOW WITH ONLY SUS BFOS finds Multi Niches - BEST SO FAR
n =100 #50
r = 20
lamda = 100
argrep = 'susonly'
f = tf.gaussian_diff_multimodal_positive
dim =2
alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false','susonly')
"""
#################################################
#2_STD REPRODUCTION SAME AS 1
"""
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false','std')
"""
#################################################
"""
#1_ORIGINAL BFOA BEST ONE - HIGH CONVERGENCE - 
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
alh = SwarmPackagePy.z_bfoa_extended(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false','std')
"""
###################################################

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

animation(alh.get_agents(), f, -r, r,dim)
animation3D(alh.get_agents(), f, -r, r,dim)
