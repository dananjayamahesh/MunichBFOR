import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import multiniche_benchmark as mbtf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt


#"""
#BEST 2 - with Swarm2-WOW - WOOOOOOOOOOWWWW
n =100 #200 #50
r = 19
lamda = 100
f = tf.gaussian_diff_multimodal4_positive
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 1#0.2 #1
d_max = 3
clust_alpha = 2
step_size = 0.1
#alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none' '''adaptive1 also''', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4_raw_for_debug(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none' '''adaptive1 also''', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
#"""

#LAST TESTED
'''rastrigin Working , Change last Visualize, LAST one is the BEST, This is EXTRA, Only Change is n=100 not 500
n =100 #50
r = 5 #lb =-1 ub=5
lamda = 100
#f = tf.f5_griewank_function
f = tf.f4_rastrigin_function_var1

dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 0.2#0.2 #1
d_max = 0.4 #1 #3 #initially 3 Although 1 is good 

clust_alpha = 2
step_size = 0.01 #0.005 #0.1 0.005 is also good
#alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -1, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4_raw_for_debug(n, f, -1, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
'''


''' ************************************************WORKING
#rastrigin BEST DEMO - WOW - NICHES and CLUSTERS are equals
n =500 #50
r = 5 #lb =-1 ub=5
lamda = 100
#f = tf.f5_griewank_function
f = tf.f4_rastrigin_function_var1

dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 0.2#0.2 #1
d_max = 0.4 #1 #3 #initially 3 Although 1 is good 

clust_alpha = 2
step_size = 0.01 #0.005 #0.1 0.005 is also good

alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -1, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
'''


''' WORKING
#M special multimodal function
n =100 #50
r = 1
lamda = 100
#f = tf.f5_griewank_function
f = tf.f7_M_function

dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 0.2#0.2 #1
d_max = 3

clust_alpha = 2
step_size = 0.005 #0.1

alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, 0, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
'''


""" Great Output - BEST DEMO
#Griewank
n =100 #50
r = 10
lamda = 100
#f = tf.f5_griewank_function
f = tf.f5_griewank_function_var1

dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 0.2#0.2 #1
d_max = 3

clust_alpha = 2
step_size = 0.01 #0.1

alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
"""


#BEST ACKLEY WITH SHARING SCHEME
'''
#ACkely 1
n =100 #50
r = 2
lamda = 100
f = tf.f3_ackley_function
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 0.2#0.2 #1
d_max = 3

clust_alpha = 2
step_size = 0.01 #0.1

alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
'''

"""
#BEST 2 - with Swarm2-WOW - WOOOOOOOOOOWWWW
n =100 #50
r = 19
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 1#0.2 #1
d_max = 3

clust_alpha = 2
step_size = 0.1
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none'adaptive1, 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
"""

''' BEST WORK - NICELY WORKING- SHARING on F3 TEST @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
n =100 #50
r = 20
lamda = 1
f = tf.F3_test #tf.F2_var1 #tf.F2_var1
dim =2

sigma_share = 0.2 #0.6 NOT WORKING
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
'''

"""
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal4_positive
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
"""

"""
n =100 #50
r = 20
lamda = 1 #NOT OWRKING
f = tf.F3_test

dim =2
sigma_share = 0.1
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.01, 0.2, 0.01, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
"""
#alh = SwarmPackagePy.z_bfoa(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false')
####################################################################





#Test Cases

##############################################
""" #QUITE WORKING NEED AA CHI SQUARE
n =100 #50
r = 1
lamda = 500
f = tf.F2_var1 #tf.F2_var1
dim =1

sigma_share = 0.2 #0.6 NOT WORKING
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)

"""
#############################################
#F3 BEST MULTI NICHE 
"""
n =100 #50
r = 20
lamda = 1 #NOT OWRKING
f = tf.F3_test

dim =2
sigma_share = 0.1
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.01, 0.2, 0.01, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
"""
##############################################

#BENCHMAR F3 - WOW RESULTS- SHARING ONLY MODEL
"""
n =100 #50
r = 20
lamda = 1 #NOT OWRKING
f = tf.F3_test

dim =2
sigma_share = 0.1
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.01, 0.2, 0.01, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
"""

##############################################
#BENCHMARK F2 - NOT OK -SOMETHING OK -Diverging Pressure
"""
n =100 #50
r = 1
lamda = 400
f = tf.F2_var1
dim =1
sigma_share = 0.1
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.01, 0.2, 0.01, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
"""
##############################################
#BENCHMARK F1 - NOT OK -SOMETHING OK - Diverging Pressure
"""
n =100 #50
r = 1
lamda = 400
f = tf.F1_var1
dim =1
sigma_share = 0.1
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.01, 0.2, 0.01, 10, lamda, 0.03, 'adaptive1', 'swarm1', 'false', sigma_share)
"""
###############################################
"""
#signma_share = 1 and 3 not working
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
sigma_share = 3
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.01, 0.2, 0.01, 10, lamda, 0.03, 'adaptive1', 'swarm1', 'false', sigma_share)
"""
##############################################
"""
#WORST 1- WITH SWARM1 - SHIT
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm1', 'false', 0.2)
"""
##################################################
#
"""
#BEST 2 - with Swarm2-WOW - WOOOOOOOOOOWWWW
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
"""
##################################################
"""
#BEST SO FAR- WITHOUT SWARMING
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal4_positive
dim =2
alh = SwarmPackagePy.z_bfoa_multiniche_sharing_v4(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'none', 'false', 0.2)
"""
###################################################
#FOr F1 [0,1]
#alh = SwarmPackagePy.z_bfoa_multiniche_sharing(50, f, -r, r, dim, 100, 8, 8, 12, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, lamda, 0.03, 'none', 'swarm1', 'false')
#alh = SwarmPackagePy.z_bfoa_multiniche_sharing(100, f, -r, r, dim, 100, 8, 8, 12, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm1', 'false')

'''
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

#Average Over Genrations
avgsteps = alh._get_avg_steps()
avgj = alh._get_avg_j()
avgjfit = alh._get_avg_jfit()

plt.plot(avgj,'b')
plt.plot(avgjfit, 'r')
plt.show()

plt.plot(avgsteps,'b')
plt.show()

numniches = alh._get_num_niches()
plt.plot(numniches,'r')
plt.show()

#animation(alh.get_agents(), f, -r, r,dim)
#animation3D(alh.get_agents(), f, -r, r,dim)


#only for assymetric domain,  M function
#animation(alh.get_agents(), f, 0, r,dim)
#animation3D(alh.get_agents(), f, 0, r,dim)


#for rastrigan, f4_rastrigin_function_var1
animation(alh.get_agents(), f, -1, r,dim)
animation3D(alh.get_agents(), f, -1, r,dim)
'''