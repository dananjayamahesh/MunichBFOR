import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import multiniche_benchmark as mbtf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt
'''
Some note on clsutering
d_max has a huge impact, 3 is not woring, but double the size of d_min with d_min=0.2 is working
'''
''' BEST WORK - NICELY WORKING- SHARING on F3 TEST
n =100 #50
r = 20
lamda = 1
f = tf.F3_test #tf.F2_var1 #tf.F2_var1
dim =2
d_min = 0.2
d_max = 0.4

sigma_share = 0.2 #0.6 NOT WORKING
step_size = 0.1 #0.05
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share)
#Adaptive
#alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
'''

''' 2018/04/17
#Clsuterign works for 4 niche, But what happens in the 2 niche -BEST DEMO
#BEST 2 - with Swarm2-WOW - WOOOOOOOOOOWWWW
n =100 #50
r = 19
lamda = 100
f = tf.gaussian_diff_multimodal4_positive
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
#d_min= 0.2, d_max = 0.3
d_min = 0.2
d_max = 0.4
 #0.3 #2 #0.3 #0.6 #0.4 #3 #0.4 #3 #3 is not quite working, 0.4 is perfect, 3 is the error
clust_alpha = 2
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.1, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
#both none and adaptive1 working
'''
##############################################################################################################

#New  Benchmark Test Cases

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
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -1, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
'''


#RASTRIGIN VAR2
''' ************************************************WORKING , I sed -1 to 2 as range, ONLY CHAGE from sharing
#rastrigin BEST DEMO - WOW - NICHES and CLUSTERS are equals
n = 200 #100#50
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

#fixed step working
#alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
#adaptive - WORKING, BUT OT WELL
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)

'''



#RATRIGIN VAR 1
''' ************************************************WORKING , I sed -1 to 2 as range, ONLY CHAGE from sharing
#rastrigin BEST DEMO - WOW - NICHES and CLUSTERS are equals
n =100 #50
r = 2 #lb =-1 ub=5
lamda = 100
#f = tf.f5_griewank_function
f = tf.f4_rastrigin_function_var1

dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 0.2#0.2 #1
d_max = 0.4 #1 #3 #initially 3 Although 1 is good 

clust_alpha = 2
step_size = 0.01 #0.005 #0.1 0.005 is also good

alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
'''




''' ************************************************WORKING
#rastrigin BEST DEMO - WOW - NICHES and CLUSTERS are equals
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

alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -1, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
'''


''' WORKING
#M special multimodal function m WOrking only for fixed length
n =100 #50
r = 1
lamda = 1000 #400 #100 #100 is not working, massive displacemets
#f = tf.f5_griewank_function
f = tf.f7_M_function

dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 0.2#0.2 #1
d_max = 0.4 #3

clust_alpha = 2
step_size = 0.005 #0.1

#alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, 0, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
#Full range -1 to 1
#alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
#adaptive4
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)

'''


""" Great Output - BEST DEMO -IDEAL for DEMO- Carefull about the stepsize
#Griewank
n =100 #50
r = 10
lamda = 100
#f = tf.f5_griewank_function
f = tf.f5_griewank_function_var1

dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 0.2#0.2 #1
d_max = 0.4 #3

clust_alpha = 2
step_size = 0.05 #0.1 #0.01 #0.1 #0.01 is BEST, 0.1 is Working, 0.05 is decent

#VGOOD alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
#ADAPTIVE is also VGOOD
"""


#BEST ACKLEY WITH SHARING SCHEME
#'''
#ACkely 1
n =100 #50
r = 2
lamda = 400#100 is the Original, but no workinh 
f = tf.f3_ackley_function
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 0.2#0.2 #1
d_max = 0.4 #0.3

clust_alpha = 2
step_size = 0.01 #0.1

#non adaptive os working fine
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
#alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, step_size, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
#adaptive Have some issues
#'''










###############################################################################################################
'''
#Clsuterign works for 4 niche, But what happens in the 2 niche
#BEST 2 - with Swarm2-WOW - WOOOOOOOOOOWWWW
n =100 #50
r = 19
lamda = 100
f = tf.gaussian_diff_multimodal4_positive
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
#d_min= 0.2, d_max = 0.3
d_min = 1 #0.2 #1
d_max = 2 #0.3 #0.6 #0.4 #3 #0.4 #3 #3 is not quite working, 0.4 is perfect, 3 is the error
clust_alpha = 2
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.1, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'none', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)
#both none and adaptive1 working
'''

#TEST CASES

######################################################

'''NOT GOOD WITH DIFF 2
n =100 #50
r = 19
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
'''

#######################################################
''' #BEST CLUSTERING d_max =3, d_min=1 d_min = 1 d_max = 5
clust_alpha = 1
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal4_positive
dim =2
d_min = 1
d_max = 5
clust_alpha = 1
sigma_share = 0.2 #0.6 NOT WORKING
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share, d_min, d_max, clust_alpha)
'''


#######################################################

'''
#BEST 2 - with Swarm2-WOW - WOOOOOOOOOOWWWW - Sharing
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal4_positive
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)

'''
#######################################################

"""
n =100 #50
r = 15
lamda = 400
f = tf.F4 #tf.F2_var1 #tf.F2_var1
dim =2

sigma_share = 0.2 #0.6 NOT WORKING
alh = SwarmPackagePy.z_bfoa_multiniche_clearing_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
"""


"""
Masssive Deceptive Multimodal
n =100 #50
r = 5
lamda = 10000
f = tf.F5_var1  #tf.F2_var1 #tf.F2_var1
dim =2

sigma_share = 0.2 #0.6 NOT WORKING
alh = SwarmPackagePy.z_bfoa_multiniche_clearing_v1(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share)
"""


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

animation(alh.get_agents(), f, -r, r,dim)
animation3D(alh.get_agents(), f, -r, r,dim)

#only for assymetric domain,  M function
#animation(alh.get_agents(), f, 0, r,dim)
#animation3D(alh.get_agents(), f, 0, r,dim)


#for rastrigan, f4_rastrigin_function_var1
#animation(alh.get_agents(), f, -1, r,dim)
#animation3D(alh.get_agents(), f, -1, r,dim)
