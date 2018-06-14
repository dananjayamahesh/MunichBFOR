import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import multiniche_benchmark as mbtf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt


#BEST 2 - with Swarm2-WOW - WOOOOOOOOOOWWWW
n =100 #50
r = 19
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
sigma_share = 0.2 #0.6 NOT WORKING
d_min = 0.2 #1
d_max = 3
clust_alpha = 1
alh = SwarmPackagePy.z_bfoa_multiniche_clustering_v2(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.1, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm2', 'false', sigma_share,d_min, d_max, clust_alpha)


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
