import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt

r = 50 #20
lamda = 100
f = tf.gaussian_multimodal4_positive_max
n =100 #50
lamda = 100
dim =2
f = tf.gaussian_diff_multimodal4_positive_max
f = tf.gaussian_diff_multimodal_positive
f = tf.gaussian_diff_unimodal_positive

lb = [-r for i in range(dim)]
ub = [ r for i in range(dim)]

#alh = SwarmPackagePy.z_bfoa_general_v1(100, f, -r, r, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'swarm2', 'false')
#alh = SwarmPackagePy.z_bfoa_general_v1(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'swarm2', 'false')
#NICE WOW
#alh = SwarmPackagePy.z_bfoa_general_v1(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'swarm2', 'false', search_type='discrete')

#WOW BEST SO FAR - CELLULAR AUTOMATUM
#alh = SwarmPackagePy.z_bfoa_general_v1(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'none', 'false', search_type='discrete')

#GOOD BUT NOT VERY MUCH. BEST VALUE VARY
#alh = SwarmPackagePy.z_bfoa_general_v1(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'swarm1', 'false', search_type='discrete')

#alh = SwarmPackagePy.z_bfoa_general_v1(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'swarm2', 'false', search_type='discrete')
#alh = SwarmPackagePy.z_bfoa_general_v1_max(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'swarm1', 'false', 'discrete','max')


#alh = SwarmPackagePy.z_bfoa_general_v1_max(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'none', 'false', 'discrete','max')

alh = SwarmPackagePy.z_bfoa_general_v1_max(100, f, lb, ub, 2, 100, 8, 8, 4, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'adaptive1', 'swarm2', 'false', 'continuous','min')


#TEST CASES

########################################################################
#DAMN WORSE - Lets change SOme Parametr Nc = 4
'''
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 2, 100, 8, 8, 12, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'none', 'swarm1', 'false')
'''

#########################################################################
''' OK
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal_positive
dim =2
alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 2, 100, 8, 8, 12, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'adaptive1', 'swarm1', 'false')
'''
##########################################################################

'''
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal4_positive
dim =2
alh = SwarmPackagePy.z_bfoa(100, f, -r, r, 2, 100, 8, 8, 12, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, 0.03, 'adaptive1', 'swarm1', 'false')
'''

##########################################################################
#NOT WORKING - For DIff Multi2
"""
n =100 #50
r = 20
lamda = 100
f = tf.gaussian_diff_multimodal4_positive
dim =2
alh = SwarmPackagePy.z_bfoa(n, f, -r, r, dim, 100, 16, 2, 8, 12, 0.05, 0.25, 0.05, 0.2, 0.05, 10, lamda, 0.03, 'adaptive1', 'swarm1', 'false')
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

animation(alh.get_agents(), f, -r, r,dim)
animation3D(alh.get_agents(), f, -r, r,dim)
