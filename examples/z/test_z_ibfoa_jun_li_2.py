import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt


#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f1_sphere_function, -100, 100, 3, 100, 8, 4, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,10 )
#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f5_griewank_function, -10, 10, 30, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
#VERYGOOD alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f3_ackley_function, -32, 32, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100 )
r = 20
f = tf.gaussian_multimodal3_positive #32 #lamda=50
#f = tf.f3_ackley_function #lamda=100 
#f = tf.f2_rosenbrock_function #lamda=400 not enough.
#ACKLEY
#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f2_rosenbrock_function, -r, r, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100 )
#alh = SwarmPackagePy.z_bfoa_swarm1_dev1(100, tf.f3_ackley_function, -32, 32, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100 )
# SO FAR BEST alh = SwarmPackagePy.z_ibfoa_jun_li(100, f, -r, r, 30, 100, 8, 8, 16, 4, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100,0.03)
#alh = SwarmPackagePy.z_ibfoa_jun_li_2(200, f, -r, r, 30, 100, 8, 8, 16, 4, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100,0.02)
alh = SwarmPackagePy.z_ibfoa_jun_li_2(100, f, -r, r, 30, 100, 8, 8, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100,0.02)

fits = alh._get_jfits()
plt.plot(fits, 'b', label='J-fit')

jcclist = alh._get_jcclist()
plt.plot(jcclist,'r', label='J-cc')

jarlist = alh._get_jarlist()
plt.plot(jarlist,'r', label='J-ar')

jlist = alh._get_jlist()
plt.plot(jlist,'y', label='J')

jblist = alh._get_jblist()
plt.plot(jblist,'g', label='J-best')

plt.legend()
plt.show()

#plt.pause(0.001)
#plt.subplot(2, 1, 2)
steps = alh._get_csteps()
#print(steps)
plt.plot(steps)
plt.show()

animation(alh.get_agents(), f, -r, r)
animation3D(alh.get_agents(), f, -r, r)

""""
alh = SwarmPackagePy.bfoa_swarm1(50, tf.f1_sphere_function, -10, 10, 2, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
animation(alh.get_agents(), tf.f1_sphere_function, -10, 10)
animation3D(alh.get_agents(), tf.f1_sphere_function, -10, 10)
"""

"""
alh = SwarmPackagePy.bfoa_swarm1(100, tf.f1_sphere_function, -32, 32, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
animation(alh.get_agents(), tf.f1_sphere_function, -32, 32)
animation3D(alh.get_agents(), tf.f1_sphere_function, -32, 32)
"""