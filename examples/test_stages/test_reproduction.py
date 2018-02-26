import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt

#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f1_sphere_function, -100, 100, 3, 100, 8, 4, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,10 )
alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f5_griewank_function, -10, 10, 30, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )

fits = alh._get_jfits()
plt.plot(fits, 'b', label='J-fit')

jcclist = alh._get_jcclist()
plt.plot(jcclist,'r', label='J-cc')

jarlist = alh._get_jarlist()

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

animation(alh.get_agents(), tf.f5_griewank_function, -10, 10)
animation3D(alh.get_agents(), tf.f5_griewank_function, -10, 10)

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