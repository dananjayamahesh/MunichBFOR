import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt

#alh = SwarmPackagePy.abfoa2(20, tf.easom_function, -10, 10, 2, 20, 16, 4, 2, 12, 0.9, 0.25, 0.1)
#animation(alh.get_agents(), tf.easom_function, -10, 10)
#animation3D(alh.get_agents(), tf.easom_function, -10, 10)

#alh = SwarmPackagePy.abfoa2(20, tf.gaussian_function, -10, 10, 2, 20, 16, 4, 2, 12, 0.9, 0.25, 0.1)
#animation(alh.get_agents(), tf.gaussian_function, -10, 10)
#animation3D(alh.get_agents(), tf.gaussian_function, -10, 10)

alh = SwarmPackagePy.abfoa2(100, tf.thriple_gaussian_function_positive, -10, 10, 50, 20, 16, 4, 2, 12, 0.9, 0.25, 1)

steps = alh._get_csteps()
print(steps)
plt.plot(steps)
plt.ylabel('Step Size')
#plt.show()

jlist = alh._get_jlist()
print(jlist)
#plt.plot(jlist,'r')
plt.show()

animation(alh.get_agents(), tf.thriple_gaussian_function_positive, -10, 10)
animation3D(alh.get_agents(), tf.thriple_gaussian_function_positive, -10, 10)

"""
alh = SwarmPackagePy.abfoa2(100, tf.thriple_wide_gaussian_function, -50, 50, 2, 20, 16, 4, 2, 12, 0.9, 0.25, 0.1)
animation(alh.get_agents(), tf.thriple_wide_gaussian_function, -50, 50)
animation3D(alh.get_agents(), tf.thriple_wide_gaussian_function, -50, 50)

"""

#SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
