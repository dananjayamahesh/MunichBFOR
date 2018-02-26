import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt

#alh = SwarmPackagePy.abfoa1(100, tf.easom_function, -10, 10, 2, 20, 16, 4, 2, 12, 0.9, 0.25, 400 )
#animation(alh.get_agents(), tf.easom_function, -10, 10)
#animation3D(alh.get_agents(), tf.easom_function, -10, 10)

alh = SwarmPackagePy.abfoa1(100, tf.thriple_gaussian_function_positive, -10, 10, 50, 20, 16, 4, 2, 12, 0.9, 0.25, 1000 )

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

#SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
