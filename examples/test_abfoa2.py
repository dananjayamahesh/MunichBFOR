import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D

#alh = SwarmPackagePy.abfoa2(20, tf.easom_function, -10, 10, 2, 20, 16, 4, 2, 12, 0.9, 0.25, 0.1)
#animation(alh.get_agents(), tf.easom_function, -10, 10)
#animation3D(alh.get_agents(), tf.easom_function, -10, 10)

#alh = SwarmPackagePy.abfoa2(20, tf.gaussian_function, -10, 10, 2, 20, 16, 4, 2, 12, 0.9, 0.25, 0.1)
#animation(alh.get_agents(), tf.gaussian_function, -10, 10)
#animation3D(alh.get_agents(), tf.gaussian_function, -10, 10)

alh = SwarmPackagePy.abfoa2(20, tf.thriple_gaussian_function, -10, 10, 2, 20, 16, 4, 2, 12, 0.9, 0.25, 0.1)
animation(alh.get_agents(), tf.thriple_gaussian_function, -10, 10)
animation3D(alh.get_agents(), tf.thriple_gaussian_function, -10, 10)

#SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
