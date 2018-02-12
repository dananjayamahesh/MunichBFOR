import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D

#alh = SwarmPackagePy.pso(50, tf.easom_function, -10, 10, 2, 20,w=0.5, c1=1, c2=1)

#alh = SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
alh = SwarmPackagePy.bfo(50, tf.easom_function, -10, 10, 2, 20, 2, 12, 0.2, 1.15)

animation(alh.get_agents(), tf.easom_function, -10, 10)
animation3D(alh.get_agents(), tf.easom_function, -10, 10)

#SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
