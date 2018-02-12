import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D

#alh = SwarmPackagePy.pso(50, tf.easom_function, -10, 10, 2, 20,w=0.5, c1=1, c2=1)

#alh = SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
#alh = SwarmPackagePy.bfo_with_swarm(20, tf.easom_function, -30, 30, 2, 10, 2, 12, 0.2, 1.15, 0.1, 0.2, 0.1, 10)
alh = SwarmPackagePy.bfo_with_env_swarm(50, tf.easom_function, -30, 30, 3, 30, 100, 4, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 0)
animation3D(alh.get_agents(), tf.easom_function, -50, 50)
animation3D(alh.get_agents(), tf.easom_function, -50, 50)

#SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
