import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D

alh = SwarmPackagePy.classic_bfo(100, tf.easom_function, -10, 10, 2, 20, 16, 4, 2, 12, 0.1, 0.25, 1/16 )
animation(alh.get_agents(), tf.easom_function, -10, 10)
animation3D(alh.get_agents(), tf.easom_function, -10, 10)

#SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
