import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D

# alh = SwarmPackagePy.bfoa(100, tf.easom_function, -10, 10, 2, 20, 16, 4, 2, 12, 0.1, 0.25, 400 )
# animation(alh.get_agents(), tf.easom_function, -10, 10)
# animation3D(alh.get_agents(), tf.easom_function, -10, 10)

#alh = SwarmPackagePy.bfoa(100, tf.easom_function, -10, 10, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 400 )
#animation(alh.get_agents(), tf.easom_function, -20, 20)
#animation3D(alh.get_agents(), tf.easom_function, -20, 20)



#alh = SwarmPackagePy.bfoa(100, tf.multiple_gaussian_function, -20, 20, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 400 )
#animation(alh.get_agents(), tf.multiple_gaussian_function, -20, 20)
#animation3D(alh.get_agents(), tf.multiple_gaussian_function, -20, 20)
"""
alh = SwarmPackagePy.bfoa_swarm1(1000, tf.thriple_gaussian_function, -10, 10, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
animation(alh.get_agents(), tf.thriple_gaussian_function, -10, 10)
animation3D(alh.get_agents(), tf.thriple_gaussian_function, -10, 10) 
"""

alh = SwarmPackagePy.bfoa_swarm1(100, tf.thriple_wide_gaussian_function, -50, 50, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
animation(alh.get_agents(), tf.thriple_wide_gaussian_function, -50, 50)
animation3D(alh.get_agents(), tf.thriple_wide_gaussian_function, -50, 50)

#SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
