'''
import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt

r = 32
f = tf.gaussian_multimodal3_positive
#f = tf.f3_ackley_function

alh = SwarmPackagePy.pso(100, f, -32, 32, 30, 100,0.5,1,1)

animation(alh.get_agents(), f, -r, r,30)
animation3D(alh.get_agents(), f, -r, r,30)
'''
import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D


alh = SwarmPackagePy.pso(50, tf.easom_function, -10, 10, 2, 20, w=0.5, c1=1, c2=1)
#animation(alh.get_agents(), tf.easom_function, -10, 10)
#animation3D(alh.get_agents(), tf.easom_function, -10, 10)

animation(alh.get_agents(), tf.easom_function, -10, 10, 2)
animation3D(alh.get_agents(), tf.easom_function, -10, 10, 2)
