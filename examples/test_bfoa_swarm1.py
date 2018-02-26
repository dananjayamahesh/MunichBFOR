import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt

# alh = SwarmPackagePy.bfoa(100, tf.easom_function, -10, 10, 2, 20, 16, 4, 2, 12, 0.1, 0.25, 400 )
# animation(alh.get_agents(), tf.easom_function, -10, 10)
# animation3D(alh.get_agents(), tf.easom_function, -10, 10)

#alh = SwarmPackagePy.bfoa(100, tf.easom_function, -10, 10, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 400 )
#animation(alh.get_agents(), tf.easom_function, -20, 20)
#animation3D(alh.get_agents(), tf.easom_function, -20, 20)



#alh = SwarmPackagePy.bfoa(100, tf.multiple_gaussian_function, -20, 20, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 400 )
#animation(alh.get_agents(), tf.multiple_gaussian_function, -20, 20)
#animation3D(alh.get_agents(), tf.multiple_gaussian_function, -20, 20)


#alh = SwarmPackagePy.bfoa_swarm1(100, tf.thriple_gaussian_function, -50, 50, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
#GOOD alh = SwarmPackagePy.bfoa_swarm1(100, tf.f2_rosenbrock_function, -100, 100, 30, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
#GOOD alh = SwarmPackagePy.bfoa_swarm1(100, tf.f2_rosenbrock_function, -32, 32, 30, 100, 16, 4, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
#NOT alh = SwarmPackagePy.bfoa_swarm1(100, tf.f3_ackley_function, -32, 32, 30, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
#GOOD alh = SwarmPackagePy.bfoa_swarm1(100, tf.f3_ackley_function, -32, 32, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
alh = SwarmPackagePy.bfoa_swarm1(100, tf.f3_ackley_function, -32, 32, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )

fits = alh._get_jfits()
#print(fits)
plt.plot(fits, 'b', label='J-fit')
#plt.ylabel('J values')
#plt.show()

jcclist = alh._get_jcclist()
#print(jcclist)
plt.plot(jcclist,'r', label='J-cc')
#plt.show()

jarlist = alh._get_jarlist()
#print(jarlist)
#plt.plot(jarlist,'g',)
#plt.show()

jlist = alh._get_jlist()
#print(jlist)
plt.plot(jlist,'y', label='J')
#plt.show()

jblist = alh._get_jblist()
#print(jblist)
plt.plot(jblist,'g', label='J-best')

plt.legend()
plt.show()

#animation(alh.get_agents(), tf.thriple_gaussian_function, -50, 50)
#animation3D(alh.get_agents(), tf.thriple_gaussian_function, -50, 50) 

animation(alh.get_agents(), tf.f3_ackley_function, -32, 32)
animation3D(alh.get_agents(), tf.f3_ackley_function, -32, 32) 


#alh = SwarmPackagePy.bfoa_swarm1(100, tf.thriple_wide_gaussian_function, -50, 50, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
#animation(alh.get_agents(), tf.thriple_wide_gaussian_function, -50, 50)
#animation3D(alh.get_agents(), tf.thriple_wide_gaussian_function, -50, 50) 

#SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
