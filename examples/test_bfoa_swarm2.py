import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D
import matplotlib.pyplot as plt

"""
alh = SwarmPackagePy.bfoa_swarm2(100, tf.thriple_gaussian_function, -50, 50, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
animation(alh.get_agents(), tf.thriple_gaussian_function, -50, 50)
animation3D(alh.get_agents(), tf.thriple_gaussian_function, -50, 50)
"""
#easom_function

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

#alh = SwarmPackagePy.bfoa_swarm2(100, tf.thriple_wide_gaussian_function, -50, 50, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
#animation(alh.get_agents(), tf.thriple_wide_gaussian_function, -50, 50)
#animation3D(alh.get_agents(), tf.thriple_wide_gaussian_function, -50, 50) 
#double_gaussian_function
#thriple_gaussian_function

alh = SwarmPackagePy.bfoa_swarm2(100, tf.thriple_gaussian_function, -10, 10, 50, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )

fits = alh._get_jfitlist()
print('Fit Val')
print(fits)
plt.plot(fits, 'b', label='J-fit')
#plt.ylabel('J values')
#plt.show()

jcclist = alh._get_jcclist()
print('jcclist')
print(jcclist)
plt.plot(jcclist,'r', label='J-cc')
#plt.show()

jarlist = alh._get_jarlist()
print('jarlist')
print(jarlist)
plt.plot(jarlist,'g', label='J-ar')
#plt.show()

jlist = alh._get_jlist()
print('jlist')
print(jlist)
plt.plot(jlist,'y', label='J')
#plt.show()

jblist = alh._get_jblist()
#print(jblist)
plt.plot(jblist,'p', label='J-Best')

plt.legend()
plt.show()


for i in range(128):
	print(fits[i],jarlist[i],jlist[i])
	#print(fits[i])
	#print(jarlist[i])
	#print(jlist[i])
	if(jlist[i] != fits[i]+jarlist[i]):
		print('i')
		print('Error')
	

animation(alh.get_agents(), tf.thriple_gaussian_function, -10, 10)
animation3D(alh.get_agents(), tf.thriple_gaussian_function, -10, 10) 

#SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
