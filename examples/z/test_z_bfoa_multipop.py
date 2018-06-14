import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation_multipop, animation3D_multipop
import matplotlib.pyplot as plt


#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f1_sphere_function, -100, 100, 3, 100, 8, 4, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,10 )
#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f5_griewank_function, -10, 10, 30, 100, 16, 4, 2, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,400 )
#VERYGOOD alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f3_ackley_function, -32, 32, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100 )

r = 20
lamda = 400
f = tf.gaussian_multimodal4_positive
n = 100
seg = 4
#f = tf.gaussian_multimodal_positive #32 #lamda=50
#f = tf.f3_ackley_function #lamda=100 
#f = tf.f2_rosenbrock_function #lamda=400 not enough.
#ACKLEY

#alh = SwarmPackagePy.z_ibfoa_jun_li(100, tf.f2_rosenbrock_function, -r, r, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100,0.01 )
#alh = SwarmPackagePy.bfoa_swarm1_dev1_rep(100, tf.f2_rosenbrock_function, -r, r, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100 )
#alh = SwarmPackagePy.z_bfoa_swarm1_dev1(100, tf.f3_ackley_function, -32, 32, 30, 100, 24, 8, 8, 12, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100 )
# SO FAR BEST alh = SwarmPackagePy.z_ibfoa_jun_li(100, f, -r, r, 30, 100, 8, 8, 16, 4, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100,0.03)
#HARSHA alh = SwarmPackagePy.z_ibfoa_jun_li(100, f, -r, r, 30, 100, 6, 8, 10, 5, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100,0.03)
#alh = SwarmPackagePy.z_ibfoa_jun_li(100, f, -r, r, 2, 100, 6, 8, 10, 5, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100,0.03)
#alh = SwarmPackagePy.bfoa_swarm1(100, f, -r, r, 30, 100, 5, 8, 10, 5, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100)
#alh = SwarmPackagePy.z_bfoa_swarm1_dev1(100, f, -r, r, 30, 100, 6, 8, 10, 5, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100)
#alh = SwarmPackagePy.z_ibfoa_jun_li(100, f, -r, r, 30, 100, 6, 8, 10, 5, 0.1, 0.25, 0.1, 0.2, 0.1, 10,lamda,0.03)

#BEST alh = SwarmPackagePy.z_bfoa_multipop(100, f, -r, r, 30, 100, 6, 8, 10, 5, 0.1, 0.25, 0.1, 0.2, 0.1, 10,100,seg, 15)
#2D
alh = SwarmPackagePy.z_bfoa_multipop(100, f, -r, r, 3, 100, 4, 8, 4, 8, 0.1, 0.25, 0.1, 0.2, 0.1, 10, 100, seg, 15)


fits = alh._get_jfits()
plt.plot(fits, 'b', label='J-fit')

jcclist = alh._get_jcclist()
plt.plot(jcclist,'r', label='J-cc')

jarlist = alh._get_jarlist()

jlist = alh._get_jlist()
plt.plot(jlist,'y', label='J')

jblist = alh._get_jblist()
plt.plot(jblist,'g', label='J-best')

plt.legend()
plt.show()

#plt.pause(0.001)
#plt.subplot(2, 1, 2)
steps = alh._get_csteps()
#print(steps)
plt.plot(steps)
plt.show()

animation_multipop(alh.get_agents(), f, -r, r, n, seg)
animation3D_multipop(alh.get_agents(), f, -r, r)
