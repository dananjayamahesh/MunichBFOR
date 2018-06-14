import SwarmPackagePy
from SwarmPackagePy import testFunctions as tf
from SwarmPackagePy import animation, animation3D,test_function_shape
import matplotlib.pyplot as plt



r =20
f = tf.F3_test
dim =2

test_function_shape(f, -r, r,dim)

