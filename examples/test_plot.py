import matplotlib.pyplot as plt
import math
import numpy as np


def function(x, A, B):
    return math.exp(A*x) * math.sin(B*x)


rr = np.arange(-0, 10, 0.001)

def y(x): 
    return np.sin(x / 2.) * np.exp(x / 4.) + 6. * np.exp(-x / 4.)

def f1(x): 
    return np.exp(-x *10)

def f2(x): 
    return np.exp(-x *1)

def f3(x): 
    return np.exp(-x *1) - np.exp(-x *10)

def f4(x): 
    return -np.exp(-x *1) + np.exp(-x *10)

plt.plot(rr, f1(rr), 'b', label='np.exp(-x *10)')

plt.plot(rr, f2(rr), 'r', label='np.exp(-x *1)')
plt.plot(rr, f3(rr), 'y', label='np.exp(-x *1) - np.exp(-x *10)')
plt.plot(rr, f4(rr), 'g', label='-np.exp(-x *1) + np.exp(-x *10)')
plt.legend()
plt.show()
"""
import matplotlib.pyplot as plt
points = 1000 #Number of points
xmin, xmax = -1, 5
xlist = map(lambda x: float(xmax - xmin)*x/points, range(points+1))
ylist = map(lambda y: function(y, -1, 5), xlist)
plt.plot(xlist, ylist)
plt.show()
"""

"""
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
"""