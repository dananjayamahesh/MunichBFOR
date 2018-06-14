from math import *
import numpy as np

def f1_sphere_function(x):
    return sum([i**2 for i in x])

def f2_rosenbrock_function(x):
    n = len(x)
    #print(n)
    return sum([(100*((x[i+1]-(x[i]**2))**2) + (x[i]-1)**2) for i in range (n-1)])

def f3_ackley_function(x):
    #print(x.shape[0])
    #https://www.sfu.ca/~ssurjano/ackley.html
    n = len(x)    
    return (-20 * exp(-0.2 * sqrt(sum([i**2 for i in x])/n ))) - exp(sum([cos(2 *np.pi*i) for i in x]) / n) + 20 + exp(1)

def f4_rastrigin_function(x):
	return sum([(i**2 - (10*cos(2*pi*i)) + 10) for i in x])

def f4_rastrigin_function_var1(x): #WORKING ONE
    n = len(x)
    return sum([(i**2 - (10*cos(2*pi*i)) + 10) for i in x]) + 10*n

def f5_griewank_function(x):
	pro = 1
	k = 1
	for j in x:
		s = sqrt(k)
		pro *= cos(j/s)
		k = k+1

	return sum([(i**2) for i in x])/4000 - pro + 1 #ERROR s

def f5_griewank_function_var1(x): #WORKING ONE
    pro = 1
    k = 1
    for j in x:
        s = sqrt(k)
        pro = pro*cos(j/s)
        k = k+1

    return (sum([(i**2) for i in x])/4000) - pro + 1 + 1

#product([(cos(j/sqrt(j))) for j in x])
def f6_H_function(x):
    return 0 

def f7_M_function(x):
    alpha = 2
    n = len(x)
    return 10 - (sum([(sin(5*pi*i)**alpha) for i in x]))/n



def ackley_function(x):
    #print(x.shape[0])
    return -exp(-sqrt(0.5 * sum([i**2 for i in x]))) - exp(0.5 * sum([cos(i) for i in x])) + 1 + exp(1)

def bukin_function(x):
    return 100 * sqrt(abs(x[1] - 0.01 * x[0]**2)) + 0.01 * abs(x[0] + 10)

def cross_in_tray_function(x):
    return round(-0.0001 * (abs(sin(x[0]) * sin(x[1]) * exp(abs(100 - sqrt(sum([i**2 for i in x])) / pi))) + 1)**0.1, 7)

def sphere_function(x):
    return sum([i**2 for i in x])

def bohachevsky_function(x):
    return x[0]**2 + 2 * x[1]**2 - 0.3 * cos(3 * pi * x[0]) - 0.4 * cos(4 * pi * x[1]) + 0.7

def sum_squares_function(x):
    return sum([(i + 1) * x[i]**2 for i in range(len(x))])

def sum_of_different_powers_function(x):
    return sum([abs(x[i])**(i + 2) for i in range(len(x))])

def booth_function(x):
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

def matyas_function(x):
    return 0.26 * sphere_function(x) - 0.48 * x[0] * x[1]

def mccormick_function(x):
    return sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1

def dixon_price_function(x):
    return (x[0] - 1)**2 + sum([(i + 1) * (2 * x[i]**2 - x[i - 1])**2
                                for i in range(1, len(x))])
def six_hump_camel_function(x):
    return (4 - 2.1 * x[0]**2 + x[0]**4 / 3) * x[0]**2 + x[0] * x[1]\
        + (-4 + 4 * x[1]**2) * x[1]**2

def three_hump_camel_function(x):
    return 2 * x[0]**2 - 1.05 * x[0]**4 + x[0]**6 / 6 + x[0] * x[1] + x[1]**2

def easom_function(x):
    return -cos(x[0]) * cos(x[1]) * exp(-(x[0] - pi)**2 - (x[1] - pi)**2)

def michalewicz_function(x):
    return -sum([sin(x[i]) * sin((i + 1) * x[i]**2 / pi)**20 for i in range(len(x))])

def beale_function(x):
    return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + \
           (2.625 - x[0] + x[0] * x[1]**3)**2

def drop_wave_function(x):
    return -(1 + cos(12 * sqrt(sphere_function(x)))) / (0.5 * sphere_function(x) + 2)

# New Functions


def gaussian_function(x):
    return -(np.exp(-4 * np.log(2) * ((x[0] - 4)**2 + (x[1] - 4)**2) / 2**2))


def multiple_gaussian_function(x):
    return -((np.exp(-4 * np.log(2) * ((x[0] - 4)**2 + (x[1] - 4)**2) / 5**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 12)**2 + (x[1] - 12)**2) / 4**2)))


def thriple_gaussian_function(x):
    return -((np.exp(-4 * np.log(2) * ((x[0] - (-4))**2 + (x[1] - (-4))**2) / 5**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 0)**2 + (x[1] - 0)**2) / 4**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 5)**2 + (x[1] - 5)**2) / 6**2)))


def thriple_wide_gaussian_function(x):
    return -((np.exp(-4 * np.log(2) * ((x[0] - (-20))**2 + (x[1] - (-20))**2) / 20**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 0)**2 + (x[1] - 0)**2) / 15**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 20)**2 + (x[1] - 20)**2) / 25**2)))


def double_gaussian_function(x):
    return -(20 * (np.exp(-4 * np.log(2) * ((x[0] - (-5))**2 + (x[1] - (-5))**2) / 20**2)) + 20 * (np.exp(-4 * np.log(2) * ((x[0] - 7)**2 + (x[1] - 7)**2) / 30**2)))


def thriple_gaussian_function_positive(x):
    return 100 - ((np.exp(-4 * np.log(2) * ((x[0] - (-4))**2 + (x[1] - (-4))**2) / 5**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 0)**2 + (x[1] - 0)**2) / 6**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 5)**2 + (x[1] - 5)**2) / 6**2)))


def thriple_gaussian_function_codegen(x):
    return -((np.exp(-4 * np.log(2) * ((x[0] - 0)**2 + (x[1] - 0)**2) / 6**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 7)**2 + (x[1] - 7)**2) / 6**2)))


def gaussian_multimodal_positive(x):
    return 10 - ((np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (-10))**2) / 6**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 10)**2 + (x[1] - 10)**2) / 6**2)))


def gaussian_diff_multimodal_positive(x):
    return 10 - ((np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (-10))**2) / 4**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 10)**2 + (x[1] - 10)**2) / 6**2)))

def gaussian_diff_unimodal_positive(x):
    return 10 - ((np.exp(-4 * np.log(2) * ((x[0] - 10)**2 + (x[1] - 10)**2) / 6**2)))


def gaussian_multimodal3_positive(x):
    return 6 - ((np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (-10))**2) / 6**2)) + (np.exp(-4 * np.log(2) * ((x[0] - (0))**2 + (x[1] - (0))**2) / 5**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 10)**2 + (x[1] - 10)**2) / 6**2)))

def gaussian_multimodal4_positive(x):
    return 10 - ((np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (-10))**2) / 6**2)) + (np.exp(-4 * np.log(2) * ((x[0] - (10))**2 + (x[1] - (-10))**2) / 6**2)) +(np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (10))**2) / 6**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 10)**2 + (x[1] - 10)**2) / 6**2)))

def gaussian_diff_multimodal4_positive(x):
    return 20 - (3*(np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (-10))**2) / 2**2)) + 4*(np.exp(-4 * np.log(2) * ((x[0] - (10))**2 + (x[1] - (-10))**2) / 4**2)) + 5*(np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (10))**2) / 6**2)) + 6*(np.exp(-4 * np.log(2) * ((x[0] - 10)**2 + (x[1] - 10)**2) / 8**2)))

def gaussian_multimodal4_positive_max(x):
    return ((np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (-10))**2) / 6**2)) + (np.exp(-4 * np.log(2) * ((x[0] - (10))**2 + (x[1] - (-10))**2) / 6**2)) +(np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (10))**2) / 6**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 10)**2 + (x[1] - 10)**2) / 6**2)))

def gaussian_diff_multimodal4_positive_max(x):
    return (3*(np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (-10))**2) / 2**2)) + 4*(np.exp(-4 * np.log(2) * ((x[0] - (10))**2 + (x[1] - (-10))**2) / 4**2)) + 5*(np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (10))**2) / 6**2)) + 6*(np.exp(-4 * np.log(2) * ((x[0] - 10)**2 + (x[1] - 10)**2) / 8**2)))

#Multi-Niche Benchmarking
def f3(x):
    A = [i*8-20 for i in range(5)]
    B = [i*8-20 for i in range(5)]

    #return 10 -(1 + sum([(1/(i+ ((x[0]-A[i])**6) + ((x[1]-B[i])**6))) for i in range(20)]))
    return 10 -(1 + sum(np.sum([[(1/(i+ ((x[0]-A[i])**6) + ((x[1]-B[j])**6))) for j in range(5)] for i in range(5) ],axis=1) ) ) 



#BENCHMARK FORM ORIGINAL MULTINICHE GA
def F1(x):
    dim =  len(x)
    if dim <= 1:
        return 5-sin(5*pi*x[0])**6
    elif x[1]==0:
        return (5-sin(5*pi*x[0])**6)
    else:
        return 10.0

def F1_var1(x):
    dim =  len(x)
    if dim <= 1:
        return 10-5*(sin(5*pi*x[0])**6)
    elif x[1]==0:
        return (10-5*(sin(5*pi*x[0])**6))
    else:
        return 10.0

def F2(x):
    dim =  len(x)
    if dim<=1:
        return exp(-2*(np.log(2))*(((x[0]-0.1)/(0.8))**2))*(sin(5*pi*x[0])**6)
    elif x[1]==0:
        return exp(-2*(np.log(2))*(((x[0]-0.1)/(0.8))**2))*(sin(5*pi*x[0])**6)
    else:
        return 10.0

def F2_var1(x):
    dim =  len(x)
    if dim<=1:
        return 10-5*exp(-2*(np.log(2))*(((x[0]-0.1)/(0.8))**2))*(sin(5*pi*x[0])**6)
    elif x[1]==0:
        return 10-5*exp(-2*(np.log(2))*(((x[0]-0.1)/(0.8))**2))*(sin(5*pi*x[0])**6)
    else:
        return 10.0

def F3(x):
    A = [i*8-20 for i in range(5)]
    B = [i*8-20 for i in range(5)]

    beta = 5

    #return 10 -(1 + sum([(1/(i + ((x[0]-A[i])**6) + ((x[1]-B[i])**6))) for i in range(20)]))
    C = [[(beta/((i*beta+j)*beta + ((x[0]-A[i])**6) + ((x[1]-B[j])**6))) for j in range(5)] for i in range(5) ]
    #print(C)
    C=np.array(C)
    #C=C.ravel()
    C=C.flatten()
    return 10 -(1 + sum(C)) 

def F3_test(x):
    b = 4
    #8
    A = [i*13-20 for i in range(b)]
    B = [i*13-20 for i in range(b)]

    beta = 1

    #return 10 -(1 + sum([(1/(i + ((x[0]-A[i])**6) + ((x[1]-B[i])**6))) for i in range(20)]))
    C = [[(1/(1 + ((x[0]-A[i])**6) + ((x[1]-B[j])**6))) for j in range(b)] for i in range(b) ]
    #print(C)
    C=np.array(C)
    #C=C.ravel()
    C=C.flatten()
    return 2-(1 + sum(C)) 

def F4(x):
    A = [-10.0,-10.0,10.0,10.0,0]
    B = [-10.0,10.0,-10.0,10.0,0]
    W = [0.02,0.5,0.01,2.0,0.1]
    H = [0.4,0.2,0.7,1.0,0.05]

    return 10-sum([(H[i]/(1 + W[i]*( (x[0]-A[i])**2 + (x[1]-B[i])**2 ))) for i in range(5)])

#BENCHMARK FROM NICHEPSO
def F5(x):
    return 200 - ((x[0]**2) + x[1] -11)**2 - (x[0] + (x[1]**2) - 7)**2

def F5_var1(x):
    return 1000 - (200 - ((x[0]**2) + x[1] -11)**2 - (x[0] + (x[1]**2) - 7)**2 )

'''
def F_revenue(x):
    #Revenue Optimization Function
    # x[0] - Booking Bucket
    #x[1] - 
'''

def D_t(x):
    print('Demand Formulation')

def D_t(x):
    print('Demand Formulation')

def F_revenue(x):
    print('Revenue Function')
    #r,m,t,k
    #t,r,m,t
    

    #Revenue Optimization Function
    # x[0] - Booking Bucket
    #x[1] - 
