from math import *
import numpy as np
import csv
import psutil
from pandas import *
import math
# from . import testFunctions


class revenue_optimization_function():

    def __init__(self, M_=3, R_=2, lrm_=3, T_=2):
        print('Initiating Revenue Function')

        self.Nr = [(i + 1) *400 for i in range(R_)]
        print('Capacities', self.Nr)
        self.N = 400 #Should Change Later #

        print('RunningCapacity', self.N)
        print(self.Nr)
        self.OptAlloc = []
        self.OptTags = []
        self.MaxRev = 0
        self.M = M_  # 3
        self.R = R_  # 2
        self.lrm = lrm_  # 3
        self.T = T_  # 2
        #self.T = self.T + 1
        self.D = np.zeros((self.R, self.T, self.M, self.lrm))
        print('Demand ', self.D)

        self.P = np.zeros((self.R, self.T, self.M, self.lrm))
        print('Price Tag ', self.D)

        self.PSelect = np.zeros((self.R, self.T, self.M))
        self.DCollect= np.zeros((self.R, self.T, self.M))

        for r in range(self.R):
            for t in range(self.T):
                for m in range(self.M):
                    for k in range(self.lrm):
                        self.D[r][t][m][k] = 50 * (r + t + m + k)

        for r in range(self.R):
            for t in range(self.T):
                for m in range(self.M):
                    for k in range(self.lrm):
                        self.P[r][t][m][k] = 5000 / self.D[r][t][m][k]

        print('Demand ', self.D)
        print('Price Tag ', self.P)

        self.Nr = [(i + 1) * 10 for i in range(self.R)]
        print(self.Nr)  # Room

    def gaussian_multimodal4_positive_max(self,x):
        print('Revenue Function', x)
        print('X ',x, ' Shape ',len(x))
        return ((np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (-10))**2) / 6**2)) + (np.exp(-4 * np.log(2) * ((x[0] - (10))**2 + (x[1] - (-10))**2) / 6**2)) +(np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (10))**2) / 6**2)) + (np.exp(-4 * np.log(2) * ((x[0] - 10)**2 + (x[1] - 10)**2) / 6**2)))

    def gaussian_diff_multimodal4_positive_max(self,x):
        print('Revenue Function', x)
        print('X ',x, ' Shape ',len(x))
        return (3*(np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (-10))**2) / 2**2)) + 4*(np.exp(-4 * np.log(2) * ((x[0] - (10))**2 + (x[1] - (-10))**2) / 4**2)) + 5*(np.exp(-4 * np.log(2) * ((x[0] - (-10))**2 + (x[1] - (10))**2) / 6**2)) + 6*(np.exp(-4 * np.log(2) * ((x[0] - 10)**2 + (x[1] - 10)**2) / 8**2)))

    def rev_function(self, x):
        # Calculate Revenue Function

        #print('Revenue Function', x)
        #x = x.clip(0)
        #print('X ',x, ' Shape ',len(x))
        #alloc = x.clip(0)
        alloc = [(e if e>0 else 0)  for e in x]
        print('Allocation ', alloc)
        print('Total Allocations: ',sum(alloc))
        ran = len(x)
        #print('Brute Force Element ', alloc)
        sum_alloc = sum(alloc)
        revenue = 0


        #if (sum_alloc <= self.N):
        profits = [0.0 for n in range(ran)]
        tags = [0 for o in range(ran)]
        demands = [0 for o in range(ran)]
        for j in range(ran):
            r = 0

            '''BUG DETECTED
            t = j // self.T
            m = j % self.T
            '''
            t = j // self.M
            m = j % self.M

            #print(j,r,t,m, self.T)
            price_tags = self.P[r][t][m]
            demand_prediction = self.D[r][t][m]
            l = len(price_tags)
            max_rev_tag = [0.0 in range(l)]
            alloc_tmp = alloc[j]
            # profit_tag = profit_tag - demand_prediction[j]
            profit_tag = alloc_tmp * price_tags
            max_val = 0
            max_tag = 0
            max_demand = 0
            for p in range(l):
                if(profit_tag[p] > max_val and alloc_tmp <= demand_prediction[p]):
                    max_val = profit_tag[p]
                    max_tag = p
                    max_demand = demand_prediction[p]

            profits[j] = max_val
            tags[j] = max_tag
            demands[j]= max_demand

        print('Demands ',demands)
        print('Profits ',profits)
        print('Tags ',tags)    
        revenue = sum(profits)
        if(revenue > self.MaxRev):
            self.MaxRev = revenue
            self.OptAlloc = alloc
            self.OptTags = tags

        #else:
        #    revenue = 0
        sigma  = 0.01 #0.001
        if (sum_alloc > self.N):
            #revenue = revenue - sigma*(sum_alloc-self.N)*revenue #Really Good
            revenue = revenue - (((sum_alloc-self.N) / self.N)*revenue)
            #revenue = revenue/2
        return revenue

    def read_paramemeters_file(self, filename, Nr, M_=3, R_=2, lrm_=3, T_=2, TTotal = 800 ):
        
        print('Read Parameters from File')
        N = 400 #not necessary        
        print('N value', N)
        self.N = 400 #Should Change Later #

        #BUG Fixed
        self.N = Nr[0]
        self.Nr = Nr
        print(self.Nr)
        self.OptAlloc = []
        self.OptTags = []
        self.MaxRev = 0
        self.M = M_  # 3
        self.R = R_  # 2
        self.lrm = lrm_  # 3
        self.T = T_  # 2
        #self.T = self.T + 1
        self.D = np.zeros((self.R, self.T, self.M, self.lrm))
        print('Demand ', self.D)

        self.P = np.zeros((self.R, self.T, self.M, self.lrm))
        print('Price Tag ', self.D)

        self.PSelect = np.zeros((self.R, self.T, self.M))
        self.DCollect = np.zeros((self.R, self.T, self.M))

        for r in range(self.R):
            for t in range(self.T):
                for m in range(self.M):
                    for k in range(self.lrm):
                        self.D[r][t][m][k] = 50 * (r + t + m + k)

        for r in range(self.R):
            for t in range(self.T):
                for m in range(self.M):
                    for k in range(self.lrm):
                        self.P[r][t][m][k] = 5000 / self.D[r][t][m][k]

        #path = 'c:\\temp\\'
        self.D = np.zeros((self.R, self.T, self.M, self.lrm))
        print('Demand ', self.D)

        self.P = np.zeros((self.R, self.T, self.M, self.lrm))
        print('Price Tag ', self.D)

        self.PSelect = np.zeros((self.R, self.T, self.M))
        self.DCollect= np.zeros((self.R, self.T, self.M))

        Price = np.zeros((self.R,self.lrm))

        PRate = [[30,50,70,90],[20,40,60,80]]

        data = []

        Par = TTotal//self.T #Parametr for taking modulus operator
        print('Modulo Parameter : ', Par, TTotal)

        #Price Rate Strategy
        file = open(filename, "r")
        reader = csv.reader(file)
        for line in reader:
            bb = line[0] # Bucking Bucket
            t = line[0]
            t = int(t)

            #modified_t = (t % self.T)
            #modified_t = math.fmod(t, self.T)
            modified_t = (t // Par)
            t = modified_t

            pr = line[3] #Pricing Tag
            p =  line[3]
            p = int(p)
            p = p-1

            ms = line[4] #Market Segment
            m = line[4]
            m = int(m)

            rt = line[5] #Room Type
            r = line[5] #Room Type
            r = int(r)

            dm = line [6] #demand
            d = line [6]
            d = int(d)

            para = (rt, bb, ms, pr, dm)
            data += [para]

            for k in range(self.lrm):
                self.P[r][t][m][k] = PRate[r][k]

            self.PSelect [r][t][m] = int(p)
            self.DCollect [r][t][m] += int(d)

            #self.P[r][t][m][k] =            

            #t=line[1],line[2]
            #print(t)

        #Demand-Price Strategy Formulation
        print(data)
        data.sort()
        print(DataFrame(data))
        print('########################### Price Selection ##############################')
        print(DataFrame(self.PSelect[0]))
        print('########################### Demand Collecttion ################################')
        print(DataFrame(self.DCollect[0]))


        for r in range(self.R):
            for t in range(self.T):
                for m in range(self.M):
                    print('Index',r,t,m)
                    print('Original', self.DCollect [r][t][m], self.PSelect [r][t][m])
                    d_tmp = int(self.DCollect [r][t][m])
                    p_tmp = int(self.PSelect [r][t][m]) #price Index
                    print(d_tmp, p_tmp, self.P[r][t][m][p_tmp])
                    pr_tmp = self.P[r][t][m][p_tmp]
                    print(d_tmp, p_tmp, pr_tmp)
                    m_eq = d_tmp/pr_tmp
                    #Function Definition
                    for k in range(self.lrm):
                        #Ceil or FLoor
                        self.D[r][t][m][k] = int(m_eq * self.P[r][t][m][k])

        print('Modified Demand ', self.D)
        print('Available Price Tag ', self.P)
        print('End Probelm FOrmulating')
        print(self.Nr,self.N)
        #T = 780


    def flush(self):
        print('Flush Operation')

    def set_parameters(self, N, M_=3, R_=2, lrm_=3, T_=2):
        print('Setting Up Revenue Parameters')
        #NCap is an array
        self.Nr = [(i + 1) *400 for i in range(R_)]
        #
        print('Capacities', self.Nr)
        print('N value', N)
        self.N = N #Should Change Later #
        print(self.Nr)
        self.OptAlloc = []
        self.OptTags = []
        self.MaxRev = 0
        self.M = M_  # 3
        self.R = R_  # 2
        self.lrm = lrm_  # 3
        self.T = T_  # 2
        #self.T = self.T + 1
        self.D = np.zeros((self.R, self.T, self.M, self.lrm))
        print('Demand ', self.D)

        self.P = np.zeros((self.R, self.T, self.M, self.lrm))
        print('Price Tag ', self.D)

        for r in range(self.R):
            for t in range(self.T):
                for m in range(self.M):
                    for k in range(self.lrm):
                        self.D[r][t][m][k] = 50 * (r + t + m + k)

        for r in range(self.R):
            for t in range(self.T):
                for m in range(self.M):
                    for k in range(self.lrm):
                        self.P[r][t][m][k] = 5000 / self.D[r][t][m][k]

        print('Demand ', self.D)
        print('Price Tag ', self.D)

        self.Nr = [(i + 1) * 10 for i in range(self.R)]
        print(self.Nr)  # Room


