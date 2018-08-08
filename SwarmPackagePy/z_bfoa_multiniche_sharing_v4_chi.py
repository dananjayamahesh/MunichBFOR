import numpy as np
from random import random
import math
from pandas import *
import sys
from . import intelligence
import os
#import psutil
import time

class z_bfoa_multiniche_sharing_v4_chi(intelligence.sw):
    """

    FInalized Sharing for BFOA - V4 with Chi Square Performance Evaluation

     Adaptive K-Means clustrring based MuNichBFOR
    -Multi Niche Sharing Bacteria Foraging Optimization-
    Holland 1975, Goldberg and Richardson -1987
    Developer: Mahesh Dananjaya
    All included
    """
    def calculate_distance_matrix(self, n):
       
        for i in range(n):
            T = np.array(self.__agents)
            T_diff = (T - T[i])
            T_diff_sq = T_diff**2
            #Distance Calculations
            T_sum = np.sum(T_diff_sq, axis=1)
            T_dis = np.sqrt(T_sum)
            self.dis_mat [i] = np.array(T_dis)
            #Sharing FUnction
            T_dis_shr = 1-(T_dis/self.sigma_share)**self.alpha
            T_dis_shr = np.array(T_dis_shr)
            T_dis_shr_clip = T_dis_shr.clip(0)
            self.shr_mat[i] = np.array(T_dis_shr_clip)
            self.niche_count[i] = np.sum(T_dis_shr_clip)

            #Just to Identidy Niches in Sharing Scheme
            T_clus_shr = 1-(T_dis/self.sigma_cluster)
            T_clus_shr_clip = T_clus_shr.clip(0)
            self.clus_mat[i] = np.array(T_clus_shr_clip)

    #Calculate the Distance between initial clusters for merge and loose
    def calculate_centroid_distance_matrix(self, n, k):       
        for i in range(k):
            #T = np.array([self.__agents[i] for i in range(k)])
            T = np.array(self.cluster_centroids)            
            T_diff = (T - T[i])
            T_diff_sq = T_diff**2
            #Distance Calculations
            T_sum = np.sum(T_diff_sq, axis=1)
            T_dis = np.sqrt(T_sum)
            self.centdis_mat[i] = np.array(T_dis) 
            #distance between centroids

    def calculate_distance_between_point_and_centro(self, n, i):
       
            #T = np.array([self.__agents[i] for i in range(k)])
            T = np.array(self.cluster_centroids)
            T_point = self.__agents[i]            
            T_diff = (T - T_point)
            T_diff_sq = T_diff**2
            #Distance Calculations
            T_sum = np.sum(T_diff_sq, axis=1)
            T_dis = np.sqrt(T_sum)

            #self.centdis_mat [i] = np.array(T_dis) 
            mini = T_dis.argmin()
            minv = T_dis[mini]
            return [mini,minv]

    def check_and_merge_centroids(self,n,k):


        while True:

            self.centdis_mat = np.zeros((k,k))
            self.calculate_centroid_distance_matrix(n,k) #Calculate Centroid Matrix
            
            #min_dis_loc = self.centdis_mat.argmin()

            min_x =0
            min_y =0
            min_val = sys.maxsize
            for p in range(k):
                for q in range(p+1,k):
                    if self.centdis_mat[p][q]<min_val:
                        min_x = p
                        min_y = q
                        min_val = self.centdis_mat[p][q] 

            min_dis_loc_raw = min_x
            min_dis_loc_col = min_y
            
            if self.centdis_mat[min_dis_loc_raw][min_dis_loc_col]<self.d_min:
                #MERGE - NEED a FLAG to TRUE
                tmp_cent = (self.cluster_centroids[min_dis_loc_raw] + self.cluster_centroids[min_dis_loc_col])/2
                #OR CAN USE WEIGHTED ONE
                self.cluster_centroids[min_dis_loc_raw] = tmp_cent
                self.cluster_centroids.pop(min_dis_loc_col)
                k=k-1
            else:
                #print('End Of Centroid Reduction')
                break

            if(k==1):break
        return k

    def cell_to_cell_function(self, agents, i):
        #print('cell-to-cell interactions')

        T = np.array(self.__agents)
        T_diff = (T - T[i])
        T_diff_sq = T_diff**2

        #Distance Calculations
        T_sum = np.sum(T_diff_sq, axis=1)

        self.dis_mat [i] = np.array(np.sqrt(T_sum))

        T_sum_a = (-self.Wa) * T_sum
        T_sum_r = (-self.Wr) * T_sum
        T_sum_a_exp = np.exp(T_sum_a)
        T_sum_r_exp = np.exp(T_sum_r)
        T_sum_aa = -self.Da * T_sum_a_exp
        T_sum_rr = self.Hr * T_sum_r_exp
        J_cc = sum(T_sum_aa) + sum(T_sum_rr)
        self.evaluations = self.evaluations +1
        # if(i == self.N / 2):
        # print(J_cc)
        return J_cc

    def __init__(self, n, function, lb, ub, dimension, iteration, Nre=16, Ned=4, Nc=2, Ns=12, C=0.1, Ped=0.25, Da=0.1, Wa=0.2, Hr=0.1, Wr=10, lamda=400, L=0.03, arga='none', argj='none', arged='false',sigma=1, d_min=1, d_max=3, clust_alpha=1):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: the number of iterations
        :param Nc: number of chemotactic steps (default value is 2)
        :param Ns: swimming length (default value is 12)
        :param C: the size of step taken in the random direction specified by
        the tumble (default value is 0.2)
        :param Ped: elimination-dispersal probability (default value is 1.15)
        """

        super(z_bfoa_multiniche_sharing_v4_chi, self).__init__()

        #Memory Usage

        #process = psutil.Process(os.getpid())
        start_time = time.time()        

        print(n, function, lb, ub, dimension, iteration, Nre, Ned, Nc,
              Ns, C, Ped, Da, Wa, Hr, Wr, lamda, L, arga, argj, arged)
        # Randomly populate the individuals in the intial population
        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        self._points(self.__agents)

        self.dis_mat = np.zeros((n,n)) #Distance Matrix
        self.sim_mat = np.array(self.dis_mat) #Similarity Matrix
        self.shr_mat = np.zeros((n,n))
        self.neigh_mat = np.zeros((n,n))
        self.niche_count = np.sum(self.shr_mat, axis=1)

        self.sigma_share = sigma#0.5 #0.2
        self.alpha = 1
        self.beta = 1

        #NICHE IDENTIFICATION
        self.clus_mat = np.zeros((n,n))        
        self.sigma_cluster = sigma    #1.5#0.5 #0.2 #MAJOR CHANGE

        #Clustering
        self.d_min = 1.0 #0.2
        self.d_max = 3 #5 #1.5 #0.5

        self.d_min = d_min #1.0 #0.2
        self.d_max = d_max #3 #5 #1.5 #0.5
        self.clust_alpha = clust_alpha #CRUCIAL

        init_k = 20
        k = init_k
        self.centdis_mat = np.zeros((init_k,init_k))

        f = open("results.txt","w+")

        n_is_even = True
        if n & 1:
            n_is_even = False

        self.Da = Da
        self.Wa = Wa
        self.Hr = Hr
        self.Wr = Wr
        self.N = n

        J = np.array([function(x) for x in self.__agents])
        J_fit = np.array([J[x] for x in range(n)])  # [for p in J] #p
        #J_fit = J
        J_cc = np.array([0.0 for x in self.__agents])
        J_ar = np.array([0.0 for x in self.__agents])

        self._points(self.__agents)

        Pbest = self.__agents[J.argmin()]
        Gbest = Pbest
        J_best = function(Gbest)

        C_a = [C for i in range(n)]
        self.__Steps = []
        # self.cell_to_cell_function(self.__agents,1)
        self.__JFitList = []
        self.__JCCList = []
        self.__JARList = []
        self.__JList = []
        self.__JBList = []

        self.__AvgSteps=[]
        self.__AvgJ=[]
        self.__AvgJFit=[]

        self.__NumNiches=[]

        generation = 0
        iteration = 0
        beta = 20  # Q value update frequency
        Ged = 0  # generation MOD 20
        Q = 1  # percentage of bacteria to be eliniated and disperes

        self.evaluations = 0
        self.gbesteval = 0

        self.cluster_centroids = []
        self.cluster_centroids_fit = []
        self.cluster_centroids_capacity = []

        for l in range(Ned):

            for k in range(Nre):

                J_chem = [J[::1]]
                #J_last = J[::1]
                J_last = np.array(J)

                #J_fit = np.array(J)

                for j in range(Nc):

                    for i in range(n):

                        # Can move inside
                        dell = np.random.uniform(-1, 1, dimension)

                        #####################################################
                        if(arga == 'adaptive1'):
                            C_a[i] = 1 / (1 + (lamda / np.abs(J_fit[i]))) #Initially we use J[i]
                        elif(arga == 'adaptive2'):
                            C_a[i] = 1 / (1 + (lamda / np.abs(J_fit[i] - J_best)))
                        elif(arga == 'improved1'):
                            C_a[i] = C_a[i] / (2 * (generation // 10))
                        else:
                            C_a[i] = C_a[i]  # Doing nothing
                        #####################################################

                        self.__agents[i] += C_a[i] * \
                            np.linalg.norm(dell) * dell

                        J_fit[i] = function(self.__agents[i])
                        J_cc[i] = self.cell_to_cell_function(self.__agents, i)

                        #####################################################

                        #print('ERROR1', J[i], J_last[i])

                        if(argj == 'swarm1'):
                            J[i] = J_fit[i] + J_cc[i]
                        elif (argj == 'swarm2'):
                            J_ar[i] = np.exp(-J_fit[i]) * J_cc[i]
                            J[i] = J_fit[i] + J_ar[i]
                        else:
                            J[i] = J_fit[i]

                        #print('ERROR2', J[i], J_last[i])
                        #####################################################
                        if J[i] < J_best:
                            Gbest = self.__agents[i]
                            J_best = J[i]
                            self.gbesteval = self.evaluations
                        ######################################################
                       # Monitoring
                        if(i == n / 2):
                            self.__JFitList.append(J_fit[i])
                            self.__JCCList.append(J_cc[i])
                            self.__JARList.append(J_ar[i])
                            self.__JList.append(J[i])
                            self.__JBList.append(J_best)
                            self.__Steps.append(C_a[i])
                            #print(C_a[i], J_fit[i], J_cc[i], J[i],(J_fit[i] + J_cc[i]), J_best)
                        #####################################################
                        # Start Swim Steps
                        #print('Here')
                        for m in range(Ns):
                            #print('Print', l,k,j,i,m)
                            #if(m == (Ns - 1)):
                            #    print('HitMax', generation)

                            # bacteria moves only when objective function is reduced
                            #print('BEST', J[i], J_last[i])
                            if J[i] < J_last[i]:
                                #print('HIT')
                                J_last[i] = J[i]

                                ##############################################################
                                if(arga == 'adaptive1'):
                                    C_a[i] = 1 / (1 + (lamda / np.abs(J_fit[i]))) #Initially J_fit[i]
                                elif(arga == 'adaptive2'):
                                    C_a[i] = 1 / \
                                        (1 + (lamda / np.abs(J_fit[i] - J_best)))
                                elif(arga == 'improved1'):
                                    C_a[i] = C_a[i] / (2 * (generation // 10))
                                else:
                                    C_a[i] = C_a[i]  # Doing nothing
                                ###############################################################
                                #C_a[i] = 1 / (1 + (lamda / np.abs(J[i]-J_best)))
                                self.__agents[i] += C_a[i] * np.linalg.norm(dell) \
                                    * dell

                                J_fit[i] = function(self.__agents[i])
                                J_cc[i] = self.cell_to_cell_function(
                                    self.__agents, i)

                                #####################################################
                                if(argj == 'swarm1'):
                                    J[i] = J_fit[i] + J_cc[i]
                                elif (argj == 'swarm2'):
                                    J_ar[i] = np.exp(-J_fit[i]) * J_cc[i]
                                    J[i] = J_fit[i] + J_ar[i]
                                else:
                                    J[i] = J_fit[i]
                                #####################################################

                                #J[i] = J_fit[i] + J_cc[i]

                                if J[i] < J_best:
                                    Gbest = self.__agents[i]
                                    J_best = J[i]
                                    self.gbesteval = self.evaluations

                                # Probing
                                if(i == n / 2):
                                    self.__JFitList.append(J_fit[i])
                                    self.__JCCList.append(J_cc[i])
                                    self.__JARList.append(J_ar[i])
                                    self.__JList.append(J[i])
                                    self.__JBList.append(J_best)
                                    self.__Steps.append(C_a[i])
                                    #print('m is of bac ', m)
                                    #print(C_a[i], J_fit[i], J_cc[i],J[i], (J_fit[i] + J_cc[i])), J_best
                                #J[i] = function(self.__agents[i])+ self.cell_to_cell_function(self.__agents, i)
                            else:
                                break

                    #After Every Chemotaxis Stage
                    self._points(self.__agents)  # Add to the animation

                    self.__AvgSteps.append(sum(C_a)/n)
                    self.__AvgJ.append(sum(J)/n)
                    self.__AvgJFit.append(sum(J_fit)/n)

                    # End of Chemotaxis
                    # Make lgorithm faster
                    #J = np.array([function(x) for x in self.__agents])
                    J_chem += [J]
                    generation += 1

                # Ending Chemotaxix Steps of all individuals
                J_chem = np.array(J_chem)
                self.calculate_distance_matrix(n)
                #self.dis_mat is available
                #print(self.dis_mat)

                #MULTINICHE CLUSTERING K-MEANS ADAPTIVE                #
                
                J_clust_new = [(J_fit[p],p,J[p],False) for p in range(n)] 
                J_clust_new.sort() #Min to Max Sorting

                clusters = []
                #Initial K
                k = init_k

                self.cluster_centroids = [self.__agents[J_clust_new[i][1]] for i in range(init_k)]
                self.cluster_centroids_with_details = [(self.__agents[J_clust_new[i][1]],J_clust_new[i][1]) for i in range(init_k)]                
                clusters_assigned = [J_clust_new[i][1] for i in range(init_k)]+ [0 for i in range(init_k,n)]
                

                #Calculating Initial Centroids
                while True:
                    self.centdis_mat = np.zeros((k,k))
                    self.calculate_centroid_distance_matrix(n,k) #Calculate Centroid Matrix 
                    
                    #print(DataFrame(self.centdis_mat))    

                    #BUG Avoid i,i element              
                    #min_dis_loc = self.centdis_mat.argmin()
                    #Have to Rewrite the ArgMin and Avoid zero
                    min_x =0
                    min_y =0
                    min_val = sys.maxsize
                    for p in range(k):
                        for q in range(p+1,k):

                            if self.centdis_mat[p][q]<min_val:
                                min_x = p
                                min_y = q
                                min_val = self.centdis_mat[p][q] 

                    #min_dis_loc_raw = min_dis_loc//k
                    #min_dis_loc_col = min_dis_loc%k

                    min_dis_loc_raw = min_x
                    min_dis_loc_col = min_y

                    if self.centdis_mat[min_dis_loc_raw][min_dis_loc_col]<self.d_min:
                        #MERGE - NEED a FLAG to TRUE
                        tmp_cent = (self.cluster_centroids[min_dis_loc_raw] + self.cluster_centroids[min_dis_loc_col])/2
                        #OR CAN USE WEIGHTED ONE
                        self.cluster_centroids[min_dis_loc_raw] = tmp_cent
                        self.cluster_centroids.pop(min_dis_loc_col)
                        k=k-1
                    else:
                        #print('End Of Centroid Reduction')
                        break #END OF WHILE

                    if(k==1):break

                #END of WHILE
                print('Number of CLusters',k)
                #Seperation of POints into Clsuters
                for p in range(init_k,n):

                    element_index = J_clust_new[p][1]
                    [cluster_index,dis] = self.calculate_distance_between_point_and_centro(n,element_index)
                    #Check D-Max
                    if dis<self.d_max:
                        tmp_cent = (self.cluster_centroids[cluster_index] + self.__agents[element_index])/2
                        self.cluster_centroids[cluster_index] = tmp_cent
                    else:
                        self.cluster_centroids = self.cluster_centroids + [self.__agents[element_index]]
                        k=k+1

                    if(k>1):   #only when more than single cluster                     
                        k=self.check_and_merge_centroids(n,k) #check this

                #Now we have New Centroids
                print('Seperation Done')
                print(print(DataFrame(self.centdis_mat)))
                print(print(DataFrame(self.cluster_centroids)))
                #REassign of Every Indivisual
                print('Number of Centroids Remain',k)
                cluster_capacity = [0 for p in range(k)] #NC

                d_c = [0.0 for p in range(n)] #Distance between points and clusters
                for p in range(n):
                    [cluster_index,dis] = self.calculate_distance_between_point_and_centro(n,p)                    
                    d_c[p] = dis
                    clusters_assigned[p] = cluster_index
                    cluster_capacity[cluster_index] = cluster_capacity[cluster_index]+1
                #Reassigned Done
                print('Number of Clusters', k)
                print(clusters_assigned) 
                print(cluster_capacity)
                cluster_niche_count = [0 for p in range(n)]
                #Clsutering NICHe COunt- MAIN DIfference between Sharing
                for p in range(n):
                    #Calculate Distance between Centroid and the Point
                    c_index = clusters_assigned[p]
                    nc = cluster_capacity[c_index]
                    cluster_niche_count[p] = nc*(1 - (d_c[p]/(2*self.d_max))**self.alpha)/n

                #print('Cluster Niche Count', cluster_niche_count)
                #print('Individual Distance from their center',d_c)
                #MAIN POINT TO CONSIDER WHEN OPTIMIZATION IS A MIN Problem

                #J_shared = [(J_fit[p]/self.niche_count[p]) for p in range(n)]
                
                #J_shared = [(J_fit[p]*self.niche_count[p]/n) for p in range(n)]
                #J_shared = [(J_fit[p]/(n-self.niche_count[p])) for p in range(n)]
                
                """
                print('SHARED FITNESS')
                print(DataFrame(J_shared))
                print('_-----------------------------------------------------------___')
                """  

                #Dnamic Nich Sharng#####################################################
                J_shared_old = [(J_fit[p]*self.niche_count[p]) for p in range(n)]
                J_shared = [(J_fit[p]*self.niche_count[p]) for p in range(n)] #original sharing heuritic
                
                #J_shared = [(J_fit[p]*cluster_niche_count[p]) for p in range(n)]
                #J_shared = [(J_fit[p]*cluster_niche_count[p]) for p in range(n)]

                
                #k = 4
                #sigma = 0.2
                #for niche in range(k):
                #for niche in range(n):
                   

                J_health_new = [(J_shared[p],p,J[p],False) for p in range(n)]
                #J_health = [(sum(J_chem[:, i]), i) for i in range(n)]
                J_health = [(sum(J_chem[:, p]), p, J[p]) for p in range(n)]
                # print(J_health)
                # Sorting: Performance#1
                J_health.sort()
                J_health_new.sort()
                #ADD
                #J_health_new.reverse()

                # Dynamic Niche Identification - Numbr of niches #####
                
                p=0
                member_selected = []
                associated_niche = [0 for p in range(n)]
                associated_niche_peak = [0 for p in range(n)]
                niche_assigned = [False for p in range(n)]
                num_niches = 0
                niches = []
                niche_peak = []

                while p<n:
                    niche_peak_id = J_health_new[p][1]
                    niche_peak_j = J_health_new[p][0]
                    if(niche_assigned[niche_peak_id]==False):
                        #niche_peak_id = J_health_new[p][1]
                        niche_peak = niche_peak + [niche_peak_id]
                        niche_members = []
                        associated_niche[niche_peak_id]= num_niches
                        associated_niche_peak[niche_peak_id]= niche_peak_id
                        niche_assigned[niche_peak_id] = True
                        niche_members = niche_members + [niche_peak_id]
                        #num_niches = num_niches+1            
                        niche_member_count = 1
                        for q in range(n):
                            if self.shr_mat[niche_peak_id][q] > 0 and self.shr_mat[niche_peak_id][q]<1 and niche_assigned[q]==False:
                            #if self.clus_mat[niche_peak_id][q] > 0 and self.clus_mat[niche_peak_id][q]<1 and niche_assigned[q]==False:
                                #FOUND a Member of te Niche
                                niche_assigned[q] = True
                                associated_niche[q]= num_niches
                                associated_niche_peak[q]= niche_peak_id
                                niche_member_count = niche_member_count+1
                                niche_members = niche_members + [q]
                        niches = niches + [(num_niches, niche_peak_id, niche_member_count,niche_members)]
                        num_niches = num_niches+1
                    p = p+1    
                ########################################################
                print('Number of Niches Found', num_niches)
                self.__NumNiches.append([num_niches])
                print(niches)
                print('Associated Niche', associated_niche)
                print('Associated Niche Peak', associated_niche_peak)

                #"""
                #########################################################
                #Stochastic Universal Selection (SUS) for Reproduction
                #ERROR J_mean = np.sum(J_health_new, axis=0)/n
                
                N_select = n/4 #originally N/2

                J_all = np.sum([J_health_new[p][0] for p in range(n)])
                J_mean = J_all/n
                
                #gamma = random()
                gamma = np.random.uniform(0,(1/N_select))
                P = J_all/N_select

                delta = gamma*J_mean

                J_sum = J_health_new[0][0]
                count = 0
                J_select = [False for p in range(n)]
                q=0
                print('JAll', J_all)
                print('Gamma',gamma)
                print('Delta', delta)

                for p in range(n):
                    #print(J_health_new[p],associated_niche[J_health_new[p][1]],associated_niche_peak[J_health_new[p][1]])
                    print(J_health_new[p],J_shared_old[J_health_new[p][1]],self.niche_count[J_health_new[p][1]],cluster_niche_count[J_health_new[p][1]],associated_niche[J_health_new[p][1]],associated_niche_peak[J_health_new[p][1]],clusters_assigned[J_health_new[p][1]],cluster_capacity[clusters_assigned[J_health_new[p][1]]],d_c[J_health_new[p][1]])

                while True:
                    if (delta<J_sum):
                        #select individual j
                        #J_health_new[q][3]=True
                        selected_index = J_health_new[q][1]
                        print('selection', q, selected_index, associated_niche[selected_index],associated_niche_peak[selected_index] )
                        J_select[q] = True
                        count = count +1
                        #delta = delta + J_sum
                        delta = delta + P
                    else:
                        q = q+1
                        J_sum = J_sum + J_health_new[q][0]

                    if(q>= (n-1)):
                        break
                #####################################################
                if(count >= n):
                    print('ERROR: Selection Out of Bound')

                print(count,' selected')
                #print([(i,J_select[i]) for i in range(n)])

                #Choose Selected Individal
                ptr1 = (n-1)
                ptr2 = 0

                while ptr1 >= 0 and count>0 :
                    if(J_select[ptr1] == False):
                        replaced_index = J_health_new[ptr1][1] 
                        while True:
                            if J_select[ptr2] == True:
                                #Relacement from selected ndividual
                                selected_index = J_health_new[ptr2][1]                                
                                self.__agents[replaced_index] = self.__agents[selected_index] 
                                J[replaced_index] =  J[selected_index]
                                J_fit[replaced_index] = J_fit [selected_index]
                                J_select[ptr1]=True
                                count = count -1
                                ptr2 = ptr2+1
                                break
                            ptr2 = ptr2+1
                            #Termination Statement
                            if(ptr2 >= n):
                                print('End of Selection Loop', ptr1, ptr2)
                                ptr2 = 0
                                break
                            #Assumpton : Selection always > unslected

                    ptr1 = ptr1-1
                #print('After selection')
                #print([(i,J_select[i]) for i in range(n)])

                #"""
                #####################################################
                


                # TMP Reproduction
                #"""
                """
                alived_agents = []
                alived_fits = []
                #for i in J_health:

                for i in J_health_new:
                    alived_agents += [list(self.__agents[i[1]])]
                    alived_fits += [i[2]]

                ######################################################################
                print('Mutlti-Niche')
                print('Sharing Function Activated')
           		

                
                if n_is_even:
                    alived_agents = 2 * alived_agents[:n // 2]
                    self.__agents = np.array(alived_agents)
                    alived_fits = 2 * alived_fits[:n // 2]
                    J = np.array(alived_fits)

                else:
                    alived_agents = 2 * \
                        alived_agents[:n // 2] + [alived_agents[n // 2]]
                    self.__agents = np.array(alived_agents)
                    alived_fits = 2 * \
                        alived_fits[:n // 2] + [alived_fits[n // 2]]
                    J = np.array(alived_fits)
                """
                #"""
                # TMP
                ######################################################################
            

            #Elimiation and Dispersal
            """
            if l < Ned - 2:
                """"""
                J_ed = [(J[i], i) for i in range(n)]
                J_ed.sort()

                Ged = generation / beta  # beta=20
                Q = 1 - (0.5 * Ged * L)

                S_ed = Q * n

                if(S_ed < 0):
                    S_ed = 0

                sed = int(S_ed)
                print('Elimination and Dispersal')
                print(generation, Ged, Q, S_ed, sed)
                """"""
                s = 0
                if(arged == 'true'):
                    s = n - sed
                else:
                    s = 0

                for i in range(s, n):
                    # for i in range(n):
                    r = random()
                    # if r >= Ped_list[t]:
                    if r >= Ped:
                        if(arged == 'true'):
                            self.__agents[J_ed[i][1]] = np.random.uniform(
                                lb, ub, dimension)
                        else:
                            self.__agents[i] = np.random.uniform(
                                lb, ub, dimension)
                        J[i] = function(self.__agents[i])
                        if J[i] < J_best:
                            Gbest = self.__agents[i]
                            J_best = J[i]
                            self.gbesteval = self.evaluations
            
            #Elimination and Dispersal
            """

        #Memory Isage
        end_time = time.time()
        print('Processing Time')
        print("--- %s seconds ---" % (time.time() - start_time))

        print('Memeory Usage')
        #print(process.memory_info().rss)

        #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        #print(process.get_memory_info()[0])

        # END of BFOA
        print('Best Fitness', J_best)
        print('Solution', Gbest)
        print('GBestEval',self.gbesteval)
        f.close()

        Pbest = self.__agents[J.argmin()]
        Gbest = Pbest
        J_best = function(Gbest)

        print('Best Fitness', J_best)
        print('Solution', Gbest)

        Pbest_fit = self.__agents[J_fit.argmin()]
        Gbest_fit = Pbest_fit
        J_best_fit = function(Gbest_fit)

        print('Best Fitness (Raw)', J_best_fit)
        print('Solution', Gbest_fit)

        print('############################### Cluster Details ##################################')
        print('Number of Clusters', k)
        print(clusters_assigned) 
        print(cluster_capacity)
        print(DataFrame(self.centdis_mat))
        print(DataFrame(self.cluster_centroids))

        print('Cluster Center Values')
        for i in range(k):
            fit_val = function(self.cluster_centroids[i])
            self.cluster_centroids_fit += [fit_val]
            print(fit_val)
        print(DataFrame(self.cluster_centroids_fit))
        self.cluster_centroids_capacity = cluster_capacity
        print('Clyster Capacity', self.cluster_centroids_capacity)
        """
        J = np.array([function(x) for x in self.__agents])
        self._points(self.__agents)
        Pbest = self.__agents[J.argmin()]
        if function(Pbest) < function(Gbest):
            Gbest = Pbest
        self._set_Gbest(Gbest)
        """

    def _get_jfits(self):
        return self.__JFitList

    def _get_jcclist(self):
        return self.__JCCList

    def _get_jarlist(self):
        return self.__JARList

    def _get_jlist(self):
        return self.__JList

    def _get_jblist(self):
        return self.__JBList

    def _get_csteps(self):
        return self.__Steps

    def _get_avg_steps(self):
        return self.__AvgSteps

    def _get_avg_j(self):
        return self.__AvgJ

    def _get_avg_jfit(self):
        return self.__AvgJFit

    def _get_num_niches(self):
        return self.__NumNiches

    def _get_cluster_centroids(self):
        return self.cluster_centroids

    def _get_cluster_centroids_fit(self):
        return self.cluster_centroids_fit

    def _get_cluster_centroids_cap(self):
        return self.cluster_centroids_capacity  


        #J = np.array([function(x) for x in self.__agents])
        #Pbest = self.__agents[J.argmin()]
        #    if function(Pbest) < function(Gbest):
        #        Gbest = Pbest
        #        J_best = function(Gbest)
