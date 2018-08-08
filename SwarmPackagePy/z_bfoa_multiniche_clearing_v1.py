import numpy as np
from random import random
import math
from pandas import *
import sys

from . import intelligence


class z_bfoa_multiniche_clearing_v1(intelligence.sw):
    """

    

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

            T_clus_shr = 1-(T_dis/self.sigma_cluster)
            T_clus_shr_clip = T_clus_shr.clip(0)
            self.clus_mat[i] = np.array(T_clus_shr_clip)


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

    def __init__(self, n, function, lb, ub, dimension, iteration, Nre=16, Ned=4, Nc=2, Ns=12, C=0.1, Ped=0.25, Da=0.1, Wa=0.2, Hr=0.1, Wr=10, lamda=400, L=0.03, arga='none', argj='none', arged='false',sigma=1):
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

        super(z_bfoa_multiniche_clearing_v1, self).__init__()

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
        self.sigma_cluster = 1.5#0.5 #0.2

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

                #print(self.dis_mat)
                self.calculate_distance_matrix(n)

                print(np.matrix(self.dis_mat))
                """
                #Print Sharing Details
                print('_-----------------------------------------------------------___')
                #print(self.dis_mat)
                print('STAT')
                print('DISTANCE MATRIX')
                print(np.matrix(self.dis_mat))
                print(DataFrame(self.dis_mat))
                print('SIMILARITY MATRIX')
                #print(np.matrix(self.shr_mat))
                print(DataFrame(self.shr_mat))
                print('NICHE COUNT')
                print(DataFrame(self.niche_count))
                """

                #f.write('GENERATION')
                #np.savetxt(f,[generation])
                np.savetxt(f,self.dis_mat)
                np.savetxt(f,self.shr_mat)
                np.savetxt(f,self.clus_mat)
                #np.savetxt(f,self.niche_count)


                #MAIN POINT TO CONSIDER WHEN OPTIMIZATION IS A MIN Problem

                #J_shared = [(J_fit[p]/self.niche_count[p]) for p in range(n)]
                
                #J_shared = [(J_fit[p]*self.niche_count[p]/n) for p in range(n)]
                #J_shared = [(J_fit[p]/(n-self.niche_count[p])) for p in range(n)]
                

                J_shared = [(J_fit[p]*self.niche_count[p]) for p in range(n)]

                """
                print('SHARED FITNESS')
                print(DataFrame(J_shared))
                print('_-----------------------------------------------------------___')
                """
                ##############################################################
                #CLEARING PROCEDURE
                MAXVAL = sys.maxsize   
                #kappa =10 is now working           
                kappa = 2 #Capacity of the niche
                nbWinners = 0 #Number of Winners according to Elitist Selection
                
                J_clear = np.zeros((1,n))
                J_clear = np.array(J_fit)

                J_fit_clear = [(J_clear[p],p,J[p],False) for p in range(n)]
                J_fit_clear.sort()
                #BE Careful When the OPtimization problem is Min or Max
                for i in range(n):
                    nbWinners = 1;
                    index = J_fit_clear[i][1]
                    if(J_clear[index]<MAXVAL):
                        for j in range (0,n):
                            if((self.dis_mat[index][j] < self.sigma_share) and J_clear[j]<MAXVAL ):
                                if(nbWinners<kappa):
                                    nbWinners = nbWinners + 1
                                else:
                                    J_clear[j] = MAXVAL 
                
                ##############################################################

                #Dnamic Nich Sharng
                
                k = 4
                #sigma = 0.2
                #for niche in range(k):
                #for niche in range(n):

                #IFDEF THIS LABEL MIN-OP                
                #J_shared = [(self.niche_count[p]/J_fit[p]) for p in range(n)]   

                J_health_new = [(J_shared[p],p,J[p],False) for p in range(n)]
                J_health_new = [(J_clear[p],p,J[p],False) for p in range(n)]

                #J_health = [(sum(J_chem[:, i]), i) for i in range(n)]
                J_health = [(sum(J_chem[:, p]), p, J[p]) for p in range(n)]
                J_health.sort()
                # print(J_health)
                # Sorting: Performance#1
                
                J_health_new.sort()

                #THEN THIS LABEL MIN-OP
                #J_health_new.reverse()
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
                            #if self.shr_mat[niche_peak_id][q] > 0 and self.shr_mat[niche_peak_id][q]<1 and niche_assigned[q]==False:
                            if self.clus_mat[niche_peak_id][q] > 0 and self.clus_mat[niche_peak_id][q]<1 and niche_assigned[q]==False:
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
                N_select = n/4
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
                    print(J_health_new[p],associated_niche[J_health_new[p][1]],associated_niche_peak[J_health_new[p][1]])
                
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
                print([(i,J_select[i]) for i in range(n)])

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
        # END of BFOA
        print('Best Fitness', J_best)
        print('Solution', Gbest)
        print('GBestEval',self.gbesteval)
        f.close() 
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

        #J = np.array([function(x) for x in self.__agents])
        #Pbest = self.__agents[J.argmin()]
        #    if function(Pbest) < function(Gbest):
        #        Gbest = Pbest
        #        J_best = function(Gbest)
