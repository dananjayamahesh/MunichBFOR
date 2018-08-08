import numpy as np
from random import random

from . import intelligence


class z_bfoa_multipop(intelligence.sw):
    """
    Bacteria Foraging Optimization
    Implementation of the Paper "Multi-Bacterial Foraging Optimization for Dynamic Environments" by Daas
    """

    def cell_to_cell_function(self, agents, i, n, p, seg):
        #print('cell-to-cell interactions')
        S = int(n/seg)
        start = p*S
        end = start+S

        T = np.array(self.__agents[start:end])
        T_diff = (T - self.__agents[i])
        T_diff_sq = T_diff**2
        T_sum = np.sum(T_diff_sq, axis=1)
        T_sum_a = (-self.Wa) * T_sum
        T_sum_r = (-self.Wr) * T_sum
        T_sum_a_exp = np.exp(T_sum_a)
        T_sum_r_exp = np.exp(T_sum_r)
        T_sum_aa = -self.Da * T_sum_a_exp
        T_sum_rr = self.Hr * T_sum_r_exp
        J_cc = sum(T_sum_aa) + sum(T_sum_rr)
        #if(i == self.N / 2):
            #print('Jcc ',J_cc)
        return J_cc

    def __init__(self, n, function, lb, ub, dimension, iteration, Nre=16, Ned=4, Nc=2, Ns=12, C=0.1, Ped=0.25, Da=0.1, Wa=0.2, Hr=0.1, Wr=10, lamda=400, seg=2, R=15):
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

        super(z_bfoa_multipop, self).__init__()
        # Randomly populate the individuals in the intial population
        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        self._points(self.__agents)

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
        J_cc = np.array([0.0 for x in self.__agents])
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

        S = int(n/seg) #Population Density per Segmentation

        S_is_even = True
        if S & 1:
            S_is_even = False

        JPopBest= np.array([J[i] for i in range(seg)])
        GPopBest = np.array([self.__agents[i] for i in range(seg)])
        
        ###############################################################
        for p in range(seg):
            start = p*S
            end = start + S
            J_pop = J[start:end]
            b = J_pop.argmin()
            print(b)
            a = self.__agents[(start + b)]
            GPopBest[p] = a
            JPopBest[p] = J[start + b]

        print(J)
        print(GPopBest)
        print('JBestCell', JPopBest) 
        print('Segments',seg,' S is ',S)

        ###############################################################

        for l in range(Ned):

            for k in range(Nre):

                J_chem = [J[::1]]
                #BUG J_last = J[::1]
                J_last = np.array(J)
                J_fit = np.array(J)

                for j in range(Nc): 

                    # p - Population Index   
                    for p in range(seg):  

                        start = p*S
                        end = start + S

                        for i in range(start, end): #Number of Bacteria for a Unit Segment
                            #i = p*S + q
                            # Can move inside
                            dell = np.random.uniform(-1, 1, dimension)
                            C_a[i] = 1 / (1 + (lamda / np.abs(J[i])))
                            self.__agents[i] += C_a[i] * \
                                np.linalg.norm(dell) * dell
    
                            J_fit[i] = function(self.__agents[i])
                            J_cc[i] = self.cell_to_cell_function(self.__agents, i, n, p, seg)
                            J[i] = J_fit[i] + J_cc[i]
    
                            if J[i] < J_best:
                                Gbest = self.__agents[i]
                                J_best = J[i]

                            #################################################################
                            if(J[i]<JPopBest[p]):
                                JPopBest[p] = J[i]
                                GPopBest[p] = self.__agents[i]
                            #################################################################

                               
                           # Monitoring
                            if(i == n / 2):
                                self.__JFitList.append(J_fit[i])
                                self.__JCCList.append(J_cc[i])
                                self.__JList.append(J[i])
                                self.__JBList.append(J_best)
                                self.__Steps.append(C_a[i])
                                #print(C_a[i], J_fit[i], J_cc[i], J[i], (J_fit[i] + J_cc[i]), J_best)
                            # Start Swim Steps
                            for m in range(Ns):
    
                                # bacteria moves only when objective function is reduced
                                if J[i] < J_last[i]:
                                    J_last[i] = J[i]
                                    C_a[i] = 1 / (1 + (lamda / np.abs(J[i])))
                                    self.__agents[i] += C_a[i] * np.linalg.norm(dell) \
                                        * dell
                                    J_fit[i] = function(self.__agents[i])
                                    J_cc[i] = self.cell_to_cell_function(self.__agents, i, n, p, seg)
                                    J[i] = J_fit[i] + J_cc[i]
    
                                    if J[i] < J_best:
                                        Gbest = self.__agents[i]
                                        J_best = J[i]

                                    #################################################################
                                    if(J[i] < JPopBest[p]):
                                        JPopBest[p] = J[i]
                                        GPopBest[p] = self.__agents[i]
                                    #################################################################
    
                                    # Probing
                                    if(i == n / 2):
                                        self.__JFitList.append(J_fit[i])
                                        self.__JCCList.append(J_cc[i])
                                        self.__JList.append(J[i])
                                        self.__JBList.append(J_best)
                                        self.__Steps.append(C_a[i])
                                        #print(C_a[i], J_fit[i], J_cc[i], J[i], (J_fit[i] + J_cc[i])), J_best
                                    #J[i] = function(self.__agents[i])+ self.cell_to_cell_function(self.__agents, i)
                                else:
                                    break
                        #ENDOFPOPULATION

                    #ENDOFPOPS
                    self._points(self.__agents)  # Add to the animation
                    # End of Chemotaxis
                    # Make lgorithm faster
                    #J = np.array([function(x) for x in self.__agents])
                    J_chem += [J]

                # Ending Chemotaxix Steps of all individuals
                J_chem = np.array(J_chem)

                for p in range(seg): #PER POPULATION

                    start = p*S
                    end = start + S
                    #J_health = [(sum(J_chem[:, i]), i) for i in range(n)]
                    J_health = [(sum(J_chem[:, i]), i, J[i]) for i in range(start, end)]
                    # print(J_health)
                    # Sorting: Performance#1
                    J_health.sort()
                    alived_agents = []
                    alived_fits = []
                    for i in J_health:
                        alived_agents += [list(self.__agents[i[1]])]
                        alived_fits += [i[2]]

                    #print(alived_agents)
                    #print(alived_fits)
    
                    if S_is_even:
                        alived_agents = 2 * alived_agents[:S // 2]
                        self.__agents[start:end] = np.array(alived_agents)
                        alived_fits = 2 * alived_fits[:S // 2]
                        #print(alived_fits)
                        J[start:end] = np.array(alived_fits)
    
                    else:
                        alived_agents = 2 * alived_agents[:S // 2] + [alived_agents[S // 2]]
                        #print('Alived Agents', alived_agents)                    
                        self.__agents[start:end] = np.array(alived_agents)
                        alived_fits = 2 * alived_fits[:S // 2] + [alived_fits[S // 2]]
                        #print('Alived Fits',alived_fits)
                        J[start:end] = np.array(alived_fits)

            #END REPRODUCTION FOR ALL POPULATIONS
            distance_matrix = [[0 for x in range(seg)] for y in range(seg)]
            pop_valid = ['true' for i in range(seg)]

            for p in range(seg):
                print('Exclusion Check')
                for q in range(seg):
                    distance_matrix[p][q] = np.sqrt(sum([(GPopBest[p][d]- GPopBest[q][d])**2 for d in range(dimension)])) #calculate_distance()
            print('Distance Matrix: ', distance_matrix)

            for p in range(seg):
                for q in range(p+1,seg):
                    if(distance_matrix[p][q] < R):
                        #MARK INVALID POPULATION
                        if(JPopBest[p] < JPopBest[q]):
                            pop_valid[q] = 'false'
                        elif(JPopBest[p] > JPopBest[q]):
                            pop_valid[p] = 'false'
                        else:
                            print('No Need to Eliminate')

            print(pop_valid)
            
            for p in range(seg): 
                start = p*S
                end = start + S
                
                if(pop_valid[p]=='false'):
                    print('Reevaluate')
                    if l < Ned - 2:
                        for i in range(start, end):
                            #r = random()
                            # if r >= Ped_list[t]:
                            #if r >= Ped:
                            self.__agents[i] = np.random.uniform(lb, ub, dimension)
                            J[i] = function(self.__agents[i])
                            if J[i] < J_best:
                                Gbest = self.__agents[i]
                                J_best = J[i]
                                
                            #################################################################
                            if(J[i] < JPopBest[p]):
                                JPopBest[p] = J[i]
                                GPopBest[p] = self.__agents[i]
                            #################################################################
            

        #END OF FBFOA
        print('Best Fitness',J_best)
        print('Solution', Gbest)
        print('JPopBest',JPopBest)
        print('GPopBest', GPopBest)

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

        #J = np.array([function(x) for x in self.__agents])
        #Pbest = self.__agents[J.argmin()]
        #    if function(Pbest) < function(Gbest):
        #        Gbest = Pbest
        #        J_best = function(Gbest)
