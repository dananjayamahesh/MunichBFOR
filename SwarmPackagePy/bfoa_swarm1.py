import numpy as np
from random import random

from . import intelligence


class bfoa_swarm1(intelligence.sw):
    """
    Bacteria Foraging Optimization
    """
    """
    def cell_to_cell_function(self,agents, i):
        print('cell-to-cell interactions')

        T = np.array(self.__agents)
        print('T')
        print(T)

        T_diff = (T - T[i])
        print('T Difference')
        print(T_diff)

        T_diff_sq = T_diff**2
        print('T Difference Squared')
        print(T_diff_sq)

        T_sum = np.sum(T_diff_sq, axis=1)
        print('T diff square Sum')
        print(T_sum)

        T_sum_a = (-self.Wa) * T_sum
        print('a sum inside')
        print(T_sum_a)

        T_sum_r = (-self.Wr) * T_sum
        print('r sum inside')
        print(T_sum_r)

        T_sum_a_exp = np.exp(T_sum_a)
        print('a sum inside exp')
        print(T_sum_a_exp)

        T_sum_r_exp = np.exp(T_sum_r)
        print('a sum inside exp')
        print(T_sum_r_exp)

        T_sum_aa = -self.Da * T_sum_a_exp
        print('a sum of exp')
        print(T_sum_aa)
#
        T_sum_rr = self.Hr * T_sum_r_exp
        print('r sum of exp')
        print(T_sum_rr)
        J_cc = sum(T_sum_aa) + sum(T_sum_rr)
        print (J_cc)
        #J_cost = J_t + J_cc
        return J_cc
    """
    def cell_to_cell_function(self,agents, i):
        #print('cell-to-cell interactions')

        T = np.array(self.__agents)
        T_diff = (T - T[i])
        T_diff_sq = T_diff**2
        T_sum = np.sum(T_diff_sq, axis=1)
        T_sum_a = (-self.Wa) * T_sum
        T_sum_r = (-self.Wr) * T_sum
        T_sum_a_exp = np.exp(T_sum_a)
        T_sum_r_exp = np.exp(T_sum_r)
        T_sum_aa = -self.Da * T_sum_a_exp
        T_sum_rr = self.Hr * T_sum_r_exp
        J_cc = sum(T_sum_aa) + sum(T_sum_rr)
        #print (J_cc)
        #J_cost = J_t + J_cc
        return J_cc


    def __init__(self, n, function, lb, ub, dimension, iteration, Nre=16, Ned=4, Nc=2, Ns=12, C=0.1, Ped=0.25, Da=0.1, Wa=0.2, Hr=0.1, Wr=10, lamda=400):
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

        super(bfoa_swarm1, self).__init__()
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

        # J = np.array([function(x) for x in self.__agents])
        # Pbest = self.__agents[J.argmin()]
        # Gbest = Pbest
        #iteration = Ned * Nre

        #C_list = [C - C * 0.9 * i / iteration for i in range(iteration)]
        # C_list = [C for i in range(iteration)]
        #Ped_list = [Ped - Ped * 0.5 * i / iteration for i in range(iteration)]

        #Just for the initialization purposes
        J = np.array([function(x) for x in self.__agents])
        self._points(self.__agents)
        Pbest = self.__agents[J.argmin()]
        Gbest = Pbest

        #J = [0 for i in range(n)]
        #J_last = J[::1]
        #Pbest = self.__agents[0]
        #Gbest = Pbest
        C_a = [C for i in range(n)]
        #self.cell_to_cell_function(self.__agents,1)

        for l in range(Ned):
            for k in range(Nre):
                J_chem = [J[::1]]

                J = np.array([function(x) for x in self.__agents])
                #print(J)
                #self._points(self.__agents)
                Pbest = self.__agents[J.argmin()]
                if function(Pbest) < function(Gbest):
                    Gbest = Pbest
                J_last = J[::1]

                for j in range(Nc):

                    self._points(self.__agents)

                    for i in range(n):

                        # Can move inside
                        dell = np.random.uniform(-1, 1, dimension)
                        self.__agents[i] += C_a[i] * \
                            np.linalg.norm(dell) * dell

                        J[i] = function(self.__agents[i]) + self.cell_to_cell_function(self.__agents, i)
                        # Start Swim Steps
                        for m in range(Ns):

                            # bacteria moves only when objective function is reduced
                            if J[i] < J_last[i]:
                                J_last[i] = J[i]
                                self.__agents[i] += C_a[i] * np.linalg.norm(dell) \
                                    * dell
                                J[i] = function(self.__agents[i])+ self.cell_to_cell_function(self.__agents, i)
                            else:
                                break
                                # This is not the original algorithm
                                #dell = np.random.uniform(-1, 1, dimension)
                                #self.__agents[i] += C_a[i] * np.linalg.norm(dell) * dell

                    # Make lgorithm faster
                    #J = np.array([function(x) for x in self.__agents])
                    J_chem += [J]

                # Ending Chemotaxix Steps of all individuals

                J_chem = np.array(J_chem)

                J_health = [(sum(J_chem[:, i]), i) for i in range(n)]

                # Sorting: Performance#1
                J_health.sort()
                alived_agents = []
                for i in J_health:
                    alived_agents += [list(self.__agents[i[1]])]

                if n_is_even:
                    alived_agents = 2 * alived_agents[:n // 2]
                    self.__agents = np.array(alived_agents)
                else:
                    alived_agents = 2 * \
                        alived_agents[:n // 2] + [alived_agents[n // 2]]
                    self.__agents = np.array(alived_agents)

            if l < Ned - 2:
                for i in range(n):
                    r = random()
                    # if r >= Ped_list[t]:
                    if r >= Ped:
                        self.__agents[i] = np.random.uniform(lb, ub, dimension)

        J = np.array([function(x) for x in self.__agents])
        self._points(self.__agents)
        Pbest = self.__agents[J.argmin()]
        if function(Pbest) < function(Gbest):
            Gbest = Pbest
        self._set_Gbest(Gbest)
