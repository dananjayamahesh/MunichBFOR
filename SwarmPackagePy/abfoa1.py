import numpy as np
from random import random

from . import intelligence


class abfoa1(intelligence.sw):
    """
    Bacteria Foraging Optimization
    """
    def __init__(self, n, function, lb, ub, dimension, iteration, Nre=16, Ned=4, Nc=2, Ns=12, C=0.1, Ped=0.25, lamda=400):
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

        super(abfoa1, self).__init__()
        # Randomly populate the individuals in the intial population
        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        self._points(self.__agents)

        n_is_even = True
        if n & 1:
            n_is_even = False

        # J = np.array([function(x) for x in self.__agents])
        # Pbest = self.__agents[J.argmin()]
        # Gbest = Pbest
        #iteration = Ned * Nre

        #C_list = [C - C * 0.9 * i / iteration for i in range(iteration)]
        # C_list = [C for i in range(iteration)]
        #Ped_list = [Ped - Ped * 0.5 * i / iteration for i in range(iteration)]

        J = np.array([function(x) for x in self.__agents])
        self._points(self.__agents)
        Pbest = self.__agents[J.argmin()]
        Gbest = Pbest

        #J = [0 for i in range(n)]
        #J_last = J[::1]
        #Pbest = self.__agents[0]
        #Gbest = Pbest
        C_a = [C for i in range(n)]

        self.__Steps = []
        self.__JList = []
        #self.__Steps.append()

        for l in range(Ned):
            for k in range(Nre):
                J_chem = [J[::1]]

                J = np.array([function(x) for x in self.__agents])
                # self._points(self.__agents)
                Pbest = self.__agents[J.argmin()]
                if function(Pbest) < function(Gbest):
                    Gbest = Pbest

                #BUG J_last = J[::1]
                J_last = np.array(J)

                for j in range(Nc):

                    self._points(self.__agents)

                    for i in range(n):

                        # Can move inside
                        dell = np.random.uniform(-1, 1, dimension)
                        C_a[i] = 1 / (1 + (lamda / np.abs(J[i])))

                       
                        self.__agents[i] += C_a[i] * \
                            np.linalg.norm(dell) * dell
                        J[i] = function(self.__agents[i])

                         #Monitor
                        if(i==n/2):
                            self.__Steps.append(C_a[i])
                            self.__JList.append(J[i])

                        # Start Swim Steps
                        for m in range(Ns):

                            # bacteria moves only when objective function is reduced
                            if J[i] < J_last[i]:
                                J_last[i] = J[i]
                                C_a[i] = 1 / (1 + (lamda / np.abs(J[i])))

                                
                                self.__agents[i] += C_a[i] * np.linalg.norm(dell) \
                                    * dell
                                J[i] = function(self.__agents[i])

                                #Monitor
                                if(i==n/2):
                                    self.__Steps.append(C_a[i])
                                    self.__JList.append(J[i])

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

    def _get_csteps(self):
        return self.__Steps

    def _get_jlist(self):
        return self.__JList
