import numpy as np
from random import random

from . import intelligence


class abfo1_with_swarm(intelligence.sw):
    """
    Bacteria Foraging Optimization with Swarming Effect
    """

    def __init__(self, n, function, lb, ub, dimension, iteration,
                 Nc=2, Ns=12, C=0.2, Ped=1.15, da=0.1, wa=0.2, hr=0.1, wr=10):
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
        :da: Depth of the atrractant (default value is 1.15)
        :wa: Width/Rate of the atrractant (default value is 1.15)
        :hr: Height of the repellent (default value is 1.15)
    :wr: Width/Rate of the repellent (default value is 1.15)
        """

        super(bfo_with_swarm, self).__init__()

        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        self._points(self.__agents)

        n_is_even = True
        if n & 1:
            n_is_even = False

        J = np.array([function(x) for x in self.__agents])
        print(J.shape)
        print(J)
        Pbest = self.__agents[J.argmin()]
        Gbest = Pbest

        C_list = [C - C * 0.9 * i / iteration for i in range(iteration)]
        Ped_list = [Ped - Ped * 0.5 * i / iteration for i in range(iteration)]

        J_last = J[::1]

        for t in range(iteration):

            J_chem = [J[::1]]

            for j in range(Nc):
                for i in range(n):
                    dell = np.random.uniform(-1, 1, dimension)
                    self.__agents[i] += C_list[t] * np.linalg.norm(dell) * dell

                    for m in range(Ns):
                        J_t = function(self.__agents[i])
                        T = np.array(self.__agents)
                        T_diff = (T - T[i])
                        T_diff_sq = T_diff**2
                        T_sum = np.sum(T_diff_sq, axis=1)
                        T_sum_a = (-wa) * T_sum
                        T_sum_r = (-wr) * T_sum
                        T_sum_a_exp = np.exp(T_sum_a)
                        T_sum_r_exp = np.exp(T_sum_r)
                        T_sum_a = -da * T_sum_a_exp
                        T_sum_r = hr * T_sum_r_exp
                        J_cc = sum(T_sum_a) + sum(T_sum_r)
                        J_cost = J_t + J_cc
                        if J_cost < J_last[i]:
                            # J_last[i] = J[i]
                            J_last[i] = J_cost
                            self.__agents[i] += C_list[t] * \
                                np.linalg.norm(dell) * dell
                        else:
                            dell = np.random.uniform(-1, 1, dimension)
                            self.__agents[i] += C_list[t] * \
                                np.linalg.norm(dell) * dell
                    #print('Inside')
                    J_t = function(self.__agents[i])
                    T = np.array(self.__agents)
                    T_diff = (T - T[i])
                    T_diff_sq = T_diff**2
                    T_sum = np.sum(T_diff_sq, axis=1)
                    T_sum_a = (-wa) * T_sum
                    T_sum_r = (-wr) * T_sum
                    T_sum_a_exp = np.exp(T_sum_a)
                    T_sum_r_exp = np.exp(T_sum_r)
                    T_sum_a = -da * T_sum_a_exp
                    T_sum_r = hr * T_sum_r_exp
                    J_cc = sum(T_sum_a) + sum(T_sum_r)
                    J_cost = J_t + J_cc
                    J[i] = J_cost

                # J = np.array([function(x) for x in self.__agents])
                J_chem += [J]

            J_chem = np.array(J_chem)

            J_health = [(sum(J_chem[:, i]), i) for i in range(n)]
            J_health.sort()
            alived_agents = []
            for i in J_health:
                alived_agents += [list(self.__agents[i[1]])]

            if n_is_even:
                alived_agents = 2 * alived_agents[:n // 2]
                self.__agents = np.array(alived_agents)
            else:
                alived_agents = 2 * alived_agents[:n // 2] +\
                    [alived_agents[n // 2]]
                self.__agents = np.array(alived_agents)

            if t < iteration - 2:
                for i in range(n):
                    r = random()
                    if r >= Ped_list[t]:
                        self.__agents[i] = np.random.uniform(lb, ub, dimension)

            J = np.array([function(x) for x in self.__agents])
            self._points(self.__agents)

            Pbest = self.__agents[J.argmin()]
            if function(Pbest) < function(Gbest):
                Gbest = Pbest

        self._set_Gbest(Gbest)
