"""
SwarmPackagePy
==============


Provides:
    1. Swarm optimization algorithms.
    2. Test functions for swarm algorithms.
    3. Animation of algorithm computation.


To compute any algorithm you need to create an object like below
----------------------------------------------------------------

  >>> alh = SwarmPackagePy.alg(n, function, lb, ub, dimension, iteration)

Where:
    alg:        name of required algorithm
    n:          number of agents
    function:   test function (function must be an object, that accepts
                               coordinates of a point as init parameter)
    lb:         lower limits for plot axes
    ub:         upper limits for plot axes
    dimension:  space dimension
    iteration:  number of iterations

!Note: almost every algorithm has additional parameters. But they are all
have default value. To view additional parameters of any algorithm type:

  >>> help(SwarmPackagePy.alg)


Example of computation pso algorithm for 2 dimensions with easom function
-------------------------------------------------------------------------

  >>> function = SwarmPackagePy.testFunctions.easom_function
  >>> alh = SwarmPackagePy.pso(15, function, -10, 10, 2, 20)

To get the best position of an algorithm use "get_Gbest()" method:

  >>> alh.get_Gbest()

To watch animation for algorithm:

  >>> animation(alh.get_agents(), function, 10, -10, sr=False)


3D version
----------

    >>> function = SwarmPackagePy.testFunctions.easom_function
    >>> alh = SwarmPackagePy.pso(15, function, -10, 10, 3, 20)
    >>> animation3D(alh.get_agents(), function, 10, -10, sr=False)

Avaible test functions (in SwarmPackagePy.testFunctions)
--------------------------------------------------------

    ackley_function
    bukin_function
    cross_in_tray_function
    sphere_function
    bohachevsky_function
    sum_squares_function
    sum_of_different_powers_function
    booth_function
    matyas_function
    mccormick_function
    dixon_price_function
    six_hump_camel_function
    three_hump_camel_function
    easom_function
    michalewicz_function
    beale_function
    drop_wave_function
"""

from SwarmPackagePy.aba import aba
from SwarmPackagePy.ba import ba
from SwarmPackagePy.bfo import bfo
from SwarmPackagePy.classic_bfo import classic_bfo
from SwarmPackagePy.bfoa import bfoa
from SwarmPackagePy.bfoa_swarm1 import bfoa_swarm1
from SwarmPackagePy.bfoa_swarm2 import bfoa_swarm2
from SwarmPackagePy.abfoa1 import abfoa1
from SwarmPackagePy.abfoa1_swarm1 import abfoa1_swarm1
from SwarmPackagePy.abfoa1_swarm2 import abfoa1_swarm2
from SwarmPackagePy.abfoa2 import abfoa2
from SwarmPackagePy.abfoa2_swarm1 import abfoa2_swarm1
from SwarmPackagePy.abfoa2_swarm2 import abfoa2_swarm2
from SwarmPackagePy.abfo1 import abfo1
from SwarmPackagePy.abfo2 import abfo2
from SwarmPackagePy.bfo_with_swarm import bfo_with_swarm
from SwarmPackagePy.abfo1_with_swarm import abfo1_with_swarm
from SwarmPackagePy.abfo2_with_swarm import abfo2_with_swarm
from SwarmPackagePy.bfo_with_env_swarm import bfo_with_env_swarm
from SwarmPackagePy.abfo1_with_env_swarm import abfo1_with_env_swarm
from SwarmPackagePy.abfo2_with_env_swarm import abfo2_with_env_swarm
from SwarmPackagePy.chso import chso
from SwarmPackagePy.cso import cso
from SwarmPackagePy.fa import fa
from SwarmPackagePy.fwa import fwa
from SwarmPackagePy.gsa import gsa
from SwarmPackagePy.gwo import gwo
from SwarmPackagePy.pso import pso
from SwarmPackagePy.ca import ca
from SwarmPackagePy.hs import hs
from SwarmPackagePy.ssa import ssa
from SwarmPackagePy.wsa import wsa
from SwarmPackagePy.animation import animation, animation3D,animation1D,test_function_shape
from SwarmPackagePy.bfoa_swarm1_dev1_rep import bfoa_swarm1_dev1_rep
from SwarmPackagePy.z_bfoa_swarm1_dev1 import z_bfoa_swarm1_dev1
from SwarmPackagePy.z_ibfoa_jun_li import z_ibfoa_jun_li
from SwarmPackagePy.z_ibfoa_jun_li_2 import z_ibfoa_jun_li_2
from SwarmPackagePy.z_bfoa import z_bfoa
from SwarmPackagePy.z_bfoa_extended import z_bfoa_extended
from SwarmPackagePy.z_bfoa_multipop import z_bfoa_multipop
from SwarmPackagePy.z_bfoa_multiniche import z_bfoa_multiniche
from SwarmPackagePy.animation_multipop import animation_multipop, animation3D_multipop
from SwarmPackagePy.z_bfoa_multiniche_sharing import z_bfoa_multiniche_sharing
from SwarmPackagePy.z_bfoa_multiniche_sharing_v2 import z_bfoa_multiniche_sharing_v2
from SwarmPackagePy.z_bfoa_multiniche_sharing_v3 import z_bfoa_multiniche_sharing_v3
from SwarmPackagePy.z_bfoa_multiniche_sharing_v4 import z_bfoa_multiniche_sharing_v4
from SwarmPackagePy.z_bfoa_multiniche_sharing_v4_raw_for_debug import z_bfoa_multiniche_sharing_v4_raw_for_debug
from SwarmPackagePy.z_bfoa_multiniche_sharing_v4_chi import z_bfoa_multiniche_sharing_v4_chi
from SwarmPackagePy.z_bfoa_multiniche_sharing_v4_chi_test import z_bfoa_multiniche_sharing_v4_chi_test
from SwarmPackagePy.z_bfoa_multiniche_clearing_v1 import z_bfoa_multiniche_clearing_v1
from SwarmPackagePy.z_bfoa_multiniche_clustering_v1 import z_bfoa_multiniche_clustering_v1
from SwarmPackagePy.z_bfoa_multiniche_clustering_v2 import z_bfoa_multiniche_clustering_v2
from SwarmPackagePy.z_bfoa_general_v1 import z_bfoa_general_v1
from SwarmPackagePy.z_bfoa_general_v1_max import z_bfoa_general_v1_max
from SwarmPackagePy.revenue_optimization_function import revenue_optimization_function
#from SwarmPackagePy.NewFeatures.bfoa_swarm1_dev1 import bfoa_swarm1_dev1
#from SwarmPackagePy.NewFeatures import NewFeatures
_version_ = '1.0.0'
