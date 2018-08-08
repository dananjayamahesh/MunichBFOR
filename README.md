
# MunichBFOR
**MunichBFOR** is a Library of evolutionary optimization algorithms based on Bacterial Foraging. It includes a class of optimization algorithms and each can be used for solving specific optimization problem. You can find the principles they operate on and pseudo codes  below.<br>

Provides:<br>
- BFO optimization algorithms.
- Test functions for BFO algorithms.
- Animation of minimum find process.<br>
<br>Every algorithm has arguments listed below:<br>
- **n**: number of agents
- **function**: test function
- **lb**: lower limits for plot axes
- **ub**: upper limits for plot axes
- **dimension**: space dimension
- **iteration**: number of iterations<br>
<br>Every algorithm has methods listed below:<br>
- **get_agents()**: returns a history of all agents of the algorithm
- **get_Gbest()**: returns the best position of algorithm<br>
<br>If an algorithm accepts some additional arguments or methods they will be described in its "Arguments" or "Methods" section.

For all questions and suggestions contact dananjayamahesh@gmail.com:<br>
* Authors - Mahesh Dananjaya (dananjayamahesh@gmail.com)

## Table of contents
* [Installation](#installation)<br>
* [Bacterial Foraging Optimization](#bacterial-foraging-optimization)<br>
* [Multi Niche Bacterial Foraging Optimization - Cluster Based](#bacterial-foraging-optimization)<br>
* [Multi Niche Bacterial Foraging Optimization - Sharing Based](#bacterial-foraging-optimization)<br>
* [Adanced Multi Niche Bacterial Foraging Optimization](#bacterial-foraging-optimization)<br>
* [Tests](#tests)<br>
* [Animation](#animation)<br>

### Installation
#### Requirements
- python (version >= 3.5 if you are going to run tests; for all other actions you can use python any version)<br>
- numpy<br>
- pytest (if you are going to run tests)<br> 
- matplotlib (if you are going to watch animation)<br> 
- pandas (if you are going to watch 3D animation)<br> 
#### Installation
You can install **SwarmPackagePy** from PyPI repositories using pip. Command bellow will do this:
```
pip install SwarmPackagePy
```
Or you can just clone this repository and at the main folder execute command:
```
cd SwarmPackagePy/
python setup.py install
```

### Bacterial Foraging Optimization
#### Description
The **Bacterial Foraging Optimization**, proposed by Passino is inspired by the social foraging behavior of Escherichia coli (next E.coli).<br>
During foraging of the real bacteria, locomotion is achieved by a set of tensile flagella.
Flagella help an E.coli bacterium to tumble or swim, which are two basic operations performed by a bacterium at the
time of foraging. When they rotate the flagella in the clockwise direction, each flagellum pulls
on the cell. That results in the moving of flagella independently and finally the bacterium tumbles with
lesser number of tumbling whereas in a harmful place it tumbles frequently to find a nutrient gradient.
Moving the flagella in the counterclockwise direction helps the bacterium to swim at a very fast rate.
In the above-mentioned algorithm the bacteria undergoes chemotaxis, where they like to move towards
a nutrient gradient and avoid noxious environment. Generally the bacteria move for a longer distance
in a friendly environment.<br>
When they get food in sufficient, they are increased in length and in presence of suitable temperature
they break in the middle to from an exact replica of itself. This phenomenon inspired Passino to
introduce an event of reproduction in BFO. Due to the occurrence of sudden environmental changes
or attack, the chemotactic progress may be destroyed and a group of bacteria may move to some other
places or some other may be introduced in the swarm of concern. This constitutes the event of
elimination-dispersal in the real bacterial population, where all the bacteria in a region are killed or a
group is dispersed into a new part of the environment.<br>
Bacterial Foraging Optimization has three main steps:
* Chemotaxis
* Reproduction
* Elimination and Dispersal
#### Mathematical model
_Chemotaxis:_ This process simulates the movement of an E.coli cell through swimming and
tumbling via flagella. Biologically an E.coli bacterium can move in two different ways. It can
swim for a period of time in the same direction or it may tumble, and alternate between these
two modes of operation for the entire lifetime.
_Reproduction:_ The least healthy bacteria eventually die while each of the healthier bacteria (those
yielding lower value of the objective function) asexually split into two bacteria, which are then
placed in the same location. This keeps the swarm size constant.
_Elimination and Dispersal:_ Gradual or sudden changes in the local environment where a
bacterium population lives may occur due to various reasons e.g. a significant local rise of
temperature may kill a group of bacteria that are currently in a region with a high concentration of
nutrient gradients. Events can take place in such a fashion that all the bacteria in a region are killed
or a group is dispersed into a new location. To simulate this phenomenon in BFO some bacteria
are liquidated at random with a very small probability while the new replacements are randomly
initialized over the search space.
#### Algorithm
<pre>
BEGIN
&nbsp;Initialize randomly the bacteria foraging optimization population
&nbsp;Calculate the fitness of each agent
&nbsp;Set global best agent to best agent
&nbsp;FOR number of iterations
&nbsp;&nbsp;FOR number of chemotactic steps
&nbsp;&nbsp;&nbsp;FOR each search agent
&nbsp;&nbsp;&nbsp;&nbsp;Move agent to the random direction
&nbsp;&nbsp;&nbsp;&nbsp;Calculate the fitness of the moved agent
&nbsp;&nbsp;&nbsp;&nbsp;FOR swimming length
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IF current fitness is better than previous
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Move agent to the same direction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ELSE
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Move agent to the random direction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;END IF
&nbsp;&nbsp;&nbsp;&nbsp;END FOR
&nbsp;&nbsp;&nbsp;END FOR
&nbsp;&nbsp;&nbsp;Calculate the fitness of each agent
&nbsp;&nbsp;END FOR
&nbsp;&nbsp;Compute and sort sum of fitness function of all chemotactic loops (health of agent)
&nbsp;&nbsp;Let live and split only half of the population according to their health
&nbsp;&nbsp;IF not the last iteration
&nbsp;&nbsp;&nbsp;FOR each search agent
&nbsp;&nbsp;&nbsp;&nbsp;With some probability replace agent with new random generated
&nbsp;&nbsp;&nbsp;END FOR
&nbsp;&nbsp;END IF
&nbsp;&nbsp;Update the best search agent
&nbsp;Calculate the fitness of each agent
END
</pre>
#### Arguments
The bfo method accepts the following arguments:<br>
**param Nc** number of chemotactic steps<br>
**param Ns** swimming length<br>
**param C** the size of step taken in the random direction specified by the tumble<br>
**param Ped** elimination-dispersal probability<br>
#### Method invocation
The method can be invoked by passing the arguments in the following order:
```
SwarmPackagePy.bfo(n, function, lb, ub, dimension, iteration, Nc, Ns, C, Ped)
```


### Tests
All algorithms were tested with different test functions. In fact, you can run tests for all the algorithms on your own. All you need to do is to open terminal (console) and insert the following line:
```
pytest -v —tb=line test.py
```
Every algorithm is tested with the following set of test functions:<br>
- Ackley function
- Bukin function
- Cross in tray function
- Sphere function
- Bohachevsky function
- Sum squares function
- Sum of different powers function
- Booth function
- Matyas function
- McCormick function
- Dixon price function
- Six hump camel function
- Three hump camel function
- Easom function
- Michalewicz function
- Beale function
- drop wave function
- Revenue Optimization Algorithms

### Animation
There are 2D animation and 3D animation of search process. The general way to start it is (example for pso algorithm):<br>
#### 2D animation
```
# Compute the algorithm
function = SwarmPackagePy.testFunctions.easom_function
alh = SwarmPackagePy.pso(15, function, -10, 10, 2, 20)
# Show animation
animation(alh.get_agents(), function, 10, -10)
```
#### 3D animation
```
# Compute the algorithm
function = SwarmPackagePy.testFunctions.easom_function
alh = SwarmPackagePy.pso(15, function, -10, 10, 2, 20)
# Show animation
animation3D(alh.get_agents(), function, 10, -10)
```
#### Save the animation
You can also save the animation of the search process. To do this, add as an animation argument sr=True. The example of saving animation:
```
# Compute the algorithm
function = SwarmPackagePy.testFunctions.easom_function
alh = SwarmPackagePy.pso(15, function, -10, 10, 2, 20)
# Show animation
animation(alh.get_agents(), function, 10, -10, sr=True)
```
