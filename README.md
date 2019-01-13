# vaeedaq
Variational AutoEncoder with Population Queue and AVS
Copyright Sourodeep Bhattacharjee and Robin Gras, 2019.

To execute the code please enter the following from command prompt console or terminal:
  Python Main.py

The parameters of the algorithm are in Main.py and described as follows:

########################
PAREMETERS OF VAEEDA Q
########################
population = []
populationHistory = []
N = 1290               # Population Size 
n = 40                 # Problem Size
percentParent = 0.5    # Percentage of Parents to Keep in the next generation
maxIter = 2000         # Maximum Number of EDA iteration to continue for
tournamentSize = 2     # Size of Tournament
tournamentProb = 0.8   # Probability of picking best in the tournament
fitnessFunc = 1        # 0 - One Max #


headLess = 0           # Graphs are plotted when headless if off(0) , otherwise run in console
supressConsole = 0     # Verbose detailed output is shown when set to 0
NKLanscapeTest=0       # Turn this to 1 to test on predefined NK Landscapes
NKkvalue=6             # Value of K to use, N is taken from above N

########################
Fitness Function List
########################
0 - One Max
1 - Trap 5
2 - Trap 7
3 - Trap 9
4 - Trap 11
5 - Trap 13
