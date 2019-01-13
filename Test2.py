import random
from sklearn.neural_network import MLPClassifier

def populationprnt():
    for elem in population:
        print(elem)

def evalpop():
    for elem in population:
        fit = 0
        for vect in elem:
            fit = fit + vect
        print(elem, " Fitness", fit)


def evalcand( elem ):
    fit = 0
    for vect in elem:
        fit = fit + vect
    return  fit

N = 10  # Population Size
n = 20  # Vector size

# Init pop
population = []
candidate = []
random.seed(9001)
for j in range(0, N):
    candidate = []
    for i in range(0, n):
        candidate.append(random.randint(0, 1))
    population.append(candidate)
#


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

clf.fit(population, population)

for elem in population:
    var = clf.predict([elem])
    fit = evalcand(elem)
    print("Act :", elem, "Fit :", fit)
    fit = evalcand(var[0])
    print("Pre :", var[0].tolist() , "Fit :", fit)
    for k in range(len(elem)*2):
        print("-", end='-')
    print()



