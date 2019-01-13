from Fitness import  getfit

def populationprnt(population):
    for elem in population:
        print(elem)


def populationfitnessprnt(population, fitnessFunc):
    for elem in population:
        fit = getfit[fitnessFunc](elem)
        print("Candidate :", elem, "Fitness :", fit)


def populationFitnessAverage(population, fitnessFunc):
    fit = 0
    for elem in population:
        fit = fit + getfit[fitnessFunc](elem)
    return fit/len(population)

def candidateActivation(candidate):
        newcandidate = []
        for vect in candidate:
            if vect > 0.5:
                temp = 1
            else:
                temp = 0
            newcandidate.append(temp)
        return newcandidate