import random
import Assist


# <editor-fold desc="Fitness Function Definitions">
def evalcandonemax(elem):
    fit = 0
    for vect in elem:
        fit = fit + vect
    return fit


def evalcandTrap5(elem):
    fit = 0
    i = 0
    while i < len(elem):
        u = elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        if u < 5:
            fit += 4 - u
        else:
            fit += 5

    return fit


def evalcandTrap5Inv(elem):
    fit = 0
    i = 0
    while i < len(elem):
        u = elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        if u > 0:
            fit += u
        else:
            fit += 5

    return fit


def evalcandTrap7(elem):
    fit = 0
    i = 0
    while i < len(elem):
        u = elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        u += elem[i]
        i += 1

        if u < 7:
            fit += 6 - u
        else:
            fit += 7

    return fit
# </editor-fold>


# <editor-fold desc=" Get Fitness Function Names">
def evalcandonemaxName():
    return "One Max"

def evalcandTrap5Name():
    return "Trap 5"

def evalcandTrap7Name():
    return "Trap 7"

def evalcandTrap5InvName():
    return "Inverted Trap 5"
# </editor-fold>

getfit = {0: evalcandonemax,
          1: evalcandTrap5,
          2: evalcandTrap7,
          3: evalcandTrap5Inv,
          }

getfitName = {0: evalcandonemaxName,
              1: evalcandTrap5Name,
              2: evalcandTrap7Name,
              3: evalcandTrap5InvName,
              }


# <editor-fold desc="Assistive Functions">
def evalpop(population):
    for elem in population:
        fit = 0
        for vect in elem:
            fit = fit + vect
        print(elem, " Fitness", fit)

def tournament(population, k, n, fitnessfunc):
    best = []
    for i in range(1, k+1):
        ind = population[random.randint(1, n)]
        if (len(best) == 0) or getfit[fitnessfunc](ind) > getfit[fitnessfunc](best):
            best = ind
    return best


def tournamentprob(population, k, n, fitnessFunc):
    # print("#####TOUR######")
    best = []
    final = []
    for i in range(1, k+1):
        ind = population[random.randint(1, n)]
        best.append(ind)

    sorted(best, key=getfit[fitnessFunc], reverse=True)[0]

    toss = random.randint(1, 100)
    # print("toss val ", toss)
    #
    # print("\n")
    #
    # print("best0 ", best[0])
    # print(" Fitness ", getfit[fitnessFunc](best[0]))
    # print("\n")
    # print("best1 ", best[1])
    # print(" Fitness ", getfit[fitnessFunc](best[1]))
    # print("\n")

    if toss < 90:
        final = best[0]
    else:
        final = best[1]

    #print("final ", final)
    #input("Press Enter to continue...")
    return final
# </editor-fold>
