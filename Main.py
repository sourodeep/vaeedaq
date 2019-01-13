import tensorflow as tf
import random
import Assist

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import Fitness
from time import time
import numpy as np
from keras import optimizers
import os
import json
from keras.models import Sequential
from keras import regularizers
from keras.utils import plot_model
from keras.layers import Dense, Activation, Input
from keras.callbacks import TensorBoard
from keras.models import Model
from IPython.display import SVG
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')
#<editor-fold desc="Setup">

population = []
populationHistory = []
N = 1290     # Population Size 1300 1290  No Help: 2580 2000 3500
n = 40             # Vector size
percentParent = 0.5
maxIter = 2000    # 1290
tournamentSize = 4
tournamentProb = 0.8
fitnessFunc = 1     # 0 - One Max

encoding_dim = 200
modelFit_Epoch = 15     # increase
modeFit_BatchSize = 64

startRec = 1
stopRec = 1   # maxIter

skipGen = 10  # Generations between training

headLess = 0        # Sexy Graphs will be plotted when not head less
supressConsole = 0  # Silence is Golden !!
consoleHead = 0     # Plain Jane Graphs will be plotted when console head is on
NKLanscapeTest=0
NKkvalue=6

#</editor-fold>

#<editor-fold desc="Initialize popopulation">

random.seed(time())
for j in range(0, N):
    candidate = []
    for i in range(0, n):
        # candidate.append(random.randint(0, 1))
        newrand = int(random.sample(range(0, 2), 1)[0])
        candidate.append(newrand)
    population.append(candidate)

#</editor-fold>

# <editor-fold desc="Setup Graphs">
if headLess == 0:

    convGraph = plt.figure(figsize=(8, 6))
    # plt.title("AEEDA Convergence Graph", fontsize=8)
    plt.ylabel('Fitness')
    plt.xlabel('Iteration')

    plt.axis([0, maxIter, 0, n+10])
    red_patch = mpatches.Patch(color='red', label='Average Fitness')
    green_patch = mpatches.Patch(color='green', label='Best Individual Fitness')
    plt.legend(handles=[red_patch, green_patch])
    plt.ion()

if consoleHead == 1:
    grid = []
    for y in range(-(n), 1):
        if y != 0:
            if y % 5 == 0:
                grid.append(str(y * -1).ljust(int(maxIter/10)))
            else:
                grid.append("|".ljust(int(maxIter / 10)))
        else:
            grid.append("+".ljust(int(maxIter/10), "-"))

    # for row in grid:
    #     print(row)
    #
    #
    #
    # input("Press Enter to continue...")
# </editor-fold>

## Test predictor

#Fitness.predictFitness(160, 1)
#input("Press Enter to continue...")
## End of Test predictor


# <editor-fold desc="VAE Model">

# <editor-fold desc="Moved VAE Definition">

sgd = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.006);
opti = "Nadam"  # 0.006
batch_size = 64
original_dim = n
intermediate_dim = 20  # Worked: 20**   // No Help: 40 100 30
latent_dim = 20  # Worked: 25**             // No Help:  15 60
epochs = 15
epsilon_std = 1  # 1.0          0.8 ^

x = Input(shape=(original_dim,))
g = Dense(intermediate_dim, activation='relu')(x)
h = Dense(10, activation='relu')(g)   # Keep this layer - decreasing makes it dumb
# k = Dense(10, activation='relu')(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')

h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

autoencoder = Model(x, x_decoded_mean)
# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
autoencoder.add_loss(vae_loss)

autoencoder.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# </editor-fold>
# autoencoder.summary()

# </editor-fold>



# </editor-fold>

for gen in range(1, maxIter):
    if startRec <= gen < stopRec:
        fh = open("G:/Dropbox/Dropbox/Uni/NNEDA Result/Dec25/Details9/"
                  + Fitness.getfitName[fitnessFunc]() +
                  "Size" + str(n) + "Pop" + str(N) + "Iter" + str(iter)
                  + "Results.txt", "a")
        fh.write("\n \n")
        otherdetails = "PROBLEM :" + str(Fitness.getfitName[fitnessFunc]()) + " Problem Size:" + str(n) \
                       + " Population: " \
                       + str(N) + "\n" \
                       + " \nModel Build Epoch: " + str(modelFit_Epoch) \
                       + "   Model Build Batch Size: " + str(modeFit_BatchSize)
        fh.write(str(otherdetails))
        fh.write("\n \n")
        hidden_neurons = "Hidden Nodes: " + str(encoding_dim) + "\n"
        fh.write(hidden_neurons)

    if supressConsole == 0:
        print("ITERATION : \n", gen)

    #<editor-fold desc="Selection based on Fitness">

    if supressConsole == 0:
        print("##################################################")
        print("########           SELECTION STEP       ##########")
        print("##################################################")

        print("########           Before      ##########")
    #Assist.populationfitnessprnt(population, fitnessFunc)
    prunedPop = []

    pool = []
    while len(prunedPop) < N/2:
            if gen % 10 == 0:
                prunedPop.append(Fitness.tournamentprob(population, tournamentSize, N - 1, fitnessFunc))
            else:
                prunedPop.append(Fitness.tournamentprob(population, tournamentSize, N - 1, fitnessFunc))

    #prunedPop = Fitness.selTournament(population, round(N/2), tournamentSize, fitnessFunc )
    #prunedPop = Fitness.selRoulette(population, round(N/2), fitnessFunc )

    #input("Press Enter to continue...")
    if supressConsole == 0:
        print("########           After      ##########")
    population = prunedPop
    # population = population[:int(N * percentParent)]
    #Assist.populationfitnessprnt(population, fitnessFunc)

    if supressConsole == 0:
        print("##################################################")
        print("########       END OF SELECTION STEP     #########")
        print("##################################################")
        print()
        print()

    #</editor-fold>


    #<editor-fold desc="Train MLP">
    if supressConsole == 0:
        print("##################################################")
        print("########           TRAIN AUTO-ENCODER   ##########")
        print("##################################################")



    #x_train = np.unique(populationnp, axis=0)
    # print(len(populationnp))
    # print(len(x_train))
    # input("Press Enter to continue...")
    if gen == 1 or (gen % skipGen) == 0:
        if gen > 1:
            populationnp = np.array(Assist.samplePopulationHistory(populationHistory))
            x_train = populationnp
        elif gen == 1:
            populationnp = np.array(population)
            x_train = populationnp
        varnp = autoencoder.fit(x_train, x_train, verbose=0)
        populationHistory = []
    elif gen == 1 or (gen % skipGen) != 0:
        populationHistory.append(population)



    # if gen == 10:
    #     Fitness.predictFitness(n, 1, populationnp)

    #G:/Dropbox/Dropbox/Uni/auto

    #print(varnp.history.keys())


    if startRec <= gen < stopRec:
        fh.write("================================================================================================= \n")
        fh.write("================================================================================================= \n")
        iterDetails = "AEEDA Iteration :  " + str(gen) + "\n"
        fh.write(iterDetails)
        fh.write("================================================================================================= \n")
        optiDetails = "OPTIMIZER :" + str(opti) + str(sgd.get_config())
        fh.write(json.dumps(optiDetails))
        fh.write(str("\n"))
        accuracy = "\nAccuracy: " + str(varnp.history['acc'][modelFit_Epoch-1]) + "\n\n"
        loss = "Loss: " + str(varnp.history['loss'][modelFit_Epoch-1]) + "\n\n"
        val_accuracy = "Validation Accuracy: " + str(varnp.history['val_acc'][modelFit_Epoch-1]) + "\n\n"
        val_loss = "Validation Loss: " + str(varnp.history['val_loss'][modelFit_Epoch-1]) + "\n\n"
        fh.write(accuracy)
        fh.write(loss)
        fh.write(val_accuracy)
        fh.write(val_loss)

    if headLess == 0:
        plt.suptitle("PROBLEM :" + str(Fitness.getfitName[fitnessFunc]()) + " Problem Size:" + str(n) + " Population: "
                     + str(N) + "\n"
                     + "OPTIMIZER :" + str(opti) + str(sgd.get_config()) + "\n"
                     # + "MODEL :" + str(model.get_config()) + "\n"
                     + " \n Model Build Epoch: " + str(modelFit_Epoch)
                     + "   Model Build Batch Size: " + str(modeFit_BatchSize), fontsize=6)

    #var = list(varnp)
    #print("Verbose Train:", varnp)



    # modelDetails = autoencoder.get_config()
    # fh.write(json.dumps(modelDetails))
    # fh.write(str("\n"))


    #score = autoencoder.evaluate(populationnp, populationnp, batch_size=32)
    # print("\n Test Score:", score)


    # if startRec <= gen < stopRec:
    #     fh.write(str("Accuracy  Loss \n"))
    #     fh.write(str(score))
    #     fh.write(str("\n \n \n"))


    # input("Press Enter to continue...")
    if supressConsole == 0:
        print("##################################################")
        print("########      END OF TRAIN AUTO ENCODER ##########")
        print("##################################################")
        print()
        print()
    #</editor-fold>

    # <editor-fold desc="Test MLP">
    if supressConsole == 0:
        print("##################################################")
        print("########           TEST AUTO-ENCODER    ##########")
        print("##################################################")

    populationTemp = []
    winner = []

    if startRec <= gen < stopRec:
        fh.write(str("\n"))

    for elem in population:

        var = list(autoencoder.predict(np.array([elem]).astype('float32'), verbose=0))
        # print("var =", var)
        var2 = Assist.candidateActivation(var[0].tolist())
        # print("var2 =", var2)
        # var2 = var[0].tolist()
        populationTemp.append(var2)



        fit = Fitness.getfit[fitnessFunc](elem)
        if supressConsole == 0:
            print("Parent :", elem, "Fit :", fit)

        if startRec <= gen < stopRec:
            parent = "Parent :" + str(elem) + "Fit :" + str(fit) + "\n"
            fh.write(str(parent))



        fit = Fitness.getfit[fitnessFunc](var2)
        if supressConsole == 0:
            print("Child  :", var2, "Fit :", fit)

        child = "Child  :" + str(var2) + "Fit :" + str(fit) + "\n"


        if startRec <= gen < stopRec:
            fh.write(str(child))
            fh.write(str("\n"))

        # if gen > stopRec:
        #     fh.close()

        # replace following later with stopping criterion

        # for k in range(len(elem)*2):
        #     print("-", end='-')
        # print()

        del var2  # Clear Memory

    if supressConsole == 0:
        print("##################################################")
        print("########    END OF TEST AUTO-ENCODER    ##########")
        print("##################################################")

        print()
        print()

        print("##################################################")
        print("########           PARENT + OFFSPRING   ##########")
        print("##################################################")

    population.extend(populationTemp)
    if supressConsole == 0:
        Assist.populationfitnessprnt(population, fitnessFunc)

    if supressConsole == 0:
        print("##################################################")
        print("########   END OF  PARENT + OFFSPRING   ##########")
        print("##################################################")
        print()
        print()

    # <editor-fold desc="Plot Convergence">
    if headLess == 0:
        best = sorted(population, key=Fitness.getfit[fitnessFunc], reverse=True)[0]
        fitBest = Fitness.getfit[fitnessFunc](best)
        fitAvg = Assist.populationFitnessAverage(population, fitnessFunc)
        fitAvgProximity = Assist.populationFitnessAverage(population, 0)
        plt.plot(gen, fitBest, 'g.')
        if fitBest > 36:
            print("BEST > 36: ", best)
            print("BEST Fitness: ", fitBest)
            #input("Press Enter to continue...")

        plt.plot(gen, fitAvg, 'r.')
        plt.plot(gen, fitAvgProximity, 'b.')
        plt.pause(0.07)

    if consoleHead == 1 and gen % 10 == 0:
        # for x in range(1, 10):
        #     value = int(x)  # get the function result as int, you can call any other function of course
        #     # if -10 <= value <= 10:  # no reason to plot outside of our grid range
        #     x = x + int(maxIter / 2)  # normalize x to the grid
        #     y = n - value  # normalize y to the grid
        #     grid_line = grid[y]
        #     grid[y] = grid_line[:x] + "o" + grid_line[x + 1:]
        best = sorted(population, key=Fitness.getfit[fitnessFunc], reverse=True)[0]

        fitAvg = Assist.populationFitnessAverage(population, fitnessFunc)
        fitBest = Fitness.getfit[fitnessFunc](best)
        fitAvgProximity = Assist.populationFitnessAverage(population, 0)


        value = int(fitAvg)  # get the function result as int, you can call any other function of course
        valuebest = int(fitBest)
        valueprox = int(fitAvgProximity)
        # if -10 <= value <= 10:  # no reason to plot outside of our grid range
        gen = gen  # normalize x to the grid
        y = n - value  # normalize y to the grid
        yBest = n - valuebest
        yProx = n - valueprox

        grid_line = grid[y]
        grid[y] = grid_line[:int(gen/10)] + "@" + grid_line[int(gen/10) + 1:]

        grid_linebest = grid[yBest]
        grid[yBest] = grid_linebest[:int(gen/10)] + "+" + grid_linebest[int(gen/10) + 1:]

        grid_lineprox = grid[yProx]
        grid[yProx] = grid_lineprox[:int(gen / 10)] + "." + grid_lineprox[int(gen / 10) + 1:]

        clear()
        for row in grid:
            print(row)

    # convGraph.savefig('G:/Dropbox/Dropbox/Uni/NNEDA Result/Dec22/convGraph' + str(gen) + '.png')
    #fig = convGraph.savefig('G:/Dropbox/Dropbox/Uni/NNEDA Result/Dec22/convGraph' + str(gen)+'.png')
    #plt.close(fig)
    # </editor-fold>

    if startRec <= gen < stopRec:
        bestDetails = "Best in Solution, " + str(best) + " fitness, " + str(Fitness.getfit[fitnessFunc](best))
        fh.write(bestDetails)
        fh.close()
    #</editor-fold>

#<editor-fold desc="End Game">
best = sorted(population, key=Fitness.getfit[fitnessFunc], reverse=True)[0]
print("Best in Solution, ", best, " fitness, ", Fitness.getfit[fitnessFunc](best))
# fh.close()
input("Press Enter to continue...")
#</editor-fold>

print("##################################################")
print("###########              END               #######")
print("##################################################")




