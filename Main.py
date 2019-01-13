
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

#<editor-fold desc="Setup">

population = []
N = 1290            # Population Size
n = 40              # Vector size
percentParent = 0.5
maxIter = 600
tournamentSize = 2
tournamentProb = 0.8
fitnessFunc = 1     # 0 - One Max

encoding_dim = 200
modelFit_Epoch = 20     # increase
modeFit_BatchSize = 64





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







for iter in range(1, maxIter):
   
    print("ITERATION : \n", iter)


    #<editor-fold desc="Selection based on Fitness">

    print("##################################################")
    print("########           SELECTION STEP       ##########")
    print("##################################################")

    print("########           Before      ##########")
    Assist.populationfitnessprnt(population, fitnessFunc)
    prunedPop = []

    pool = []
    while len(prunedPop) < N/2:
        prunedPop.append(Fitness.tournamentprob(population, tournamentSize, N-1, fitnessFunc))

    # input("Press Enter to continue...")
    print("########           After      ##########")
    population = prunedPop
    # population = population[:int(N * percentParent)]
    Assist.populationfitnessprnt(population, fitnessFunc)

    print("##################################################")
    print("########       END OF SELECTION STEP     #########")
    print("##################################################")
    print()
    print()

    #</editor-fold>

    # <editor-fold desc="Setup Seq Model (Input shape and Compilation)">

    # <editor-fold desc="Old Model">

    # model = Sequential()
    # model.add(Dense(20, activation='relu',  input_dim=n, activity_regularizer=regularizers.l1(10e-5)))
    # model.add(Dense(n, activation='sigmoid'))

    # </editor-fold>
    opti = ""
    #sgd = optimizers.SGD(lr=0.01, momentum=0.7, decay=0.4, nesterov=False);    opti = "SGD"
    # sgd = optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0); opti = "Adam"
    # sgd = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0); opti = "RMSProp"
    # sgd = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0); opti = "AdaDelta"
    sgd = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.006); opti = "Nadam"  # 0.006
    # <editor-fold desc="Autoecoder Model">
    # this is our input placeholder
    input_img = Input(shape=(n,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(n, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    # </editor-fold>


    autoencoder.compile(optimizer='sgd',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])





    # </editor-fold>

    #<editor-fold desc="Train MLP">
    print("##################################################")
    print("########           TRAIN AUTO-ENCODER   ##########")
    print("##################################################")

    populationnp = np.array(population)
    #varnp = model.fit(populationnp, populationnp)

    
    varnp = autoencoder.fit(populationnp, populationnp, epochs=modelFit_Epoch, batch_size=modeFit_BatchSize,
                            shuffle=True, validation_data=(populationnp, populationnp),
                            verbose=2)


    score = autoencoder.evaluate(populationnp, populationnp, batch_size=32)

    # if startRec <= iter < stopRec:
    #     fh.write(str("Accuracy  Loss \n"))
    #     fh.write(str(score))
    #     fh.write(str("\n \n \n"))

    print("\n Test Score:", score)
    # input("Press Enter to continue...")
    print("##################################################")
    print("########      END OF TRAIN AUTO ENCODER ##########")
    print("##################################################")
    print()
    print()
    #</editor-fold>

    # <editor-fold desc="Test MLP">

    print("##################################################")
    print("########           TEST AUTO-ENCODER    ##########")
    print("##################################################")

    populationTemp = []
    winner = []

   

    for elem in population:

        var = list(autoencoder.predict(np.array([elem]), verbose=1))
        # print("var =", var)
        var2 = Assist.candidateActivation(var[0].tolist())
        # print("var2 =", var2)
        # var2 = var[0].tolist()
        populationTemp.append(var2)



        fit = Fitness.getfit[fitnessFunc](elem)
        print("Parent :", elem, "Fit :", fit)

     

        fit = Fitness.getfit[fitnessFunc](var2)
        print("Child  :", var2, "Fit :", fit)

        child = "Child  :" + str(var2) + "Fit :" + str(fit) + "\n"

      
        # if iter > stopRec:
        #     fh.close()

        # replace following later with stopping criterion

        for k in range(len(elem)*2):
            print("-", end='-')
        print()

        del var2  # Clear Memory

        # if len(winner) > 0:
        #     print("Fit ones at iter:", iter)
        #     Assist.populationfitnessprnt(winner, fitnessFunc)
        #     # input("Press Enter to continue...")
        # winner.clear()



    print("##################################################")
    print("########    END OF TEST AUTO-ENCODER    ##########")
    print("##################################################")

    print()
    print()

    print("##################################################")
    print("########           PARENT + OFFSPRING   ##########")
    print("##################################################")

    population.extend(populationTemp)
    Assist.populationfitnessprnt(population, fitnessFunc)

    print("##################################################")
    print("########   END OF  PARENT + OFFSPRING   ##########")
    print("##################################################")
    print()
    print()

    # <editor-fold desc="Plot Convergence">
    best = sorted(population, key=Fitness.getfit[fitnessFunc], reverse=True)[0]
    fitBest = Fitness.getfit[fitnessFunc](best)
    fitAvg = Assist.populationFitnessAverage(population, fitnessFunc)
  



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




