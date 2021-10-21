import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import multiprocessing
import pickle
from sklearn.datasets import make_blobs, make_moons, make_circles
from matplotlib import pyplot
from pandas import DataFrame

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP
from rpy2.robjects import IntVector, Formula
pandas2ri.activate()

cont = 0
bobj = 0.4
P = [12]
SCALES = [1]
tread = ""
ok = "0"
NGEN = 100
CXPB = 0.7
MUTPB = 0.2
INDPB = 0.05
POP = 100

n_instancias = 100
n_features = 3
centers = 1
metricas = ""
noise = 0.3

while ok == "0":
  print("Escolha que tipo de base deseja gerar:")
  print("Escolha 1 - Para bolhas de pontos com uma distribuição gaussiana.")
  print("Escolha 2 - Para gerar um padrão de redemoinho, ou duas luas.")
  print("Escolha 3 - Para gerar um problema de classificação com conjuntos de dados em círculos concêntricos.")

  dataset = input("Opção 1 - 2  - 3: ")

  n_instancias = input("Quantas instancias (Exemplos) deseja utilizar? ")
  n_features = input("Quantos atributos (features) deseja utilizar? ")

  if(dataset == "1"):
      centers = input("Quantas bolhas (centers) deseja utilizar?")
      print(type(centers))
      X, y = make_blobs(n_samples=int(n_instancias), centers=int(
          centers), n_features=int(n_features))
      if n_features == "2":
          df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
      else:
          df = DataFrame(dict(x=X[:, 0], y=X[:, 1], z=X[:, 2], label=y))
      # , 2:'green', 3:'orange', 4:'pink'}
      colors = {0: 'red', 1: 'blue', 2: 'orange'}
      fig, ax = pyplot.subplots()
      grouped = df.groupby('label')
      for key, group in grouped:
          group.plot(ax=ax, kind='scatter', x='x',
                      y='y', label=key, color=colors[key])
      print(X)
      print(y)
      pyplot.show()
      ok = input(
          "Esse é o dataset que deseja utilizar? 1 - sim / 0 - não ")
      ok = "1"

  if (dataset == "2"):
      noise = input("Quanto de ruido deseja utilizar? entre 0 e 1")
      X, y = make_moons(n_samples=int(n_instancias), noise=float(noise))
      # scatter plot, dots colored by class value
      df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
      colors = {0: 'red', 1: 'blue'}
      fig, ax = pyplot.subplots()
      grouped = df.groupby('label')
      for key, group in grouped:
          group.plot(ax=ax, kind='scatter', x='x',
                      y='y', label=key, color=colors[key])
      print(X)
      print(y)
      pyplot.show()
      ok = input(
          "Esse é o dataset que deseja utilizar? 1 - sim / 0 - não ")
      ok = "1"

  if (dataset == "3"):
      #noise = input("Quanto de ruido deseja utilizar? entre 0 e 1")
      X, y = make_circles(n_samples=int(
          n_instancias), noise=float(noise))
      # scatter plot, dots colored by class value
      df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
      colors = {0: 'red', 1: 'blue'}
      fig, ax = pyplot.subplots()
      grouped = df.groupby('label')

      for key, group in grouped:
          group.plot(ax=ax, kind='scatter', x='x',
                      y='y', label=key, color=colors[key])
      print(X)
      print(y)
      pyplot.show()
      ok = input(
          "Esse é o dataset que deseja utilizar? 1 - sim / 0 - não ")
      ok = "1"

filename = "Ferramenta"
print("Escolha quais métricas deseja otimizar (separe com espaço)")
print("Class imbalance C2 = 1")
print("Linearity L2 = 2")
print("Neighborhood N2 = 3")
print("Network ClsCoef = 4")
print("Dimensionality T2 = 5")
print("Feature-based F1 = 6")

metricas = input("Métrica: ")

metricasList = metricas.split()
N_ATTRIBUTES = int(n_instancias)
NOBJ = len(metricasList)

if ("1" in metricasList):
    objetivo = input(
        "Escolha os valores que deseja alcançar para: Class imbalance C2")
    globalBalance = float(objetivo)
if ("2" in metricasList):
    objetivo = input(
        "Escolha os valores que deseja alcançar para: Linearity L2")
    globalLinear = float(objetivo)
if ("3" in metricasList):
    objetivo = input(
        "Escolha os valores que deseja alcançar para: Neighborhood N2")
    globalN2 = float(objetivo)
if ("4" in metricasList):
    objetivo = input(
        "Escolha os valores que deseja alcançar para: Network ClsCoef")
    globalClsCoef = float(objetivo)
if ("5" in metricasList):
    objetivo = input(
        "Escolha os valores que deseja alcançar para: Dimensionality T2")
    globalt2 = float(objetivo)
if ("6" in metricasList):
    objetivo = input(
        "Escolha os valores que deseja alcançar para: Feature-based F1")
    globalf1 = float(objetivo)


dic = {}

# reference points
ref_points = [tools.uniform_reference_points(
    NOBJ, p, s) for p, s in zip(P, SCALES)]
ref_points = np.concatenate(ref_points)
_, uniques = np.unique(ref_points, axis=0, return_index=True)
ref_points = ref_points[uniques]


ecol = rpackages.importr('ECoL')


def my_evaluate(individual):
    vetor = []
    dataFrame['label'] = individual
    robjects.globalenv['dataFrame'] = dataFrame
    fmla = Formula('label ~ .')
    if("1" in metricasList):
        ##imbalance
        imbalanceVector = ecol.balance_formula(
            fmla, dataFrame, measures="C2", summary="return")
        imbalance = imbalanceVector.rx(1)
        vetor.append(abs(globalBalance - imbalance[0][0]))
    if ("2" in metricasList):
        ## -- linearity
        linearityVector = ecol.linearity_formula(
            fmla, dataFrame, measures="L2", summary="return")
        linearity = linearityVector.rx(1)
        vetor.append(abs(globalLinear - linearity[0][0]))
    if ("3" in metricasList):
        ## -- neighborhood N2
        n2Vector = ecol.neighborhood_formula(
            fmla, dataFrame, measures="N2", summary="return")
        n2 = n2Vector.rx(1)
        vetor.append(abs(globalN2 - n2[0][0]))
    if ("4" in metricasList):
        ## -- Network ClsCoef
        ClsCoefVector = ecol.network_formula(
            fmla, dataFrame, measures="ClsCoef", summary="return")
        ClsCoef = ClsCoefVector.rx(1)
        vetor.append(abs(globalClsCoef - ClsCoef[0][0]))
    if ("5" in metricasList):
        ## -- Dimensionality T2
        t2Vector = ecol.dimensionality_formula(
            fmla, dataFrame, measures="T2", summary="return")
        t2 = t2Vector.rx(1)
        vetor.append(abs(globalt2 - t2[0]))
    if ("6" in metricasList):
        ## -- Feature-based F1
        f1Vector = ecol.overlapping_formula(
            fmla, dataFrame, measures="F1", summary="return")
        f1 = f1Vector.rx(1)
        vetor.append(abs(globalf1 - f1[0][0]))
    ## --
    if(len(vetor) == 1):
        return vetor[0],
    if(len(vetor) == 2):
        return vetor[0], vetor[1],
    elif(len(vetor) == 3):
        return vetor[0], vetor[1], vetor[2],
    elif(len(vetor) == 4):
        return vetor[0], vetor[1], vetor[2], vetor[3],


def print_evaluate(individual):
    vetor = []
    dataFrame['label'] = individual
    robjects.globalenv['dataFrame'] = dataFrame
    fmla = Formula('label ~ .')
    if("1" in metricasList):
        ##imbalance
        imbalanceVector = ecol.balance_formula(
            fmla, dataFrame, measures="C2", summary="return")
        imbalance = imbalanceVector.rx(1)
        vetor.append(imbalance[0][0])
    if ("2" in metricasList):
        ## -- linearity
        linearityVector = ecol.linearity_formula(
            fmla, dataFrame, measures="L2", summary="return")
        linearity = linearityVector.rx(1)
        vetor.append(linearity[0][0])
    if ("3" in metricasList):
        ## -- neighborhood N2
        n2Vector = ecol.neighborhood_formula(
            fmla, dataFrame, measures="N2", summary="return")
        n2 = n2Vector.rx(1)
        vetor.append(n2[0][0])
    if ("4" in metricasList):
        ## -- Network ClsCoef
        ClsCoefVector = ecol.network_formula(
            fmla, dataFrame, measures="ClsCoef", summary="return")
        ClsCoef = ClsCoefVector.rx(1)
        vetor.append(ClsCoef[0][0])
    if ("5" in metricasList):
        ## -- Dimensionality T2
        t2Vector = ecol.dimensionality_formula(
            fmla, dataFrame, measures="T2", summary="return")
        t2 = t2Vector.rx(1)
        vetor.append(t2[0])
    if ("6" in metricasList):
        ## -- Feature-based F1
        f1Vector = ecol.overlapping_formula(
            fmla, dataFrame, measures="F1", summary="return")
        f1 = f1Vector.rx(1)
        vetor.append(f1[0][0])
    ## --
    if(len(vetor) == 1):
        return vetor[0],
    if(len(vetor) == 2):
        return vetor[0], vetor[1],
    elif(len(vetor) == 3):
        return vetor[0], vetor[1], vetor[2],
    elif(len(vetor) == 4):
        return vetor[0], vetor[1], vetor[2], vetor[3],


creator.create("FitnessMin", base.Fitness, weights=(-1.0,)*NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)

RANDINT_LOW = 0
RANDINT_UP = 1

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, RANDINT_LOW, RANDINT_UP)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_int, N_ATTRIBUTES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", my_evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=INDPB)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)


def main(seed=None):
    random.seed(64)
    pool = multiprocessing.Pool(processes=12)
    toolbox.register("map", pool.map)
    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(POP)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # Compile statistics about the population
    record = stats.compile(pop)

    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)
    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, POP)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
    return pop, logbook


if __name__ == '__main__':
    cont1 = 0
    cont0 = 0
    #dataFrame = pd.read_csv(str(N_ATTRIBUTES) + '.csv')
    #dataFrame = dataFrame.drop('c0', axis=1)
    dataFrame = df
    results = main()
    print("logbook")
    print(results[0][0])
    for x in range(len(results[0])):
        dic[print_evaluate(results[0][x])] = results[0][x]
        outfile = open(filename, 'wb')
        pickle.dump(dic, outfile)
        outfile.close()

    df['label'] = results[0][0]
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x',
                   y='y', label=key, color=colors[key])
    print(X)
    print(y)
    pyplot.show()
