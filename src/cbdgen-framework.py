import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import multiprocessing
import pickle
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.datasets import load_iris
from matplotlib import pyplot
from pandas import DataFrame

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import rpy2.robjects as robjects

import complexity as complx
import generate

cont = 0
bobj = 0.4
P = [12]
SCALES = [1]
tread = ""
select_new_dataset = "N"
NGEN = 1000
CXPB = 0.7
MUTPB = 0.2
INDPB = 0.05
POP = 100

n_instancias = 100
n_features = 3
centers = 1
metricas = ""
noise = 0.3

while select_new_dataset == "N":
    print("Escolha que tipo de base deseja gerar:")
    print("Escolha 1 - Para bolhas de pontos com uma distribuição gaussiana.")
    print("Escolha 2 - Para gerar um padrão de redemoinho, ou duas luas.")
    print("Escolha 3 - Para gerar um problema de classificação com conjuntos de dados em círculos concêntricos.")

    dataset = input("Opção 1 - 2  - 3: ")

    n_instancias = int(input("Quantas instancias (Exemplos) deseja utilizar? "))
    n_features = int(input("Quantos atributos (features) deseja utilizar? "))

    if(dataset == "1"):
        centers = int(input("Quantas bolhas (centers) deseja utilizar?"))
        df = generate.blobs(n_instancias, centers, n_features)
    if (dataset == "2"):
        noise = input("Quanto de ruido deseja utilizar? entre 0 e 1")
        df = generate.moons(n_instancias, noise)
    if (dataset == "3"):
        # noise = input("Quanto de ruido deseja utilizar? entre 0 e 1")
        df = generate.circles(n_instancias, noise)
    if (dataset == "4"):
        df = generate.classification(n_instancias, n_features)

    print(df.head)
    ax1 = df.plot.scatter(x=0, y=1, c='Blue')
    pyplot.show()
    select_new_dataset = 'N' if input("Esse é o dataset que deseja utilizar? (y/N)") != 'y' else 'y'

filename = "NGEN=" + str(NGEN)

print("Você deseja basear as métricas a um dataset já existente? (y/N)")
escolha = input()

print("Escolha quais métricas deseja otimizar (separe com espaço)")
print("Class imbalance C2 = 1")
print("Linearity L2 = 2")
print("Neighborhood N2 = 3")
print("Network ClsCoef = 4")
print("Dimensionality T2 = 5")
print("Feature-based F1 = 6")

metricas = input("Métrica: ")

metricasList = metricas.split()

if (escolha == 'y'):
    base_dataset = load_iris()
    base_df = pd.DataFrame(data=np.c_[base_dataset['data'], base_dataset['target']], columns=base_dataset['feature_names'] + ['target'])
    target = "target"

    if ("1" in metricasList):
        globalBalance = complx.balance(base_df, target, "C2")
        filename += "-C2"
    if ("2" in metricasList):
        globalLinear = complx.linearity(base_df, target, "L2")
        print(globalLinear)
        filename += "-L2"
    if ("3" in metricasList):
        globalN2 = complx.neighborhood(base_df, target, "N2")
        filename += "-N2"
    if ("4" in metricasList):
        globalClsCoef = complx.network(base_df, target, "ClsCoef")
        print(globalClsCoef)
        filename += "-CLSCOEF"
    if ("5" in metricasList):
        globalt2 = complx.dimensionality(base_df, target, "T2")
        print(globalt2)
        filename += "-T2"
    if ("6" in metricasList):
        globalf1 = complx.feature(base_df, target, "F1")
        filename += "-F1"
else:
    if ("1" in metricasList):
        objetivo = input(
            "Escolha os valores que deseja alcançar para: Class imbalance C2")
        globalBalance = float(objetivo)
        filename += "-C2"
    if ("2" in metricasList):
        objetivo = input(
            "Escolha os valores que deseja alcançar para: Linearity L2")
        globalLinear = float(objetivo)
        filename += "-L2"
    if ("3" in metricasList):
        objetivo = input(
            "Escolha os valores que deseja alcançar para: Neighborhood N2")
        globalN2 = float(objetivo)
        filename += "-N2"
    if ("4" in metricasList):
        objetivo = input(
            "Escolha os valores que deseja alcançar para: Network ClsCoef")
        globalClsCoef = float(objetivo)
        filename += "-CLSCOEF"
    if ("5" in metricasList):
        objetivo = input(
            "Escolha os valores que deseja alcançar para: Dimensionality T2")
        globalt2 = float(objetivo)
        filename += "-T2"
    if ("6" in metricasList):
        objetivo = input(
            "Escolha os valores que deseja alcançar para: Feature-based F1")
        globalf1 = float(objetivo)
        filename += "-F1"

N_ATTRIBUTES = int(n_instancias)
NOBJ = len(metricasList)

dic = {}

# reference points
ref_points = [tools.uniform_reference_points(
    NOBJ, p, s) for p, s in zip(P, SCALES)]
ref_points = np.concatenate(ref_points)
_, uniques = np.unique(ref_points, axis=0, return_index=True)
ref_points = ref_points[uniques]

def my_evaluate(individual):
    vetor = []
    dataFrame['label'] = individual
    robjects.globalenv['dataFrame'] = dataFrame
    target = "label"

    if("1" in metricasList):
        imbalance = complx.balance(dataFrame, target, "C2")
        vetor.append(abs(globalBalance - imbalance))
    if ("2" in metricasList):
        linearity = complx.linearity(dataFrame, target, "L2")
        vetor.append(abs(globalLinear - linearity))
    if ("3" in metricasList):
        n2 = complx.neighborhood(dataFrame, target, "N2")
        vetor.append(abs(globalN2 - n2))
    if ("4" in metricasList):
        ClsCoef = complx.network(dataFrame, target, "ClsCoef")
        vetor.append(abs(globalClsCoef - ClsCoef))
    if ("5" in metricasList):
        t2 = complx.dimensionality(dataFrame, target, "T2")
        vetor.append(abs(globalt2 - t2))
    if ("6" in metricasList):
        f1 = complx.feature(dataFrame, target, "F1")
        vetor.append(abs(globalf1 - f1))
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
    target = "label"
    if("1" in metricasList):
        imbalance = complx.balance(dataFrame, target, "C2")
        vetor.append(abs(imbalance))
    if ("2" in metricasList):
        linearity = complx.linearity(dataFrame, target, "L2")
        vetor.append(abs(linearity))
    if ("3" in metricasList):
        n2 = complx.neighborhood(dataFrame, target, "N2")
        vetor.append(abs(n2))
    if ("4" in metricasList):
        ClsCoef = complx.network(dataFrame, target, "ClsCoef")
        vetor.append(abs(ClsCoef))
    if ("5" in metricasList):
        t2 = complx.dimensionality(dataFrame, target, "T2")
        vetor.append(abs(t2))
    if ("6" in metricasList):
        f1 = complx.feature(dataFrame, target, "F1")
        vetor.append(abs(f1))
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
    df.to_csv(str(filename)+".csv")
    ax1 = df.plot.scatter(x=0, y=1, c='label', colormap='Paired')
    pyplot.show()
