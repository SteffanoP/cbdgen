import multiprocessing
import pickle
import random

import numpy as np
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from rpy2 import robjects

import extractor
import preprocess
import setup.setup_framework as setup
from meta_features.ecol import Ecol
from instances_generator.generator import InstancesGenerator

# TODO: Implement Setup in a minimal main()
options = setup.get_options()

# TODO: Implement Generator of Instances in a minimal main()
def generate_instances(samples, attributes, classes, maker: tuple[int,str]
                       ) -> pd.DataFrame:
    gen_instances = InstancesGenerator(samples, attributes,
                                       classes=classes,
                                       maker_option=maker[1])
    return gen_instances.generate(maker[0])

metrics = options['measures']

# TODO: Implement fitness global measures in a minimal main()
global_measures = []
def complexity_extraction(measures: list[str], *,
                          dataframe_label: tuple[pd.DataFrame,str]=None,
                          complexity_values: dict) -> tuple[np.float64]:
    if dataframe_label is not None:
        # Copying Columns names
        # df.columns = preprocess.copyFeatureNamesFrom(base_df, label_name=target)

        # Extraction of Data Complexity Values
        return tuple(extractor.complexity(dataframe_label[0],
                                          dataframe_label[1],
                                          measures))
    return tuple(complexity_values[cm] for cm in measures)

# TODO: Build a clever architecture for the filename
def build_filename(filename: str='', *, ngen: int) -> str:
    filename = filename if filename != "" else "NGEN="+ \
        str(options['NGEN'])
    filename += '-' + '-'.join(metrics)
    return filename

def my_evaluate(individual):
    vetor = []
    dataFrame['label'] = individual
    ecol_dataFrame.update_label(individual)
    robjects.globalenv['dataFrame'] = dataFrame

    for global_value, metrica in zip(global_measures, metrics):
        complexity_value = extractor.ecol_complexity(ecol_dataFrame, metrica)
        vetor.append(abs(global_value - complexity_value))

    return tuple(vetor)

def print_evaluate(individual):
    vetor = []
    dataFrame['label'] = individual
    ecol_dataFrame.update_label(individual)
    robjects.globalenv['dataFrame'] = dataFrame

    for metrica in metrics:
        complexity_value = extractor.ecol_complexity(ecol_dataFrame, metrica)
        vetor.append(abs(complexity_value))

    return tuple(vetor)

def setup_engine(options):
    N_ATTRIBUTES = int(options['samples']) # mispelled variable name
    NOBJ = len(options['measures'])

    # reference points
    ref_points = [tools.uniform_reference_points(
        NOBJ, p, s) for p, s in zip(options['P'], options['SCALES'])]
    ref_points = np.concatenate(ref_points)
    _, uniques = np.unique(ref_points, axis=0, return_index=True)
    ref_points = ref_points[uniques]

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)*NOBJ)
    creator.create("Individual", list, fitness=creator.FitnessMin)

    RANDINT_LOW = 0
    RANDINT_UP = options['classes'] - 1

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, RANDINT_LOW, RANDINT_UP)
    toolbox.register("individual", tools.initRepeat,
                    creator.Individual, toolbox.attr_int, N_ATTRIBUTES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", my_evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    indpb = options['INDPB']
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=indpb)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    return toolbox

def results(seed=None):
    pop = options['POP']
    cxpb = options['CXPB']
    mutpb = options['MUTPB']
    ngen = options['NGEN']
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

    tool_pop = toolbox.population(pop)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in tool_pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # Compile statistics about the population
    record = stats.compile(tool_pop)

    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)
    # Begin the generational process
    for gen in range(1, ngen):
        offspring = algorithms.varAnd(tool_pop, toolbox, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Select the next generation population from parents and offspring
        tool_pop = toolbox.select(tool_pop + offspring, pop)

        # Compile statistics about the new population
        record = stats.compile(tool_pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
    return tool_pop, logbook


if __name__ == '__main__':
    dataFrame = df
    # This Ecol object should be called according to the variable dataFrame.
    # If dataFrame is renamed, then ecol_dataFrame should be renamed 
    # accordingly.
    ecol_dataFrame = Ecol(dataframe=dataFrame, label='label')
    results = results()

    for x in range(len(results[0])):
        dic[print_evaluate(results[0][x])] = results[0][x]
        outfile = open(filename, 'wb')
        pickle.dump(dic, outfile)
        outfile.close()

    df['label'] = results[0][0]
    # Scale to original Dataset (Optional) #TODO: Improve preprocessing
    # df = preprocess.scaleColumnsFrom(base_df, df, label_column='label')
    df.to_csv(str(filename)+".csv")
