import multiprocessing
import pickle
import random

import numpy as np
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from pymfe.mfe import MFE
# from meta_features.ecol import ECoL

import _internal

from extractor import CBDGENExtractor
import preprocess
import setup.setup_framework as setup
from instances_generator.generator import InstancesGenerator

def generate_instances(samples, attributes, classes, maker: tuple[int,str]
                       ) -> pd.DataFrame:
    """
    Function responsible for the Generatation of Instances, highly dependent
    of a InstancesGenerator object.

    Parameters
    ----------
        samples : Number of instances to be generated.
        attributes : Number of Attributes/Features to be generated.
        classes : Number of classes to be classified to a instance.
        maker : The type of maker that will generate the set of instances.

    Returns
    -------
        pandas.DataFrame
    """
    gen_instances = InstancesGenerator(samples, attributes,
                                       classes=classes,
                                       maker_option=maker[1])
    return gen_instances.generate(maker[0])

# TODO: Build a clever architecture for the filename
def build_filename(filename: str='', *, ngen: int, metrics: list) -> str:
    """
    Function that builds a filename based on the number of generations and
    metrics used to optimize.

    Parameters
    ----------
        filename : Name or Prefix of the File that contains the result of the
            optimization process.
        ngen : Number of generations of the current run of optimization.
        metrics : A list of metrics used to optimize.
    """
    filename = filename if filename != "" else "NGEN="+ \
        str(ngen)
    filename += '-' + '-'.join(metrics)
    return filename

def my_evaluate(individual):
    extractor.update_label(individual)
    return tuple([abs(g - l) for g,l in zip(global_measures,extractor.complexity())])

def print_evaluate(individual):
    extractor.update_label(individual)
    return tuple(extractor.complexity())

def setup_engine(options):
    """
    Function that set up a deap.base.toolbox for the search-engine process

    Parameters
    ----------
        options : Dictionary of setup parameters highly necessary to how the
            search engine will find the solutions

    Returns
    -------
        deap.base.Toolbox
    """
    samples = int(options['samples'])
    n_objectives = len(options['measures'])

    # reference points
    ref_points = [tools.uniform_reference_points(
        n_objectives, p, s) for p, s in zip(options['P'], options['SCALES'])]
    ref_points = np.concatenate(ref_points)
    _, uniques = np.unique(ref_points, axis=0, return_index=True)
    ref_points = ref_points[uniques]

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)*n_objectives)
    creator.create("Individual", list, fitness=creator.FitnessMin)

    randint_down = 0
    randint_up = options['classes'] - 1

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, randint_down, randint_up)
    toolbox.register("individual", tools.initRepeat,
                    creator.Individual, toolbox.attr_int, samples)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", my_evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    indpb = options['INDPB']
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=indpb)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    return toolbox

def results(options: dict, toolbox: base.Toolbox):
    """
    Function that operates the search engine process by operating an
    evolutional algorithm to find the best results.

    Parameters
    ----------
        options : Dictionary of setup parameters highly necessary to how the
            search engine will find the solutions.
        toolbox : A Toolbox for evolution that contains evolutionary operators.

    Returns
    -------
        deap.base.toolbox.population : A population of the best individuals
            from the search engine process.
        deap.tools.logbook : A logbook that contains evolutionary and
            statistics information about the search process.
    """
    pop = options['POP']
    cxpb = options['CXPB']
    mutpb = options['MUTPB']
    ngen = options['NGEN']
    random.seed(64)
    pool = multiprocessing.Pool()
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

def main():
    options = setup.get_options()
    metrics = options['measures']

    if options['filepath'] != '':
        base_df = pd.read_csv(options['filepath'])
        base_complx = _internal.extract_complexity_dataframe(base_df,
                                                             options['label_name'],
                                                             metrics)
        for metric, complx in zip(metrics, base_complx):
            print(metric, complx)
            options[metric] = complx

    global global_measures
    global_measures = [options[measure] for measure in metrics]

    dataframe = generate_instances(options['samples'], options['attributes'],
                                   options['classes'], options['maker'])

    complexity_values = {}
    for measure in metrics:
        complexity_values[measure] = options[measure]

    global extractor
    label = dataframe.pop(options['label_name']).values
    extractor = CBDGENExtractor(MFE, dataframe, label, metrics)
    # extractor = CBDGENExtractor(ECoL, dataframe, label, metrics)

    filename = build_filename(options['filename'],
                              ngen=options['NGEN'],
                              metrics=metrics)

    print(metrics, len(metrics))
    print(global_measures)
    toolbox = setup_engine(options)
    result = results(options, toolbox)

    compiled_results = {}
    for x in range(len(result[0])):
        compiled_results[print_evaluate(result[0][x])] = result[0][x]
        outfile = open(filename, 'wb')
        pickle.dump(compiled_results, outfile)
        outfile.close()

    dataframe['label'] = result[0][0]
    # Scale to original Dataset (Optional) #TODO: Improve preprocessing
    # df = preprocess.scaleColumnsFrom(base_df, df, label_column='label')
    dataframe.to_csv(str(filename)+".csv")

if __name__ == '__main__':
    main()
