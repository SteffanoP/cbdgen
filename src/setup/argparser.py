import argparse

from setup.complexity_types.cm import cm

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="These are common options used in various situations:"
    )
    parser.add_argument('-i','--interative',
                        dest='option_interative',
                        help="optional choice to run the framework on "
                        "interative mode.",
                        action='store_true'
                        )
    parser.add_argument('--classes',
                        dest='number_of_classes',
                        help="number of classes to be targeted"
                        "(default: %(default)d classes)",
                        type=int,
                        default=2
                        )
    # parser.add_argument('--distribution', # Distribution is not the right name
    #                     dest="distribution",
    #                     help="distribution to generate a random dataset.",
    #                     choices=['blobs', 'moons', 'circles',
    #                              'classf', 'multi_classf'],
    #                     type=str,
    #                     default='classf'
    #                     )
    parser.add_argument('--features',
                        dest='number_of_features',
                        help="number of features to be generated"
                        "(default: %(default)d "
                        "features).",
                        type=int,
                        default=2
                        )
    parser.add_argument('--filename',
                        dest='filename',
                        help="prefix of the filename generated." 
                        "(default: %(default)s)",
                        type=str,
                        default=None
                        )
    parser.add_argument('--instances',
                        dest="number_of_instances",
                        help="number of instances to be generated " 
                        "(default: %(default)d "
                        "instances).",
                        type=int,
                        default=100
                        )
    parser.add_argument('--ngen',
                        dest="number_of_generations",
                        help="Number of generations for the non sorted "
                        "domination algorithm (NSGA) " 
                        "(default: %(default)d generations).",
                        type=int,
                        default=1000
                        )
    parser.add_argument('--cm', 
                        dest='cm',
                        help="List of Complexity Measures to be optimized",
                        type=cm, 
                        nargs='+'
                        )

    return parser.parse_args()

args = parse_args()

print(args)
