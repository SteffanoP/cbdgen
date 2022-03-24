import argparse

from setup.options_types.cm import cm
from setup.options_types.maker import maker

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
    parser.add_argument('--attributes',
                        dest='number_of_attributes',
                        help="number of attributes to be generated"
                        "(default: %(default)d "
                        "attributes).",
                        type=int,
                        default=2
                        )
    parser.add_argument('--classes',
                        dest='number_of_classes',
                        help="number of classes to be targeted"
                        "(default: %(default)d classes)",
                        type=int,
                        default=2
                        )
    parser.add_argument('--cm', 
                        dest='cm',
                        help="List of Complexity Measures to be optimized"
                            " (e.g. `N1:0.5 L2:0.4 C2:0.35 F2:0.80`)",
                        type=cm, 
                        nargs='+'
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
    parser.add_argument('--maker',
                        dest="maker",
                        help="The maker to generate a random dataset.",
                        type=maker,
                        default=4
                        )
    parser.add_argument('--ngen',
                        dest="number_of_generations",
                        help="Number of generations for the non sorted "
                        "domination algorithm (NSGA) " 
                        "(default: %(default)d generations).",
                        type=int,
                        default=1000
                        )

    return parser.parse_args()