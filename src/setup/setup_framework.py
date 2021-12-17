import setup.options as options

def setup_non_interative(args):
    samples = args.number_of_instances
    features = args.number_of_features
    classes = args.number_of_classes
    NGEN = args.number_of_generations
    filename = args.filename

    switch_distribution = {
        'blobs': '1',
        'moons': '2',
        'circles': '3',
        'classf': '4',
        'multi_classf': '5'
    }
    distribution = switch_distribution.get(args.distribution)
    print(distribution)
    return NGEN, distribution, samples, features, classes, filename

def setup_interative():
    print()

def get_args():
    args = options.parse_args()
    if args.option_interative:
        setup_interative()
    else:
        return setup_non_interative(args)

