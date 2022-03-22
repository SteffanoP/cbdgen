import setup.argparser as argparser
import setup.interactor as interactor

def get_options() -> dict:
    args = argparser.parse_args()
    if args.option_interative:
        return setup_interative()
    return setup_non_interative(args)

def setup_interative() -> dict:
    return interactor.terminal_input()

def setup_non_interative(args) -> dict:
    options = {}
    options['classes'] = args.number_of_classes
    options['attributes'] = args.number_of_attributes
    options['samples'] = args.number_of_instances
    options['NGEN'] = args.number_of_generations
    options['filename'] = args.filename
    options['maker'] = args.maker
    # Separating Measures
    if args.cm != None:
        options['measures'] = []
        for measure in args.cm:
            options['measures'].append(measure[0])
            options[measure[0]] = measure[1]
    
    return options
