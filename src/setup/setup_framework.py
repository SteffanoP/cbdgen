import setup.argparser as argparser
import setup.terminal_interaction as interaction

def get_options() -> dict:
    args = argparser.parse_args()
    if args.option_interative:
        return setup_interative()
    return setup_non_interative(args)

def setup_interative() -> dict:
    return interaction.interact()

def setup_non_interative(args) -> dict:
    options = {}
    options['classes'] = args.number_of_classes
    options['attributes'] = args.number_of_attributes
    options['samples'] = args.number_of_instances
    options['NGEN'] = args.number_of_generations
    options['filename'] = args.filename

    # Separating Measures
    if args.cm != None:
        for measure in args.cm:
            options[measure[0]] = measure[1]
    
    return options
