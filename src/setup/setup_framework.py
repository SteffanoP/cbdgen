import pandas as pd
import setup.argparser as argparser
import setup.interactor as interactor

# TODO: It is necessary to implement a config.json or a parameters.json for 
# hyperparameters about the search-engine.
HYPERPARAMETERS = {
    'P' : [12],
    'SCALES' : [1],
    'CXPB' : 0.7,
    'MUTPB' : 0.2,
    'INDPB' : 0.05,
    'POP' : 100
}

def get_options() -> dict:
    args = argparser.parse_args()
    return (setup_interative() if args.option_interative
            else setup_non_interative(args)
           ) | HYPERPARAMETERS

def setup_interative() -> dict:
    options = {}
    # Populates the dictionary with the necessary options
    options['maker'] = interactor.maker_type_input()
    filepath, label_name = interactor.based_mode_input()  
    options['filepath'] = filepath
    options['label_name'] = label_name
    measures = interactor.measures_input()
    
    # Check if is in normal mode or based-on mode
    # Refers to the normal mode
    if filepath == "":
        options['samples'] = interactor.samples_input()
        options['attributes'] = interactor.attributes_input()
        options['classes'] = interactor.classes_input()

    # Refers to the based-on mode
    if filepath != "":
        df, label = read_based_on_dataset(filepath, label_name)
        samples, attributes, classes = extract_properties(df, label)
        options |= {'samples': samples,
                    'attributes': attributes,
                    'classes': classes}
        # TODO: Extract Measures from the real dataset
    
    options['filename'] = interactor.filename_input()

    # Separating Measures
    if measures != None:
        options['measures'] = []
        for measure in measures:
            options['measures'].append(measure[0])
            options[measure[0]] = measure[1]
    
    return options

def setup_non_interative(args) -> dict:
    options = {}
    options['maker'] = args.maker
    filepath = args.option_based_on_filepath_label[0]
    label_name = args.option_based_on_filepath_label[1]
    options['filepath'] = filepath
    options['label_name'] = label_name
    measures = args.cm
    
    # Check if is in normal mode or based-on mode
    # Refers to normal mode
    if filepath == "":
        options['classes'] = args.number_of_classes
        options['attributes'] = args.number_of_attributes
        options['samples'] = args.number_of_instances
        
    # Refers to based-on mode
    if filepath != "":
        df, label = read_based_on_dataset(filepath, label_name)
        samples, attributes, classes = extract_properties(df, label)
        options |= {'samples':samples,
                    'attributes':attributes,
                    'classes':classes}
        # TODO: Extract Measures from the real dataset
    
    options['NGEN'] = args.number_of_generations
    options['filename'] = args.filename
    
    # Separating Measures
    if measures != None:
        options['measures'] = []
        for measure in args.cm:
            options['measures'].append(measure[0])
            options[measure[0]] = measure[1]
    
    return options

def extract_properties(df: pd.DataFrame, label: str) -> tuple[int, int, int]:
    rows, cols = df.shape
    classes = df[label].nunique()
    return rows, cols, classes

def read_based_on_dataset(filepath: str, label: str) -> tuple[pd.DataFrame, str]:
    if filepath[-4:] != '.csv':
        filepath += '.csv'

    df = pd.read_csv(filepath)
    if label in df.columns: return (df, label)
    raise NameError("Attribute not found in the dataset")
