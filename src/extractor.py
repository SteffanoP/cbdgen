"""
Extractor is a module that contains functions that extract meta-features values
from Data Sets.

Features
--------
    complexity: extract data complexity values.
"""
from pandas import DataFrame
from meta_features.ecol import Ecol

COMPLEXITY_MEASURES_PREFIX = {
    'F': 'feature_based',
    'N': 'neighborhood',
    'L': 'linearity',
    'T': 'dimensionality',
    'C': 'class_balance',
    'S': 'smoothness'
}

def complexity(dataframe: DataFrame,
                         label: str,
                         measures: list[str]) -> list:
    """
    Complexity extractor is a function that is able to extract Data Complexity
    Values from a given Data Set. For now, this function is strongly dependent
    of the ECoL package/object to extract each Complexity Measure.

    This function extracts a list of values by creating an ECoL object and
    specifying a list of measures desired to extract.

    Parameters
    ----------
        dataframe : Data Set to extract data complexity.
        label : Column name of the class attribute of the Data Set.
        measures : List of Complexity Measures to extract data complexity from
        the Data Set.

    Returns
    -------
        list : List of Data Complexity Values extracted.
    """
    ecol = Ecol(dataframe=dataframe, label=label)
    return [getattr(ecol, COMPLEXITY_MEASURES_PREFIX[measure[0]])(measure)
            for measure in measures]

def complexity_optimized(ecol: Ecol, measure: str):
    """
    This function extracts a complexity value given an ECoL object to extract
    from.

    Parameters
    ----------
        ecol : Ecol object to extract our measures.
        measure : the measure to be extracted.

    Returns
    -------
        Complexity value of the measure.

    Details
    -------
    Complexity extractor is a function that is able to extract Data Complexity
    Values from a given Data Set. For now, this function is strongly dependent
    of the ECoL package/object to extract each Complexity Measure.
    """
    return getattr(ecol, COMPLEXITY_MEASURES_PREFIX[measure[0]])(measure)
