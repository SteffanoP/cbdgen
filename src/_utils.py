"""A module dedicated for utils for cbdgen
"""
from meta_features.ecol import ECoL

COMPLEXITY_MEASURES_PREFIX = {
    'F': 'feature_based',
    'N': 'neighborhood',
    'L': 'linearity',
    'T': 'dimensionality',
    'C': 'class_balance',
    'S': 'smoothness'
}

def ecol_complexity(ecol: ECoL, measure: str):
    """
    This function extracts a complexity value given an ECoL.py object to
    extract from.

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
