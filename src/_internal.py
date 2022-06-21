"""This module provides useful functions for CBDGEN
"""
import numpy as np
from pandas import DataFrame

from extractor import CBDGENExtractor

def extract_complexity_dataframe(dataframe: DataFrame, target_name: str,
                                 features: list[str]) -> tuple[np.float64]:
    """
    Function that extracts complexity data from a DataFrame.

    Parameters
    ----------
    dataframe : :obj:`pd.DataFrame`
        DataFrame to extract complexity data.

    target_name : :obj:`str`
        Name of the target attribute in the DataFrame.

    features : :obj:`list`
        A list of complexity extraction methods names for complexity
        extraction.
    """
    target = dataframe.pop(target_name).values
    data = dataframe.values
    return _extract_pymfe_complexity(data, target, features)

def _extract_pymfe_complexity(data: np.ndarray, target: np.ndarray,
                              features: list[str]) -> tuple[np.float64]:
    extractor = CBDGENExtractor(data, target, features)
    return extractor.complexity()
