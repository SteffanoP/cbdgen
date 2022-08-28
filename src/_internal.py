"""This module provides useful functions for CBDGEN
"""
import numpy as np
from pandas import DataFrame

from pymfe.mfe import MFE

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
    return _extract_pymfe_complexity(dataframe, target, features)

def _extract_pymfe_complexity(data: DataFrame, target: np.ndarray,
                              features: list[str]) -> tuple[np.float64]:
    extractor = CBDGENExtractor(MFE, data, target, features)
    return extractor.complexity()
