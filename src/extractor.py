"""A Module dedicated to the extraction of complexity measures
"""
import typing as t
import numpy as np
import pandas as pd

from pymfe.mfe import MFE
from meta_features.ecol import ECoL

class CBDGENExtractor:
    """
    Core class dedicated for the extraction of complexity data values.

    Attributes
    ----------
    data : :obj:`pd.DataFrame`
        Attributes of the dataset.

    label : :obj:`np.ndarray`
        Label attributes of the dataset.

    features : :obj:`list`
        A list of complexity extraction methods names for complexity
        extraction.

    mfe : :obj:`MFE`
        Meta-feature object extractor dedicated to extract meta-features, in
        this scenario to fit and extract complexity data.

    summary : :obj:`list`, optional
        Summary functions names for features summarization.
    """

    def __init__(self,
                 extractor: t.Union[MFE, ECoL],
                 data: pd.DataFrame,
                 label: np.ndarray,
                 features: list,
                 summary: list = None) -> None:

        if extractor is MFE:
            self.data = data.values
            self.label = label
            self.mfe = MFE(groups=['complexity'],
                           features=features,
                           summary=['mean'])

            if summary is not None:
                self.mfe = MFE(groups=['complexity'],
                                features=features,
                                summary=summary)

            self.mfe.fit(self.data, self.label)

            self.update_label = self._update_label
            self.complexity = self._complexity

        # Support for ECoL legacy extractor
        elif extractor is ECoL:
            default_label_name = 'label'
            self.data = data
            self.data[default_label_name] = label
            self.label_name = default_label_name

            self.ecol = ECoL(data, default_label_name, features)

            # Support for ECoL version of CBDGENExtractor
            self.update_label = self._update_label_ecol
            self.complexity = self._complexity_ecol

        else:
            raise ValueError("Invalid value for Extractor Object or Extractor"+
                             " not supported")

    def _update_label(self, label: t.Union[list, np.ndarray]) -> None:
        """
        Update label attributes of the dataset.

        Parameters
        ----------
        label : :obj:`np.ndarray`
        """
        self.mfe.fit(self.data, label)


    def _complexity(self) -> tuple[np.float64]:
        """
        Extracts complexity data based on previously fitted data and label
        attributes.

        Returns
        -------
        :obj:`tuple` of :obj:`float` complexity data values.
        """
        complx = self.mfe.extract()
        return tuple(complx[1][::-1])

    def _update_label_ecol(self, label: np.ndarray) -> None:
        """
        Update label attributes of the dataset.

        Legacy function that uses :obj:`ECoL` object.

        Parameters
        ----------
        label : :obj:`np.ndarray`
        """
        self.ecol.update_label(label)

    def _complexity_ecol(self) -> tuple[np.float64]:
        """
        Extracts complexity data based on previously fitted data and label
        attributes.

        Legacy function that uses :obj:`ECoL` object.

        Returns
        -------
        :obj:`tuple` of :obj:`float` complexity data values.
        """
        return self.ecol.extract()
