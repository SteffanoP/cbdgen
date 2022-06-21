"""A Module dedicated to the extraction of complexity measures
"""
import typing as t
import numpy as np
from pymfe.mfe import MFE

class CBDGENExtractor:
    """
    Core class dedicated for the extraction of complexity data values.

    Attributes
    ----------
    data : :obj:`np.ndarray`
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

    def __init__(self, data: np.ndarray, label: np.ndarray, features: list,
                 summary: list = None):
        self.data = data
        self.label = label
        self.mfe = MFE(groups=['complexity'],
                        features=features,
                        summary=['mean'])

        if summary is not None:
            self.mfe = MFE(groups=['complexity'],
                            features=features,
                            summary=summary)

        self.mfe.fit(self.data, self.label)

    def update_label(self, label: t.Union[list, np.ndarray]) -> None:
        """
        Update label attributes of the dataset.

        Parameters
        ----------
        label : :obj:`np.ndarray`
        """
        self.mfe.fit(self.data, label)

    def complexity(self) -> tuple[np.float64]:
        """
        Extracts complexity data based on previously fitted data and label
        attributes.

        Returns
        -------
        :obj:`tuple` of :obj:`float` complexity data values.
        """
        complx = self.mfe.extract()
        return tuple(complx[1][::-1])
