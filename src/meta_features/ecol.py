"""
The Extended Complexity Library (ECoL) is the implementation in R of a set of
measures to characterize the complexity of classification and regression
problems based on aspects that quantify the linearity of the data, the presence
of informative feature, the sparsity and dimensionality of the datasets.

Originally implement in R, this package uses ryp2 to convert ECoL package to
Python method/functions.

More details at: <https://github.com/lpfgarcia/ECoL/tree/0.3.0>
"""
from pandas import DataFrame
from rpy2.robjects import pandas2ri, Formula
import rpy2.robjects.packages as rpackages

class Ecol:
    def __init__(self, dataframe: DataFrame, label: str) -> None:
        pandas2ri.activate()
        self.r_df, self.fml = self._conversion_formula(dataframe, label)
        self.ecol = rpackages.importr('ECoL')

    #TODO: Create a method to extract all measures from a dataset
    #TODO: Create a method to extract a list of measures from multiple subgroups
    def feature_based(self, *measure: str) -> tuple:
        if len(measure) == 1:
            feature_vector = self.ecol.overlapping(self.fml, self.r_df,
                                                   measures=measure[0])
            return feature_vector[0][0], feature_vector[0][1]
        #TODO: Implement extraction of multiple measures for subgroups

    def neighborhood(self, *measure: str) -> tuple:
        if len(measure) == 1:
            neighborhood_vector = self.ecol.neighborhood(self.fml, self.r_df,
                                                         measures=measure[0])
            return neighborhood_vector[0][0], neighborhood_vector[0][1]

    def linearity(self, *measure: str) -> tuple:
        if len(measure) == 1:
            linearity_vector = self.ecol.linearity(self.fml, self.r_df,
                                                   measures=measure[0])
            return linearity_vector[0][0], linearity_vector[0][1]

    def dimensionality(self, *measure: str):
        if len(measure) == 1:
            dimensionality_vector = self.ecol.dimensionality(self.fml,
                                                             self.r_df,
                                                             measures=
                                                             measure[0])
            return dimensionality_vector[0]

    def class_balance(self, *measure: str):
        if len(measure) == 1:
            balance_vector = self.ecol.balance(self.fml, self.r_df,
                                               measures=measure[0])
            return balance_vector[0][0]

    def structural(self, *measure: str):
        if len(measure) == 1:
            network_vector = self.ecol.network(self.fml, self.r_df,
                                               measures=measure[0])
            return network_vector[0][0]

    def feature_correlation(self, *measure: str) -> tuple:
        if len(measure) == 1:
            correlation_vector = self.ecol.correlation(self.fml, self.r_df,
                                                       measures=measure[0])
            return correlation_vector[0][0], correlation_vector[0][1]

    def smoothness(self, *measure: str) -> tuple:
        if len(measure) == 1:
            smoothness_vector = self.ecol.smoothness(self.fml, self.r_df,
                                                     measures=measure[0])
            return smoothness_vector[0][0], smoothness_vector[0][1]

    @staticmethod
    def _conversion_formula(dataframe: DataFrame, label: str) -> tuple[DataFrame, Formula]:
        r_df = pandas2ri.py2rpy(dataframe)
        fml = Formula(label + ' ~ .')
        return r_df, fml
