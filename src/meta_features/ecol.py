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
    """
    Package Ecol is a conversion from the original Extended Complexity Library
    (ECoL) implemented in R as a Complexity Extractor package. This package
    does treat and convert ECoL to a similar level using rpy2r package to
    convert its functions.
    """
    def __init__(self,
                 dataframe: DataFrame,
                 label: str,
                 summary: str = None) -> None:
        pandas2ri.activate()

        # Set up of DataFrame and ECoL formula
        self.r_df, self.fml = self._conversion_formula(dataframe, label)
        self.ecol = rpackages.importr('ECoL')

        """
        Summarization of complexity multiple complexity values
        Default summary is None which corresponds to return only mean values
        """
        if summary is None:
            self.feature_based = self._feature_based
            self.neighborhood = self._neighborhood
            self.linearity = self._linearity
            self.dimensionality = self._dimensionality
            self.class_balance = self._class_balance
            self.structural = self._structural
            self.feature_correlation = self._feature_correlation
            self.feature_correlation = self._smoothness
        #TODO: Factory functions for summaries (e.g. "mean", "sd", "median")

    #TODO: Create a method to extract all measures from a dataset
    #TODO: Create a method to extract a list of measures from multiple subgroups
    #TODO: Implement extraction of multiple measures for subgroups
    def _feature_based(self, *measure: str):
        """
        Method to extract feature based measures (some known as overlapping).

        Parameters
        ----------
            measure : A list of measures names.

        Returns
        -------
            Mean value of feature based complexity.

        Details
        -------
            "F1" : Maximum Fisher's Discriminant Ratio (F1) measures the
            overlap between the values of the features and takes the value of
            the largest discriminant ratio among all the available features.

            "F1v" : Directional-vector maximum Fisher's discriminant ratio
            (F1v) complements F1 by searching for a vector able to separate two
            classes after the training examples have been projected into it.

            "F2" : Volume of the overlapping region (F2) computes the overlap
            of the distributions of the features values within the classes. F2
            can be determined by finding, for each feature its minimum and
            maximum values in the classes.

            "F3" : The maximum individual feature efficiency (F3) of each
            feature is given by the ratio between the number of examples that
            are not in the overlapping region of two classes and the total
            number of examples. This measure returns the maximum of the values
            found among the input features.

            "F4" : Collective feature efficiency (F4) get an overview on how
            various features may work together in data separation. First the
            most discriminative feature according to F3 is selected and all
            examples that can be separated by this feature are removed from the
            dataset. The previous step is repeated on the remaining dataset
            until all the features have been considered or no example remains.
            F4 returns the ratio of examples that have been discriminated.

        References
        ----------
        Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho,
        T. K. (2019). How Complex Is Your Classification Problem?: A Survey on
        Measuring Classification Complexity. ACM Computing Surveys (CSUR),
        52:1-34.

        Albert Orriols-Puig, Nuria Macia and Tin K Ho. (2010). Documentation
        for the data complexity library in C++. Technical Report. La Salle -
        Universitat Ramon Llull.
        """
        if len(measure) == 1:
            feature_vector = self.ecol.overlapping(self.fml, self.r_df,
                                                   measures=measure[0])
            return feature_vector[0][0]
        return None

    def _neighborhood(self, *measure: str):
        """
        Method to extract neighborhood measures.

        Parameters
        ----------
            measure : A list of measures names.

        Returns
        -------
            Mean value of neighborhood complexity.

        Details
        -------
            "N1" : Fraction of borderline points (N1) computes the percentage
            of vertexes incident to edges connecting examples of opposite
            classes in a Minimum Spanning Tree (MST).

            "N2" : Ratio of intra/extra class nearest neighbor distance (N2)
            computes the ratio of two sums: intra-class and inter-class. The
            former corresponds to the sum of the distances between each example
            and its closest neighbor from the same class. The later is the sum
            of the distances between each example and its closest neighbor from
            another class (nearest enemy).

            "N3" : Error rate of the nearest neighbor (N3) classifier
            corresponds to the error rate of a one Nearest Neighbor (1NN)
            classifier, estimated using a leave-one-out procedure in dataset.

            "N4" : Non-linearity of the nearest neighbor classifier (N4)
            creates a new dataset randomly interpolating pairs of training
            examples of the same class and then induce a the 1NN classifier on
            the original data and measure the error rate in the new data points
            .

            "T1" : Fraction of hyperspheres covering data (T1) builds
            hyperspheres centered at each one of the training examples, which
            have their radios growth until the hypersphere reaches an example
            of another class. Afterwards, smaller hyperspheres contained in
            larger hyperspheres are eliminated. T1 is finally defined as the
            ratio between the number of the remaining hyperspheres and the
            total number of examples in the dataset.

            "LSC" : Local Set Average Cardinality (LSC) is based on Local Set
            (LS) and defined as the set of points from the dataset whose
            distance of each example is smaller than the distance from the
            examples of the different class. LSC is the average of the LS.

        References
        ----------
        Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho,
        T. K. (2019). How Complex Is Your Classification Problem?: A Survey on
        Measuring Classification Complexity. ACM Computing Surveys (CSUR),
        52:1-34.

        Albert Orriols-Puig, Nuria Macia and Tin K Ho. (2010). Documentation
        for the data complexity library in C++. Technical Report. La Salle -
        Universitat Ramon Llull.
        """
        if len(measure) == 1:
            neighborhood_vector = self.ecol.neighborhood(self.fml, self.r_df,
                                                         measures=measure[0])
            return neighborhood_vector[0][0]
        return None

    def _linearity(self, *measure: str):
        """
        Method to extract linearity measures.

        Parameters
        ----------
            measure : A list of measures names.

        Returns
        -------
            Mean value of linearity complexity.

        Details
        -------
        "L1" : Sum of the error distance by linear programming (L1) computes
        the sum of the distances of incorrectly classified examples to a linear
        boundary used in their classification.

        "L2" : Error rate of linear classifier (L2) computes the error rate
        of the linear SVM classifier induced from dataset.

        "L3" : Non-linearity of a linear classifier (L3) creates a new
        dataset randomly interpolating pairs of training examples of the same
        class and then induce a linear SVM on the original data and measure
        the error rate in the new data points.

        References
        ----------
        Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho,
        T. K. (2019). How Complex Is Your Classification Problem?: A Survey on
        Measuring Classification Complexity. ACM Computing Surveys (CSUR),
        52:1-34.

        Albert Orriols-Puig, Nuria Macia and Tin K Ho. (2010). Documentation
        for the data complexity library in C++. Technical Report. La Salle -
        Universitat Ramon Llull.
        """
        if len(measure) == 1:
            linearity_vector = self.ecol.linearity(self.fml, self.r_df,
                                                   measures=measure[0])
            return linearity_vector[0][0]
        return None

    def _dimensionality(self, *measure: str):
        """
        Method to extract dimensionality measures.

        Parameters
        ----------
            measure : A list of measures names.

        Returns
        -------
            Complexity value for dimensionality.

        Details
        -------
        "T2" : Average number of points per dimension (T2) is given by the
        ratio between the number of examples and dimensionality of the dataset.

        "T3" : Average number of points per PCA (T3) is similar to T2, but
        uses the number of PCA components needed to represent 95% of data
        variability as the base of data sparsity assessment.

        "T4" : Ratio of the PCA Dimension to the Original (T4) estimates the
        proportion of relevant and the original dimensions for a dataset.

        References
        ----------
        Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho,
        T. K. (2019). How Complex Is Your Classification Problem?: A Survey on
        Measuring Classification Complexity. ACM Computing Surveys (CSUR),
        52:1-34.

        Albert Orriols-Puig, Nuria Macia and Tin K Ho. (2010). Documentation
        for the data complexity library in C++. Technical Report. La Salle -
        Universitat Ramon Llull.
        """
        if len(measure) == 1:
            dimensionality_vector = self.ecol.dimensionality(self.fml,
                                                             self.r_df,
                                                             measures=
                                                             measure[0])
            return dimensionality_vector[0]
        return None

    def _class_balance(self, *measure: str):
        """
        Method to extract class balance measures.

        Parameters
        ----------
            measure : A list of measures names.

        Returns
        -------
            Complexity value for class balance.

        Details
        -------
        "C1" : The entropy of class proportions (C1) capture the imbalance in
          a dataset based on the proportions of examples per class.

        "C2" : The imbalance ratio (C2) is an index computed for measuring
          class balance. This is a version of the measure that is also suited
          for multiclass classification problems.

        References
        ----------
        Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho,
        T. K. (2019). How Complex Is Your Classification Problem?: A Survey on
        Measuring Classification Complexity. ACM Computing Surveys (CSUR),
        52:1-34.

        Albert Orriols-Puig, Nuria Macia and Tin K Ho. (2010). Documentation
        for the data complexity library in C++. Technical Report. La Salle -
        Universitat Ramon Llull.
        """
        if len(measure) == 1:
            balance_vector = self.ecol.balance(self.fml, self.r_df,
                                               measures=measure[0])
            return balance_vector[0][0]
        return None

    def _structural(self, *measure: str):
        """
        Method to extract structural measures (some known as network measures).

        Parameters
        ----------
            measure : A list of measures names.

        Returns
        -------
            Complexity value for structural.

        Details
        -------
        "Density" : Average Density of the network (Density) represents the
        number of edges in the graph, divided by the maximum number of edges
        between pairs of data points.

        "ClsCoef" : Clustering coefficient (ClsCoef) averages the clustering
        tendency of the vertexes by the ratio of existent edges between its
        neighbors and the total number of edges that could possibly exist
        between them.

        "Hubs" : Hubs score (Hubs) is given by the number of connections it
        has to other nodes, weighted by the number of connections these
        neighbors have.


        References
        ----------
        Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho,
        T. K. (2019). How Complex Is Your Classification Problem?: A Survey on
        Measuring Classification Complexity. ACM Computing Surveys (CSUR),
        52:1-34.

        Albert Orriols-Puig, Nuria Macia and Tin K Ho. (2010). Documentation
        for the data complexity library in C++. Technical Report. La Salle -
        Universitat Ramon Llull.
        """
        if len(measure) == 1:
            network_vector = self.ecol.network(self.fml, self.r_df,
                                               measures=measure[0])
            return network_vector[0][0]
        return None

    def _feature_correlation(self, *measure: str):
        """
        Method to extract feature correlation measures.

        Parameters
        ----------
            measure : A list of measures names.

        Returns
        -------
            Mean value of feature correlation complexity.

        Details
        -------
        "C1" : Maximum feature correlation to the output (C1) calculate the
        maximum absolute value of the Spearman correlation between each feature
        and the outputs.

        "C2" : Average feature correlation to the output (C2) computes the
        average of the Spearman correlations of all features to the output.

        "C3" : Individual feature efficiency (C3) calculates, for each
        feature, the number of examples that must be removed from the dataset
        until a high Spearman correlation value to the output is achieved.

        "C4" : Collective feature efficiency (C4) computes the ratio of
        examples removed from the dataset based on an iterative process of
        linear fitting between the features and the target attribute.


        References
        ----------
        Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho,
        T. K. (2019). How Complex Is Your Classification Problem?: A Survey on
        Measuring Classification Complexity. ACM Computing Surveys (CSUR),
        52:1-34.

        Albert Orriols-Puig, Nuria Macia and Tin K Ho. (2010). Documentation
        for the data complexity library in C++. Technical Report. La Salle -
        Universitat Ramon Llull.
        """
        if len(measure) == 1:
            correlation_vector = self.ecol.correlation(self.fml, self.r_df,
                                                       measures=measure[0])
            return correlation_vector[0][0], correlation_vector[0][1]
        return None

    def _smoothness(self, *measure: str):
        """
        Method to extract smoothness measures.

        Parameters
        ----------
            measure : A list of measures names.

        Returns
        -------
            Mean value of smoothness complexity.

        Details
        -------
        "S1" : Output distribution (S1) monitors whether the examples
        joined in the MST have similar output values. Lower values indicate
        simpler problems, where the outputs of similar examples in the input
        space are also next to each other.

        "S2" : Input distribution (S2) measure how similar in the input space
        are data items with similar outputs based on distance.

        "S3" : Error of a nearest neighbor regressor (S3) calculates the mean
        squared error of a 1-nearest neighbor regressor  using leave-one-out.

        "S4" : Non-linearity of nearest neighbor regressor (S4) calculates
        the mean squared error of a 1-nearest neighbor regressor to the new
        randomly interpolated points.



        References
        ----------
        Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho,
        T. K. (2019). How Complex Is Your Classification Problem?: A Survey on
        Measuring Classification Complexity. ACM Computing Surveys (CSUR),
        52:1-34.

        Albert Orriols-Puig, Nuria Macia and Tin K Ho. (2010). Documentation
        for the data complexity library in C++. Technical Report. La Salle -
        Universitat Ramon Llull.
        """
        if len(measure) == 1:
            smoothness_vector = self.ecol.smoothness(self.fml, self.r_df,
                                                     measures=measure[0])
            return smoothness_vector[0][0]
        return None

    @staticmethod
    def _conversion_formula(dataframe: DataFrame, label: str) -> tuple[DataFrame, Formula]:
        r_df = pandas2ri.py2rpy(dataframe)
        fml = Formula(label + ' ~ .')
        return r_df, fml
