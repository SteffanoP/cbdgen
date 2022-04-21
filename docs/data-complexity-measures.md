# Data Complexity Measures

Data Complexity Measures or Complexity Measures is a set of descriptors that quantifies how difficult is to create a decision boundary for classification problems.

## Feature-based Measures

Feature-based measures quantifies how effective a single feature dimension is for class separation.

### F1

Maximum Fisher's Discriminant Ratio (F1) measures the overlap between the values of the features and takes the value of the largest discriminant ratio among all the available features.

### F1v

Directional-vector maximum Fisher's discriminant ratio (F1v) complements F1 by searching for a vector able to separate two classes after the training examples have been projected into it.

### F2

Volume of the overlapping region (F2) computes the overlap of the distributions of the features values within the classes. F2 can be determined by finding, for each feature its minimum and maximum values in the classes.

### F3

The maximum individual feature efficiency (F3) of each feature is given by the ratio between the number of examples that are not in the overlapping region of two classes and the total number of examples. This measure returns the maximum of the values found among the input features.

### F4

Collective feature efficiency (F4) get an overview on how various features may work together in data separation. First the most discriminative feature according to F3 is selected and all examples that can be separated by this feature are removed from the dataset. The previous step is repeated on the remaining dataset until all the features have been considered or no example remains. F4 returns the ratio of examples that have been discriminated.

## Neighborhood Measures

Neighborhood Measures quantifies the shape (such as density and presence of instances in the same or different classes) of the decision boundary and characterize the class overlap by analyzing neighborhoods.

### N1

Fraction of borderline points (N1) computes the percentage of vertexes incident to edges connecting examples of opposite classes in a Minimum Spanning Tree (MST).

### N2

Ratio of intra/extra class nearest neighbor distance (N2) computes the ratio of two sums: intra-class and inter-class. The former corresponds to the sum of the distances between each example and its closest neighbor from the same class. The later is the sum of the distances between each example and its closest neighbor from another class (nearest enemy).

### N3

Error rate of the nearest neighbor (N3) classifier
 corresponds to the error rate of a one Nearest Neighbor (1NN)
 classifier, estimated using a leave-one-out procedure in dataset.

### N4

Non-linearity of the nearest neighbor classifier (N4)
 creates a new dataset randomly interpolating pairs of training
 examples of the same class and then induce a the 1NN classifier on
 the original data and measure the error rate in the new data points

### T1

Fraction of hyperspheres covering data (T1) builds
 hyperspheres centered at each one of the training examples, which
 have their radios growth until the hypersphere reaches an example
 of another class. Afterwards, smaller hyperspheres contained in
 larger hyperspheres are eliminated. T1 is finally defined as the
 ratio between the number of the remaining hyperspheres and the
 total number of examples in the dataset.

### LSC

Local Set Average Cardinality (LSC) is based on Local Set
 (LS) and defined as the set of points from the dataset whose
 distance of each example is smaller than the distance from the
 examples of the different class. LSC is the average of the LS.

## Linearity Measures

Linearity Measures quantifies how much the problem classes can be linearly classified.

### L1

Sum of the error distance by linear programming (L1) computes the sum of the distances of incorrectly classified examples to a linear boundary used in their classification.

### L2

Error rate of linear classifier (L2) computes the error rate of the linear SVM classifier induced from dataset.

### L3

Non-linearity of a linear classifier (L3) creates a new dataset randomly interpolating pairs of training examples of the same class and then induce a linear SVM on the original data and measure the error rate in the new data points.

## Dimensionality Measures

Dimensionality Measures quantifies the data sparsity based on the dimensionality of samples in a dataset.

### T2

Average number of points per dimension (T2) is given by the ratio between the number of examples and dimensionality of the dataset.

### T3

Average number of points per PCA (T3) is similar to T2, but uses the number of PCA components needed to represent 95% of data variability as the base of data sparsity assessment.

### T4

Ratio of the PCA Dimension to the Original (T4) estimates the proportion of relevant and the original dimensions for a dataset.

## Class balance Measures

Class balance Measures quantifies the ratio of the number of examples between classes.

### C1

The entropy of class proportions (C1) capture the imbalance in a dataset based on the proportions of examples per class.

### C2

The imbalance ratio (C2) is an index computed for measuring class balance. This is a version of the measure that is also suited for multiclass classification problems.

## Structural Measures

### Density

Average Density of the network (Density) represents the number of edges in the graph, divided by the maximum number of edges between pairs of data points.

### ClsCoef

Clustering coefficient (ClsCoef) averages the clustering tendency of the vertexes by the ratio of existent edges between its neighbors and the total number of edges that could possibly exist between them.

### Hubs

Hubs score (Hubs) is given by the number of connections it has to other nodes, weighted by the number of connections these neighbors have.

## Feature Correlation Measures

### C1-Correlation

Maximum feature correlation to the output (C1) calculate the maximum absolute value of the Spearman correlation between each feature and the outputs.

### C2-Correlation

Average feature correlation to the output (C2) computes the average of the Spearman correlations of all features to the output.

### C3-Correlation

Individual feature efficiency (C3) calculates, for each feature, the number of examples that must be removed from the dataset until a high Spearman correlation value to the output is achieved.

### C4-Correlation

Collective feature efficiency (C4) computes the ratio of examples removed from the dataset based on an iterative process of linear fitting between the features and the target attribute.

## Smoothness Measures

### S1

Output distribution (S1) monitors whether the examples joined in the MST have similar output values. Lower values indicate simpler problems, where the outputs of similar examples in the input space are also next to each other.

### S2

Input distribution (S2) measure how similar in the input space are data items with similar outputs based on distance.

### S3

Error of a nearest neighbor regressor (S3) calculates the mean squared error of a 1-nearest neighbor regressor  using leave-one-out.

### S4

Non-linearity of nearest neighbor regressor (S4) calculates the mean squared error of a 1-nearest neighbor regressor to the new randomly interpolated points.

## References

Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho,
T. K. (2019). How Complex Is Your Classification Problem?: A Survey on
Measuring Classification Complexity. ACM Computing Surveys (CSUR),
52:1-34, <https://github.com/lpfgarcia/ECoL>.

Albert Orriols-Puig, Nuria Macia and Tin K Ho. (2010). Documentation
for the data complexity library in C++. Technical Report. La Salle -
Universitat Ramon Llull, <https://github.com/nmacia/dcol>.
