import pandas as pd
import numpy as np

# Supported dataset generators
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification
from sklearn.datasets import make_multilabel_classification as make_mlabel_classification

# Use same random seed for multiple calls to make_multilabel_classification to
# ensure same distributions
RANDOM_SEED = np.random.randint(2 ** 10)

def blobs(samples, centers, features):
    """
    Generate isotropic Gaussian blobs for clustering, but resumes to 3 main
    parameters.

    See more at <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html>

    Parameters
    ----------
        samples : int
            The total number of points equally divided among clusters.
        centers : int
            The number of centers to generate, or the fixed center locations.
        features : int
            The number of features for each sample.

    Returns
    -------
        DataFrame : pandas.DataFrame
            A DataFrame of the generated samples grouped by x and y.
    """
    X, y = make_blobs(n_samples=samples, centers=centers, 
                        n_features=features)
    return _create_pd_dataframe(X, y)

def moons(samples, noise):
    """
    Make two interleaving half circles.

    See more at <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html>

    Parameters
    ----------
        samples : int
            The total number of points generated.
        noise : int
            Standard deviation of Gaussian noise added to the data.

    Returns
    -------
        DataFrame : pandas.DataFrame
            A DataFrame of the generated samples grouped by x and y.
    """
    X, y = make_moons(n_samples=samples, noise=noise)
    return _create_pd_dataframe(X, y)

def circles(samples, noise):
    """
    Make a large circle containing a smaller circle in 2d.

    See more: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html>

    Parameters
    ----------
        samples : int
            The total number of points generated.
        noise : int
            Standard deviation of Gaussian noise added to the data.

    Returns
    -------
        DataFrame : pandas.DataFrame
            A DataFrame of the generated samples grouped by x and y.
    """
    X, y = make_circles(n_samples=samples, noise=noise)
    return _create_pd_dataframe(X, y)

def classification(samples, features, classes):
    """
    Generate a random n-class classification problem.

    See more at <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html>

    Parameters
    ----------
        samples : int
            The total number of points.
        features : int
            The number of informative features.
        features : int
            The number of features for each sample.

    Returns
    -------
        DataFrame : pandas.DataFrame
            A DataFrame of the generated samples grouped by x and y.
    """
    X, y = make_classification(
        n_samples=samples, 
        n_features=features, 
        n_classes=classes,
        n_redundant=0, 
        n_informative=2, 
        n_clusters_per_class=1
        )
    return _create_pd_dataframe(X, y)

def multilabel_classification(samples, features, classes, labels):
    X, y = make_mlabel_classification(
        n_samples=samples,
        n_features=features,
        n_classes=classes,
        n_labels=labels,
        allow_unlabeled=False,
        random_state=RANDOM_SEED
    )
    return _create_pd_dataframe(X, y)

def _create_pd_dataframe(samples, label):
    df = pd.DataFrame(samples)
    df['label'] = label
    return df
