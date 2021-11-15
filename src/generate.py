import pandas as pd

# Supported dataset generators
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification

def blobs(samples, centers, features):
    X, y = make_blobs(n_samples=samples, centers=centers, 
                        n_features=features)
    df = pd.DataFrame(X)
    df['label'] = y
    return df

def moons(samples, noise):
    X, y = make_moons(n_samples=samples, noise=noise)
    df = pd.DataFrame(X)
    df['label'] = y
    return df

def circles(samples, noise):
    X, y = make_circles(n_samples=samples, noise=noise)
    df = pd.DataFrame(X)
    df['label'] = y
    return df

def classification(samples, features):
    X, y = make_classification(
        n_samples=samples, 
        n_features=features, 
        n_redundant=0, 
        n_informative=2, 
        n_clusters_per_class=1
        )
    df = pd.DataFrame(X)
    df['label'] = y
    return df
