from rpy2.robjects import pandas2ri

import rpy2.robjects.packages as rpackages
from rpy2.robjects import Formula
ecol = rpackages.importr('ECoL')

pandas2ri.activate()

def feature(df, label, measure):
    r_df, fml = __conversion_formula(df, label)
    f1Vector = ecol.overlapping_formula(
        fml, r_df, measures=measure, summary="return")
    f1 = f1Vector.rx(1)
    return float(f1[0][0])
    
def neighborhood(df, label, measure):
    r_df, fml = __conversion_formula(df, label)
    n2Vector = ecol.neighborhood_formula(
        fml, r_df, measures=measure, summary="return")
    n2 = n2Vector.rx(1)
    return float(n2[0][0])

def linearity(df, label, measure):
    r_df, fml = __conversion_formula(df, label)
    l2Vector = ecol.linearity_formula(
        fml, r_df, measures=measure, summary="return")
    l2 = l2Vector.rx(1)
    return float(l2[0][0])

def dimensionality(df, label, measure):
    r_df, fml = __conversion_formula(df, label)
    t2Vector = ecol.dimensionality_formula(
        fml, r_df, measures=measure, summary="return")
    t2 = t2Vector[0]
    return float(t2)

def balance(df, label, measure):
    r_df, fml = __conversion_formula(df, label)
    c2Vector = ecol.balance_formula(
        fml, r_df, measures=measure, summary="return")
    c2 = c2Vector.rx(1)
    return float(c2[0][0])

def network(df, label, measure):
    r_df, fml = __conversion_formula(df, label)
    clscoefVector = ecol.network_formula(
        fml, r_df, measures=measure, summary="return")
    clscoef = clscoefVector.rx(1)
    return float(clscoef[0][0])

def __conversion_formula(df, label):
    r_df = pandas2ri.py2rpy(df)
    formula = Formula(label + ' ~ .')
    return r_df, formula;
