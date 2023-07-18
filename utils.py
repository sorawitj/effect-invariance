import itertools
from functools import reduce

import numpy as np
import pandas as pd
import rpy2.robjects as robj
from rpy2.robjects import pandas2ri

pandas2ri.activate()


def to_r(x):
    return robj.FloatVector(x)


def to_r_df(df):
    return robj.conversion.py2rpy(df)


def unzip(comb):
    ret = list(zip(*comb))
    names = reduce(lambda x1, x2: x1 + '+' + x2, ret[0])
    idx = reduce(lambda x1, x2: np.concatenate([x1, x2]), ret[1])

    return names, idx


def create_subset(features, min_size=3):
    subsets = set()
    combinations = range(min_size, len(features) + 1)
    for c in combinations:
        combi = list(itertools.combinations(features, c))
        subsets.update(combi)

    return subsets


def clip_weight(weight, p=.98):
    cut_point = np.quantile(weight, p)
    weight[weight > cut_point] = cut_point

    return weight


def quantile_clip(arr, q=.99):
    threshold = np.quantile(arr, q)
    arr[arr > threshold] = threshold

    return arr


def get_weights(A, pi):
    return (pi ** A) * ((1 - pi) ** (1 - A))


def to_intervels(arrs):
    return pd.arrays.IntervalArray.from_arrays(arrs[0], arrs[1])
