import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

from heartstep_experiment.direct_icp import direct_icp_test
import warnings

warnings.filterwarnings("ignore")

from rpy2.robjects import numpy2ri

numpy2ri.activate()


def rnorm(n):
    return np.random.normal(size=(n,))


def gen_data(n):
    E = np.random.binomial(1, p=0.5, size=n)
    g_e = E + -(1 - E)
    mu_e = 2 * E + -2 * (1 - E)
    U = rnorm(n)
    X = g_e * U + rnorm(n)
    A = ((1 + X + rnorm(n)) > 0).astype(int)
    R = mu_e + X + A * (1 + X) + rnorm(n)
    dat_df = pd.DataFrame(np.vstack([X, A, R, E]).T, columns=['X', 'A', 'R', 'E'])

    return dat_df


def einv_test(dat_df):
    x_set = ['X']

    interaction = str.join('+', ['A*E*' + x for x in x_set])
    f = 'R ~ E + X*E +' + interaction

    model = ols(formula=f, data=dat_df).fit()

    wald_str = 'A:E = 0,' + str.join(',', ['A:E:{} = 0'.format(s) for s in x_set])
    pval = model.wald_test('({})'.format(wald_str), use_f=False, scalar=False).pvalue.item()

    return pval


np.random.seed(1)

dat_df = gen_data(5000)

print("e-invariance {}".format(einv_test(dat_df)))

icpdat = [dat_df[dat_df.E == 0][['X', 'R']].to_numpy(), dat_df[dat_df.E == 1][['X', 'R']].to_numpy()]

print("full invariance {}".format(direct_icp_test(1, [0], icpdat)))
