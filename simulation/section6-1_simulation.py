import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.formula.api import wls, glm
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from utils import get_weights
import warnings

warnings.filterwarnings("ignore")

from rpy2.robjects import numpy2ri, packages

numpy2ri.activate()


def rnorm(n):
    return np.random.normal(size=(n,))


def get_pi(dfX):
    cut_off = .1
    logits = 0.5 + dfX['X1'] - dfX['X2'] * .5 + dfX['X3'] * .3
    pi = 1 / (1 + np.exp(-logits))
    pi[pi < cut_off] = cut_off
    pi[pi > (1 - cut_off)] = (1 - cut_off)
    return pi


def gen_data(n, e_params, model, policy=get_pi):
    E = np.random.binomial(1, p=0.5, size=n)
    g1_e, g2_e, g3_e, mu_e = [arr.flatten() for arr in np.hsplit(e_params[E, :], (1, 2, 3))]
    U1 = rnorm(n)
    U2 = rnorm(n)
    X3 = U1 * g3_e + rnorm(n)
    X2 = U2 * g2_e + rnorm(n)
    X1 = X2 + U1 * g1_e + rnorm(n)
    X = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})
    pi = policy(X)
    A = np.random.binomial(1, p=pi, size=n)

    if model == 'lin-lin':
        R = mu_e + U1 + U2 + X2 - 0.5 * X2 * X3 + X3 + 1. * A + A * (.5 * X2 + .5 * U1) + rnorm(n)
    elif model == 'nonlin-lin':
        R = mu_e + U1 + X2 + X3 + A + A * (.5 * X2 + .5 * U1) + rnorm(n)
    elif model == 'nonlin-nonlin':
        R = mu_e + U1 + U2 + X2 - 0.5 * X2 * X3 + X3 + 1. * A + A * (.5 * X2 ** 2 + .2 * X2 ** 3 + .5 * U1) + rnorm(n)
    else:
        raise Exception('incorrect specification of the "model" parameter')

    dat_df = pd.DataFrame(np.vstack([X1, X2, X3, A, R, E]).T, columns=['X1', 'X2', 'X3', 'A', 'R', 'E'])

    return dat_df


# get env-specific params
e_params = np.load("simulation/e_params.npy")


def einv_test(test, model, x_set, n, rng):
    np.random.seed(rng)

    dat_df = gen_data(n, e_params, model, get_pi)
    dat_df['pA1'] = get_pi(dat_df)

    dat_df['pi'] = get_weights(dat_df.A, dat_df['pA1'])
    p_f = str.join('+', ['E*' + x for x in x_set])
    pi_model = glm("A ~ {}".format(p_f), data=dat_df, family=sm.families.Binomial()).fit()
    pA1 = pi_model.predict(dat_df)
    dat_df['pi_til'] = get_weights(dat_df.A, pA1)

    if test == 'Wald':
        dat_df['A_C'] = dat_df['A'] - pA1

        interaction = str.join('+', ['A_C*E*' + x for x in x_set])
        f = 'R ~ E + X1*E + X2*E + X3*E +' + interaction

        weights = dat_df['pi_til'] / dat_df['pi']

        M2 = wls(formula=f, data=dat_df, weights=weights).fit(cov_type='HC3')

        wald_str = 'A_C:E = 0,' + str.join(',', ['A_C:E:{} = 0'.format(s) for s in x_set])
        pval = M2.wald_test('({})'.format(wald_str), use_f=False, scalar=False).pvalue.item()
    elif test == 'DR':

        wGCM = packages.importr("weightedGCM")

        D1 = dat_df.sample(frac=.5)
        D2 = dat_df[~dat_df.index.isin(D1.index)]
        # main effect
        X, Y = D1[['X1', 'X2', 'X3', 'E']].to_numpy(), D1['R'].to_numpy()
        main_A0 = RandomForestRegressor()
        main_A1 = RandomForestRegressor()

        main_A0.fit(X[D1['A'] == 0], Y[D1['A'] == 0])
        main_A1.fit(X[D1['A'] == 1], Y[D1['A'] == 1])

        XD2 = D2[['X1', 'X2', 'X3', 'E']]
        muA1 = main_A1.predict(XD2) + D2['A'] * (D2['R'] - main_A1.predict(XD2)) / D2['pA1']
        muA0 = main_A0.predict(XD2) + (1 - D2['A']) * (D2['R'] - main_A0.predict(XD2)) / (1 - D2['pA1'])

        D2['new_R'] = (muA1 - muA0)

        Xs, Y, E = D2[x_set].to_numpy(), D2['new_R'].to_numpy(), D2['E'].to_numpy()
        ret = wGCM.wgcm_fix(X=E, Y=Y, Z=Xs, regr_meth='xgboost', weight_num=7)
        pval = ret.item()
    else:
        raise Exception("Invalid test choice")
    return pval


candidate_sets = [['X1'], ['X2'], ['X3'], ['X1', 'X2'], ['X1', 'X3'], ['X2', 'X3'], ['X1', 'X2', 'X3']]
res_df = pd.DataFrame()
for test in ['Wald', 'DR']:
    for model in ['lin-lin', 'nonlin-lin', 'nonlin-nonlin']:
        for x_set in candidate_sets:
            for n in [100, 200, 400, 800]:
                random_state = np.random.randint(np.iinfo(np.int32).max, size=100)
                ret = Parallel(n_jobs=-1)(delayed(einv_test)(test, model, x_set, n, rng) for rng in tqdm(random_state))

                inner_df = pd.DataFrame(ret, columns=['pval'])
                inner_df['test'] = test
                inner_df['model'] = model
                inner_df['set'] = str.join(',', x_set)
                inner_df['sample size'] = n

                res_df = pd.concat([res_df, inner_df], ignore_index=True)

res_df['Rejection Rate'] = res_df['pval'] < 0.05
res_df.to_csv("results/wald_dr_compare.csv", index=False)
