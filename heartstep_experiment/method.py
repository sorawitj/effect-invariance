import re

import numpy as np
import pandas as pd
import patsy
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from statsmodels.formula.api import ols
import warnings

from heartstep_experiment.direct_icp import direct_icp_test

warnings.filterwarnings("ignore")

class DirectICP(object):
    def __init__(self, interaction, y_label='y'):
        self.y_label = y_label
        self.interaction = interaction
        self.y_idx = None

    def transform_data(self, df):
        f = f'{self.y_label} ~ -1 + ' + str.join('+', self.interaction)

        y, X = patsy.dmatrices(f, df, return_type='dataframe')
        X['y'] = y
        X = X.to_numpy()

        data = []
        for e in df['E'].unique():
            e_idx = df['E'] == e
            data += [X[e_idx]]

        return data

    def invariant_search(self, df, candidate_sets_direct):
        data = self.transform_data(df)
        y_idx = data[0].shape[1] - 1

        res_df = pd.DataFrame(columns=['pval', 'set'])
        for x_set, s_idx in candidate_sets_direct.items():
            pval = direct_icp_test(y_idx, set(s_idx), data)

            res_df = res_df.append(
                {'pval': pval, 'set': str.join(',', x_set)},
                ignore_index=True)

        return res_df

class InvariantSearch(object):
    def __init__(self, activity_feature, control_feature, y_label='y'):
        self.activity_feature = activity_feature
        self.control_feature = control_feature
        self.main_effect = activity_feature + control_feature
        self.y_label = y_label

    def invariant_search(self, df, candidate_sets, method='full'):
        res_df = pd.DataFrame(columns=['pval', 'set'])

        if method == 'full':
            f_main = f'{self.y_label} ~ ' + str.join('+', self.control_feature)
        elif method == 'effect':
            f_main = f'{self.y_label} ~ ' + str.join('+', ['C(E)*' + x for x in self.main_effect])
        else:
            raise Exception(f"incorrect choice of method: {method}. Only 'full' and 'effect' are supported")

        for x_set in candidate_sets:

            f_interaction = str.join('+', ['A*C(E)*' + x for x in x_set])
            f = f_main + '+' + f_interaction

            model = ols(formula=f, data=df).fit(cov_type='HC1')

            if method == 'full':
                wald_str = str.join(' = 0,', [x for x in model.params.index if 'C(E)' in x])
            elif method == 'effect':
                wald_str = str.join(' = 0,',
                                    [x for x in model.params.index if bool(re.match("^A:C\(E\)\[T.[0-9]+\]", x))]
                                    )
            else:
                raise Exception(f"incorrect choice of method: {method}. Only 'full' and 'effect' are supported")

            pval_wald = model.wald_test('({})'.format(wald_str), use_f=None).pvalue.item()

            res_df = res_df.append(
                {'pval': pval_wald, 'set': str.join(',', x_set)},
                ignore_index=True)

        res_df['set_size'] = res_df.set.str.split(',').apply(len)
        return res_df


class DRLearnerTest(object):
    def __init__(self, main_effect, x_set, a_label='send', y_label='y', name='DR'):
        self.name = name
        self.x_set = x_set
        self.a_label = a_label
        self.f_main = f'{y_label} ~ -1 + ' + str.join('+', ['C(E)*' + x for x in main_effect])
        self.f_interaction = f'{y_label} ~ -1 + ' + str.join('+', self.x_set)
        self.model_y_t0 = StatsModelsLinearRegression()
        self.model_y_t1 = StatsModelsLinearRegression()
        self.model_final = StatsModelsLinearRegression()

    def transform_data(self, df, train=True):
        _, X = patsy.dmatrices(self.f_interaction, df, return_type='matrix')
        X = np.asarray(X)
        T = df[self.a_label].to_numpy()
        if train:
            y, W = patsy.dmatrices(self.f_main, df, return_type='matrix')
            y = np.asarray(y).ravel()
            W = np.asarray(W)
        else:
            return T, X
        return y, T, X, W

    def fit_nuisance(self, df):
        y, T, X, W = self.transform_data(df)
        t0_idx, t1_idx = df[self.a_label] == 0, df[self.a_label] == 1
        self.model_y_t0.fit(W[t0_idx], y[t0_idx])
        self.model_y_t1.fit(W[t1_idx], y[t1_idx])

    def create_pseudo(self, df):
        y, T, X, W = self.transform_data(df)
        a_prob = T.mean()
        self.fit_nuisance(df)
        mu0 = self.model_y_t0.predict(W)
        mu1 = self.model_y_t1.predict(W)

        pd_out = mu1 - mu0 + (T * (y - mu1) / a_prob) - \
                 ((1 - T) * (y - mu0) / (1 - a_prob))

        return pd_out

    def get_res(self, df):
        T, X = self.transform_data(df, train=False)
        pd_out = self.create_pseudo(df)
        self.model_final.fit(X, pd_out)
        pred = self.model_final.predict(X)
        res = pd_out - pred

        return res


class ConstClassifier:
    def __init__(self, probs):
        self._probs = probs

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        probs = np.zeros(shape=(X.shape[0], 2))
        probs[:, 0] = 1 - self._probs
        probs[:, 1] = self._probs
        return probs