import numpy as np
import patsy
from econml.dml import LinearDML

from heartstep_experiment.method import ConstClassifier
from sklearn.ensemble import RandomForestRegressor
import warnings

from utils import to_intervels

warnings.filterwarnings("ignore")


class RandomPolicy(object):
    def __init__(self, p, name):
        self.p = p
        self.name = name

    def train(self, df):
        # do nothing
        None

    def suggest(self, df):
        actions = np.random.binomial(1, p=self.p, size=df.shape[0])

        return actions


class LinearDMLPolicy(object):
    def __init__(self, main_effect, x_set, a_label='send', y_label='y', name='DML'):
        self.name = name
        self.x_set = x_set
        self.a_label = a_label
        self.f_interaction = f'{y_label} ~ -1 + ' + str.join('+', self.x_set)
        self.f_main = f'{y_label} ~ -1 + C(E) +' + str.join("+", main_effect)

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

    def train(self, df, cate_intercept=True, inference=None, cache_values=False):
        y, T, X, W = self.transform_data(df)
        a_prob = T.mean()
        model_t = ConstClassifier(probs=a_prob)

        self.model = LinearDML(model_y=RandomForestRegressor(),
                       model_t=model_t,
                       linear_first_stages=False,
                       discrete_treatment=True,
                       categories=[0, 1], cv=2,
                       fit_cate_intercept=True)

        self.model.fit(y, T, X=X, W=W, inference=inference, cache_values=cache_values)

    def effect(self, df):
        T, X = self.transform_data(df, train=False)
        effect_pred = self.model.effect(X)
        return effect_pred

    def effect_interval(self, df):
        _, X = self.transform_data(df, train=False)
        intervals = to_intervels(self.model.effect_interval(X))
        return intervals

    def suggest(self, df):
        T, X = self.transform_data(df, train=False)
        effect = self.model.effect(X)
        actions = (effect > 0).ravel().astype(int)
        return actions


class BestSubsetDML(LinearDMLPolicy):
    def __init__(self, main_effect, candidate_sets, name, a_label='send', y_label='y'):
        self.name = name
        self.a_label = a_label
        self.candidate_sets = candidate_sets
        self.f_main = f'{y_label} ~ -1 +' + str.join('+', ['C(E)*' + x for x in main_effect])
        self.f_interaction_candidates = {}
        for s in candidate_sets:
            x_set = s.split(',')
            self.f_interaction_candidates[s] = f'{y_label} ~ -1 +' + str.join('+', x_set)
        self.models = {}
        self.selected_model = None

    def effect(self, df):
        T, X = self.transform_data(df, train=False)
        effect_pred = self.selected_model.effect(X)
        return effect_pred

    def effect_interval(self, df):
        _, X = self.transform_data(df, train=False)
        intervals = to_intervels(self.selected_model.effect_interval(X))
        return intervals

    def train(self, df):
        for s in self.candidate_sets:
            self.f_interaction = self.f_interaction_candidates[s]
            super().train(df)
            self.models[s] = self.model

    def suggest(self, test_df):
        ret_actions = None
        max_effect = -np.infty
        selected_set = self.candidate_sets[0]
        for s in self.candidate_sets:
            self.f_interaction = self.f_interaction_candidates[s]
            self.model = self.models[s]
            _, X = self.transform_data(test_df, train=False)
            actions = super().suggest(test_df)
            if ret_actions is None:
                ret_actions = actions
            effect = (self.model.effect(X, T0=0, T1=actions)).mean()
            if effect > max_effect:
                self.selected_model = self.model
                selected_set = s
                max_effect = effect
                ret_actions = actions
        print(f"selected set: {selected_set}")
        return ret_actions