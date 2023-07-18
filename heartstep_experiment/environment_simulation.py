from statsmodels.formula.api import ols
import numpy as np


class LinearEnv(object):
    def __init__(self, main_effect,
                 interaction,
                 y_label='y'):
        self.interaction = interaction
        f_main = f'{y_label} ~ ' + str.join('+', ['A*' + x for x in main_effect])
        f_interaction = str.join('+', ['A*' + x for x in self.interaction])
        self.f = f_main + '+' + f_interaction
        self.model = None
        self.res_df = None

    def train(self, df, n_obs=10000):
        temp_df = df.copy()
        temp_df['A'] = temp_df['send']
        self.model = ols(formula=self.f, data=temp_df).fit()
        temp_df['res'] = self.model.resid
        self.res_df = temp_df.sample(n_obs, replace=True)

    def evaluate(self, policy):
        temp_df = self.res_df.copy()
        temp_df['A'] = policy.suggest(temp_df)
        pred_reward = self.model.predict(temp_df)
        reward = pred_reward + temp_df['res']

        return np.mean(reward)
