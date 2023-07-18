import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

reward_df = pd.read_csv("reward_df.csv")

X = reward_df[reward_df.model == 'Inv'].sort_values(by='test_user')['exp_reward']
Y = reward_df[reward_df.model == 'Full'].sort_values(by='test_user')['exp_reward']

wilcoxon(x = X, y= Y, alternative='greater')

