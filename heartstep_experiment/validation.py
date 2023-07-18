import functools

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from heartstep_experiment.policy import BestSubsetDML


def compute_reward(model, df, treatment_label='send', y_label='y'):
    inner_df = df.copy()
    # compute propensity score
    a_prob = inner_df[treatment_label].mean()
    inner_df.loc[:, 'prob'] = (inner_df[treatment_label] * a_prob) + (1 - inner_df[treatment_label]) * (1 - a_prob)

    suggests = model.suggest(inner_df)
    df_rev = inner_df[suggests == inner_df[treatment_label]]
    reward = df_rev[y_label] / df_rev['prob']
    reward = reward.sum() / inner_df.shape[0]
    return reward


# perform leave-one-environment-out validation
def validate_simu(df, search_algo,
                  env,
                  candidate_sets,
                  main_effect, baseline_policies,
                  zero_policy, n_jobs=-2):
    def inner_fn(environment, rng):
        np.random.seed(rng)
        inner_df = pd.DataFrame(columns=['test_user', 'exp_reward', 'model'])
        train_df = df.loc[df['E'] != environment]
        test_df = df.loc[df['E'] == environment]

        effect_inv = search_algo.invariant_search(train_df, candidate_sets, method='effect')
        effect_inv_set = effect_inv[effect_inv.pval > 0.05]
        max_set = effect_inv_set[effect_inv_set.set_size == effect_inv_set.set_size.max()]
        inv_policy = BestSubsetDML(main_effect, list(max_set.set), 'Inv')
        candidate_policies = [inv_policy] + baseline_policies

        env.train(test_df, n_obs=1000)

        for policy in candidate_policies:
            policy.train(train_df)
            reward = env.evaluate(policy) - env.evaluate(zero_policy)

            temp_res = dict(test_user=environment,
                            exp_reward=reward,
                            model=policy.name)
            inner_df = pd.concat([inner_df, pd.DataFrame([temp_res])], ignore_index=True)

        return inner_df

    env_list = df['E'].unique()
    random_state = np.random.randint(np.iinfo(np.int32).max, size=len(env_list))
    ret = Parallel(n_jobs=n_jobs)(delayed(inner_fn)(e, rng) for e, rng in tqdm(list(zip(env_list, random_state))))
    ret_df = functools.reduce(lambda df1, df2: pd.concat([df1, df2], ignore_index=True), ret)

    return ret_df
