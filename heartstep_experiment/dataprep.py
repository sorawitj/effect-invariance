import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

def extract_suggestions():
    url = "https://github.com/klasnja/HeartStepsV1/blob/main/data_files/suggestions.csv?raw=true"
    suggest = pd.read_csv(url, low_memory=False)

    suggest = suggest.rename({'user.index': 'user'}, axis=1)
    suggest = suggest.rename({'sugg.select.slot': 'slot'}, axis=1)
    suggest = suggest.rename({'dec.location.category': 'location.category'}, axis=1)
    suggest = suggest.rename({'dec.temperature': 'temperature'}, axis=1)
    suggest['slot'] = suggest['slot'].astype(int)
    suggest['jbsteps30'] = suggest['jbsteps30'].fillna(0)
    suggest['sugg.decision.utime'] = (pd.to_datetime(suggest['sugg.decision.utime']) - pd.Timedelta(hours=4)).dt.floor('d')
    start_dates = suggest.groupby('user')['sugg.decision.utime'].min().reset_index()
    start_dates = start_dates.rename(columns={'sugg.decision.utime': 'start_date'})
    suggest = suggest.merge(start_dates, on='user', how='inner')
    suggest['study.day'] = (suggest['sugg.decision.utime'] - suggest['start_date']).dt.days
    suggest['study.day'] = suggest['study.day'].fillna(0)
    suggest['study.day'] = suggest['study.day'].astype(int)

    return suggest

def get_activity():
    suggest = extract_suggestions()
    suggest = suggest.dropna(
        subset=['jbsteps30', 'jbsteps30pre'])  # Drop rows with NA values in step count before/after DP
    suggest = suggest[suggest['avail'] == True]  # Only when they are available
    suggest = suggest[suggest['intransit'] == False]  # Only when they are not in transit
    suggest = suggest[suggest['connect'] == True]  # Only when they are connected
    suggest = suggest[suggest['snooze.status'] == False]  # Only when they not snoozed

    activity = {}
    # Dosage and square root of steps yesterday
    lambda_dosage = 0.95
    dosage = np.ones(suggest.shape[0])
    square_root_yesterday = np.ones(suggest.shape[0])
    engagements = np.ones(suggest.shape[0])
    variation = np.zeros(suggest.shape[0])
    for user in suggest['user'].unique():
        user_index = suggest['user'] == user
        user_df = suggest.loc[
            user_index, ['decision.index', 'send', 'jbsteps120', 'jbsteps60', 'study.day', 'slot', 'interaction.count']]
        study_days = user_df['study.day']

        # Engagement indicator
        user_engagements = np.zeros(user_df.shape[0])
        interactions_daily = np.zeros(study_days.max() + 1)
        user_df['interaction.count'] = user_df['interaction.count'].replace(np.nan, 0)
        interaction_counts = user_df.groupby(['study.day'])['interaction.count'].sum()
        interactions_daily[interaction_counts.index] = interaction_counts
        for i in range(user_df.shape[0]):
            row = user_df.iloc[i, :]
            day = row['study.day']
            if day > 0:
                engagements_upto = interactions_daily[:(day - 1)]
                if engagements_upto.shape[0] > 0:
                    quantile = np.quantile(engagements_upto, 0.4)
                    user_engagements[i] = int(interactions_daily[day - 1] > quantile)
        engagements[user_index] = user_engagements

        # User's dosage
        decision_index = user_df.loc[:, 'decision.index']
        push_at_index = np.zeros(user_df['decision.index'].max() + 1)
        push_at_index[decision_index] = user_df['send']
        dosage_at_index = np.zeros(push_at_index.shape)
        for index in decision_index:
            if index > 0:
                prev_dose = dosage_at_index[index - 1]
                message_sent = push_at_index[index - 1]
                if message_sent == 0:
                    dosage_at_index[index] = lambda_dosage * prev_dose
                else:
                    dosage_at_index[index] = lambda_dosage * prev_dose + 1

        dosage[user_index] = dosage_at_index[decision_index]

        # Users previous day step count
        study_day_counts = user_df.groupby(['study.day'])['jbsteps120'].sum()
        user_previous_day = np.zeros(user_df.shape[0])
        every_day_count = np.empty(study_days.max() + 1)
        every_day_count[:] = np.nan
        every_day_count[study_day_counts.index] = study_day_counts  # for imputing
        for study_day in study_days:
            if ((study_day - 1 in study_days.unique()) and study_day > 0):
                user_previous_day[user_df['study.day'] == study_day] = np.sqrt(study_day_counts[study_day - 1])
            else:
                past_week = every_day_count[study_day - 7:study_day]
                if (np.nansum(past_week) == 0):
                    user_previous_day[user_df['study.day'] == study_day] = 0
                else:
                    user_previous_day[user_df['study.day'] == study_day] = np.sqrt(np.nanmean(past_week))
        square_root_yesterday[user_index] = user_previous_day

        # Variation indicator
        every_day_count = np.empty((study_days.max() + 1, 5))  # 5 time slots
        every_day_count[:] = np.nan
        # imputing
        for i, row in user_df.iterrows():  # fill known values
            day = row['study.day']
            timeslot = row['slot']
            every_day_count[day, timeslot - 1] = row['jbsteps60']

        for i in range(every_day_count.shape[0]):
            for j in range(every_day_count.shape[1]):
                curr_count = every_day_count[i, j]
                if np.isnan(curr_count):  # impute with past 7 days
                    past_vals = every_day_count[:i, j]
                    past_vals = past_vals[~np.isnan(past_vals)]
                    if past_vals.shape[0] > 0:
                        average = np.mean(past_vals[-7:])
                    else:
                        average = 0
                    every_day_count[i, j] = average

        user_variation = np.ones(user_df.shape[0])
        for i in range(user_df.shape[0]):
            row = user_df.iloc[i, :]
            # get 60 min step counts for past 7 days at same time slot
            day = row['study.day']
            timeslot = row['slot']

            data_upto = every_day_count[:day, timeslot - 1]
            # median of standard dev up to day d
            std_devs = np.array([np.std(data_upto[j - 7:j]) for j in range(7, data_upto.shape[0])])
            if (std_devs.shape[0] > 0):
                median = np.median(std_devs)
            else:
                median = np.nan

            # std past 7 days
            data_pastweek = every_day_count[day - 7:day, timeslot - 1]
            if (data_pastweek.shape[0] > 0):
                todays_std = np.std(data_pastweek)
            else:
                todays_std = np.nan
            if todays_std <= median and np.isnan(todays_std) == False and np.isnan(median) == False:
                user_variation[i] = 0
        variation[user_index] = user_variation

    # Interaction
    suggest['location'] = 'other'
    suggest.loc[suggest['location.category'] == 'home', 'location'] = 'home'
    suggest.loc[suggest['location.category'] == 'work', 'location'] = 'work'
    suggest['location'] = pd.Categorical(suggest['location'], ['work', 'home', 'other'])
    activity['location_other'] = (~suggest['location.category'].isin(['home', 'work'])).astype(int)
    activity['location_work'] = suggest['location.category'].isin(['work']).astype(int)
    activity['decision_idx'] = suggest['decision.index']
    activity['interaction'] = suggest['interaction.count']

    activity['location'] = suggest['location']
    activity['slot'] = suggest['slot']
    activity['dosage'] = dosage
    activity['variation_indicator'] = variation
    activity['engagement'] = engagements

    # Main effect
    activity['square_root_yesterday'] = square_root_yesterday
    activity['temperature'] = suggest['temperature']
    activity['pre_treatment_steps'] = np.log(0.5 + suggest['jbsteps30pre'])

    activity['user'] = suggest['user']
    activity['y'] = np.log(0.5 + suggest['jbsteps30'])
    activity['send'] = suggest['send'].astype(int)
    activity['send_active'] = suggest['send.active']
    activity['send_sedentary'] = suggest['send.sedentary']
    activity_df = pd.DataFrame(data=activity)

    # # Winsorizing continuous features
    activity_df['temperature'] = winsorize(activity_df['temperature'], (0.02, 0.02))
    activity_df['pre_treatment_steps'] = winsorize(activity_df['pre_treatment_steps'], (0.0, 0.02))
    activity_df['square_root_yesterday'] = winsorize(activity_df['square_root_yesterday'], (0.0, 0.02))
    activity_df['dosage'] = winsorize(activity_df['dosage'], (0.0, 0.02))

    return activity_df


def groupby_quantile(x):
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    x.loc[x < q1] = 1
    x.loc[(q1 <= x) & (x < q2)] = 2
    x.loc[x >= q2] = 3

    return x


def pre_process(activity_df, a_label, e_label):
    # filter only users with at least one interaction
    user_interact = activity_df.groupby('user')['interaction'].mean()
    valid_user = user_interact[user_interact > 0].index
    all_df = activity_df[activity_df.user.isin(valid_user)].reset_index(drop=True)

    # define actions and environments
    all_df['A'] = all_df[a_label] - all_df[a_label].mean()
    all_df['E'] = all_df[e_label]
    # compute propensity score
    a_prob = all_df[a_label].mean()
    all_df['prob'] = (all_df[a_label] * a_prob) + (1 - all_df[a_label]) * (1 - a_prob)

    # compute decision time index bucket
    all_df['decision_bucket'] = all_df.groupby('E')['decision_idx'].transform(groupby_quantile)
    bucket_ate = all_df.groupby(['send', 'decision_bucket'])['y'].mean().reset_index()
    bucket_ate = bucket_ate.pivot(index=['decision_bucket'], columns='send')
    bucket_ate['ate'] = bucket_ate[('y', 1)] - bucket_ate[('y', 0)]

    all_df = pd.merge(all_df, bucket_ate[['ate']].reset_index().droplevel(1, axis=1), on=['decision_bucket'],
                      how='inner')

    return all_df
