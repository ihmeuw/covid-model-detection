from pathlib import Path
from typing import List
from functools import reduce
from tqdm import tqdm

import pandas as pd
import numpy as np

from loguru import logger

from covid_model_detection.utils import ss_from_ci, se_from_ss, linear_to_logit
from covid_model_detection.aggregate import aggregate_data_from_md


def load_serosurveys(model_inputs_root: Path) -> pd.DataFrame:
    '''
    COLUMNS:
        'nid', 'location_id', 'location', 'date', 'survey_series', 'value',
        'units', 'lower', 'upper', 'study_start_age', 'study_end_age',
        'sample_size', 'correction_status', 'bias', 'bias_type',
        'geo_accordance', 'source_population', 'notes', 'link'
    '''
    # load
    data = pd.read_csv(model_inputs_root / 'serology' / 'global_serology_summary.csv',
                       encoding='latin1')
    logger.info(f'Initial observation count: {len(data)}')

    # date formatting
    data['date'] = data['date'].str.strip().replace('.202$', '.2020')
    data.loc[(data['location_id'] == 570) & (data['date'] == '11.08.2021'), 'date'] = '11.08.2020'
    data.loc[(data['location_id'] == 533) & (data['date'] == '13.11.2.2020'), 'date'] = '13.11.2020'
    data.loc[data['date'] == '05.21.2020', 'date'] = '21.05.2020'
    data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')

    # convert to m/l/u to 0-1, sample size to numeric
    if not (data['units'].str.lower().unique() == 'percentage').all():
        raise ValueError('Units other than percentage present.')
    data['lower'] = data['lower'].str.strip().replace('not specified', np.nan).astype(float)
    data['upper'] = data['upper'].str.strip().replace('not specified', np.nan).astype(float)
    data['seroprev_mean'] = data['value'] / 100
    data['seroprev_lower'] = data['lower'] / 100
    data['seroprev_upper'] = data['upper'] / 100
    data['sample_size'] = data['sample_size'].str.strip().replace(('unchecked', 'not specified'), np.nan).astype(float)
    
    data['bias'] = data['bias'].str.strip().replace(('unchecked', 'not specified'), np.nan).astype(float)
    
    outliers = []
    data['manual_outlier'] = data['manual_outlier'].fillna(0)
    manual_outlier = data['manual_outlier']
    outliers.append(manual_outlier)
    logger.info(f'{manual_outlier.sum()} rows from sero data flagged as outliers in ETL.')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## SOME THINGS
    # 1)
    #    Question: How to get complete SS?
    #    Current approach: CI -> SE -> SS where possible; fill with min(SS) where we also don't have CI (very few rows).
    #    Final solution: ...
    ss = ss_from_ci(data['seroprev_mean'], data['seroprev_lower'], data['seroprev_upper'])
    n_missing_ss = (data['sample_size'].isnull() & ss.notnull()).sum()
    n_missing_ss_ci = (data['sample_size'].isnull() & ss.isnull()).sum()
    data['sample_size'] = data['sample_size'].fillna(ss)
    data['sample_size'] = data['sample_size'].fillna(data['sample_size'].min())
    logger.info(f'Inferring sample size from CI for {n_missing_ss} studies; '
                f'filling missing sample size with min observed for {n_missing_ss_ci} that also do not report CI.')
    del n_missing_ss, n_missing_ss_ci
    
    # 2)
    #    Question: What if survey is only in adults? Only kids?
    #    Current approach: Drop beyond some threshold limits.
    #    Final solution: ...
    max_start_age = 20
    min_end_age = 60
    data['study_start_age'] = data['study_start_age'].str.strip().replace('not specified', np.nan).astype(float)
    data['study_end_age'] = data['study_end_age'].str.strip().replace('not specified', np.nan).astype(float)
    too_old = data['study_start_age'] > 20
    too_young = data['study_end_age'] < min_end_age
    age_outlier = (too_old  | too_young).astype(int)
    outliers.append(age_outlier)
    logger.info(f'{age_outlier.sum()} rows from sero data do not have enough '
                f'age coverage (at least ages {max_start_age} to {min_end_age}).')
    
    # 3)
    #    Question: Use of geo_accordance?
    #    Current approach: Drop non-represeentative (geo_accordance == 0).
    #    Final solution: ...
    data['geo_accordance'] = data['geo_accordance'].str.strip().replace(('unchecked', np.nan), '0').astype(int)
    geo_outlier = data['geo_accordance'] == 0
    outliers.append(geo_outlier)
    logger.info(f'{geo_outlier.sum()} rows from sero data do not have `geo_accordance`.')
    data['correction_status'] == data['correction_status'].str.strip().replace(('unchecked', np.nan), '0').astype(int)
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    keep_columns = ['nid', 'location_id', 'date',
                    'seroprev_mean', 'sample_size',
                    'bias', 'bias_type',
                    'correction_status', 'geo_accordance',
                    'is_outlier', 'manual_outlier']
    data['is_outlier'] = pd.concat(outliers, axis=1).max(axis=1).astype(int)
    data = (data
            .loc[:, keep_columns]
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))
    
    logger.info(f"Final inlier count: {len(data.loc[data['is_outlier'] == 0])}")
    logger.info(f"Final outlier count: {len(data.loc[data['is_outlier'] == 1])}")
    
    return data


def load_output_measure(model_inputs_root:Path, measure: str, hierarchy: pd.DataFrame) -> pd.DataFrame:
    data = pd.read_csv(model_inputs_root / 'output_measures' / measure / 'cumulative.csv')
    data['date'] = pd.to_datetime(data['date'])
    is_all_ages = data['age_group_id'] == 22
    is_both_sexes = data['sex_id'] == 3
    data = data.loc[is_all_ages & is_both_sexes]
    data = data.rename(columns={'value':f'cumulative_{measure}'})
    
    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: fill_dates(x, [f'cumulative_{measure}']))
            .reset_index(drop=True))
    data = data.dropna()
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    
    logger.info(f'Aggregating {measure} data.')
    data = aggregate_data_from_md(data, hierarchy, f'cumulative_{measure}')

    return data


def load_testing(testing_root: Path, pop_data: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    # raw_data = pd.read_csv(testing_root / 'data_smooth.csv')
    # raw_data['date'] = pd.to_datetime(raw_data['date'])
    # raw_data = (raw_data
    #             .loc[:, ['location_id', 'date', 'daily_total_reported']]
    #             .dropna()
    #             .reset_index(drop=True))
    # raw_data['cumulative_tests_raw'] = raw_data.groupby('location_id')['daily_total_reported'].cumsum()
    # raw_data = (raw_data.groupby('location_id', as_index=False)
    #             .apply(lambda x: fill_dates(x, ['cumulative_tests_raw']))
    #             .reset_index(drop=True))
    # raw_data['daily_tests_raw'] = (raw_data
    #                                .groupby('location_id')['cumulative_tests_raw']
    #                                .apply(lambda x: x.diff())
    #                                .fillna(raw_data['cumulative_tests_raw']))
    
    data = pd.read_csv(testing_root / 'forecast_raked_test_pc_simple.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    del data['pop']
    del data['population']
    data = data.merge(pop_data)
    data['daily_tests'] = data['test_pc'] * data['population']
    data['cumulative_tests'] = data.groupby('location_id')['daily_tests'].cumsum()
    data = (data
            .loc[:, ['location_id', 'date', 'cumulative_tests']]
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))
    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: fill_dates(x, ['cumulative_tests']))
            .reset_index(drop=True))
    logger.info('Aggregating testing data.')
    data = aggregate_data_from_md(data, hierarchy, 'cumulative_tests')
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    data['daily_tests'] = (data
                           .groupby('location_id')['cumulative_tests']
                           .apply(lambda x: x.diff()))
    data = data.dropna()
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    data['test_days'] = (data['date'] - data.groupby('location_id')['date'].transform(min)).dt.days
    # add 1 so first day is 1, and another since we are starting at t+1
    data['test_days'] = data['test_days'] + 2
    
    # data = data.merge(raw_data, how='left')
    data = data.loc[:, ['location_id', 'date',
                        'daily_tests',  # 'daily_tests_raw',
                        'cumulative_tests',  # 'cumulative_tests_raw',
                        'test_days']]
    
    return data


def load_ifr(infection_fatality_root: Path) -> pd.DataFrame:
    data = pd.read_csv(infection_fatality_root / 'allage_ifr_by_loctime.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'ifr':'ratio'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, 'ratio'])
    
    return data


def load_infections(model_inputs_root:Path, infection_fatality_root: Path, hierarchy: pd.DataFrame) -> pd.DataFrame:
    logger.info('Providing 7-day rolling average of deaths / IFR as infections (indexed on date of death).')
    cumul_deaths = load_output_measure(model_inputs_root, 'deaths', hierarchy)

    ifr = load_ifr(infection_fatality_root)
    
    daily_deaths = (cumul_deaths
                   .sort_values(['location_id', 'date'])
                   .groupby('location_id')
                   .apply(lambda x: x.set_index('date')['cumulative_deaths'].diff())
                   .rename('daily_deaths'))
    daily_deaths = (daily_deaths
                    .reset_index()
                    .groupby('location_id')
                    .apply(lambda x: pd.Series(x['daily_deaths'].rolling(window=7, min_periods=7, center=True).mean().values,
                                            index=x['date']))
                    .dropna())

    cumul_infections = (daily_deaths / ifr).rename('daily_infections').dropna().sort_index().reset_index()
    cumul_infections['cumulative_infections'] = cumul_infections.groupby('location_id')['daily_infections'].cumsum()
    del cumul_infections['daily_infections']
    cumul_infections = aggregate_data_from_md(cumul_infections, hierarchy, 'cumulative_infections')
        
    daily_infections = (cumul_infections
                        .sort_values(['location_id', 'date'])
                        .groupby('location_id')
                        .apply(lambda x: x.set_index('date')['cumulative_infections'].diff())
                        .rename('daily_infections'))
    
    return daily_infections.dropna().to_frame().reset_index()
    

def fill_dates(data: pd.DataFrame, interp_vars: List[str]) -> pd.DataFrame:
    data = data.set_index('date').sort_index()
    data = data.asfreq('D').reset_index()
    data[interp_vars] = data[interp_vars].interpolate(axis=0)
    data['location_id'] = data['location_id'].fillna(method='pad')
    data['location_id'] = data['location_id'].astype(int)

    return data[['location_id', 'date'] + interp_vars]


def load_hierarchy(model_inputs_root:Path) -> pd.DataFrame:
    data = pd.read_csv(model_inputs_root / 'locations' / 'modeling_hierarchy.csv')
    data = data.sort_values('sort_order').reset_index(drop=True)
    
    return data


def load_population(model_inputs_root: Path) -> pd.DataFrame:
    data = pd.read_csv(model_inputs_root / 'output_measures' / 'population' / 'all_populations.csv')
    is_2019 = data['year_id'] == 2019
    is_bothsex = data['sex_id'] == 3
    is_alllage = data['age_group_id'] == 22
    data = (data
            .loc[is_2019 & is_bothsex & is_alllage, ['location_id', 'population']]
            .reset_index(drop=True))

    return data


def get_infection_weighted_avg_testing(infections: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([tests, infections], axis=1)
    data = data.loc[data['daily_tests'].notnull()]
    data['daily_infections'] = data['daily_infections'].fillna(method='bfill')
    if data.isnull().any().any():
        raise ValueError(f"Missing tail infections for location_id {data.reset_index()['location_id'].unique().item()}.")
    if not data.empty:
        infwavg_tests = np.average(data['daily_tests'], weights=(data['daily_infections'] + 1))

        return pd.DataFrame({'infwavg_daily_tests':infwavg_tests},
                            index=data.index[[-1]])
    else:
        return pd.DataFrame()


def prepare_model_data(hierarchy: pd.DataFrame,
                       sero_data: pd.DataFrame,
                       case_data: pd.DataFrame,
                       test_data: pd.DataFrame,
                       infection_data: pd.DataFrame,
                       pop_data: pd.DataFrame,
                       pcr_days: int,
                       sero_days: int,
                       death_days: int,
                       dep_var: str,
                       dep_var_se: str,
                       indep_vars: List[str], **kwargs) -> pd.DataFrame:
    data = reduce(lambda x, y: pd.merge(x, y, how='outer'), [case_data, test_data, pop_data])
    #md_locations = hierarchy.loc[hierarchy['most_detailed'] == 1, 'location_id'].to_list()
    if not data.set_index(['location_id', 'date']).index.is_unique:
        raise ValueError('Non-unique location-date values in combination of case + testing + population data.')

    sero_data = sero_data.copy()
    if sero_days > pcr_days:
        sero_data['date'] -= pd.Timedelta(days=sero_days - pcr_days)
    elif sero_days < pcr_days:
        sero_data['date'] += pd.Timedelta(days=pcr_days - sero_days)
    data = sero_data.merge(data, how='outer')
    
    data['cumulative_case_rate'] = data['cumulative_cases'] / data['population']
    
    infection_data['date'] -= pd.Timedelta(days=death_days-pcr_days)
    sero_location_dates = sero_data[['location_id', 'date']].drop_duplicates()
    sero_location_dates = sero_location_dates.loc[sero_location_dates['location_id'] != 79]
    sero_location_dates = list(zip(sero_location_dates['location_id'], sero_location_dates['date']))
    infwavg_daily_tests = []
    for location_id, date in sero_location_dates:
        infwavg_daily_tests.append(
            get_infection_weighted_avg_testing(
                (infection_data
                 .loc[(infection_data['location_id']==location_id) & (infection_data['date'] <= date)]
                 .set_index(['location_id', 'date'])
                 .loc[:, 'daily_infections']),
                (test_data
                 .loc[(test_data['location_id']==location_id) & (test_data['date'] <= date)]
                 .set_index(['location_id', 'date'])
                 .loc[:, 'daily_tests'])
            )
        )
    infwavg_daily_tests = pd.concat(infwavg_daily_tests)
    data = data.merge(infwavg_daily_tests.reset_index(), how='left')
    
    data['log_avg_daily_testing_rate'] = np.log(data['cumulative_tests'] / (data['population'] * data['test_days']))
    data['daily_testing_rate'] = data['daily_tests'] / data['population']
    data['infwavg_daily_testing_rate'] = data['infwavg_daily_tests'] / data['population']
    data['log_daily_testing_rate'] = np.log(data['daily_testing_rate'])
    data['log_infwavg_daily_testing_rate'] = np.log(data['infwavg_daily_testing_rate'])
    
    data['intercept'] = 1
    data['idr'] = data['cumulative_case_rate'] / data['seroprev_mean']
    # data.loc[data['idr'] > (1 - 1e-4), 'idr'] = (1 - 1e-4)
    # data.loc[data['idr'] < 1e-4, 'idr'] = 1e-4
    data['idr_se'] = se_from_ss(data['idr'], (data['seroprev_mean'] * data['sample_size']))
    data['logit_idr'], data['logit_idr_se'] = linear_to_logit(data['idr'], data['idr_se'])
    # 01/15/21 -- equally weight all points like IFR/IHR models
    data['idr_se'] = 1
    data['logit_idr_se'] = 1
    
    # assign variable for India subnationals
    ind_in_hierarchy = hierarchy['path_to_top_parent'].apply(lambda x: '163' in x.split(','))
    ind_location_ids = hierarchy.loc[ind_in_hierarchy, 'location_id'].to_list()
    ind_in_data = data['location_id'].isin(ind_location_ids).astype(int)
    data['india'] = ind_in_data
    if 'india_test_cov' in indep_vars and 'log_infwavg_daily_testing_rate' in indep_vars:
        data['india_test_cov'] = ind_in_data * data['log_infwavg_daily_testing_rate']
        data['log_infwavg_daily_testing_rate'] *= np.abs(1 - ind_in_data)

        data['india_test_cov_pred'] = ind_in_data * data['log_daily_testing_rate']
        data['log_daily_testing_rate'] *= np.abs(1 - ind_in_data)
    elif 'india_test_cov' in indep_vars:
        raise ValueError('Did not find expected slope variable for India covariate.')
        
    # assign variable for SSA locations
    ssa_in_hierarchy = hierarchy['path_to_top_parent'].apply(lambda x: '166' in x.split(','))
    ssa_location_ids = hierarchy.loc[ssa_in_hierarchy, 'location_id'].to_list()
    ssa_in_data = data['location_id'].isin(ssa_location_ids).astype(int)
    data['ssa'] = ssa_in_data
    if 'ssa_test_cov' in indep_vars and 'log_infwavg_daily_testing_rate' in indep_vars:
        data['ssa_test_cov'] = ssa_in_data * data['log_infwavg_daily_testing_rate']
        data['log_infwavg_daily_testing_rate'] *= np.abs(1 - ssa_in_data)

        data['ssa_test_cov_pred'] = ssa_in_data * data['log_daily_testing_rate']
        data['log_daily_testing_rate'] *= np.abs(1 - ssa_in_data)
    elif 'ssa_test_cov' in indep_vars:
        raise ValueError('Did not find expected slope variable for SSA covariate.')

    
    #logger.info('Trimming out low and high testing points.')
    #data.loc[data['log_avg_daily_testing_rate'] < -7.75, 'is_outlier'] = 1
    
    #logger.info('Trimming out low and high IDR points.')
    #data.loc[data['logit_idr'] < -4, 'is_outlier'] = 1

    data = data.replace((-np.inf, np.inf), np.nan)
    need_vars = ['location_id', 'date', dep_var, dep_var_se] + indep_vars
    data['is_missing'] = data[need_vars].isnull().any(axis=1).astype(int)
    data = data.sort_values(['location_id', 'date', 'nid']).reset_index(drop=True)
    data['data_id'] = data.index
    
    data.loc[data['idr'] >= 1, 'is_outlier'] = 1
    data.loc[data['idr'] <= 0, 'is_outlier'] = 1
    
    is_inlier = data['is_outlier'] == 0
    has_data = data['is_missing'] == 0
    model_data = data.loc[is_inlier & has_data, ['nid', 'data_id'] + need_vars].copy()
    
    logger.info('Modeling with all national and below data.')
    model_locations = hierarchy.loc[hierarchy['level'] >= 3, 'location_id'].to_list()
    model_data = model_data.loc[model_data['location_id'].isin(model_locations)].reset_index(drop=True)
    
    logger.info(f'Final model observations: {len(model_data)}')
    
    return data, model_data


def determine_mean_date_of_infection(location_dates: List,
                                     cumul_cases: pd.DataFrame,
                                     pred_idr: pd.Series) -> pd.DataFrame:
    daily_cases = (cumul_cases
                   .sort_values(['location_id', 'date'])
                   .groupby('location_id')
                   .apply(lambda x: x.set_index('date')['cumulative_cases'].diff())
                   .rename('daily_cases'))
    daily_cases = (daily_cases
                   .fillna(cumul_cases
                           .set_index(['location_id', 'date'])
                           .sort_index()
                           .loc[:,'cumulative_cases']))
    daily_infections = (daily_cases / pred_idr).rename('daily_infections').dropna()

    dates_data = []
    for location_id, date in location_dates:
        data = daily_infections[location_id]
        data = data.reset_index()
        data = data.loc[data['date'] <= date].reset_index(drop=True)
        if not data.empty:
            avg_date_of_infection_idx = int(np.round(np.average(data.index, weights=(data['daily_infections'] + 1))))
            avg_date_of_infection = data.loc[avg_date_of_infection_idx, 'date']
            dates_data.append(pd.DataFrame({'location_id':location_id, 'date':date, 'avg_date_of_infection':avg_date_of_infection}, index=[0]))
    dates_data = pd.concat(dates_data).reset_index(drop=True)

    return dates_data
