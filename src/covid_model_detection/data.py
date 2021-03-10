from pathlib import Path
from typing import List
from functools import reduce
from tqdm import tqdm

import pandas as pd
import numpy as np

from loguru import logger

from covid_model_detection.utils import ss_from_ci, se_from_ss, linear_to_logit
from covid_model_detection.aggregate import aggregate_data_from_md


def str_fmt(str_col: pd.Series):
    fmt_str_col = str_col.copy()
    fmt_str_col = fmt_str_col.str.lower()
    fmt_str_col = fmt_str_col.str.strip()
    return fmt_str_col


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
    data['date'] = str_fmt(data['date']).replace('.202$', '.2020')
    data.loc[(data['location_id'] == 570) & (data['date'] == '11.08.2021'), 'date'] = '11.08.2020'
    data.loc[(data['location_id'] == 533) & (data['date'] == '13.11.2.2020'), 'date'] = '13.11.2020'
    data.loc[data['date'] == '05.21.2020', 'date'] = '21.05.2020'
    data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')

    # convert to m/l/u to 0-1, sample size to numeric
    if not (str_fmt(data['units']).unique() == 'percentage').all():
        raise ValueError('Units other than percentage present.')
    data['lower'] = str_fmt(data['lower']).replace('not specified', np.nan).astype(float)
    data['upper'] = str_fmt(data['upper']).replace('not specified', np.nan).astype(float)
    data['seroprev_mean'] = data['value'] / 100
    data['seroprev_lower'] = data['lower'] / 100
    data['seroprev_upper'] = data['upper'] / 100
    data['sample_size'] = str_fmt(data['sample_size']).replace(('unchecked', 'not specified'), np.nan).astype(float)
    
    data['bias'] = str_fmt(data['bias']).replace(('unchecked', 'not specified'), np.nan).astype(float)
    
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
    data['study_start_age'] = str_fmt(data['study_start_age']).replace('not specified', np.nan).astype(float)
    data['study_end_age'] = str_fmt(data['study_end_age']).replace('not specified', np.nan).astype(float)
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
    data['geo_accordance'] = str_fmt(data['geo_accordance']).replace(('unchecked', np.nan), '0').astype(int)
    geo_outlier = data['geo_accordance'] == 0
    outliers.append(geo_outlier)
    logger.info(f'{geo_outlier.sum()} rows from sero data do not have `geo_accordance`.')
    data['correction_status'] == str_fmt(data['correction_status']).replace(('unchecked', 'not specified', np.nan), '0').astype(int)
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
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    data['daily_tests'] = (data
                           .groupby('location_id')['cumulative_tests']
                           .apply(lambda x: x.diff()))
    data = data.dropna()
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    data['testing_capacity'] = data.groupby('location_id')['daily_tests'].cummax()
    testing_capacity_data = aggregate_data_from_md(data.loc[:, ['location_id', 'date', 'testing_capacity']].copy(),
                                                   hierarchy, 'testing_capacity')
    del data['testing_capacity']
    data = data.merge(testing_capacity_data, how='left')
    
    data = data.loc[:, ['location_id', 'date',
                        'daily_tests',
                        'testing_capacity',
                        'cumulative_tests',]]
    
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


def load_infections(model_inputs_root:Path, infection_fatality_root: Path, hierarchy: pd.DataFrame,
                    death_days: int,) -> pd.DataFrame:
    logger.info('Providing 7-day rolling average of deaths / IFR as infections.')
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
    
    cumul_infections['date'] -= pd.Timedelta(days=death_days)
        
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


def get_infection_weighted_avg_testing(infections: pd.DataFrame, tests: pd.DataFrame, varname: str) -> pd.DataFrame:
    data = pd.concat([tests.rename('tests'), infections.rename('infections')], axis=1)
    data = data.loc[data['tests'].notnull()]
    data['infections'] = data['infections'].fillna(method='bfill')
    if data.isnull().any().any():
        #raise ValueError(f"Missing tail infections for location_id {data.reset_index()['location_id'].unique().item()}.")
        logger.warning(f"Missing tail infections for location_id {data.reset_index()['location_id'].unique().item()}.")
        data['infections'] = data['infections'].fillna(method='ffill')
    if not data.empty:
        infwavg_tests = np.average(data['tests'], weights=(data['infections'] + 1))

        return pd.DataFrame({varname:infwavg_tests},
                            index=data.index[[-1]])
    else:
        return pd.DataFrame()
    

def add_location_fes(data: pd.DataFrame, parent_id: int, loc_label: str, hierarchy: pd.DataFrame,
                     model_test_var: str, pred_test_var: str, indep_vars: List[str]):
    parent_in_hierarchy = hierarchy['path_to_top_parent'].apply(lambda x: str(parent_id) in x.split(','))
    child_location_ids = hierarchy.loc[parent_in_hierarchy, 'location_id'].to_list()
    child_data = data['location_id'].isin(child_location_ids).astype(int)
    data[loc_label] = child_data
    if f'{loc_label}_test_cov' in indep_vars:
        data[f'{loc_label}_test_cov'] = child_data * data[model_test_var]
        data[model_test_var] *= np.abs(1 - child_data)

        data[f'{loc_label}_test_cov_pred'] = child_data * data[pred_test_var]
        data[pred_test_var] *= np.abs(1 - child_data)
        
    return data


def prepare_model_data(hierarchy: pd.DataFrame,
                       sero_data: pd.DataFrame,
                       case_data: pd.DataFrame,
                       test_data: pd.DataFrame,
                       infection_data: pd.DataFrame,
                       pop_data: pd.DataFrame,
                       pcr_days: int,
                       sero_days: int,
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
    
    infection_data['date'] += pd.Timedelta(days=pcr_days)
    sero_location_dates = sero_data[['location_id', 'date']].drop_duplicates()
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
                 .loc[:, 'testing_capacity']),
                'infwavg_testing_capacity'
            )
        )
    infwavg_daily_tests = pd.concat(infwavg_daily_tests)
    data = data.merge(infwavg_daily_tests.reset_index(), how='left')
    
    data['testing_rate_capacity'] = data['testing_capacity'] / data['population']
    data['infwavg_testing_rate_capacity'] = data['infwavg_testing_capacity'] / data['population']
    data['log_testing_rate_capacity'] = np.log(data['testing_rate_capacity'])
    data['log_infwavg_testing_rate_capacity'] = np.log(data['infwavg_testing_rate_capacity'])
    
    data['intercept'] = 1
    data['idr'] = data['cumulative_case_rate'] / data['seroprev_mean']
    # data.loc[data['idr'] > (1 - 1e-4), 'idr'] = (1 - 1e-4)
    # data.loc[data['idr'] < 1e-4, 'idr'] = 1e-4
    data['idr_se'] = se_from_ss(data['idr'], (data['seroprev_mean'] * data['sample_size']))
    data['logit_idr'], data['logit_idr_se'] = linear_to_logit(data['idr'], data['idr_se'])
    # 01/15/21 -- equally weight all points like IFR/IHR models
    data['idr_se'] = 1
    data['logit_idr_se'] = 1
    
    # assign variable for India, SSA, and Mexico
    for parent_id, loc_label in [(163, 'india'), (166, 'ssa'), (130, 'mexico')]:
        data = add_location_fes(data,
                                parent_id, loc_label,
                                hierarchy,
                                'log_infwavg_testing_rate_capacity', 'log_testing_rate_capacity',
                                indep_vars)
        
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
