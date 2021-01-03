from pathlib import Path
from typing import List
from functools import reduce

import pandas as pd
import numpy as np

from loguru import logger

from covid_model_detection.utils import ss_from_ci, se_from_ss, linear_to_logit

def load_serosurveys(model_inputs_root: Path) -> pd.DataFrame:
    '''
    COLUMNS:
        'nid', 'location_id', 'location', 'date', 'survey_series', 'value',
        'units', 'lower', 'upper', 'study_start_age', 'study_end_age',
        'sample_size', 'correction_status', 'bias', 'bias_type',
        'geo_accordance', 'source_population', 'notes', 'link'
    '''
    # load
    data = pd.read_csv('/ihme/covid-19/snapshot-data/best/covid_onedrive/Serological studies/global_serology_summary.csv',
                       encoding='latin1')
    logger.info(f'Initial observation count: {len(data)}')

    # date formatting
    data['date'] = data['date'].str.replace('.202$|.2021$', '.2020')
    data.loc[data['date'] == '05.21.2020', 'date'] = '21.05.2020'
    data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')

    # convert to m/l/u to 0-1, sample size to numeric
    if not (data['units'].unique() == 'percentage').all():
        raise ValueError('Units other than percentage present.')
    data[['lower', 'upper']] = data[['lower', 'upper']].replace('not specified', np.nan).astype(float)
    data['seroprev_mean'] = data['value'] / 100
    data['seroprev_lower'] = data['lower'] / 100
    data['seroprev_upper'] = data['upper'] / 100
    data['sample_size'] = data['sample_size'].replace(('unchecked', 'not specified'), np.nan).astype(float)
    
    # only keep geo_accordance data ("nationally representative" ?)
    is_representative = data['geo_accordance'] == '1'
    start_len = len(data)
    data = data[is_representative]
    end_len = len(data)
    logger.info(f'Dropping {start_len - end_len} rows from sero data due to not having `geo_accordance`.')
    del start_len, end_len
    data['correction_status'] == data['correction_status'].replace('unchecked', '0').astype(int)
    
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## SOME THINGS
    # 1) 
    #    Question: What if survey is only in adults? Only kids?
    #    Current approach: Drop at some threshold age
    #    Final solution: ...
    max_start_age = 20
    min_end_age = 60
    data[['study_start_age', 'study_end_age']] = data[['study_start_age', 'study_end_age']].replace('not specified', np.nan).astype(float)
    too_old = data['study_start_age'] > 20
    too_young = data['study_end_age'] < min_end_age
    start_len = len(data)
    data = data.loc[~too_old  & ~too_young]
    end_len = len(data)
    logger.info(f'Dropping {start_len - end_len} rows from sero data due to not having enough '
                f'age coverage (at least ages {max_start_age} to {min_end_age}).')
    del start_len, end_len


    # 2) 
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
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    keep_columns = ['nid', 'location_id', 'date',
                    'seroprev_mean', 'sample_size',
                    'bias', 'bias_type', 'correction_status']
    data = (data
            .loc[:, keep_columns]
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))
    
    return data


def load_cases(model_inputs_root:Path) -> pd.DataFrame:
    data = pd.read_csv(model_inputs_root / 'output_measures' / 'cases' / 'cumulative.csv')
    data['date'] = pd.to_datetime(data['date'])
    is_all_ages = data['age_group_id'] == 22
    is_both_sexes = data['sex_id'] == 3
    data = data.loc[is_all_ages & is_both_sexes]
    data = data.rename(columns={'value':'cumulative_cases'})
    
    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: fill_dates(x, ['cumulative_cases']))
            .reset_index(drop=True))
    data = data.dropna()
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)

    return data


def load_testing(testing_root: Path) -> pd.DataFrame:
    raw_data = pd.read_csv(testing_root / 'data_smooth.csv')
    raw_data['date'] = pd.to_datetime(raw_data['date'])
    raw_data = (raw_data
                .loc[:, ['location_id', 'date', 'daily_total_reported']]
                .dropna()
                .reset_index(drop=True))
    raw_data['cumulative_tests_raw'] = raw_data.groupby('location_id')['daily_total_reported'].cumsum()
    raw_data = (raw_data.groupby('location_id', as_index=False)
                .apply(lambda x: fill_dates(x, ['cumulative_tests_raw']))
                .reset_index(drop=True))
    raw_data['daily_tests_raw'] = (raw_data
                                   .groupby('location_id')['cumulative_tests_raw']
                                   .apply(lambda x: x.diff())
                                   .fillna(raw_data['cumulative_tests_raw']))
    
    data = pd.read_csv(testing_root / 'forecast_raked_test_pc_simple.csv')
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    data['daily_tests'] = data['test_pc'] * data['population']
    data['cumulative_tests'] = data.groupby('location_id')['daily_tests'].cumsum()
    data = data.merge(raw_data, how='left')
    
    first_date_data = pd.read_csv(testing_root / 'first_case_date.csv')
    first_date_data['first_case_date'] = pd.to_datetime(first_date_data['first_case_date'])
    data = data.merge(first_date_data)
    data['case_days'] = (data['date'] - data['first_case_date']).dt.days + 1
    data = data.loc[:, ['location_id', 'date',
                        'daily_tests_raw', 'daily_tests',
                        'cumulative_tests_raw', 'cumulative_tests',
                        'case_days']]
    
    return data


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


def prepare_model_data(hierarchy: pd.DataFrame,
                       sero_data: pd.DataFrame,
                       case_data: pd.DataFrame,
                       test_data: pd.DataFrame,
                       pop_data: pd.DataFrame,
                       pcr_days: int,
                       sero_days: int,
                       indep_var: str = 'logit_idr',
                       indep_var_se: str = 'logit_idr_se',
                       dep_vars: List[str] = ['intercept', 'log_avg_daily_testing_rate']) -> pd.DataFrame:
    data = reduce(lambda x, y: pd.merge(x, y, how='outer'), [case_data, test_data, pop_data])
    md_locations = hierarchy.loc[hierarchy['most_detailed'] == 1, 'location_id'].to_list()
    data = data.loc[data['location_id'].isin(md_locations)]

    if sero_days > pcr_days:
        data['date'] += pd.Timedelta(days=sero_days - pcr_days)
    elif sero_days < pcr_days:
        data['date'] += pd.Timedelta(days=pcr_days - sero_days)
    data = sero_data.merge(data, how='outer')
    
    data['cumulative_case_rate'] = data['cumulative_cases'] / data['population']
    
    data['log_avg_daily_testing_rate'] = np.log(data['cumulative_tests'] / (data['population'] * data['case_days']))
    data['log_daily_testing_rate'] = np.log(data['daily_tests'] / data['population'])
    
    data['idr'] = data['cumulative_case_rate'] / data['seroprev_mean']
    data['idr_se'] = se_from_ss(data['idr'], (data['seroprev_mean'] * data['sample_size']))
    data['logit_idr'], data['logit_idr_se'] = linear_to_logit(data['idr'], data['idr_se'])
    data['intercept'] = 1

    data = data.replace((-np.inf, np.inf), np.nan)
    need_vars = ['location_id', 'date', indep_var, indep_var_se] + dep_vars
    no_nan = data[need_vars].notnull().all(axis=1)
    all_data = data.copy()
    data = data.loc[no_nan, ['nid'] + need_vars]
    data = data.sort_values(['location_id', 'date', 'nid']).reset_index(drop=True)
    
    has_date = all_data['date'].notnull()
    all_data = all_data.loc[has_date]
    all_data = all_data.sort_values(['location_id', 'date', 'nid']).reset_index(drop=True)
    
    logger.info(f'Final observation count: {len(data)}')
    
    return all_data, data
