from pathlib import Path
import dill as pickle
from loguru import logger

import pandas as pd
import numpy as np

from covid_shared import cli_tools
from covid_model_detection import data, model, idr_floor
from covid_model_detection.utils import SERO_DAYS, PCR_DAYS, DEATH_DAYS, logit

## TODO:
##     - timeline input (currently saying PCR positive is 11 days and antibody positive is 15)
##     - add bias covariate(s)
##     - check aggregation

def main(app_metadata: cli_tools.Metadata,
         model_inputs_root: Path, testing_root: Path, infection_fatality_root: Path,
         output_root: Path, n_draws: int):
    hierarchy = data.load_hierarchy(model_inputs_root)
    pop_data = data.load_population(model_inputs_root)
    sero_data = data.load_serosurveys(model_inputs_root)
    case_data = data.load_output_measure(model_inputs_root, 'cases', hierarchy)
    test_data = data.load_testing(testing_root, pop_data, hierarchy)
    infection_data = data.load_infections(model_inputs_root, infection_fatality_root, hierarchy)
    
    var_args = {'dep_var': 'logit_idr',
                'dep_var_se': 'logit_idr_se',
                'indep_vars': ['intercept', 'log_infwavg_daily_testing_rate',  # , 'bias'
                               #'india', 'india_test_cov',
                               #'ssa',
                               'ssa_test_cov',
                              ],
                'group_vars': ['log_infwavg_daily_testing_rate'],
                'pred_exclude_vars': []}  # 'bias'
    pred_replace_dict = {'log_daily_testing_rate': 'log_infwavg_daily_testing_rate',
                         #'india_test_cov_pred': 'india_test_cov',
                         'ssa_test_cov_pred': 'ssa_test_cov',
                        }
    model_space_suffix = 'infwavg_testing'
    
    all_data, model_data = data.prepare_model_data(
        hierarchy=hierarchy.copy(),
        sero_data=sero_data.copy(),
        case_data=case_data.copy(),
        test_data=test_data.copy(),
        infection_data=infection_data.copy(),
        pop_data=pop_data.copy(),
        pcr_days=PCR_DAYS,
        sero_days=SERO_DAYS,
        death_days=DEATH_DAYS,
        **var_args
    )
    
    mr_model, fixed_effects, random_effects = model.idr_model(model_data=model_data, **var_args)
    pred_idr, pred_idr_fe = model.predict(
        all_data, hierarchy, fixed_effects, random_effects,
        pred_replace_dict, **var_args
    )
    pred_idr_all_data_model_space, pred_idr_all_data_model_space_fe = model.predict(
        all_data, hierarchy, fixed_effects, random_effects,
        {}, **var_args
    )
    '''
    # ADD DEFAULT DIAGNOSTIC PLOTS
    plot_data = all_data.merge(pred_idr.rename('pred_idr').reset_index(),
                               how='left')
    plot_data = plot_data.merge(pred_idr_all_data_model_space.rename(f'pred_idr_{model_space_suffix}').reset_index(),
                                how='left')
    plot_data = plot_data.loc[plot_data['seroprev_mean'].notnull()]
    plot_data = plot_data.loc[plot_data['idr'] <= 1]

    plt.scatter(plot_data['infwavg_daily_testing_rate'], plot_data['idr'])
    plt.show()

    plt.scatter(plot_data['log_infwavg_daily_testing_rate'], plot_data['logit_idr'])
    plt.show()

    plt.scatter(expit(plot_data['logit_idr']), plot_data['pred_idr_infwavg_testing'])
    plt.plot((0, 1), (0, 1), color='red')
    plt.show()
    '''
    all_data = all_data.merge(pred_idr_all_data_model_space.rename(f'pred_idr_{model_space_suffix}').reset_index(),
                              how='left')
    all_data = all_data.merge(pred_idr_all_data_model_space_fe.rename(f'pred_idr_fe_{model_space_suffix}').reset_index(),
                              how='left')
    all_data['in_model'] = all_data['data_id'].isin(model_data['data_id'].to_list()).astype(int)
    all_data.loc[all_data['in_model'] != 1, f'pred_idr_{model_space_suffix}'] = np.nan
    all_data.loc[all_data['in_model'] != 1, f'pred_idr_fe_{model_space_suffix}'] = np.nan

    r2_data = all_data.loc[all_data['in_model'] == 1]
    r2_linear = model.r2_score(r2_data['idr'], r2_data[f'pred_idr_{model_space_suffix}'])
    r2_logit = model.r2_score(logit(r2_data['idr']), logit(r2_data[f'pred_idr_{model_space_suffix}']))
    r2_fe_linear = model.r2_score(r2_data['idr'], r2_data[f'pred_idr_fe_{model_space_suffix}'])
    r2_fe_logit = model.r2_score(logit(r2_data['idr']), logit(r2_data[f'pred_idr_fe_{model_space_suffix}']))
    logger.info(f'R^2 of fixed effects - Logit: {r2_fe_logit}, Linear: {r2_fe_linear}')
    logger.info(f'R^2 including random effects: - Logit: {r2_logit}, Linear: {r2_linear}')
    
    idr_rmse_data, idr_floor_data = idr_floor.find_idr_floor(
        idr=pred_idr.copy(),
        cumul_cases=case_data.set_index(['location_id', 'date'])['cumulative_cases'].copy(),
        serosurveys=all_data.loc[all_data['in_model'] == 1].set_index(['location_id', 'date'])['seroprev_mean'].copy(),
        population=pop_data.set_index('location_id')['population'].copy(),
        hierarchy=hierarchy.copy(),
        test_range=(1, 11),
        ceiling=1.,
    )
    pred_idr = (pred_idr
                .reset_index()
                .set_index('location_id')
                .join(idr_floor_data, how='left'))
    pred_idr = (pred_idr
                .groupby('location_id')
                .apply(lambda x: idr_floor.rescale_idr(x.loc[:, ['date', 'idr']], x['idr_floor'].unique().item(), ceiling=1))
                .rename('idr'))
    
    data_path = output_root / 'all_data.csv'
    all_data.to_csv(data_path, index=False)
    
    idr_rmse_path = output_root / 'idr_rmse.csv'
    idr_rmse_data.to_csv(idr_rmse_path, index=False)

    idr_floor_path = output_root / 'idr_floor.csv'
    idr_floor_data.reset_index().to_csv(idr_floor_path, index=False)
    
    idr_plot_data = all_data.loc[all_data['seroprev_mean'].notnull()].copy()
    idr_plot_data = idr_plot_data.loc[idr_plot_data['location_id'].isin(hierarchy['location_id'].to_list())]
    idr_plot_data.loc[idr_plot_data['in_model'] == 0, 'is_outlier'] = 1
    idr_plot_data.loc[idr_plot_data['seroprev_mean'] == 0, 'idr'] = 1
    idr_plot_data['idr'] = idr_plot_data['idr'].clip(0, 1)
    dates_data = data.determine_mean_date_of_infection(
        location_dates=idr_plot_data[['location_id', 'date']].drop_duplicates().values.tolist(),
        cumul_cases=case_data.copy(),
        pred_idr=pred_idr.copy()
    )
    idr_plot_data = idr_plot_data.merge(dates_data, how='left')
    if len(idr_plot_data.loc[(idr_plot_data['avg_date_of_infection'].isnull()) & (idr_plot_data['is_outlier'] != 1)]) > 0:
        raise ValueError('Cannot find avg date of infection for data in model.')
    else:
        idr_plot_data['avg_date_of_infection'] = idr_plot_data['avg_date_of_infection'].fillna(idr_plot_data['date'])
    idr_plot_data = (idr_plot_data
                     .loc[:, ['location_id', 'avg_date_of_infection', 'idr', 'is_outlier']]
                     .reset_index(drop=True))
    idr_plot_data = idr_plot_data.rename(columns={'avg_date_of_infection':'date'})
    idr_plot_data_path = output_root / 'idr_plot_data.csv'
    idr_plot_data.to_csv(idr_plot_data_path, index=False)
    
    sero_path = output_root / 'sero_data.csv'
    sero_data = sero_data.rename(columns={'date':'survey_date'})
    sero_data['infection_date'] = sero_data['survey_date'] - pd.Timedelta(days=SERO_DAYS)
    sero_data.to_csv(sero_path, index=False)
    
    test_path = output_root / 'test_data.csv'
    test_data.to_csv(test_path, index=False)
    
    model_path = output_root / 'idr_model.pkl'
    with model_path.open('wb') as file:
        pickle.dump((mr_model, fixed_effects, random_effects), file, -1)
    
    # only save most-detailed predictions for IDR until aggregation post-scaling can be applied
    pred_path = output_root / 'pred_idr.csv'
    md_locs = hierarchy.loc[hierarchy['most_detailed'] == 1, 'location_id'].to_list()
    md_locs = [i for i in md_locs if i in all_data['location_id']]
    pred_idr = pred_idr.loc[md_locs]
    pred_idr_fe = pred_idr_fe.loc[md_locs]
    pred_idr = pd.concat([pred_idr, pred_idr_fe], axis=1)
    pred_idr.reset_index().to_csv(pred_path, index=False)
