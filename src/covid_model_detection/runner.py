from pathlib import Path
import dill as pickle
from loguru import logger

import pandas as pd
import numpy as np

from covid_shared import cli_tools
from covid_model_detection import data, cascade, model, idr_floor
from covid_model_detection.utils import SERO_DAYS, PCR_DAYS, logit

## TODO:
##     - timeline input (currently saying PCR positive is 11 days and antibody positive is 15)
##     - add bias covariate(s)
##     - write out some more metadata stuff (in different file)

def main(app_metadata: cli_tools.Metadata,
         model_inputs_root: Path, testing_root: Path,
         output_root: Path, n_draws: int):
    np.random.seed(34251)
    
    hierarchy = data.load_hierarchy(model_inputs_root)
    pop_data = data.load_population(model_inputs_root)
    sero_data = data.load_serosurveys(model_inputs_root)
    case_data = data.load_cases(model_inputs_root, hierarchy)
    test_data = data.load_testing(testing_root, pop_data, hierarchy)
    
    var_args = {'dep_var': 'logit_idr',
                'dep_var_se': 'logit_idr_se',
                'indep_vars': ['intercept', 'log_avg_daily_testing_rate'],  # , 'test_days'
                'prior_dict': {'log_avg_daily_testing_rate':{'prior_beta_uniform':np.array([0, np.inf])}},
                'group_vars': [],}
    pred_replace_dict = {'log_daily_testing_rate': 'log_avg_daily_testing_rate'}
    pred_exclude_vars = []
    model_space_suffix = 'avg_testing'
    
    all_data, model_data = data.prepare_model_data(
        hierarchy=hierarchy.copy(),
        sero_data=sero_data.copy(),
        case_data=case_data.copy(),
        test_data=test_data.copy(),
        pop_data=pop_data.copy(),
        pcr_days=PCR_DAYS,
        sero_days=SERO_DAYS,
        **var_args
    )
    
    mr_model_dicts, prior_dicts = cascade.run_cascade(
        model_data=model_data.copy(),
        hierarchy=hierarchy.copy(),
        var_args=var_args.copy(),
        level_lambdas={
            0: 1000.,
            1: 1000.,
            2: 1000.,
            3: 1000.,
            4: 1000.,
        },
    )
    
    pred_idr, pred_idr_fe = cascade.predict_cascade(
        all_data=all_data.copy(),
        hierarchy=hierarchy.copy(),
        mr_model_dicts=mr_model_dicts.copy(),
        pred_replace_dict=pred_replace_dict.copy(),
        pred_exclude_vars=pred_exclude_vars.copy(),
        var_args=var_args.copy(),
    )
    pred_idr_all_data_model_space, pred_idr_all_data_model_space_fe = cascade.predict_cascade(
        all_data=all_data.copy(),
        hierarchy=hierarchy.copy(),
        mr_model_dicts=mr_model_dicts.copy(),
        pred_replace_dict={},
        pred_exclude_vars=[],
        var_args=var_args.copy(),
    )
    all_data = all_data.merge(pred_idr_all_data_model_space.rename(f'pred_idr_{model_space_suffix}').reset_index())
    all_data = all_data.merge(pred_idr_all_data_model_space_fe.rename(f'pred_idr_fe_{model_space_suffix}').reset_index())
    all_data['in_model'] = all_data['data_id'].isin(model_data['data_id'].to_list()).astype(int)

    r2_data = all_data.loc[all_data['in_model'] == 1]
    r2_linear = model.r2_score(r2_data['idr'], r2_data[f'pred_idr_{model_space_suffix}'])
    r2_logit = model.r2_score(logit(r2_data['idr']), logit(r2_data[f'pred_idr_{model_space_suffix}']))
    r2_fe_linear = model.r2_score(r2_data['idr'], r2_data[f'pred_idr_fe_{model_space_suffix}'])
    r2_fe_logit = model.r2_score(logit(r2_data['idr']), logit(r2_data[f'pred_idr_fe_{model_space_suffix}']))
    logger.info(f'R^2 of fixed effects: {r2_fe_logit}')
    logger.info(f'R^2 including random effects: {r2_logit}')
    
    idr_rmse_data, idr_floor_data = idr_floor.find_idr_floor(
        idr=pred_idr.copy(),
        cumul_cases=case_data.set_index(['location_id', 'date'])['cumulative_cases'].copy(),
        serosurveys=all_data.loc[all_data['in_model'] == 1].set_index(['location_id', 'date'])['seroprev_mean'].copy(),
        population=pop_data.set_index('location_id')['population'].copy(),
        hierarchy=hierarchy.copy(),
        test_range=(2, 9),
        ceiling=0.7,
    )
    pred_idr *= (idr_floor_data / pred_idr).clip(1, np.inf)
    pred_idr_fe *= (idr_floor_data / pred_idr_fe).clip(1, np.inf)
    
    data_path = output_root / 'all_data.csv'
    all_data.to_csv(data_path, index=False)
    
    idr_rmse_path = output_root / 'idr_rmse.csv'
    idr_rmse_data.to_csv(idr_rmse_path, index=False)
    
    idr_floor_path = output_root / 'idr_floor.csv'
    idr_floor_data.reset_index().to_csv(idr_floor_path, index=False)
    
    idr_plot_data = all_data.loc[all_data['seroprev_mean'].notnull()].copy()
    idr_plot_data.loc[idr_plot_data['in_model'] == 0, 'is_outlier'] = 1
    idr_plot_data.loc[idr_plot_data['seroprev_mean'] == 0, 'idr'] = 1
    idr_plot_data['idr'] = idr_plot_data['idr'].clip(0, 1)
    dates_data = data.determine_mean_date_of_infection(
        location_dates=idr_plot_data[['location_id', 'date']].drop_duplicates().values.tolist(),
        cumul_cases=case_data.copy(),
        pred_idr=pred_idr.copy()
    )
    start = len(idr_plot_data)
    idr_plot_data = idr_plot_data.merge(dates_data)
    end = len(idr_plot_data)
    if start != end:
        raise ValueError('Mismatch in pairing average date of infection.')
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
    
    model_path = output_root / 'idr_cascade_models.pkl'
    with model_path.open('wb') as file:
        pickle.dump(mr_model_dicts, file, -1)
    
    pred_path = output_root / 'pred_idr.csv'
    pred_idr = pd.concat([pred_idr, pred_idr_fe], axis=1)
    pred_idr.reset_index().to_csv(pred_path, index=False)
