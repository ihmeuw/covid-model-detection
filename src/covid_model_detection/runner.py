from pathlib import Path
import dill as pickle

import pandas as pd

from covid_shared import cli_tools
from covid_model_detection import data, model
from covid_model_detection.utils import SERO_DAYS, PCR_DAYS

## TODO:
##     - timeline input (currently saying PCR positive is 11 days and antibody positive is 15)
##     - confirm `first_case_date.csv` is OK (probably just use earlier of first date of test data or first case date)
##     - add bias covariate(s)
##     - aggregate cases/testing? should check # of aggregates in sero data (i.e. country where we model subnat)
##     - see justification for every dropped data point

def main(app_metadata: cli_tools.Metadata,
         model_inputs_root: Path, testing_root: Path,
         output_root: Path, n_draws: int):
    hierarchy = data.load_hierarchy(model_inputs_root)
    pop_data = data.load_population(model_inputs_root)
    sero_data = data.load_serosurveys(model_inputs_root)
    case_data = data.load_cases(model_inputs_root, hierarchy)
    test_data = data.load_testing(testing_root, hierarchy)
    
    var_args = {'dep_var': 'logit_idr',
                'dep_var_se': 'logit_idr_se',
                'indep_vars': ['intercept', 'log_avg_daily_testing_rate']}
    pred_replace_dict = {'log_daily_testing_rate': 'log_avg_daily_testing_rate'}
    model_space_suffix = 'avg_testing'
    
    all_data, model_data = data.prepare_model_data(
        hierarchy=hierarchy,
        sero_data=sero_data,
        case_data=case_data,
        test_data=test_data,
        pop_data=pop_data,
        pcr_days=PCR_DAYS,
        sero_days=SERO_DAYS,
        **var_args
    )
    
    mr_model, fixed_effects, random_effects = model.idr_model(model_data=model_data, **var_args)
    pred_idr, pred_idr_fe = model.predict(all_data, hierarchy, fixed_effects, random_effects, pred_replace_dict, **var_args)

    pred_idr_all_data, _ = model.predict(all_data, hierarchy, fixed_effects, random_effects,
                                         pred_replace_dict, **var_args)
    pred_idr_all_data_model_space, _ = model.predict(all_data, hierarchy, fixed_effects, random_effects,
                                                     {}, **var_args)
    all_data = all_data.merge(pred_idr_all_data.rename('pred_idr').reset_index())
    all_data = all_data.merge(pred_idr_all_data_model_space.rename(f'pred_idr_{model_space_suffix}').reset_index())
    all_data['in_model'] = all_data['data_id'].isin(model_data['data_id'].to_list()).astype(int)
    
    sero_path = output_root / 'sero_data.csv'
    sero_data.to_csv(sero_path, index=False)
    
    test_path = output_root / 'test_data.csv'
    test_data.to_csv(test_path, index=False)
    
    data_path = output_root / 'all_data.csv'
    all_data.to_csv(data_path, index=False)
    
    model_path = output_root / 'idr_model.pkl'
    with model_path.open('wb') as file:
        pickle.dump((mr_model, fixed_effects, random_effects), file, -1)
    
    pred_path = output_root / 'pred_idr.csv'
    pred_idr = pd.concat([pred_idr, pred_idr_fe], axis=1)
    pred_idr.reset_index().to_csv(pred_path, index=False)
