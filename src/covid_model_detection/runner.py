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
    case_data = data.load_cases(model_inputs_root)
    test_data = data.load_testing(testing_root)
    
    var_args = {'indep_var': 'logit_idr',
                'indep_var_se': 'logit_idr_se',
                'dep_vars': ['intercept', 'log_avg_daily_testing_rate']}
    pred_replace_dict = {'log_daily_testing_rate': 'log_avg_daily_testing_rate'}
    
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
    pred_idr = model.predict(all_data, fixed_effects, random_effects, pred_replace_dict, **var_args)
    
    pred_idr_model_space = model.predict(all_data, fixed_effects, random_effects, {}, **var_args)
    all_data = all_data.merge(pred_idr_model_space.rename('pred_idr_model_space').reset_index())
    
    data_out = output_root / 'all_data.csv'
    all_data.to_csv(data_out, index=False)
    
    model_out = output_root / 'idr_model.pkl'
    with model_out.open('wb') as file:
        pickle.dump(mr_model, file, -1)
    
    pred_out = output_root / 'pred_idr.csv'
    pred_idr.reset_index().to_csv(pred_out, index=False)
