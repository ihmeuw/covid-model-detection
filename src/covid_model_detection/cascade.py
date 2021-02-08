from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from mrtool.core.other_sampling import extract_simple_lme_specs, extract_simple_lme_hessian
from covid_model_detection import model


def run_cascade_lh(model_data: pd.DataFrame,
                   hierarchy: pd.DataFrame,
                   var_args: Dict,
                   level_lambdas: Dict):
    '''
    level_lambdas = {
        1:10,
        2:5,
        3:5,
        4:1,
        5:1,
    }
    '''



def run_cascade(lambda_groups: Tuple[int, Dict], lambdas: Dict[int, List]):
    '''
    '''
    pass


def run_location(model_data: pd.DataFrame, prior_dict: Dict, var_args: Dict):
    mr_model, *_ = model.idr_model(model_data=model_data,
                                   prior_dict={'intercept':{'prior_beta_gaussian':beta_soln[:,[0]]},
                                               'log_avg_daily_testing_rate':{'prior_beta_gaussian':beta_soln[:,[1]]},},
                                   **var_args)
    model_specs = extract_simple_lme_specs(mr_model)
    beta_mean = model_specs.beta_soln
    beta_std = np.sqrt(np.diag(np.linalg.inv(extract_simple_lme_hessian(model_specs))))
    beta_soln = np.vstack([beta_mean, beta_std])
    
    return mr_model, beta_soln
