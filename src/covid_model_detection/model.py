from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

from mrtool import MRData, LinearCovModel, MRBRT

from covid_model_detection.utils import expit

def idr_model(model_data: pd.DataFrame,
              dep_var: str,
              dep_var_se: str,
              indep_vars: List[str],
              inlier_pct: float = 1.) -> Tuple[MRBRT, pd.Series, pd.Series]:
    mr_data = MRData(
        obs=model_data[dep_var].values,
        obs_se=model_data[dep_var_se].values,
        covs={indep_var: model_data[indep_var].values for indep_var in indep_vars},
        study_id=model_data['location_id'].values
    )

    cov_models = [LinearCovModel(indep_var, use_re=indep_var=='intercept') for indep_var in indep_vars]

    mr_model = MRBRT(mr_data, cov_models, inlier_pct=inlier_pct)
    mr_model.fit_model(outer_max_iter=500)

    fixed_effects = pd.Series(
        mr_model.beta_soln,
        name='fixed_effects',
        index=indep_vars
    )
    location_ids = model_data['location_id'].unique()
    random_effects_values = mr_model.extract_re(location_ids)
    if not random_effects_values.shape[1] == 1:
        raise ValueError('Only expecting random intercept.')
    random_effects_values = np.hstack(random_effects_values)
    random_effects = pd.Series(
        data=random_effects_values,
        name='random_effects',
        index=pd.Index(location_ids, name='location_id')
    )

    # trimmed = pd.concat([model_data[['location_id', 'nid', 'date']],
    #                      pd.Series(mr_model.w_soln, name='trimmed')], axis=1)
    
    return mr_model, fixed_effects, random_effects


def predict(all_data: pd.DataFrame,
            fixed_effects: pd.Series, random_effects: pd.Series,
            pred_replace_dict: Dict,
            dep_var: str,
            indep_vars: List[str], **kwargs) -> pd.Series:
    keep_vars = list(pred_replace_dict.keys()) + indep_vars
    if len(set(keep_vars)) != len(keep_vars):
        raise ValueError('Duplicate in replace_var + indep_vars.')
    pred_data = (all_data
                 .loc[:, ['location_id', 'date'] + keep_vars]
                 .drop_duplicates()
                 .set_index(['location_id', 'date'])
                 .sort_index())
    pred_data = pred_data.drop(list(pred_replace_dict.values()), axis=1)
    pred_data = pred_data.rename(columns=pred_replace_dict)
    pred_data_fe = (pred_data
                    .multiply(fixed_effects)
                    .sum(axis=1)
                    .rename(f'{dep_var}_fe'))
    pred_data_w_re = ((pred_data_fe + random_effects)
                       .dropna()
                       .rename(f'{dep_var}_w_re'))
    pred_data = pd.concat([pred_data, pred_data_fe, pred_data_w_re], axis=1)
    pred_data = (pred_data[f'{dep_var}_w_re']
                 .fillna(pred_data[f'{dep_var}_fe'])
                 .rename('idr'))
    if dep_var == 'logit_idr':
        pred_data = expit(pred_data)
    elif dep_var == 'log_idr':
        pred_data = np.exp(pred_data)
    elif dep_var != 'idr':
        raise ValueError('Unexpected transformation of IDR in model; cannot predict.')
    
    return pred_data
