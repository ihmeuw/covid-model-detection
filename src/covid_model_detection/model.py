from typing import List, Tuple, Dict
from loguru import logger
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
            hierarchy: pd.DataFrame,
            fixed_effects: pd.Series, random_effects: pd.Series,
            pred_replace_dict: Dict,
            dep_var: str,
            indep_vars: List[str], **kwargs) -> pd.Series:
    keep_vars = list(pred_replace_dict.keys()) + indep_vars
    if len(set(keep_vars)) != len(keep_vars):
        raise ValueError('Duplicate in replace_var + indep_vars.')
    pred_data = (all_data
                 .loc[:, ['location_id', 'date'] + keep_vars]
                 .dropna()
                 .drop_duplicates()
                 .set_index(['location_id', 'date'])
                 .sort_index())
    pred_data = pred_data.drop(list(pred_replace_dict.values()), axis=1)
    pred_data = pred_data.rename(columns=pred_replace_dict)
    pred_data_fe = (pred_data
                    .multiply(fixed_effects)
                    .sum(axis=1)
                    .rename(f'{dep_var}_fe'))
    
    # duplicate REs for child locations
    parent_ids = [location_id for location_id in random_effects.index if location_id in hierarchy.loc[hierarchy['most_detailed'] == 0, 'location_id'].to_list()]
    child_ids_lists = [hierarchy.loc[hierarchy['path_to_top_parent'].str.contains(f',{parent_id},'), 'location_id'].to_list() for parent_id in parent_ids]
    child_ids_lists = [list(set(child_ids) - set(random_effects.index)) for child_ids in child_ids_lists]
    parent_children_pairs = list(zip(parent_ids, child_ids_lists))
    parent_children_pairs = [(parent_id, child_ids) for parent_id, child_ids in parent_children_pairs if len(child_ids) > 0]
    
    parent_random_effects = []
    for parent_id, child_ids in parent_children_pairs:
        parent_name = hierarchy.loc[hierarchy['location_id'] == parent_id, 'location_name'].item()
        child_names = hierarchy.loc[hierarchy['location_id'].isin(child_ids), 'location_name'].to_list()
        child_names = ', '.join(child_names)
        logger.info(f'Using parent {parent_name} RE for {child_names}.')
        parent_random_effects.append(pd.Series(random_effects[parent_id], index=pd.Index(child_ids, name='location_id')))
    pd.concat([random_effects] + parent_random_effects)
    random_effects = pd.concat([random_effects] + parent_random_effects).sort_index()
    if not random_effects.index.is_unique:
        raise ValueError('Duplicated random effect in process of applying parents.')
    
    pred_data_w_re = ((pred_data_fe + random_effects)
                       .dropna()
                       .rename(f'{dep_var}_w_re'))
    pred_data = pd.concat([pred_data, pred_data_fe, pred_data_w_re], axis=1)
    pred_data_fe = pred_data[f'{dep_var}_fe']
    pred_data = (pred_data[f'{dep_var}_w_re']
                 .fillna(pred_data[f'{dep_var}_fe'])
                 .rename('idr'))
    pred_data_fe = pred_data_fe.rename('idr_fe')
    if dep_var == 'logit_idr':
        pred_data = expit(pred_data)
        pred_data_fe = expit(pred_data_fe)
    elif dep_var == 'log_idr':
        pred_data = np.exp(pred_data)
        pred_data_fe = np.exp(pred_data_fe)
    elif dep_var != 'idr':
        raise ValueError('Unexpected transformation of IDR in model; cannot predict.')
    
    return pred_data, pred_data_fe
