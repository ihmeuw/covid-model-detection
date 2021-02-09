from typing import Dict, List, Tuple
from loguru import logger

import pandas as pd
import numpy as np

from mrtool.core.other_sampling import extract_simple_lme_specs, extract_simple_lme_hessian
from covid_model_detection import model


def run_cascade(model_data: pd.DataFrame,
                hierarchy: pd.DataFrame,
                var_args: Dict,
                level_lambdas: Dict):
    '''
    NOTE: `level_lambdas` apply to the stdev of the level to which they are keyed, and thus
        as priors for the next level. If no new data is added, it is multiplicatively applied.
        
    NOTE: If country level data is present, only using that in country model, and is thus the 
        estimate on which predictions for subnational locations without data are based.
    '''
    locs_in_model_path = hierarchy.loc[hierarchy['location_id'].isin(model_data['location_id'].to_list()), 'path_to_top_parent'].to_list()
    locs_in_model_path = list(set([int(l) for p in locs_in_model_path for l in p.split(',')]))
    is_cascade_location = hierarchy['location_id'].isin(locs_in_model_path)
    cascade_hierarchy = (hierarchy
                         .loc[is_cascade_location, ['location_id', 'level']])
    cascade_hierarchy = [(level, cascade_hierarchy.loc[cascade_hierarchy['level'] == level, 'location_id'].to_list()) for level in sorted(cascade_hierarchy['level'].unique())]
    
    uninformative_prior_dict = {indep_var:{'prior_beta_gaussian':np.array([[0], [np.inf]])} for indep_var in var_args['indep_vars']}
    mr_model_dicts = {}
    prior_dicts = {1:uninformative_prior_dict}
    for level, location_ids in cascade_hierarchy:
        logger.info(f'Modeling hierarchy level {level} ({len(location_ids)} location-models).')
        level_mr_model_dicts, level_prior_dict = run_level(
            level_lambda=level_lambdas[level],
            level=level,
            location_ids=location_ids,
            model_data=model_data,
            hierarchy=hierarchy,
            prior_dicts=prior_dicts,
            var_args=var_args,
        )
        if level == 0:
            prior_dicts = {}
        mr_model_dicts.update(level_mr_model_dicts)
        prior_dicts.update(level_prior_dict)
        
    return mr_model_dicts, prior_dicts


def run_level(level_lambda: int,
              level: int,
              location_ids: List[int],
              model_data: pd.DataFrame,
              hierarchy: pd.DataFrame,
              prior_dicts: Dict,
              var_args: Dict,
              child_cutoff_level: int = 3,):
    level_mr_model_dicts = {}
    level_prior_dicts = {}
    for location_id in location_ids:
        parent_id = hierarchy.loc[hierarchy['location_id'] == location_id, 'parent_id'].item()
        parent_prior_dict = prior_dicts[parent_id]
        location_in_path_hierarchy = hierarchy['path_to_top_parent'].apply(lambda x: str(location_id) in x.split(','))
        if level <= child_cutoff_level and location_id in model_data['location_id'].to_list():
            child_locations = [location_id]
        else:
            child_locations = hierarchy.loc[location_in_path_hierarchy, 'location_id'].to_list()
        location_in_path_model = model_data['location_id'].isin(child_locations)
        location_model_data = model_data.loc[location_in_path_model].copy()
        location_mr_model, location_prior_dict = run_location(
            model_data=location_model_data,
            prior_dict=parent_prior_dict,
            level_lambda=level_lambda,
            var_args=var_args
        )
        level_mr_model_dicts.update({location_id:location_mr_model})
        level_prior_dicts.update({location_id:location_prior_dict})
    
    return level_mr_model_dicts, level_prior_dicts


def run_location(model_data: pd.DataFrame, prior_dict: Dict, level_lambda: int, var_args: Dict):
    mr_model, *_ = model.idr_model(
        model_data=model_data,
        prior_dict=prior_dict,
        **var_args
    )
    model_specs = extract_simple_lme_specs(mr_model)
    beta_mean = model_specs.beta_soln
    beta_std = np.sqrt(np.diag(np.linalg.inv(extract_simple_lme_hessian(model_specs))))
    beta_std *= level_lambda
    beta_soln = np.vstack([beta_mean, beta_std])
    prior_dict = {indep_var:{'prior_beta_gaussian':beta_soln[:,[i]]} for \
                  i, indep_var in enumerate(var_args['indep_vars'])}

    return mr_model, prior_dict


def find_nearest_modeled_parent(path_to_top_parent_str = str,
                                modeled_locations = List[int],):
    path_to_top_parent = list(reversed([int(l) for l in path_to_top_parent_str.split(',')]))
    for location_id in path_to_top_parent:
        if location_id in modeled_locations:
            return location_id
    raise ValueError(f'No modeled location present in hierarchy for {path_to_top_parent[0]}.')


def predict_cascade(all_data: pd.DataFrame,
                    hierarchy: pd.DataFrame,
                    mr_model_dicts: Dict,
                    pred_replace_dict: Dict,
                    var_args: Dict,):
    logger.info('Compiling predictions.')
    random_effects = pd.DataFrame(index=pd.Index([], name='location_id'))
    modeled_locations = list(mr_model_dicts.keys())
    model_location_map = {l: find_nearest_modeled_parent(p, modeled_locations) for l, p in zip(hierarchy['location_id'].to_list(), hierarchy['path_to_top_parent'].to_list())}
    
    pred_idr = []
    pred_idr_fe = []
    for location_id in hierarchy['location_id'].to_list():
        global_effect = pd.Series({k: v.item() for k, v in mr_model_dicts[1].fe_soln.items()}).rename('fixed_effects')
        location_pred_idr_fe, _ = model.predict(
            all_data=all_data.loc[all_data['location_id'] == location_id],
            hierarchy=hierarchy,
            fixed_effects=global_effect,
            random_effects=random_effects,
            pred_replace_dict=pred_replace_dict,
            **var_args
        )
        pred_idr_fe += [location_pred_idr_fe.rename('idr_fe')]
        
        model_location_id = model_location_map[location_id]
        if location_id != model_location_id:
            logger.info(f'Using model for {model_location_id} in prediction for {location_id}.')
        location_specific_effect = pd.Series({k: v.item() for k, v in mr_model_dicts[model_location_id].fe_soln.items()}).rename('fixed_effects')
        location_pred_idr, _ = model.predict(
            all_data=all_data.loc[all_data['location_id'] == location_id],
            hierarchy=hierarchy,
            fixed_effects=location_specific_effect,
            random_effects=random_effects,
            pred_replace_dict=pred_replace_dict,
            **var_args
        )
        pred_idr += [location_pred_idr]
    pred_idr = pd.concat(pred_idr)
    pred_idr_fe = pd.concat(pred_idr_fe)
    
    return pred_idr, pred_idr_fe
