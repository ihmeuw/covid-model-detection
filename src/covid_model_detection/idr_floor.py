from typing import Tuple
from tqdm import tqdm

import pandas as pd
import numpy as np

from loguru import logger


def manual_floor_setting(rmse: pd.DataFrame,
                         best_floor: pd.Series,
                         hierarchy: pd.DataFrame,) -> Tuple[pd.DataFrame, pd.Series]:
    logger.warning('Manually setting IDR floor of 1% in SSA.')
    is_ssa_location = hierarchy['path_to_top_parent'].apply(lambda x: '166' in x.split(','))
    ssa_location_ids = hierarchy.loc[is_ssa_location, 'location_id'].to_list()
    
    is_ssa_rmse = rmse['location_id'].isin(ssa_location_ids)
    rmse.loc[is_ssa_rmse, 'rmse'] = np.nan
    rmse.loc[is_ssa_rmse, 'floor'] = 0.01
    
    best_floor[ssa_location_ids] = 0.01
    
    return rmse, best_floor
    

def find_idr_floor(idr: pd.Series,
                   cumul_cases: pd.Series,
                   serosurveys: pd.Series,
                   population: pd.Series,
                   hierarchy: pd.DataFrame,
                   test_range: Tuple[int, int],
                   ceiling: float,) -> Tuple[pd.DataFrame, pd.Series]:
    daily_cases = (cumul_cases
                   .sort_index()
                   .reset_index()
                   .groupby('location_id')
                   .apply(lambda x: x.set_index('date')['cumulative_cases'].diff()))
    daily_cases = daily_cases.fillna(cumul_cases)
    
    rmse_floor = []
    for floor in range(*test_range):        
        rmse = test_floor_value(idr=idr,
                                daily_cases=daily_cases,
                                serosurveys=serosurveys,
                                population=population,
                                hierarchy=hierarchy,
                                floor=floor/100,
                                ceiling=ceiling,)
        rmse_floor.append(rmse.reset_index())
    rmse = pd.concat(rmse_floor).reset_index(drop=True)
    
    best_floor = rmse.groupby('location_id').apply(lambda x: x.sort_values('rmse')['floor'].values[0]).rename('idr_floor')
    
    rmse, best_floor = manual_floor_setting(rmse, best_floor, hierarchy)
    
    return rmse, best_floor

    
def test_floor_value(idr: pd.Series,
                     daily_cases: pd.Series,
                     serosurveys: pd.Series,
                     population: pd.Series,
                     hierarchy: pd.DataFrame,
                     floor: float,
                     ceiling: float,) -> pd.DataFrame:
    logger.info(f'Testing {int(floor*100)}% IDR floor.')
    daily_infections = daily_cases / idr.clip(floor, ceiling)
    daily_infections = daily_infections.dropna()
    cumul_infections = daily_infections.groupby('location_id').cumsum()
    seroprevalence = (cumul_infections / population).rename('seroprev_mean_pred')

    fit_data = serosurveys.to_frame().join(seroprevalence.to_frame(), how='left')
    if fit_data.isnull().any().any():
        raise ValueError('Cannot compare prediction to data; NA present.')
    residuals = (fit_data['seroprev_mean_pred'] - fit_data['seroprev_mean']).rename('residuals')

    rmses = pd.Series([],
                      name='rmse',
                      index=pd.Index([], name='location_id'))
    for location_id in tqdm(hierarchy.sort_values('level')['location_id']):
        in_path = hierarchy['path_to_top_parent'].apply(lambda x: str(location_id) in x.split(','))
        child_ids = hierarchy.loc[in_path, 'location_id'].to_list()
        is_location = hierarchy['location_id'] == location_id
        parent_id = hierarchy.loc[is_location, 'parent_id'].item()
        # check if location_id is present
        if location_id in residuals.reset_index()['location_id'].to_list():
            rmse = np.sqrt((residuals[location_id]**2).mean())
        # check if children are present
        elif any([l in residuals.reset_index()['location_id'].to_list() for l in child_ids]):
            child_residuals = residuals.reset_index()
            child_residuals = (child_residuals
                               .loc[child_residuals['location_id'].isin(child_ids)]
                               .set_index('date')
                               .loc[:, 'residuals'])
            rmse = np.sqrt((child_residuals**2).mean())
        # check if parent is present
        elif parent_id in rmses.index:
            rmse = rmses[parent_id]
        # we have a problem
        else:
            raise ValueError(f'No source of seroprevalence RMSE for location_id {location_id}.')
        rmse = pd.Series(rmse,
                         name='rmse',
                         index=pd.Index([location_id], name='location_id'))
        rmses = pd.concat([rmses, rmse])
    rmses = rmses.to_frame()
    rmses['floor'] = floor
    
    return rmses