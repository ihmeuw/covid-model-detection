from typing import List
import pandas as pd
import numpy as np


def aggregate(data: pd.DataFrame, parent_id: int, md_child_ids: List[int], agg_var: str) -> pd.Series:
    # not most efficient to go from md each time, but safest since dataset is not square (and not a ton of data)
    if data.empty:
        return data.loc[:, ['location_id', 'date', agg_var]]
    else:
        n_md_locations = data['location_id'].unique().size
        data = data.groupby('date')[agg_var].agg(['sum','count'])
        is_complete = data['count'] == n_md_locations
        data = data.loc[is_complete, 'sum'].rename(agg_var).reset_index()
        data['location_id'] = parent_id
        
        return data.loc[:, ['location_id', 'date', agg_var]]


def aggregate_data_from_md(data: pd.DataFrame, hierarchy: pd.DataFrame, agg_var: str) -> pd.Series:
    # if not agg_var.startswith('cumulative'):
    #     raise ValueError('Expecting cumulative data (double check logic is applicable if used on daily).')
    if data[agg_var].max() < 1:
        raise ValueError(f'Data in {agg_var} looks like rates - need counts for aggregation.')
    
    data = data.copy()
    
    is_md = hierarchy['most_detailed'] == 1
    md_location_ids = hierarchy.loc[is_md, 'location_id'].to_list()
    parent_location_ids = hierarchy.loc[~is_md, 'location_id'].to_list()
    
    md_data = data.loc[data['location_id'].isin(md_location_ids)]
    
    md_child_ids_lists = [(hierarchy
                           .loc[is_md & (hierarchy['path_to_top_parent'].apply(lambda x: str(parent_location_id) in x.split(','))),
                                'location_id']
                           .to_list()) for parent_location_id in parent_location_ids]
    parent_children_pairs = list(zip(parent_location_ids, md_child_ids_lists))
    
    parent_data = [aggregate(md_data.loc[md_data['location_id'].isin(md_child_ids)],
                             parent_id, md_child_ids, agg_var)
                   for parent_id, md_child_ids in parent_children_pairs]
    data = pd.concat([md_data] + parent_data)
    
    return data


