from pathlib import Path
import dill as pickle

import pandas as pd

from covid_shared import cli_tools
from covid_model_detection import data, model, idr_floor
from covid_model_detection.utils import SERO_DAYS, PCR_DAYS

## TODO:
##     - timeline input (currently saying PCR positive is 11 days and antibody positive is 15)
##     - confirm `first_case_date.csv` is OK (probably just use earlier of first date of test data or first case date)
##     - add bias covariate(s)
##     - aggregate cases/testing? should check # of aggregates in sero data (i.e. country where we model subnat)
##     - see justification for every dropped data point
##     - assumption that if data exists at national and subnational, it is not redundant.

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
                'indep_vars': ['intercept', 'log_avg_daily_testing_rate'],  # , 'bias'
                'group_vars': ['intercept'],
                'pred_exclude_vars': []}  # 'bias'
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

    # pred_idr_all_data, _ = model.predict(all_data, hierarchy, fixed_effects, random_effects,
    #                                      pred_replace_dict, **var_args)
    pred_idr_all_data_model_space, pred_idr_all_data_model_space_fe = model.predict(
        all_data, hierarchy, fixed_effects, random_effects,
        {}, **var_args
    )
    # all_data = all_data.merge(pred_idr_all_data.rename('pred_idr').reset_index())
    all_data = all_data.merge(pred_idr_all_data_model_space.rename(f'pred_idr_{model_space_suffix}').reset_index())
    all_data = all_data.merge(pred_idr_all_data_model_space_fe.rename(f'pred_idr_fe_{model_space_suffix}').reset_index())
    all_data['in_model'] = all_data['data_id'].isin(model_data['data_id'].to_list()).astype(int)
    if all_data.loc[all_data['in_model'] == 1, 'avg_date_of_test'].isnull().any():
        raise ValueError('Unable to identify average testing date for a modeled data point.')    
        
    idr_rmse_data, idr_floor_data = idr_floor.find_idr_floor(
        idr=pred_idr.copy(),
        cumul_cases=case_data.set_index(['location_id', 'date'])['cumulative_cases'].copy(),
        serosurveys=all_data.loc[all_data['in_model'] == 1].set_index(['location_id', 'date'])['seroprev_mean'].copy(),
        population=pop_data.set_index('location_id')['population'].copy(),
        hierarchy=hierarchy.copy(),
        test_range=(1, 9),
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
    idr_plot_data = (idr_plot_data
                     .loc[:, ['location_id', 'avg_date_of_test', 'idr', 'is_outlier']]
                     .reset_index(drop=True))
    idr_plot_data = idr_plot_data.rename(columns={'avg_date_of_test':'date'})
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
    
    pred_path = output_root / 'pred_idr.csv'
    pred_idr = pd.concat([pred_idr, pred_idr_fe], axis=1)
    pred_idr.reset_index().to_csv(pred_path, index=False)

'''
## SOME MISCELLANEOUS PLOTTING STUFF
import matplotlib.pyplot as plt
from covid_model_detection.utils import logit

plot_data = all_data.loc[all_data['in_model'] == 1]
fig, ax = plt.subplots(1, 2, figsize=(11, 8.5))
ax[0].scatter(plot_data['idr'], plot_data['pred_idr_fe_avg_testing'])
ax[0].plot((0, 1.), (0, 1.), color='red')
ax[1].scatter(plot_data['idr'], plot_data['pred_idr_avg_testing'])
ax[1].plot((0, 1.), (0, 1.), color='red')
fig.show()

plot_data = all_data.loc[all_data['in_model'] == 1]
fig, ax = plt.subplots(1, 2, figsize=(11, 8.5))
ax[0].scatter(logit(plot_data['idr']), logit(plot_data['pred_idr_fe_avg_testing']))
ax[0].plot((-6, 8), (-6, 8), color='red')
ax[1].scatter(logit(plot_data['idr']), logit(plot_data['pred_idr_avg_testing']))
ax[1].plot((-6, 8), (-6, 8), color='red')
ax[1].set_xlim(-6, 4)
ax[1].set_ylim(-6, 4)
fig.show()

plot_data = all_data.loc[all_data['in_model'] == 1]
plt.scatter(plot_data['log_avg_daily_testing_rate'],
            (plot_data['idr'] - plot_data['pred_idr_fe_avg_testing']),
            alpha=0.5)
plt.axhline(0, linestyle='--', color='red')
plt.show()

plot_data = all_data.loc[all_data['in_model'] == 1]
xmin = plot_data[indep_vars[:2]].sort_values(indep_vars[1]).values[0]
ymin = (xmin * fixed_effects[:2]).sum()
xmin = xmin[1]
xmax = plot_data[indep_vars[:2]].sort_values(indep_vars[1]).values[-1]
ymax = (xmax * fixed_effects[:2]).sum()
xmax = xmax[1]
plt.scatter(plot_data['log_avg_daily_testing_rate'],
            plot_data['logit_idr'], alpha=0.25)
plt.plot((xmin, xmax), (ymin, ymax), color='red')
#plt.xlim(-9, -5.5)
#plt.ylim(-4, 2)
plt.show()
'''
