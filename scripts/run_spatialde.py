import sys
import os
import logging
sys.path.insert(0,'../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import NaiveDE
import SpatialDE

from io_utils import save_data_to_file, load_data_from_file
from general_utils import evaluate_rejections, norm_mtx

def read_trial_data(sim_dir, entry):
    lfn = os.path.join(sim_dir, entry['locs_fn'])
    locs = load_data_from_file(lfn, 'csv')
    dfn = os.path.join(sim_dir, entry['expr_fn'])
    data = load_data_from_file(dfn, 'csv')
    return locs, data
    
sim_dir = '/scratch/PI/sabatti/space_comp_sim/20191104'
sim_fn = os.path.join(sim_dir, 'sim_setup.csv')
sim_df = load_data_from_file(sim_fn, 'csv')
print('Loaded parameters: {}'.format(sim_df.shape))

alpha = 0.05
meth_dir = 'result_spatialde'
cache=False
just_load=False

meth_dir = os.path.join(sim_dir, meth_dir)
os.makedirs(meth_dir, exist_ok=True)

eval_fn = 'eval_spatialde.csv'
eval_fn = os.path.join(sim_dir, eval_fn)
eval_df = pd.DataFrame()


for i, entry in sim_df.iterrows():
    locs, data = read_trial_data(sim_dir, entry)
    data = norm_mtx(data)
    sample_info = pd.DataFrame({'total_counts': data.sum(axis=1)})
    print(i)
    assert locs.shape[0] == data.shape[0], 'Mismatch samples'
    if just_load:
        continue
    fn = os.path.join(meth_dir, 'result_{}.csv'.format(i))
    if cache and os.path.exists(fn):
        result = load_data_from_file(fn, 'csv')
    else:
        result= SpatialDE.run(locs, data)
        save_data_to_file(result, fn, 'csv')   
        
    rej_idx = np.array(result.loc[result['qval'] < alpha].g).astype(int)
    nn_idx = np.arange(entry['n_per_reg'] * entry['n_regs'])
    evals = evaluate_rejections(set(rej_idx), set(nn_idx))
    print('{}-{}: Power: {:.4f}, FDP: {:.4f}'.format(entry['temp'], entry['seed'], 
                                              evals['Power'], evals['FDP']))
    eval_df = eval_df.append(pd.Series(evals, name=i))
save_data_to_file(eval_df, eval_fn, 'csv')