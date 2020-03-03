import sys
import os
import logging
sys.path.insert(0,'../src')

from io_utils import save_data_to_file, load_data_from_file
from general_utils import evaluate_rejections
from scGCO import *

def read_trial_data(sim_dir, entry):
    lfn = os.path.join(sim_dir, entry['locs_fn'])
    locs = load_data_from_file(lfn, 'csv')
    dfn = os.path.join(sim_dir, entry['expr_fn'])
    data = load_data_from_file(dfn, 'csv')
    return locs, data
    
sim_dir = '/share/PI/sabatti/feat_viz/space_comp_sim/20191104'
sim_fn = os.path.join(sim_dir, 'sim_setup.csv')
sim_df = load_data_from_file(sim_fn, 'csv')

alpha = 0.05
meth_dir = 'result_scgco'
cache = True
just_load=False


meth_dir = os.path.join(sim_dir, meth_dir)
os.makedirs(meth_dir, exist_ok=True)

eval_fn = 'eval_scgco.csv'
eval_fn = os.path.join(sim_dir, eval_fn)
eval_df = pd.DataFrame()

for i, entry in sim_df.iterrows():
    locs, data = read_trial_data(sim_dir, entry)
    assert locs.shape[0] == data.shape[0], 'Mismatch samples'

    fn = os.path.join(meth_dir, 'result_{}.csv'.format(i))
    print(fn)
    if just_load:
        continue
    if cache and os.path.exists(fn):
        result = load_data_from_file(fn, 'csv')
#         rej_idx = result.loc[result['fdr'] < alpha]
    else:
        sf = estimate_smooth_factor(locs.values, 
                                    data, 
                                    start_sf = 20, fdr_cutoff=0.01,iterations=2)
        print('Smooth factor: {}'.format(sf))
        result_df=identify_spatial_genes(locs.values, data, smooth_factor=sf)
        result = result_df[['exp_p_value', 'exp_fdr', 'exp_diff', 'fdr', 'smooth_factor']]
        # save result to file
        save_data_to_file(result, fn, 'csv')        
    # evaluate for tracking
    rej_idx = np.array(result.loc[result['fdr'] < alpha].index).astype(int)
    nn_idx = np.arange(entry['n_per_reg'] * entry['n_regs'])
    evals = evaluate_rejections(set(rej_idx), set(nn_idx))
    print('{}-{}: Power: {:.4f}, FDP: {:.4f}'.format(entry['temp'], entry['seed'], 
                                              evals['Power'], evals['FDP']))
    eval_df = eval_df.append(pd.Series(evals, name=i))
save_data_to_file(eval_df, eval_fn, 'csv')
