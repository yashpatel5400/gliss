import numpy as np
import pandas as pd
import os
import logging
import pickle
import time
import timeit
from numba import njit, autojit

from graph_utils import dist_corr, dist_kernel_corr, kernel_dist
from graph_utils import laplacian_score, construct_kernal_graph, construct_knn_naive, get_graph_spectrum
from scipy.stats import spearmanr, pearsonr
from general_utils import norm_mtx, evaluate_rejections
from io_utils import save_data_to_file, load_data_from_file, flag_complete
from pcurve_utils import initial_pcurve

from main_methods import generate_correlation_map, batch_perm_lap_score
from main_methods import infer_lambda, compute_feature_pvals, multitest_rejections
from main_methods import run_unsupervised

logger = logging.getLogger("feat_viz")

def noiseless_models(x, mode="mid_peak"):
    # use sinusoid as spike in for now
    offset = (max(x) + min(x)) / 2
    lr_range = max(x) - min(x)
    if mode == "mid_peak":
        out = np.sin(2*np.pi*(1/4 + (x-offset)/lr_range))
    if mode == "right_span_peak":
        out = np.sin(1*np.pi*((x-offset)/lr_range))
    if mode == "left_span_peak":
        out = np.sin(1*np.pi*(1 + (x-offset)/lr_range))
    if mode == "multi_left_peak":
        out = -np.sin(3*np.pi*(0.25 + (x-offset)/lr_range))
    if mode == "multi_right_peak":
        out = -np.sin(3*np.pi*(0.75 + (x-offset)/lr_range))
    if mode == "freq_peak":
        out = np.sin(20*np.pi*(0.75 + (x-offset)/lr_range))
    if mode == "right_peak":
        out = np.sin(1.5*np.pi*((x-offset+1.25)/lr_range))
        out[x > 0.9] = 1
        out[x < 0.25] = -1
    if mode == "left_peak":
        out = np.sin(1.5*np.pi*((x-offset+0.75)/lr_range))
        out[x < 0.1] = 1
        out[x > 0.75] = -1
    if mode == "mid_r_peak":
        out = - np.sin(2*np.pi*(1/4 + (x-offset)/lr_range))
    # rescale to positive range 0,1 
    out = (out - min(out)) / (max(out) - min(out))
    return out

def spike_in_models(x, mode="mid_peak", rel_noise=0, scale=1, seed=0):
    out_noiseless = scale * noiseless_models(x, mode=mode)  
    # noise model
    sigma = rel_noise * scale
    out_noisy = add_gaussian_noise(out_noiseless, sigma, seed=seed)
    out_noisy[out_noisy < 0] = 0
    return out_noisy

def generate_spike_mtx(lam, 
                       spike_grp = ["left", "mid", "right"],
                       n_repetitions=3, 
                       rel_noise_list=[0, 1e-1, 1e-2],
                       seed_offset = 1,
                       scale=1):
    spike_valid =  ["right", "left", "mid", "mid_r", "multi_left", "multi_right", "freq"]
    for spike in spike_grp:
        assert spike in spike_valid, "Spike {} not found".format(spike)
    n_per_grp = n_repetitions * len(rel_noise_list)
    spike_pfx = np.repeat(spike_grp, n_per_grp)
    spike_mtx = np.zeros((len(lam), len(spike_pfx)))
    spike_noise = np.tile(np.repeat(rel_noise_list, n_repetitions), len(spike_grp))
    # TODO: optimize in the future if needed
    for i, pfx in enumerate(spike_pfx):
        y_sim = spike_in_models(lam, 
                                rel_noise=spike_noise[i], 
                                scale=scale, 
                                mode="{}_peak".format(pfx),
                                seed=i+seed_offset)
        spike_mtx[:, i] = y_sim
    spike_names = ["Spike_{}".format(s) for s in spike_pfx]
    spike_names = ["{}_{}".format(s, i) for i, s in enumerate(spike_names)]
    spike_df = {"gene_ids": spike_names,
                "n_cells": np.sum(spike_mtx > 0, axis=0),
                "mean": spike_mtx.mean(axis=0), 
                "std": spike_mtx.std(axis=0),
                "npc": np.sum(spike_mtx > 0, axis=0)}
    spike_df = pd.DataFrame(spike_df, index=spike_names)
#     logger.debug("Adding spike in matrix {}".format(spike_mtx.shape))
    return spike_mtx, spike_df

def generate_low_dim_shape(n_samp, shape="circle", param=None):
    np.random.seed(1)
    t_vals = np.sort(np.random.uniform(size=n_samp))

    if shape == "circle":
        mat_s = np.stack((np.cos(2*np.pi*t_vals), np.sin(2*np.pi*t_vals))).T
    if shape == "line":
        sig_corr = np.array([1, -1])
        mat_s = np.expand_dims(t_vals,1) * sig_corr
    if shape == "centroids":
        np.random.seed(1)
        if param is None:
            n_groups = 10
            n_dims = 2
            noise_var = np.log(n_groups)
        else:
            n_groups = param["n_groups"]
            n_dims = param["n_dims"]
            noise_var = np.sqrt(n_groups)
        group_size = int(np.floor(n_samp / n_groups))
        group_pos =  np.random.normal(size=(n_groups, n_dims))
        # fill with the last group coordinate
        mat_s = np.reshape(np.tile(group_pos[-1,:], n_samp),(n_samp,n_dims))
        n_fill = int(group_size * n_groups)
        mat_s[:n_fill, :] = np.reshape(np.tile(group_pos, group_size), (n_fill, n_dims))
        # noise scaling
        mat_s = mat_s * noise_var
        t_vals = (n_groups - 1) * np.ones(n_samp)
        t_vals[:n_fill] = np.repeat(range(n_groups), group_size)
        t_vals = t_vals.astype(int)

    logger.info("Noiseless matrix dimension: {}".format(mat_s.shape))

    return  mat_s, t_vals
    
def add_uniform_noise(mat, noise, seed=0):
    """ add noise based on the noise level
    """
#     np.random.seed(seed)
    nmat = np.sqrt(noise) * np.random.uniform(size=np.prod(mat.shape))
    nmat = np.reshape(nmat, mat.shape) + mat
    return nmat

def add_gaussian_noise(mat, noise, seed=0):
    """ add noise based on the noise level
    """
#     np.random.seed(seed)
    nmat = np.sqrt(noise) * np.random.normal(size=np.prod(mat.shape))
    nmat = np.reshape(nmat, mat.shape) + mat
    return nmat

def add_correlated_noise(x_null, null_struct, seed=0, trunc=True, scale=1):
    n_samps = x_null.shape[0]
    n_vars = x_null.shape[1]
    bsize = null_struct["block_size"]
    cov = np.ones((bsize, bsize)) * null_struct['corr_value']
    np.fill_diagonal(cov, 1)
    mean = 0.5 * np.ones(bsize)
    n_null_grps = int(np.ceil(n_vars / bsize))
    mvn_mtx = np.random.multivariate_normal(mean=mean, cov=cov, size=n_samps*n_null_grps)
    mvn_mtx = mvn_mtx.reshape(n_samps, n_null_grps * bsize)
    mvn_mtx = mvn_mtx[:, :n_vars] * scale
    if trunc:
        mvn_mtx[mvn_mtx > 1] = 1
        mvn_mtx[mvn_mtx < 0] = 0
    return mvn_mtx + x_null

def generate_shifted_obs_vars(n_samp, n_shifts = 10, n_null_v = 10):
    np.random.seed(10)
    shifts = np.round(np.arange(0,n_shifts) * n_samp / n_shifts)
    mat_v = np.zeros((n_samp, n_shifts))
    temp_vec = np.ones(n_samp)
    temp_vec[int(n_samp/2):] = 0
    for i, s in enumerate(shifts):
        mat_v[:, i] = (add_gaussian_noise(np.roll(temp_vec, int(s)),2))
    mat_v = np.concatenate([mat_v, 
            np.random.normal(size=(n_samp, n_null_v))], axis=1)
    var_type = ["non-null"] * n_shifts + ["null"] * n_null_v
    return mat_v, var_type

def summarize_correlations(mat_s, mat_v, g_mat_s, v_type=None):
    corr_mtx = generate_correlation_map(mat_s.T, mat_v.T)
    corr_df = pd.DataFrame(np.abs(corr_mtx.T), columns = ["corr_v0", "corr_v1"])
    corr_df["lap_score"] = laplacian_score(mat_v, g_mat_s)
    corr_df["max_corr"] = corr_df[["corr_v0", "corr_v1"]].max(axis=1)
    corr_df["dist_corr"] =  np.apply_along_axis(dist_corr, 0, mat_v, mat_s)
    corr_df["gauss_kern"] =  np.apply_along_axis(dist_kernel_corr, 0, mat_v, mat_s)
    if v_type is not None:
        corr_df["type"] = v_type
    return corr_df

def measure_all_correlations(x, y):
    out = {}
    out["dist_corr"] = dist_kernel_corr(x, y, kernel="euclidean")
    out["gauss_kern"] = dist_kernel_corr(x, y, kernel="gauss")
    out["dist_gauss_kern"] = dist_kernel_corr(x, y, kernel="dist_gauss")
    univ_vec = True
    if x.ndim == 2: # fill up to a vector
        if x.shape[1] > 1:
            univ_vec = False
    if y.ndim == 2: # fill up to a vector
        if y.shape[1] > 1:
            univ_vec = False
    if univ_vec:
#         out["pearson"] = generate_correlation_map(x.T, y.T)[0][0]
        out["pearsonr"] = pearsonr(x,y)[0][0]
        out["spearmanr"] = spearmanr(x, y)[0]
    return out


@njit(cache=True, fastmath=True)
def univ_ss_dist(vec, dist):
    # sum of square distance of pairwise univariate
    # vec is a univariate vector of dim (n_obs, )
    dist.fill(0)
    dist += np.expand_dims(vec, axis=0)
    dist -= np.expand_dims(vec, axis=1)
    dist *= dist

@njit(cache=True, fastmath=True)
def univ_ss_gauss_kernel_dist(vec, dist, row_means):
    sigma = 1
    univ_ss_dist(vec, dist)
    n_obs = dist.shape[0]
    # use apply the kernel here if need be
    for i in range(n_obs):
        for j in range(n_obs):
            dist[i, j] = np.exp(- sigma * dist[i, j])
    # kernel adjustment
    dist += 1
    dist -= np.expand_dims(np.exp(-sigma * vec**2), axis=0) 
    dist -= np.expand_dims(np.exp(-sigma * vec**2), axis=1)
    # distance centering
    for i in range(n_obs):
        row_means[i] = dist[i, :].mean()
    grand_mean = dist.mean()
    dist -= np.expand_dims(row_means, axis=0) 
    dist -= np.expand_dims(row_means, axis=1)
    dist += grand_mean

@njit(cache=True, fastmath=True)
def dcorr_comp(A, B):
    n_obs = A.shape[0]
    dcov = np.multiply(A, B).sum() / (n_obs * n_obs)
    Xcor = np.multiply(A, A).sum() / (n_obs * n_obs)
    Ycor = np.multiply(B, B).sum() / (n_obs * n_obs)
    dcorr = np.sqrt(dcov) / np.sqrt((np.sqrt(Xcor) * np.sqrt(Ycor)))
    return dcorr

@njit(cache=True)
def perm_dist(A, B, perm_id, perm_out):
    n_obs = A.shape[0]
    perm_id = np.arange(n_obs)
    perm_out.fill(0)
    n_perms = len(perm_out)
    np.random.seed(0)
    for i_perm in range(n_perms):
        np.random.shuffle(perm_id)
        B = B[perm_id, :]
        B = B[:, perm_id]
        perm_out[i_perm] = dcorr_comp(A, B)

def batch_perm_dist(A, Y, n_perms):
    n_obs, n_batch_vars = Y.shape
    rmns = np.ones(n_obs)
    B = np.ones((n_obs, n_obs))
    perm_id = np.arange(n_obs)
    perm_out = np.zeros(n_perms)
    batch_output = np.zeros((n_perms, n_batch_vars))
    
    for i_var in range(n_batch_vars):
        univ_ss_gauss_kernel_dist(Y[:, i_var], B, rmns)
        perm_dist(A, B, perm_id, perm_out)
        batch_output[:, i_var] = perm_out
        
    return batch_output

def batch_comp_dist_corr(x, y):
    n_obs, n_vars = y.shape
    rmns = np.ones(n_obs)
    A = kernel_dist(x,  centered=True, kernel="dist_gauss")
    B = np.ones((n_obs, n_obs))
    score_vec = np.zeros(n_vars) 
    start = time.time()
    for i_var in range(n_vars):        
        univ_ss_gauss_kernel_dist(y[:, i_var], B, rmns)
        score_vec[i_var] = dcorr_comp(A, B)    
        if i_var % 1000 == 0:
            end = time.time()
            print("computed {} features... ({:.1f}s)".format(i_var, end-start))
            start = end
    return score_vec

def run_batch_dist_corr_perm(params, score="dist_corr"):
    """
        The core pipeline for full permutation handling. The batch setup is used to handle a large number of variables.
        Requirements needed for this function to run:
        - x_data.npy, y_data.npy (column-scaled structural and repsonse variables)
        - batch_job_params.json -> params: parameters for the permutation experiment
        
    """
    n_perms = params["n_perms"]
    batch_size = params["batch_size"]
    tmp_dir = params["tmp_dir"]
    i_batch = params["i_batch"]
    # load the data and infer batch size and indices 
    x = np.load(os.path.join(tmp_dir, "x_data.npy"))
    y = np.load(os.path.join(tmp_dir, "y_data.npy"))
    n_obs, n_vars = y.shape
    n_batches = int(np.ceil(n_vars/batch_size))
    print('Loaded data from {}'.format(tmp_dir))  
    # batch-specific computation
    if i_batch == (n_batches-1): # the last batch
        n_batch_vars = n_vars % batch_size
    else:
        n_batch_vars = batch_size
    beg_idx = i_batch * batch_size
    end_idx = (1 + i_batch) * batch_size
    assert n_batch_vars == y[:, beg_idx : end_idx].shape[1], "dimension error"
    print('Computing batch {} with {} vars...'.format(i_batch, n_batch_vars))  
    Y = y[:, beg_idx : end_idx]
    start = time.time()
    if score == "dist_corr":
        A = kernel_dist(x,  centered=True, kernel="dist_gauss")
        batch_output = batch_perm_dist(A, Y, params["n_perms"])
    if score == "lap_score":
        kg = construct_kernal_graph(x)
        batch_output = batch_perm_lap_score(kg, Y, params["n_perms"])
    end = time.time()
    print('Total time: {} seconds'.format(end - start))
    # save the batch result
    fname = "{}_batch_{}.npy".format(score, i_batch)
    fname = os.path.join(tmp_dir, fname) 
    np.save(fname, batch_output)
    print("Saved data to file {}".format(fname))
    print(batch_output[:5, :5])
    
def run_dist_corr_perm(x, y, tmp_dir, n_perms=1000, batch_size=1000):
    """
    TODO: 1. use numba for speed up
          2. handle parallel processing in the future
          3. save as hd5 file or make converter
    """
    n_obs, n_vars = y.shape
    n_batches = int(np.ceil(n_vars/batch_size))
    A = kernel_dist(x,  centered=True, kernel="dist_gauss")
    params = {"n_perms": n_perms, "batch_size": batch_size}
    
    for i_batch in range(n_batches):
        np.random.seed(i_batch)
        if i_batch == (n_batches-1): # the last batch
            n_batch_vars = n_vars % batch_size
        else:
            n_batch_vars = batch_size
        print('Computing batch {} with {} vars...'.format(i_batch, n_batch_vars))     
        beg_idx = i_batch * batch_size
        end_idx = (1 + i_batch) * batch_size
        assert n_batch_vars == y[:, beg_idx : end_idx].shape[1], "dimension error"
        Y = y[:, beg_idx : end_idx]
        start = time.time()
        batch_perm_dist(A, Y, params)
        end = time.time()
        print('Used time: {} seconds'.format(end - start))
        # store the batch output
        fname = "dist_corr_batch_{}.npy".format(i_batch)
        fname = os.path.join(tmp_dir, fname) 
        print("Saved data to file {}".format(fname))

def multi_vs_one_dist_corr(X, S, perm=False, n_perm=10000, **kwargs):
    
    A = kernel_dist(S, **kwargs)
    n = A.shape[0] 
    if perm:
        n_vars = n_perm
    else:
        n_vars = X.shape[1]
    print("computing a total of {} features".format(n_vars))
    out_vec = np.zeros(n_vars)
    for i in range(n_vars):
        if perm: 
            value = X[:,np.random.choice(range(X.shape[1]))]
            y_vec = np.random.permutation(value)
        else:
            y_vec = X[:,i]
        B = kernel_dist(y_vec, **kwargs)
        dcov = np.multiply(A, B).sum() / (n * n )
        Xcor = np.multiply(A, A).sum() / (n * n )
        Ycor = np.multiply(B, B).sum() / (n * n )
        dcorr = np.sqrt(dcov) / np.sqrt((np.sqrt(Xcor) * np.sqrt(Ycor)))
        out_vec[i] = dcorr
        if i % 1000 == 0:
            print("computed {} features...".format(i))
    return(np.array(out_vec))


def scale_corr_input(in_dat):
    out_dat = (in_dat - in_dat.mean(axis=0)) / in_dat.std(axis=0)
    return out_dat

def get_corr_scores(x_mat, y_mat, 
                    tmp_dir, 
                    methods=["lap_score", "dist_corr"], 
                    use_cache=True):
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    assert x_mat.shape[0] == y_mat.shape[0], "sample size mismatch"
    # 1. center and scale each variable 
    x = scale_corr_input(x_mat)
    y = scale_corr_input(y_mat)
    print("Finsihed scaling: x: {}, y: {} ".format(x.shape, y.shape))
    all_results = {}
    # 2. iterate through each correlation method
    for score in methods:
        fname = os.path.join(tmp_dir, "feat_{}.pkl".format(score))
        if os.path.exists(fname) and use_cache:
            result = pickle.load(open(fname, "rb" ))
            print("Loaded cache: {}".format(fname))
            assert y_mat.shape[1] == len(result["feat_scores"]), "num feat mismatch"
            all_results[score] = result
        else:
            print("Computing {}".format(score))
            if score == "lap_score":
                from graph_utils import laplacian_score, construct_kernal_graph
                graph = construct_kernal_graph(x)
                feat_scores = laplacian_score(y, graph)
                perm_scores = laplacian_score(y, graph, perm=True)
            if score == "dist_corr":
                kn = "dist_gauss"
                # 1. standardize each feature to mean zero and unit variance
                # 2. run the (slow) test to compute this 
                feat_scores = multi_vs_one_dist_corr(y, x, kernel=kn)
                perm_scores = multi_vs_one_dist_corr(y, x, perm=True, kernel=kn)
            template = {"perm_scores": perm_scores, "feat_scores": feat_scores}    
            pickle.dump(template, open(fname, "wb"))
    return all_results

def comp_perm_pval(stat_vals, perm_vals, ascending=True):
    """
    Args:
        stats: an np array of observed statistics
        perm_vals: an np array of permutation samples from hist
        ascending: if True, computes left-sided p-value (and right otherwise)
    Returns:
        an np array of stats with p-values
    """
    if ascending: 
        sort_perm = np.sort(perm_vals)
        sort_idx = np.argsort(stat_vals)
    else: # descending order of the perm. values
        sort_perm = np.sort(perm_vals)[::-1]
        sort_idx = np.argsort(stat_vals)[::-1]
    sort_stats = stat_vals[sort_idx]
    perm_cnt = 0
    curr_idx = 0
    n_perms = len(perm_vals)
    out_stats = n_perms * np.ones(len(sort_stats))
    while (perm_cnt < n_perms and 
           curr_idx < len(sort_stats)):
        if ascending:
            condition_true = sort_perm[perm_cnt] <= sort_stats[curr_idx]
        else:
            condition_true = sort_perm[perm_cnt] >= sort_stats[curr_idx]
        if condition_true:
            perm_cnt += 1
        else:
            out_stats[sort_idx[curr_idx]] = perm_cnt
            curr_idx += 1
    out_stats = out_stats / n_perms 
    # TODO: handle the +1 problems later...
    #     print(stat_vals)
    #     print(out_stats)
    return out_stats
    
    
# Apr.11.19 simulation

def get_sim_id_classes():
    full_sim_ids = ["main_sim_{}".format(i) for i in range(7)]
    strict_sim_ids = ["main_sim_strict_1"]
    return full_sim_ids, strict_sim_ids

def get_sim_params(sim_id):
    full_sim_ids, strict_sim_ids = get_sim_id_classes()
    all_sim_ids = full_sim_ids + strict_sim_ids
    assert sim_id in all_sim_ids, "{} not specified".format(sim_id)
    nn_pattern = ["right", "left", "mid", "mid_r", "multi_left", "multi_right"]
    example_null_struct = {"model": "normal", "block_size": 150, "corr_value": 0.5}
    example_all_methods = ["unsup_pc", "unsup_graph", "pc", "graph"]
    if sim_id in full_sim_ids:
        if sim_id == "main_sim_1":
            params = {
                "seed": 1,
                "n_samps": 1500,
                "z_param": {"spike_grp": ["left", "right"],
                             "n_repetitions": 1,
                             "rel_noise_list": [1, 1e-1]},
                "x_param": {"spike_grp": ["right", "left", "mid", "mid"],
                             "n_repetitions": None, 
                             "rel_noise_list": None,
                             "seed_offset": 0},
                "n_trials": 100,
                "noise_levs": [0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3],
                "sparsities": [0.001, 0.005, 0.01, 0.05, 0.10],
                "target_vars": 8000,
                "methods": ["pc", "graph"],
                "graph_k": 10,
                "n_perms": 10000,
                "alpha": 0.05,
                "perm_method": "pool",
                "sim_dir": "/share/PI/sabatti/feat_viz/{}".format(sim_id),
                "save": True,
                "check_skip": False,
            }
        else:
            params = {
                "seed": 1,
                "n_samps": 1500,
                "z_param": {"spike_grp": ["left", "right"],
                            "n_repetitions": 1,
                            "rel_noise_list": [1, 1e-1]},
                "x_param": {"spike_grp": nn_pattern,
                            "n_repetitions": None, 
                            "rel_noise_list": None,
                            "seed_offset": 0},
                "n_trials": 100,
                "noise_levs": [0.1, 0.5, 1, 1.5, 2, 3],
                "sparsities": [0.01, 0.05, 0.10, 0.20],
                "target_vars": 9000,
                "methods": ["pc", "graph"],
                "graph_k": 10,
                "n_perms": 10000,
                "alpha": 0.05,
                "perm_method": "pool",
                "sim_dir": "/share/PI/sabatti/feat_viz/{}".format(sim_id),
                "save": True,
                "check_skip": False,
            } 
            if sim_id == "main_sim_3":
                params["z_param"] = {"spike_grp": ["left", "right", "mid", "mid_r"],
                                     "n_repetitions": 1,
                                     "rel_noise_list": [1e-1]}
                params["sparsities"] = [0.05, 0.10, 0.20]
        if sim_id == "main_sim_4"  or  sim_id == "main_sim_6" :
            params["null_struct"] = example_null_struct
            params["methods"] = example_all_methods
            params["sparsities"] = [0.05, 0.10, 0.20]
        elif sim_id == "main_sim_5":
            params["null_struct"] = example_null_struct
            params["null_struct"]["corr_value"] = 0
            params["methods"] = example_all_methods
            params["sparsities"] = [0.05, 0.10, 0.20]
        else:
            params["null_struct"] = {}
            
    # strick simulation overwrites prevoius parameters
    if sim_id == "main_sim_strict_1":
        params = {
            "seed": 0,
            "n_samps": 1500,
            "n_samps_list": [500, 1000, 1500, 2000, 5000],
            "n_trials": 50,
            "x_param": {"spike_grp": ["left", "right"],
                         "n_repetitions": 1,
                         "rel_noise_list": [1, 1e-1]},
            "y_param": {"spike_grp": nn_pattern,
                         "n_repetitions": 4, 
                         "rel_noise_list": [5e-1]},
            "target_vars": 80,
            "method": None,
            "graph_k": 10,
            "sim_dir": "/share/PI/sabatti/feat_viz/{}".format(sim_id),
        }
        target_vars = params["target_vars"]
        rep_unit = params["y_param"]["n_repetitions"]
        params["n_var_list"] = np.arange(0, target_vars, rep_unit).astype(int).tolist()
    return params

def generate_regime_info(sim_id):
    full_sim_ids, strict_sim_ids = get_sim_id_classes()
    all_sim_ids = full_sim_ids + strict_sim_ids
    assert sim_id in all_sim_ids, "{} not specified".format(sim_id)
    params = get_sim_params(sim_id)
    fn_param = os.path.join(params["sim_dir"], "param_all.json")
    fn_regime = os.path.join(params["sim_dir"], "regime_info.csv") # Regime_id, Noise, Sparsity
    os.makedirs(params["sim_dir"], exist_ok=True)
    if sim_id in full_sim_ids:
        df1 = pd.DataFrame({'n_samps': params["n_samps"], 
                            'sparsity': params['sparsities']})
        df2 = pd.DataFrame({'n_samps': params["n_samps"], 
                            'noise_lev': params['noise_levs']})
        df = pd.merge(df1, df2, on="n_samps")
    if sim_id in strict_sim_ids:
        params["n_samps"] = None
        df = pd.DataFrame({"n_samps": params["n_samps_list"], 
                           "n_trials": params["n_trials"]})
    logger.info("{}".format(params))
    save_data_to_file(params, fn_param, "json")
    save_data_to_file(df, fn_regime, "csv")
    logger.info("Saved regimes to: {}".format(fn_regime))
    return df

def get_regime_info(sim_id):
    sim_params = get_sim_params(sim_id)
    init_sim_dir = sim_params["sim_dir"]
    fn_regime = os.path.join(init_sim_dir, "regime_info.csv")
    df_regime = load_data_from_file(fn_regime, "csv")   
    df_regime["Regime"] = df_regime.index
    return df_regime

def load_regime_results(sim_id, check_only=False):
    full_sim_ids, strict_sim_ids = get_sim_id_classes()
    sim_params = get_sim_params(sim_id)
    df_regime = get_regime_info(sim_id)  
    init_sim_dir = sim_params["sim_dir"]
    logger.info("sim. dir.: {}".format(init_sim_dir))
    df_result = pd.DataFrame()
    for i_regime, regime in df_regime.iterrows():
        sub_dir = os.path.join(init_sim_dir, "regime_results", "regime_{}".format(i_regime))
        if sim_id in full_sim_ids:
            fn_result = os.path.join(sub_dir, "eval_scores.csv")
        if sim_id in strict_sim_ids:
            fn_result = os.path.join(sub_dir, "all_accuracy.csv")
        if flag_complete(sub_dir, "check"):
            if check_only:
                logger.info("Regime: {} ready to load".format(i_regime))
            result = load_data_from_file(fn_result, "csv")
            result["Regime"] = i_regime
            df_result = pd.concat([df_result, result], axis=0)
        else:
            logger.info("Regime: {} Incomplete!".format(i_regime))
            logger.info("To relaunch, run:\nsbatch {}/slurm_job.sh".format(sub_dir))
        if check_only:
            continue
    if check_only:
        return
    else:
        logger.info("Loaded all data: {}".format(df_result.shape))
        return df_result
    
def generate_synthetic_data(params, linspace=False, sort=False):
    logger.info(params)
    n_samps = params["n_samps"]
    y_p = params["y_param"]
    n_nonnull = y_p["n_repetitions"] * len(y_p["spike_grp"])
    max_nvar = max(params["n_var_list"]) - n_nonnull
    # generate latent variable
    np.random.seed(params["seed"])
    if linspace:
        lam = np.linspace(0, 1, n_samps)
    else:
        lam = np.random.uniform(size=n_samps)
    # generate structural matrix
    x, _ = generate_spike_mtx(lam, **params["x_param"])
    # generate non-null response matrix 
    y, _ = generate_spike_mtx(lam, **params["y_param"])
    # generate null response matrix 
    nn = np.zeros((n_samps, max_nvar))
    nn = add_uniform_noise(nn, 1)
    if sort:
        sort_idx = np.argsort(lam)
        lam = lam[sort_idx]
        x = x[sort_idx, :]
        y = y[sort_idx, :]
        nn = nn[sort_idx, :]
    return lam, x, y, nn


def generate_x_mtx(lam, x_params, global_params, no_noise=False):
    
    n_samps = global_params["n_samps"]
    target_vars = global_params["target_vars"]
    
    seed = x_params["seed_offset"]
    n_rep = x_params["n_repetitions"]
    n_proto = len(x_params["spike_grp"])
    
    np.random.seed(seed)
    x_nn, _ = generate_spike_mtx(lam, **x_params)
    x_null = np.zeros((n_samps, target_vars-n_rep*n_proto))
    # handle the non-null structure here
    if no_noise:
        pass
    else:
        if global_params["null_struct"]:
            logger.info("Null structure: {}".format(global_params["null_struct"]))
            x_null = add_correlated_noise(x_null, global_params["null_struct"])
        else:
            x_null = add_uniform_noise(x_null, 1)
    x = np.concatenate([x_nn, x_null], axis=1)
    
    return x

def run_regime_feat_sel(params, p_params):
    # within each directory, e.g., regime_i save the data and ground truth
    # truth_lam.npy
    # truth_nnidx.npy
    # eval_scores.csv: Regime, Trial, Method, FDP, Power, Corr

    # trial_result/
    # under trial results save trial_j/...
    # within each trial save the following:
    # Optional data (z, x)

    # _COMPLETE_{METHOD}, 
    # {METHOD}_pval.npy, 
    # {METHOD}_lamh.npy, 
    # {METHOD}_eval.json,
    logger.info(params)
    # load the parameters
    methods = params["methods"]
    n_samps = params["n_samps"]
    graph_k = params["graph_k"]
    n_trials = params["n_trials"]
    noise_levs = params["noise_levs"]
    target_vars = params["target_vars"]
    n_perms = params["n_perms"]
    z_param = params["z_param"]
    x_param = params["x_param"]
    sparsities = params["sparsities"]
    alpha = params["alpha"]
    perm_method = params["perm_method"]
    check_skip = params["check_skip"] 
    save = params["save"]
    
    sparsity = p_params["sparsity"]
    noise = p_params["noise_lev"]
    i_regime = p_params["i_regime"]
    sub_dir = p_params["sub_dir"]

    np.random.seed(params["seed"])
    lam = np.random.uniform(size=n_samps)
    z, _ = generate_spike_mtx(lam, **params["z_param"]) # save 
    
    auxs = {}
    for method in methods:
        if method in ["graph", "pc"]:          
            lam_hat, aux = infer_lambda(method, z, knn=graph_k)
            scor = abs(spearmanr(lam_hat, lam).correlation)
            logger.info("{}: init corr: {:.4f} ".format(method, scor))
            if method == "graph":
                auxs[method] = aux["graph"]
            if method == "pc":
                auxs[method] = lam_hat
    # -----------------------------------------------------------------------------
    # book-keeping files and parameters
    flag_complete(sub_dir, "remove")
    # -----------------------------------------------------------------------------
    df_trials = pd.DataFrame(columns=["Trial", "Method", "Corr", "FDP", "Power"])
    time_cumm = 0

    n_proto = len(x_param["spike_grp"])
    n_rep = int(sparsity * target_vars / n_proto)
    nn_idx = np.arange(n_rep*n_proto)
    logger.info("Current num_var: {} ({:.4f}); noise: {}".format(
        n_rep*n_proto, n_rep*n_proto/target_vars, noise))
    for i_trial in range(n_trials):
        time_start = time.time()
        # simulate features in x
        x_param["rel_noise_list"] = [noise]
        x_param["n_repetitions"] = n_rep
        x_param["seed_offset"] = i_trial*100
#         x_param["n_samps"] = n_samps
#         x_param["target_vars"] = target_vars
        x = generate_x_mtx(lam, x_param, params)
        trial_dir = os.path.join(sub_dir, "trial_results", "trial_{}".format(i_trial))
        os.makedirs(trial_dir, exist_ok=True)
        # -----------------------------------------------------------------------------
        for method in methods:
            meth_start = time.time()
            # book-keeping files and parameters
            pfn = "{}_pval.npy".format(method)
            lfn = "{}_lamh.npy".format(method)
            efn = "{}_eval.json".format(method)
            if check_skip and flag_complete(trial_dir, "check", sufx=method):
                logger.info("Trial {}: already complete - skipping".format(i_trial))
                eval_res = load_data_from_file(os.path.join(trial_dir, efn), "json")
                df_trials = df_trials.append(eval_res, ignore_index=True)
                continue
            # main procedure to select features and reconstruct latent variable
            if method in auxs: # partial knowledge framework
                pvals = compute_feature_pvals(method, x, auxs[method], 
                                              params["n_perms"], params["perm_method"])
                rej_idx = multitest_rejections(pvals, alpha, method="BH")
                new_x = np.concatenate([z, x[:,rej_idx]], axis=1)
                new_lam_hat, aux = infer_lambda(method, new_x, knn=graph_k)
            else: # unsupervised learning framework
                run_method = method.split("unsup_")[1]
                u_params = {"method": run_method, "graph_k": graph_k}
                new_lam_hat = run_unsupervised(z, x, u_params)
                rej_idx = np.arange(x.shape[1])
            # evaluation of the procedure output
            corr = abs(spearmanr(new_lam_hat, lam).correlation)
            eval_res = {"Trial": i_trial, "Method": method, "Corr": corr}
            mt_res = evaluate_rejections(set(rej_idx), set(nn_idx))
            eval_res.update(mt_res)
            # book-keeping files and parameters
            df_trials = df_trials.append(eval_res, ignore_index=True)
            if save:
                if method in auxs:
                    save_data_to_file(pvals, os.path.join(trial_dir, pfn), "npy")
                save_data_to_file(new_lam_hat, os.path.join(trial_dir, lfn), "npy")
                save_data_to_file(eval_res, os.path.join(trial_dir, efn), "json")
                flag_complete(trial_dir, "add", sufx=method)
            meth_end = time.time()
            logger.info("Ran: {}, used {:.2f}s".format(method, meth_end-meth_start))
        # -----------------------------------------------------------------------------
        # timer for book keeping
        time_end = time.time()
        time_diff = time_end - time_start
        time_cumm += time_diff
        time_avg = time_cumm / (i_trial + 1)
        time_remain = time_avg * (n_trials - (i_trial + 1))
        logger.info("Trial {}: used {:.2f}s".format(i_trial, time_diff))
        if not check_skip:
            logger.info("Remain {:.2f}s, Avg {:.2f}s".format(time_remain, time_avg))
    # save regime specific data 
    if save:
        save_data_to_file(df_trials, os.path.join(sub_dir, "eval_scores.csv"), "csv")
        save_data_to_file(lam, os.path.join(sub_dir, "truth_lam.npy"), "npy")
        save_data_to_file(nn_idx, os.path.join(sub_dir, "truth_nnidx.npy"), "npy")
        flag_complete(sub_dir, "add")
    return df_trials

def launch_strict_order_pipeline(sim_id, i_regime, skip=False, test=False):
    # regime_map for sample sizes
    df_regime = get_regime_info(sim_id)   
    init_sim_dir = get_sim_params(sim_id)["sim_dir"]
    n_samps = df_regime.iloc[i_regime]["n_samps"]
    n_trials = df_regime.iloc[i_regime]["n_trials"]
    if test:
        n_trials = 2
    # create directory to save the seed results
    sub_dir = os.path.join(init_sim_dir, "regime_results", "regime_{}".format(i_regime))
    df_reg_res = pd.DataFrame()
    fn_regime = os.path.join(sub_dir, "all_accuracy.csv")
    for i_seed in range(n_trials):
        sim_params = get_sim_params("main_sim_strict_1")
        sim_params["seed"] = i_seed
        sim_params["n_samps"] = n_samps
        trial_dir = os.path.join(sub_dir, "trial_results", "trial_{}".format(i_seed))
        os.makedirs(trial_dir, exist_ok=True)
        fn_trial = os.path.join(trial_dir, "accuracy.csv")
        if skip and flag_complete(trial_dir, "check"):
            logger.info("Skipping trial {}".format(i_seed))
            df_tot = load_data_from_file(fn_trial, "csv") 
        else:
            # -----------------------------------------------------------
            # start simulation
            flag_complete(trial_dir, "remove")
            lam, x_mtx, y_mtx, n_mtx = generate_synthetic_data(sim_params)
            # run PC-based learning
            sim_params["method"] = "pc"
            sim_params["graph_k"] = -1
            df_pc, aux_pc = run_strict_order_analysis(lam, x_mtx, y_mtx, n_mtx, sim_params)
            # run graph-based learning
            sim_params["method"] = "graph"
            sim_params["graph_k"] = 10
            df_graph, aux_graph = run_strict_order_analysis(lam, x_mtx, y_mtx, n_mtx, sim_params)
            # combine results
            df_pc["method"] = "pca"
            df_graph["method"] = "graph"
            df_tot = pd.concat([df_pc, df_graph])
            df_tot["seed"] = sim_params["seed"]
            # end simulation
            save_data_to_file(df_tot, fn_trial, "csv")
            flag_complete(trial_dir, "add")
            # -----------------------------------------------------------
        df_reg_res = pd.concat([df_reg_res, df_tot], axis=0)
    save_data_to_file(df_reg_res, fn_regime, "csv")
    flag_complete(sub_dir, "add")
    return
        
def launch_regime_feat_sel(sim_id, i_regime, test=False):
    i_regime = int(i_regime)
    init_sim_dir = get_sim_params(sim_id)["sim_dir"]
    sub_dir = os.path.join(init_sim_dir, "regime_results", "regime_{}".format(i_regime))
    fn_param = os.path.join(init_sim_dir, "param_all.json")
    fn_regime = os.path.join(init_sim_dir, "regime_info.csv")
    params = load_data_from_file(fn_param, "json")
    assert params["sim_dir"] == init_sim_dir, "simulation directory mismatch"
    logger.info("Results dir: {} (Regime: {})".format(sub_dir, i_regime))
    regime_info = load_data_from_file(fn_regime, "csv")    
    p_params = dict(regime_info.loc[i_regime])
    p_params["i_regime"] = i_regime
    p_params["sub_dir"] = sub_dir
    logger.info("{}".format(p_params))
    if test:
        params["save"] = False
        params["n_trials"] = 2 
    df = run_regime_feat_sel(params, p_params)
    logger.info("{}".format(df.head()))
    return df

def initialize_slurm_header(b_file, job_dir, python_params, time_str="01:20:00"):
    b_file.write("#!/bin/bash\n\n#SBATCH -p normal,hns\n#SBATCH --time={}\n".format(time_str))
    b_file.write("#SBATCH --error={}/slurm_job.err\n".format(job_dir))
    b_file.write("#SBATCH --output={}/slurm_job.out\n\n".format(job_dir))
    # python environment
    b_file.write("source  ~/utils/private_utils/configs/pyvenv-setup.bash \n")
    b_file.write("workon feat_viz \n\n")
    # run code
    b_file.write("cd ~/source_code/feat_viz \n")
    param_string = " ".join(python_params)
    b_file.write("python {}\n".format(param_string))

def create_trial_jobs(sim_id, check_complete=False):
    params = get_sim_params(sim_id)
    tmp_dir = params["sim_dir"]
    fn_regime = os.path.join(tmp_dir, "regime_info.csv")
    df_regime = load_data_from_file(fn_regime, "csv") 
    fn_launch_all = os.path.join(tmp_dir, "launch_all.sh")
    
    with open(fn_launch_all, "w") as f_launch_all:
        for i_regime, regime in df_regime.iterrows():
            sub_dir = os.path.join(tmp_dir, "regime_results", "regime_{}".format(i_regime))
            os.makedirs(sub_dir, exist_ok=True)
            # create the launch script in sub_dir
            fn_sub_launch = os.path.join(sub_dir, "slurm_job.sh")
            # the command to launch a file, TODO: do no launch if complete
            f_launch_all.write("sbatch {}\n".format(fn_sub_launch)) 
            python_params = ["src/script_run_sim.py", sim_id, str(i_regime)]
            if sim_id == "main_sim_strict_1":
                if regime["n_samps"] > 2000:
                    time_str="20:00:00"
                else:
                    if regime["n_samps"] > 1500:
                        time_str="10:00:00"
                    else:
                        time_str="02:00:00"
            else:
                time_str="05:00:00"
            with open(fn_sub_launch, "w") as sub_file:
                initialize_slurm_header(sub_file, sub_dir, python_params, time_str=time_str)
            logger.info("Created: {}".format(fn_sub_launch))  
    logger.info("Launch all the slurm jobs with the following command:\nbash {}".format(fn_launch_all))
    return

def run_strict_order_analysis(lam, x_mtx, y_mtx, n_mtx, params):
    logger.info(params)
    time_start = time.time()
    method = params["method"]
    n_var_list = params["n_var_list"]
    graph_k = params["graph_k"]
    df = pd.DataFrame(columns=["n_init", "n_cand", "corr"])
    cand_mtx = np.concatenate([y_mtx, n_mtx], axis=1)
    sel_scores = -1 * np.ones((len(n_var_list), cand_mtx.shape[1]))
    for i_num_var, num_var in enumerate(n_var_list):
        s_mtx = cand_mtx[:, :num_var]
        in_mtx = np.concatenate([x_mtx, s_mtx], axis=1)  
        lam_hat, aux = infer_lambda(method, in_mtx, knn=graph_k)
        if method == "graph":
            ssval = laplacian_score(cand_mtx, aux["graph"])
        if method == "pc":
            lmtx= np.expand_dims(lam, axis=0)
            ssval = abs(generate_correlation_map(lmtx, cand_mtx.T))
        sel_scores[i_num_var, :] = ssval
        score = abs(spearmanr(lam_hat, lam).correlation)
        df = df.append({"n_init": x_mtx.shape[1], 
                        "n_cand": s_mtx.shape[1], 
                        "corr": score}, 
                       ignore_index=True)
        nnulls = max(0, num_var-y_mtx.shape[1])
        if nnulls > 0:
            add_note = "(*{} nulls)".format(nnulls)
        else:
            add_note = ""
        logger.info("curr. score {:4f} aug-dim: {} {}".format(
            score, s_mtx.shape, add_note))
    aux["sel_scores"] = sel_scores
    time_end = time.time()
    time_diff = time_end - time_start
    logger.info("Trial used {:.2f}s".format(time_diff))
    return df, aux

def run_pipeline_nstruct(params,
                         verbose=True):
    method = params["method"]
    n_samps = params["n_samps"]
    spike_grp = params["spike_grps"]
    rel_noise_list = params["noise_levs"]
    n_var_list = params["n_struct_vars"]
    k_param = params["graph_k"]
    logger.info("Inferring lambda with {}".format(method))
    np.random.seed(1)
    lam = np.random.uniform(size=n_samps)
    spike_os = max(n_var_list) * len(rel_noise_list)
    spike_mtx, spike_df = generate_spike_mtx(lam, 
                                             spike_grp=spike_grp, 
                                             n_repetitions=max(n_var_list),
                                             rel_noise_list=rel_noise_list)
    df = pd.DataFrame(columns=["nvar", "nlev", "corr"])
    example_data = {"lam": None, "mtx": None, "aux": None}
    for i_noise, noise in enumerate(rel_noise_list):
        logger.debug("Current noise: {}".format(noise))
        for i_num_var, num_var in enumerate(n_var_list):
            logger.debug("Current num_var: {}".format(num_var))
            var_idx = []
            for i_grp, grp in enumerate(spike_grp):
                os = i_noise * max(n_var_list) + i_grp * spike_os
                var_idx += list(np.arange(os, (os+num_var)))
            in_mtx = spike_mtx[:, var_idx] # will be scaled later 
            lam_hat, aux = infer_lambda(method, in_mtx, knn=k_param)
            scor = abs(spearmanr(lam_hat, lam).correlation)
            df = df.append({"nvar": num_var, "nlev": noise, "corr": scor}, ignore_index=True)
            if i_noise == 2:
                logger.info("Current dimension: {}".format(in_mtx.shape))
                if i_num_var == len(n_var_list) - 1:    
                    logger.info("Outputting example ({}, {}) matrix".format(noise, num_var))
                    example_data["lam"] = lam
                    example_data["mtx"] = in_mtx
                    example_data["aux"] = aux
    return df, example_data

def run_pipeline_nnvars(params,
                        verbose=True):
    method = params["method"]
    n_samps = params["n_samps"]
    spike_grp = params["spike_grps"]
    rel_noise_list = params["noise_levs"]
    n_var_list = params["n_struct_vars"]    
    k_param = params["graph_k"]
    n_nvar_list = np.linspace(0,500,11).astype(int)
    assert len(rel_noise_list) == 1, "Only one noise value allowed now"
    noise = rel_noise_list[0]
    
    # generate the latent variable
    np.random.seed(1)
    lam = np.random.uniform(size=n_samps)
    display(n_nvar_list)
    spike_mtx, spike_df = generate_spike_mtx(lam, 
                                             spike_grp=spike_grp, 
                                             n_repetitions=max(n_var_list),
                                             rel_noise_list=[noise])
    # generate the maximum noise matrix
    noise_mtx = np.zeros((n_samps, max(n_nvar_list)))
    noise_mtx = add_uniform_noise(noise_mtx, 1)
    
    np.random.seed(1)
    lam = np.random.uniform(size=n_samps)
    spike_os = max(n_var_list) * 1
    df = pd.DataFrame(columns=["nvar", "nnum", "corr"])
    example_data = {"lam": None, "mtx": None, "aux": None}
    for i_num_var, num_var in enumerate(n_var_list):
        logger.info("Current structrual variables: {}".format(num_var))
        var_idx = []
        for i_grp, grp in enumerate(spike_grp):
            os = 0 * max(n_var_list) + i_grp * spike_os
            var_idx += list(np.arange(os, (os+num_var)))
        s_mtx = spike_mtx[:, var_idx]
        for i_nv, n_nvar in enumerate(n_nvar_list):
            n_mtx = noise_mtx[:, :n_nvar]
            in_mtx = np.concatenate([s_mtx, n_mtx], axis=1)
            lam_hat, aux = infer_lambda(method, in_mtx, knn=k_param)
            scor = abs(spearmanr(lam_hat, lam).correlation)
            df = df.append({"nvar": num_var, "nnum": n_nvar, "corr": scor}, ignore_index=True)
    logger.info("Outputting example ({}, {}) matrix...\n".format(noise, num_var))
    example_data["lam"] = lam
    example_data["mtx"] = in_mtx
    example_data["aux"] = aux
    return df, example_data


# -----------------------------------
# simulation for correlation analysis
def get_corr_sim(sim_id = "corr_sim", regime=0):
    if regime == 0:
        corr_value = 0.01
    elif regime == 1:
        corr_value = 0.2
    elif regime == 2:
        corr_value = 0.4
    else:
        assert 0, 'Regime not found: {}'
    sim_params = get_sim_params("main_sim_6")
    x_param = sim_params["x_param"]
    x_param["rel_noise_list"] = [0.1]
    x_param["n_repetitions"] = 150
    z_param = sim_params["z_param"]
    z_param['spike_grp'] =  ['left', 'mid', 'right']
    z_param['rel_noise_list'] =  [0.1, 0.1]
    sim_params['target_vars'] = 6000 - len(z_param['spike_grp']) * len(z_param['rel_noise_list'])
    sim_params['noise_levs'] = []
    sim_params['sparsities'] = []
    sim_params['methods'] = ['unsup_graph', 'graph']
    sim_params['sim_dir'] = '/share/PI/sabatti/feat_viz/{}/regime_{}'.format(sim_id, regime)
    sim_params['method_params'] =  {
        "method": "graph",
        "n_perms": 10000,
        "perm_method": "pool",
        "alpha": 0.05, 
        "graph_k": 10,
    }
    noise_params = sim_params["null_struct"]
    noise_params['corr_value'] = corr_value
    noise_params['scale'] = 0.5
    noise_params['seed'] = 10
    return sim_params

def get_variable_group_ids(sim_params):
    x_param = sim_params["x_param"]
    n_proto = len(x_param["spike_grp"])
    n_rep = x_param["n_repetitions"] * len(x_param["rel_noise_list"])
    var_ids = np.ones(sim_params["target_vars"]) * -1
    for i in range(n_proto):
        var_ids[i*(n_rep): (i+1)*n_rep] = i
    return var_ids

def generate_invariant_data(sim_params):
    np.random.seed(sim_params["seed"])
    lam = np.random.uniform(size=sim_params["n_samps"])
    z, _ = generate_spike_mtx(lam, **sim_params["z_param"]) # landmark matrix
    x = generate_x_mtx(lam, sim_params["x_param"], sim_params, no_noise=True) # remaining matrix
    nn_grp = get_variable_group_ids(sim_params).astype(int)
    var_df = pd.DataFrame({'var_id': np.arange(x.shape[1]), 'nn_grp': nn_grp})
    return lam, z, x, var_df

def model_corr_noise(null_struct, z, x):
    # process data
    orig_mtx = np.concatenate([z, x], axis=1) 
    n_lm_genes = z.shape[1]
    grp_size = null_struct['block_size']

    np.random.seed(null_struct['seed'])
    noise_mtx = np.zeros(orig_mtx.shape) # add structured noise in blocks
    noise_mtx = add_correlated_noise(noise_mtx, 
                                     null_struct, 
                                     scale=null_struct["scale"], 
                                     trunc=False)
    
    map_idx = np.random.permutation(orig_mtx.shape[1])
    rev_idx = [0] * len(map_idx)
    x_corr_grp = np.array([0] * x.shape[1])
    for old_id, curr_id in enumerate(map_idx):
        rev_idx[curr_id] = old_id
        x_id = curr_id - n_lm_genes
        if x_id >= 0: # non-landmark genes
            x_corr_grp[x_id] = old_id // grp_size
    noise_mtx = noise_mtx[:, rev_idx]
    mtx = orig_mtx + noise_mtx
    # extract the genes correlated with the lm genes
    # identify the group of the lm genes
#     lm_cor_grp_idx = []
    x_lm_corr = np.array([False] * x.shape[1])
    for i in range(n_lm_genes):
        noise_idx = rev_idx[i]
        # find the group it belongs to and the indices in between
        grp_id = noise_idx // grp_size
        grp_beg = grp_id * grp_size
        grp_end = (grp_id + 1) * grp_size
        grp_ids = map_idx[grp_beg:grp_end]
        assert i in grp_ids, '{} not in group'.format(i)
        grp_ids = grp_ids - n_lm_genes # original index
        x_lm_corr[grp_ids] = True
#         lm_cor_grp_idx.append(grp_ids)
#     new_nn_idx = np.concatenate(lm_cor_grp_idx)
#     new_nn_idx = new_nn_idx-n_lm_genes 
#     new_nn_idx = new_nn_idx[new_nn_idx >=0]
    z = mtx[:, :n_lm_genes]
    x = mtx[:, n_lm_genes:]
    # scale the outputs
    z = norm_mtx(z)
    x = norm_mtx(x)

    # true/false correlated with lm genes
    df = pd.DataFrame({'corr_grp': x_corr_grp, 'lm_corr': x_lm_corr})
    return z, x, df

def selction_eval(result, lam_true, sim_params, var_df):
    x_param = sim_params["x_param"]
    n_proto = len(x_param["spike_grp"])
    n_rep = x_param["n_repetitions"] * len(x_param["rel_noise_list"])
    nn_idx = np.arange(n_rep*n_proto)
    new_nn_idx = var_df.loc[var_df['lm_corr']].sort_values('corr_grp')['var_id']
    nn_set = set(nn_idx).union(set(new_nn_idx))
    # add the correlated noise variables to the FDR
    rej_idx = result["rejections"]
    mt_res = evaluate_rejections(set(rej_idx), nn_set)
    mt_res["Corr"] = abs(spearmanr(result["lam_update"], lam_true).correlation) 
    mt_res["Num_Nonnulls"] = len(nn_set)
    mt_res["Num_Rejections"] = len(rej_idx)
    print(mt_res)
    