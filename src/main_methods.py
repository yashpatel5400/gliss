import numpy as np
import pandas as pd
import json
import os
import pickle
import logging

from general_utils import norm_mtx, evaluate_rejections
from pcurve_utils import initial_pcurve
from graph_utils import construct_kernal_graph, construct_knn_naive
from graph_utils import laplacian_score, get_graph_spectrum
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger("feat_viz")

def multitest_rejections(raw_pvals, alpha, method="Bonferroni"):
    raw_pvals = np.array(raw_pvals) # numpy array
    n_tests = len(raw_pvals)
    # logger.info("Multi-test correction for {} hypotheses".format(n_tests))
    if method == "Bonferroni":
        adjust_pvals = raw_pvals  * n_tests
        reject_set = np.where( adjust_pvals < alpha )[0]
    if method == "BH":
        # estimates FDR by V(t) / max(R(t), 1) via threshold t
        # goal is to find the maximum t such that FDR(t) < alpha
        thresholds = np.sort(raw_pvals)
        V_t = thresholds * n_tests # estimate of false rejections
        R_t = np.arange(len(thresholds)) + 1 # number of rejections
        FDR_t = V_t / R_t
        valid_thresholds = FDR_t <= alpha
        if (np.sum(valid_thresholds) > 0):
            final_threshold = np.max(thresholds[valid_thresholds])
        else:
            final_threshold = 0
        reject_set = np.where( raw_pvals <= final_threshold )[0]
    return reject_set

# @njit(cache=True)
def pool_perm_variables(X, n_perm=10000):
    np.random.seed(10101)
    if X.ndim == 1: # fill up to a vector
        x_mtx = np.expand_dims(X, axis=1)
    else:
        x_mtx = X
    assert x_mtx.ndim == 2, "error in input X dimensions"
    n_samp, n_vars = x_mtx.shape
    perm_mat = np.zeros((n_samp, n_perm))
    for i in range(n_perm):
        value = x_mtx[:,np.random.choice(range(n_vars))]
        perm_mat[:,i] = np.random.permutation(value)
    return(perm_mat)

def batch_perm_lap_score(kgraph, Y, n_perms):
    n_obs, n_batch_vars = Y.shape
    perm_id = np.arange(n_obs)
    batch_output = np.zeros((n_perms, n_batch_vars))
    for i_perm in range(n_perms):
        if i_perm > 0 and i_perm % 100 == 0:
            logger.info("Finished {} permutations".format(i_perm))
        np.random.shuffle(perm_id)
        batch_output[i_perm, :] = laplacian_score(Y[perm_id, :], kgraph)
    return batch_output

def generate_correlation_map(x, y):
    """Correlate each n with each m.
    https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays?rq=1
    
    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def infer_lambda(method, in_mtx, knn=10):
    assert method in ["pc", "graph"], "{} not defined".format(method)
    aux = {"func": None, "graph": None}
    in_mtx = norm_mtx(in_mtx, verbose=False)  # scaling here
    if method == "pc":
        lam_hat, fun = initial_pcurve(in_mtx)
        aux["func"] = fun
    if method == "graph":
        assert knn >= 0, "Graph param must be non-negative"
        if knn == 0:
            graph = construct_kernal_graph(in_mtx)
        else:
            graph = construct_knn_naive(in_mtx, k=knn)
        lam_hat = get_graph_spectrum(graph, num=1)
        aux["graph"] = graph
    return lam_hat, aux

def compute_corr(method, x, z_repr):
    # TODO: handle the dimension problem
    if method == "graph":
        out = laplacian_score(x, z_repr)
    if method == "pc":
        # typically z_repr for lambda is going to be 1 dimensional
        in_z = np.expand_dims(z_repr, axis=0)
        out = abs(generate_correlation_map(in_z, x.T))[0]
    return out

def compute_feature_pvals(method, x, z_repr, n_perms, perm_method, seed=0):
    assert method in ["pc", "graph"], "{} not defined".format(method)
    np.random.seed(seed)
    obs_stat = compute_corr(method, x, z_repr)
    if perm_method == "pool":
        x_perm = pool_perm_variables(x, n_perm=n_perms)
        perm_data = compute_corr(method, x_perm, z_repr)
        # TODO: speed up
        pvals = -1 * np.ones(len(obs_stat))
        for i_var, stat in enumerate(obs_stat):
            if method == "graph":
                cnt_exceed = np.sum(perm_data <= stat)
            else:
                cnt_exceed = np.sum(perm_data >= stat)
            pvals[i_var] = (cnt_exceed + 1) / (n_perms + 1)
    else:
        assert 1, "{} permutation not available".format(perm_method)
        if method == "graph":
            perm_data = batch_perm_lap_score(z_repr, x, n_perms)
        
        cnt_exceed = np.sum(perm_data <= obs_stat, axis=0)
        pvals = (cnt_exceed + 1) / (n_perms + 1)
    return pvals


def run_procedure(x_k, x_d, params, lam_in=None, fn=None):
    method = params["method"]
    n_perms = params["n_perms"]
    perm_method = params["perm_method"]
    alpha = params["alpha"]
    graph_k = params["graph_k"]
    
    if lam_in:
        logger.info("Using pre-computed latent variable".format(method))
        if method == "graph":
            x_in = np.expand_dims(np.array(lam_in), axis=1)
            lam_hat, aux = infer_lambda(method, x_in, knn=graph_k)
        else: # pc or hybrid
            lam_hat = lam_in
            lam_repr = lam_hat
    else:
        logger.info("Running {}-based procedure".format(method))
        if method == "hybrid":
            use_method = "pc"
        else:
            use_method = method
        lam_hat, aux = infer_lambda(use_method, x_k, knn=graph_k)
        logger.info("Inferred initial latent variables")

    logger.info("Selecting {}-based features...".format(method))
    use_method = method
    if method == "hybrid":
        x_in = np.expand_dims(np.array(lam_hat), axis=1)
        _, aux = infer_lambda("graph", x_in, knn=graph_k)
        lam_repr = aux["graph"]
        use_method = "graph"
    elif method == "graph":
        lam_repr = aux["graph"]
    else: # pc
        lam_repr = lam_hat
        
    pvals = compute_feature_pvals(use_method, x_d, lam_repr, n_perms, perm_method)
    rej_idx = multitest_rejections(pvals, alpha, method="BH")
    logger.info("Updated latent variables...")
    new_x = np.concatenate([x_k, x_d[:,rej_idx]], axis=1)
    new_lam_hat, aux = infer_lambda(use_method, new_x, knn=graph_k)
    result = {
        "lam_init": lam_hat,
        "lam_update": new_lam_hat,
        "p_vals": pvals,
        "rejections": rej_idx,
    }
    if fn:
        logger.info("Saving results to: {}".format(fn))
        pickle.dump(result, open(fn, "wb"))
    return result

def run_unsupervised(x_k, x_d, params, fn=None):
    method = params["method"]
    assert method in ["graph", "pc"], "{} not defined".format(method)
    graph_k = params["graph_k"]
    new_x = np.concatenate([x_k, x_d], axis=1)
    new_lam_hat, aux = infer_lambda(method, new_x, knn=graph_k)
    if fn:
        logger.info("Saving results to: {}".format(fn))
        pickle.dump(new_lam_hat, open(fn, "wb"))
    return new_lam_hat
    
def evaluate_result(result, lam_ref=None, rej_ref=None):
    logger.info("Number of selected variables: {}".format(len(result["rejections"])))
    if lam_ref:
        init_corr = abs(spearmanr(lam_ref, result["lam_init"]).correlation)
        updated_corr = abs(spearmanr(lam_ref, result["lam_update"]).correlation)
#         init_corr = abs(pearsonr(lam_ref, result["lam_init"])[0])
#         updated_corr = abs(pearsonr(lam_ref, result["lam_update"])[0])
        logger.info("Correlation: {:.5f} -> {:.5f}".format(init_corr, updated_corr))

def setup_cmp_df(res_dict):
    mets = list(res_dict.keys())
    cmp_sets = [set(res_dict[m]["rejections"]) for m in mets]
    cmp_pvals = {m: res_dict[m]["p_vals"] for m in mets}
    pval_df = pd.DataFrame(cmp_pvals)
    pval_thres = []
    for m in mets:
        rej_list = res_dict[m]["rejections"]
        pt = max(res_dict[m]["p_vals"][rej_list])
        logger.info("{} threshold: {:.4f}".format(m, pt))
        pval_thres.append(pt) 
    return pval_df, pval_thres, cmp_sets, mets


def get_unique_rejection_df(res_dict, var_df):
    pval_df, pval_thres, cmp_sets, mets = setup_cmp_df(res_dict)
    pval_df = pd.concat([var_df.reset_index(drop=True), pval_df], axis=1)
    df_list = []
    for i in range(2):
        j = 1 * (i == 0)
        sub_df = pval_df.loc[pval_df[mets[i]] <= pval_thres[i]]
        sub_df = sub_df.loc[sub_df[mets[j]] > pval_thres[j]]
        sub_df["exclusive_rejection"] = mets[i]
        logger.info("Set selection: {}".format(sub_df.shape)) 
        sub_df = sub_df.set_index("gene_ids")
        sub_df = sub_df.sort_values([mets[i], mets[j]], ascending=[1, 0])
        df_list.append(sub_df)
    df_comp = pd.concat(df_list, axis=0) # should never conflict
    df_dict = dict(zip(mets, df_list))
    return df_dict, df_comp