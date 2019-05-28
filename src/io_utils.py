import os
import json
import pandas as pd
import numpy as np
import pickle
import logging

logger = logging.getLogger("feat_viz")

def compute_batch_pval(i_batch, gene_sub_df, metric, tmp_dir):
    # load_batch_data and p-values from the full permutation
    fname = "{}_batch_{}.npy".format(metric, i_batch)
    fname = os.path.join(tmp_dir, fname) 
    perm_data = np.load(fname)
    obs_stat = gene_sub_df[metric].values
    n_perms = perm_data.shape[0]
    if metric == "lap_score":
        cnt_exceed = np.sum(perm_data < obs_stat, axis=0)
    else:
        cnt_exceed = np.sum(perm_data > obs_stat, axis=0)
    pvals = (cnt_exceed + 1 ) / (n_perms + 1)
    return pvals

def compute_batch_ci(i_batch, metric, tmp_dir, ci=90):
    # load_batch_data and p-values from the full permutation
    # print("Loading {} CI with {} (batch {})".format(ci, metric, i_batch))
    fname = "{}_batch_{}.npy".format(metric, i_batch)
    fname = os.path.join(tmp_dir, fname) 
    perm_data = np.load(fname)
    lr_size = (100 - ci) / 2
    left = np.percentile(perm_data, lr_size, axis=0)
    right = np.percentile(perm_data, ci + lr_size, axis=0)
    return np.stack([left, right]).T

def compute_pool_pval(gene_df, metric, tmp_dir):
    fname = "{}_pooled.npy".format(metric)
    fname = os.path.join(tmp_dir, fname) 
    perm_data = np.load(fname)
    n_perms = len(perm_data)
    obs_stat = gene_df[metric].values
    # TODO: speed up
    pvals = -1 * np.ones(len(obs_stat))
    for i, stat in enumerate(obs_stat):
        if metric == "lap_score":
            cnt_exceed = np.sum(perm_data < stat)
        else:
            cnt_exceed = np.sum(perm_data > stat)
        pvals[i] = (cnt_exceed + 1 ) / (n_perms + 1)
    return pvals


def query_perm_data(gene_index, tmp_dir, metric):
    # searches for the batch with the correct index
    with open(os.path.join(tmp_dir, 'batch_job_params.json'), 'r') as f:
        params = json.load(f)
    i_batch = gene_index // params["batch_size"]
    j_column =  gene_index % params["batch_size"]
    fname = "{}_batch_{}.npy".format(metric, i_batch)
    fname = os.path.join(tmp_dir, fname) 
    perm_column = np.load(fname)[:, j_column]
    return perm_column


def get_pool_perm_intvls(tmp_dir, sig_level=0.05, metrics=["lap_score", "dist_corr"]):
    pool_perm_intervals = {}
    lr_size = sig_level * 100 
    ci = 100 - 2 * lr_size
    for metric in metrics:
        print("Loading pooled permutation data from {}".format(metric))
        fname = "{}_pooled.npy".format(metric)
        fname = os.path.join(tmp_dir, fname) 
        perm_data = np.load(fname)
        print("Number of permutated samples: {}".format(perm_data.shape))
        left = np.percentile(perm_data, lr_size)
        right = np.percentile(perm_data, ci + lr_size)
        pool_perm_intervals[metric] = [left, right]
        print("Constructed {}% interval: [{:.5f}, {:.5f}]\n".format(ci, left, right))    
    return pool_perm_intervals


def load_gene_perm_summary(tmp_dir, auxiliary=False, conf_intvl=90):
    # loading for only the lap_score and dist_corr result
    with open(os.path.join(tmp_dir, 'batch_job_params.json'), 'r') as f:
        params = json.load(f)
        print(params)
    if auxiliary:
        x = np.load(os.path.join(tmp_dir, "x_data.npy"))
        y = np.load(os.path.join(tmp_dir, "y_data.npy"))
        print("Loaded: x {} and y {}".format(x.shape, y.shape))
        assert y.shape[1] == params["n_variables"], "nvar error"
    gfname = os.path.join(tmp_dir, "gene_summary.csv")
    gene_scores = pd.read_csv(gfname, index_col=0)
    gene_scores["gene_index"] = np.arange(gene_scores.shape[0])
    assert params["n_variables"] == gene_scores.shape[0], "nvar error"
    print("Number of permuted vars: {}".format(params["n_variables"]))
    print("Number of permutations per var: {}".format(params["n_perms"]))

    metrics = ["lap_score", "dist_corr"]
    for met in metrics:
        assert met in gene_scores.columns, "unknown metric {}".format(met)
        # get the p-values from the pooled permutation
        pval_col = "{}_pool_pval".format(met)
        gene_scores[pval_col] = compute_pool_pval(gene_scores, met, tmp_dir)
        # get the p-values from the full permutation
        pval_col = "{}_full_pval".format(met)
        pvals = (-1) * np.ones(params["n_variables"])
        ci = (-1) * np.ones((params["n_variables"], 2))
        # gene_scores[pval_col] = -1
        for i_batch in range(params["n_batches"]):
            beg_idx = i_batch * params["batch_size"]
            end_idx = min(params["n_variables"], beg_idx + params["batch_size"])
            sub_df = gene_scores.iloc[beg_idx:end_idx]
            pvals[beg_idx:end_idx] = compute_batch_pval(i_batch, sub_df, met, tmp_dir) 
            ci[beg_idx:end_idx, :] = compute_batch_ci(i_batch, met, tmp_dir, ci=conf_intvl) 
        gene_scores[pval_col] = pvals
        gene_scores["{}_left_ci".format(pval_col)] = ci[:, 0]
        gene_scores["{}_right_ci".format(pval_col)] = ci[:, 1]
        
    n_cells = np.load(os.path.join(tmp_dir, "x_data.npy")).shape[0]
    print("Number of samples: {}".format(n_cells))
    gene_scores["sparsity"] = 1 - gene_scores["npc"] / n_cells
    return gene_scores

def load_most_recent_results(sig_level, tmp_dir, mfn):
    conf_intvl = 100 * (1 - sig_level * 2)
    gene_summary = load_gene_perm_summary(tmp_dir, auxiliary=False, conf_intvl=conf_intvl)
    print(gene_summary.shape)
    mdf = pd.read_csv(mfn)
    mdf.index = gene_summary.index
    print(mdf.shape)
    full_df = pd.concat([gene_summary, mdf], axis=1)
    return full_df

def save_data_to_file(data, fname, ftype):
    # double-check suffix
    sufx = fname.split(".")[-1]
    assert sufx == ftype, "Suffix '{}' does to match!".format(sufx)
    if ftype == "npy":
        np.save(fname, data)
    elif ftype == "csv":
        data.to_csv(fname, index=False)
    elif ftype == "pkl":
        with open(fname, "wb") as outfile:
            pickle.dump(data, outfile,
                        protocol=pickle.HIGHEST_PROTOCOL)
    elif ftype == "json":
        with open(fname, "w") as outfile:
            json.dump(data, outfile)
    else:
        assert 0, "File type: {} not recognized".format(ftype)
    if ftype in ["npy", "csv"]:
        msg = "Data (shape {})".format(data.shape)
    else:
        msg = "Data (length {})".format(len(data))
    logger.debug("{} saved as: {}".format(msg, fname))

def load_data_from_file(fname, ftype):
    # double-check suffix
    sufx = fname.split(".")[-1]
    assert sufx == ftype, "Suffix '{}' does to match!".format(sufx)
    if ftype == "npy":
        data = np.load(fname)
    elif ftype == "csv":
        data = pd.read_csv(fname)
    elif ftype == "pkl":
        with open(fname, "rb") as infile:
            data = pickle.load(infile)
    elif ftype == "json":
        with open(fname, "r") as infile:
            data = json.load(infile)
    else:
        assert 0, "File type: {} not recognized".format(ftype)
    if ftype in ["npy", "csv"]:
        msg = "Data (shape {})".format(data.shape)
    else:
        msg = "Data (length {})".format(len(data))
    logger.debug("{} loaded from: {}".format(msg, fname))

    return data

def flag_complete(data_dir, action, sufx=None):
    if sufx:
        flagfn = "_COMPLETE_{}".format(sufx)
    else:
        flagfn = "_COMPLETE"
    filename = os.path.join(data_dir, flagfn)
    if action == "add":
        open(filename, 'w').close()
    if action == "remove":
        try:
            os.remove(filename)
        except OSError:
            pass
    if action == "check":
        return os.path.exists(filename)
    
def load_all_pipeline_results(RDIR):
    # for the pipeline methods
    pipe_res = {}
    for method in ["pc", "graph", "hybrid"]:
        for mtype in ["vanilla", "oracle"]:
            key = "{}_{}".format(method, mtype)
            fn =  "result_{}_{}.plk".format(method, mtype)
            fn = os.path.join(RDIR, fn)
            pipe_res[key] = pickle.load(open(fn, "rb"))
    # for the unsupervised methods        
    all_lams = {}
    for method in ["graph", "pc"]:
        key = "unsup_{}".format(method)
        fn =  "lam_{}_unsupervised.plk".format(method)
        fn = os.path.join(RDIR, fn)
        all_lams[key] = pickle.load(open(fn, "rb"))
    return pipe_res, all_lams