import os
import pandas as pd
import numpy as np
import scanpy.api as sc
from anndata import AnnData, read_h5ad

import seaborn as sns

from io_utils import save_data_to_file, load_data_from_file

import logging
logger = logging.getLogger("feat_viz")


def prepare_adata_init(dat, min_genes=200, min_cells=10, max_genes=5000, verbose=False):
    # dat has columns as samples and rows as genes
    # Proceed with the scanpy pipeline
    adata = AnnData(dat.T.values)
    adata.obs_names = np.array(dat.columns)
    adata.var_names = np.array(dat.index)
    adata.var['gene_ids'] = dat.index
    adata.var_names_make_unique()
    if verbose:
        sc.pl.highest_expr_genes(adata, n_top=10)
    # filter out zero genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata.raw = adata.copy()
    sc.pp.filter_genes(adata, min_cells=10)
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    if verbose:
        sc.pl.violin(adata, ['n_genes', 'n_counts'], jitter=0.4, multi_panel=True)
        nz_genes = len(adata.var['gene_ids'])
        print("Filtered out: {} genes; remaining {}".format(len(dat.index) - nz_genes, nz_genes))

    # filter cells with too many counts (potential doublets)
    cell_sel = adata.obs['n_genes'] < max_genes
    adata = adata[cell_sel, :]
    if verbose:
        print("Filtered out: {} doublet cells".format(np.sum(np.logical_not(cell_sel))))
        print(adata)
    return adata

def compute_log_gene_summary(adata):
    in_mat = np.log1p(adata.X) # natural log
    log_gene_df = pd.DataFrame(index=adata.var['gene_ids'])
    log_gene_df["mean"] = in_mat.mean(axis=0)
    log_gene_df["std"] = in_mat.std(axis=0)
    log_gene_df["cv"] = log_gene_df["std"] / log_gene_df["mean"] 
    log_gene_df["sparsity"] = (in_mat < 1e-5).sum(axis=0) / in_mat.shape[0]
    return log_gene_df

def plot_gene_mean_std(log_gene_df):
    sns.jointplot(x="mean", y="std", 
                  data=log_gene_df,
                  height=4.5,
                  joint_kws={"s": 1}, 
                  marginal_kws=dict(bins=100, rug=True));
    # std filter results (log2 scale sweep)
    sweeps = [0.05, 0.1, 0.15, 0.2]
    num_genes = log_gene_df.shape[0]
    for sweep in sweeps:
        num_kept = log_gene_df.loc[log_gene_df["std"] > sweep].shape[0]
        logger.info("Kept {} / {} genes with log filter at {}".format(
            num_kept, num_genes, sweep))
        
def transform_filter_anndata(adata, std_ln_filter = 0.1, scale_pfx="original", verbose=True):
    # scale total counts of each cell ('n_counts') to be the same (or median)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    
    # select genes based on their log-scale variance
    log_gene_df = compute_log_gene_summary(adata)
    keep_genes = log_gene_df["std"] > std_ln_filter
    log_gene_df["keep"] = keep_genes
    logger.info("Keeping {} / {} genes with log1p and std > {}".format(
         np.sum(keep_genes), len(keep_genes), std_ln_filter))
    adata = adata[:, log_gene_df["keep"]]
    
    # log-transform the umi counts
    sc.pp.log1p(adata)
    logger.info("Log-transformed anndata")
    
    # optionally regress out the size factors
    if scale_pfx == "size_norm":
        print("Regressing out size effects from anndata")
        sc.pp.regress_out(adata, ['n_counts'])
    
    # optional further processing
    # sc.pp.normalize_per_cell(adata) # median normalization not use here
    # if necessary: center and scale each gene later
    if verbose:
        plot_gene_mean_std(log_gene_df)
    return adata

def write_unscaled_processed_data(out_dir, obs_df, var_df, mtx, npy=True):
    os.makedirs(out_dir, exist_ok=True)
    save_data_to_file(obs_df, os.path.join(out_dir, "obs_info.csv"), "csv")
    save_data_to_file(var_df, os.path.join(out_dir, "var_info.csv"), "csv")
    save_data_to_file(mtx, os.path.join(out_dir, "matrix_unscaled.csv"), "csv")
    if npy:
        save_data_to_file(mtx, os.path.join(out_dir, "matrix_unscaled.npy"), "npy")
        
def read_unscaled_processed_data(out_dir, npy=True):
    obs_df = load_data_from_file(os.path.join(out_dir, "obs_info.csv"), "csv")
    var_df = load_data_from_file(os.path.join(out_dir, "var_info.csv"), "csv")
    if npy:
        mtx = load_data_from_file(os.path.join(out_dir, "matrix_unscaled.npy"), "npy")
    else:
        mtx = np.loadtxt(os.path.join(out_dir, "matrix_unscaled.csv"), delimiter=",")
        logger.info("Loaded matrix: {}".format(mtx.shape))
    return obs_df, var_df, mtx

def summarize_gene_df(adata):
    gene_scores = {"mean": adata.X.mean(axis=0), 
                   "std": adata.X.std(axis=0),
                   "npc": np.sum(adata.X > 0, axis=0)
                  }
    gene_scores = pd.DataFrame(gene_scores, index=adata.var.index)
    gene_scores = adata.var.join(gene_scores, how="left")
    return gene_scores

def split_by_lm_genes(genes, var_df, mtx):
    new_df = var_df.set_index("gene_ids")
    anchor = var_df['gene_ids'].isin(genes)
    return mtx[:, anchor], mtx[:, -anchor]
    

def get_gene_df(genes, var_df, mtx):
    new_df = var_df.set_index("gene_ids")
    inds = [int(var_df.index[var_df["gene_ids"]==g][0]) for g in genes]
    sub_mtx = mtx[:, inds]
    sub_df = pd.DataFrame(sub_mtx, columns=genes)
    logger.info("x_star_df dim: {}".format(sub_df.shape))
    return sub_df
