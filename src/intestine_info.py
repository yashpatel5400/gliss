import os
import pandas as pd
import numpy as np
import scanpy.api as sc
from anndata import AnnData, read_h5ad
from plot_utils import adjust_xy_labels, plot_scatter_discrete
import matplotlib.pyplot as plt

from scrna_utils import write_unscaled_processed_data, summarize_gene_df
from scrna_utils import read_unscaled_processed_data
from general_utils import norm_mtx

import logging
logger = logging.getLogger("feat_viz")


def load_processed_enterocyte_data(RDIR, center=True, scale=False):
    out_dir = os.path.join(RDIR, "entero_data", "data")
    obs_df, var_df, mtx = read_unscaled_processed_data(out_dir)
    mtx = norm_mtx(mtx, center=center, scale=scale)
    return obs_df, var_df, mtx

def output_processed_enterocyte_data(RDIR, adata):
#     RDIR = "/share/PI/sabatti/feat_viz/real_analysis_result/analysis_060719"
    out_dir = os.path.join(RDIR, "entero_data", "data")
    var_df = summarize_gene_df(adata) # gene_info
    write_unscaled_processed_data(out_dir, adata.obs, var_df, adata.X)

def get_intestine_rna_lm_genes():
    # load landmark genes selected from rna genes
    # based on the analysis from the original paper
    dat_dir = "/share/PI/sabatti/sc_data/intestine2k"
    fn = os.path.join(dat_dir, "extracted", "RNAseq_gene_summary.csv")
    rna_df = pd.read_table(fn, delimiter=",")
    logger.info("Loaded {} genes".format(rna_df.shape[0]))
    threshold = 1e-3
    # rna_df = rna_df.loc[(rna_df.m > threshold) & (rna_df.gene_name.isin(raw_dat.columns))]
    rna_df = rna_df.loc[rna_df.m > threshold]

    logger.info("Kept {} genes with max expr > {} ".format(rna_df.shape[0], threshold))
    # top genes
    high_thres = 3.5
    high_zone = 5
    high_rna_df = rna_df.loc[(rna_df.com > high_thres) & (rna_df.mx == high_zone)]
    logger.info("Kept {} high zone genes with geom avg > {} ".format(high_rna_df.shape[0], high_thres))
    high_genes = list(high_rna_df.gene_name)
#     print(high_genes)
    # bottom genes
    low_thres = 2.5
    low_zone = 1
    low_rna_df = rna_df.loc[(rna_df.com < low_thres) & (rna_df.mx == low_zone)]
    logger.info("Kept {} low zone genes with geom avg < {} ".format(low_rna_df.shape[0], low_zone))
    low_genes = list(low_rna_df.gene_name)
#     print(low_genes)
    return {"high": high_genes, "low": low_genes}

def get_smFISH_validated_genes():
    dat_dir = "/share/PI/sabatti/sc_data/intestine2k"
    fn = os.path.join(dat_dir, "extracted", "smFISH_genes.csv")
    smFISH_genes = pd.read_table(fn, header=None)
    smFISH_genes = set(smFISH_genes[0])
    logger.info("Loaded {} smFISH genes".format(len(smFISH_genes)))
    return smFISH_genes

def load_enterocyte_raw_data(dat_dir):
    fn = os.path.join(dat_dir, "table_B_scRNAseq_UMI_counts.tsv")
    raw_dat = pd.read_table(fn, delimiter="\t")
    raw_dat = raw_dat.set_index("gene").T # load the gene expression
    raw_dat.columns.names = [None]
    raw_dat.index.names = ["cell_id"]
    # Note: the cell ids seem to be permuted compared to tableC
    # -> probably a bug from the paper (but things look ok when in order...)
    fn = os.path.join(dat_dir, "table_C_scRNAseq_tsne_coordinates_zones.tsv")
    cell_df = pd.read_table(fn, delimiter="\t")
    raw_dat.index = cell_df["cell_id"]
    return raw_dat

def load_enterocyte_meta_data(dat_dir):
    fn = os.path.join(dat_dir, "table_C_scRNAseq_tsne_coordinates_zones.tsv")
    cell_df = pd.read_table(fn, delimiter="\t")
    cell_df = cell_df.set_index("cell_id")
    proj = cell_df[["tSNE_coordinate_1", "tSNE_coordinate_2"]]
    fig, ax = plt.subplots(1, 1, figsize=(2.7, 2.5))
    plot_scatter_discrete(proj, cell_df["zone"], ax, ms=1)
    adjust_xy_labels(ax)
    plt.show()
    return cell_df

def load_enterocyte_data(dat_dir, verbose = True):
    dat = load_enterocyte_raw_data(dat_dir)
    adata = AnnData(dat.values)
    adata.obs_names = np.array(dat.index)
    adata.var_names = np.array(dat.columns)
    # adata.raw.obs_names = adata.obs_names
    # adata.raw.var_names = adata.var_names
    adata.var['gene_ids'] = dat.columns
    adata.var_names_make_unique()
    if verbose:
        sc.pl.highest_expr_genes(adata, n_top=10)
    print("Input: {} genes; {} samples".format(len(dat.columns), len(dat.index)))
    # filter out zero genes
    sc.pp.filter_cells(adata, min_genes=200)
    adata.raw = adata.copy()
    sc.pp.filter_genes(adata, min_cells=10)
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    if verbose:
        sc.pl.violin(adata, ['n_genes', 'n_counts'], jitter=0.4, multi_panel=True)
        nz_genes = len(adata.var['gene_ids'])
        print("Filtered out: {} genes; remaining {}".format(len(dat.columns) - nz_genes, nz_genes))

    # filter cells with too many counts (potential doublets)
    # cell_sel = adata.obs['n_genes'] < 5000
    # adata = adata[cell_sel, :]
    # if verbose:
    #     print("Filtered out: {} doublet cells".format(np.sum(np.logical_not(cell_sel))))
    #     print(adata)
    return adata