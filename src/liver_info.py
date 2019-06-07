import os
import pandas as pd
import numpy as np
import scanpy.api as sc
from anndata import AnnData, read_h5ad

from sim_utils import scale_corr_input
from general_utils import norm_mtx

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

import logging
logger = logging.getLogger("feat_viz")

def load_processsed_hepatocyte_data(main_dir, center=True, scale=False):
    tmp_dir = os.path.join(main_dir, "data")
    obs_df = pd.read_csv(os.path.join(tmp_dir, "obs_info.csv"), index_col=0)
    var_df = pd.read_csv(os.path.join(tmp_dir, "var_info.csv"), index_col=0)
    x = np.load(os.path.join(tmp_dir, "x_data_unscaled.npy"))
    y = np.load(os.path.join(tmp_dir, "y_data_unscaled.npy"))
    logger.info("Data directory: {}".format(tmp_dir))
    logger.info("Loaded data: x {} and y {}".format(x.shape, y.shape))
    x = norm_mtx(x, center=center, scale=scale)
    y = norm_mtx(y, center=center, scale=scale)
    return x, y, obs_df, var_df


def output_processed_data(main_dir, processed_dir):
    tmp_dir = os.path.join(main_dir, "data")
    obs_df = pd.read_csv(os.path.join(tmp_dir, "obs_info.csv"), index_col=0)
    var_df = pd.read_csv(os.path.join(tmp_dir, "var_info.csv"), index_col=0)
    x = np.load(os.path.join(tmp_dir, "x_data_unscaled.npy"))
    y = np.load(os.path.join(tmp_dir, "y_data_unscaled.npy"))
    out_dir = os.path.join(processed_dir, "data")
    os.makedirs(out_dir, exist_ok=True)
    obs_df.to_csv(os.path.join(out_dir, "obs_info.csv"))
    var_df.to_csv(os.path.join(out_dir, "var_info.csv"))
    np.savetxt(os.path.join(out_dir, "matrix_unscaled.csv"), y, delimiter=",")
    logger.info("Saved outputs to: {}".format(out_dir))


def load_zonation_result(dat_dir, just_vals=False):
    # load the zonation data 
    meta_dat = pd.read_csv(os.path.join(dat_dir, "table_s2_reform.csv"))
    meta_dat = meta_dat.set_index("Cell #")
    meta_dat.index = [cellid.replace(" ", "") for cellid in meta_dat.index]
    zone_prob_cols = [col for col in meta_dat if col.startswith("Layer")]
    layer_data = meta_dat[zone_prob_cols]
    out = layer_data.idxmax(axis=1).tolist()
    if just_vals:
        return [int(z.split(" ")[1]) for z in out]
    else:
        return out


def get_known_liver_markers():
        known_markers = {
            "zonation": [
                'Cyp2e1',
                'Cyp2f2',
                'Alb',
                'Ass1',
                'Asl',
                'Glul'
            ],
            "hepatocyte": [
                "Apoa1", 
                "Glul", 
                "Acly",
                "Asl",
                "Cyp2e1",
                "Cyp2f2",
                "Ass1",
                "Alb",
                "Mup3",
                "Pck1",
                "G6pc"
            ],
            "kupffer": [
                "Irf7",
                "Spic",
                "Clec4f"
            ],
            "enodthelial" : [
                "Ushbp1",
                "Myf6",
                "Oit3",
                "Il1a",
                "F8",
                "Bmp2",
                "C1qtnf1",
                "Mmrn2",
                "Pcdh12",
                "Dpp4"
            ],
            "kanako": [
                "Cyp2e1",
                "Cyp2f2",
                "Glul", 
                "Axin2",
                "Lgr5",
                "Lgr4",
                "Znrf3",
                "Rnf43",
                "Rspo3",
                "Wnt2",
                "Wnt9b",
                "Krt19",
                "Krt9",
                "Fah"
            ],
            "frizzled": ["Fzd{}".format(i+1) for i in range(10)]
        }
        return known_markers

def get_marker_color_dict(marker, colmap="tab10"):
    genes = get_known_liver_markers()[marker]
    n_genes = len(genes)
    cmap = plt.cm.get_cmap(colmap)
    scale_vals = np.linspace(0,1, len(genes))
    # rgb_values = [cmap(x) for x in scale_vals]
    rgb_values =  sns.color_palette(colmap, len(genes))
    color_map = dict(zip(genes, rgb_values))
    color_map["other"] = colors.to_rgba('lightgrey')
    return color_map

    

def load_cell_meta_data(dat_dir):
    adata = load_hepatocyte_data(dat_dir)

def load_hepatocyte_data(dat_dir, verbose=False):
    # parse the file that was manually reformattted
    # the last column is empty - remove
    # the first column is the gene - make index
    dat = pd.read_table(os.path.join(dat_dir, "table_s1_reform.txt"), delimiter=" ")
    dat = dat[dat.columns[:-1]].set_index("gene")
    dat.shape # Proceed with the scanpy pipeline

    adata = AnnData(dat.T.values)
    adata.obs_names = np.array(dat.columns)
    adata.var_names = np.array(dat.index)
    adata.var['gene_ids'] = dat.index
    adata.var_names_make_unique()
    if verbose:
        sc.pl.highest_expr_genes(adata, n_top=10)

    # filter out zero genes
    sc.pp.filter_cells(adata, min_genes=200)
    adata.raw = adata.copy()
    sc.pp.filter_genes(adata, min_cells=10)
    adata.obs['n_counts'] = adata.X.sum(axis=1)
    if verbose:
        sc.pl.violin(adata, ['n_genes', 'n_counts'], jitter=0.4, multi_panel=True)
        nz_genes = len(adata.var['gene_ids'])
        print("Filtered out: {} genes; remaining {}".format(len(dat.index) - nz_genes, nz_genes))

    # filter cells with too many counts (potential doublets)
    cell_sel = adata.obs['n_genes'] < 5000
    adata = adata[cell_sel, :]
    if verbose:
        print("Filtered out: {} doublet cells".format(np.sum(np.logical_not(cell_sel))))
        print(adata)
    return adata
    

def save_x_y_data(main_dir, x, y, pfx):
    tmp_dir = os.path.join(main_dir, "data")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    np.save(os.path.join(tmp_dir, "x_data_{}.npy".format(pfx)), x)
    np.save(os.path.join(tmp_dir, "y_data_{}.npy".format(pfx)), y)
    print("Saved: x {} and y {} at {}".format(
        x.shape, y.shape, tmp_dir))
    
def save_processed_data(main_dir, x, y, obs_df, var_df):
    tmp_dir = os.path.join(main_dir, "data")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    save_x_y_data(main_dir, x, y, "unscaled")
    obs_df.to_csv(os.path.join(tmp_dir, "obs_info.csv"))
    var_df.to_csv(os.path.join(tmp_dir, "var_info.csv"))
    print("Saved: x {} and y {} at {}".format(x.shape, y.shape, tmp_dir))
    print(obs_df.head())
    print(var_df.head())
    
def load_x_y_data(main_dir, center=True, scale=False):
    tmp_dir = os.path.join(main_dir, "data")
    obs_df = pd.read_csv(os.path.join(tmp_dir, "obs_info.csv"), index_col=0)
    var_df = pd.read_csv(os.path.join(tmp_dir, "var_info.csv"), index_col=0)
    x = np.load(os.path.join(tmp_dir, "x_data_unscaled.npy"))
    y = np.load(os.path.join(tmp_dir, "y_data_unscaled.npy"))
    print("Data directory: {}".format(tmp_dir))
    print("Loaded data: x {} and y {}".format(x.shape, y.shape))
    x = norm_mtx(x, center=center, scale=scale)
    y = norm_mtx(y, center=center, scale=scale)
#     if center:
#         x = x - x.mean(axis=0)
#         y = y - y.mean(axis=0)
#         print("x and y are centered")
#         if scale:
#             x = x / x.std(axis=0)
#             y = y / y.std(axis=0)
#             print("x and y are scaled")
#         else:
#             print("x and y are not scaled")
#     else:
#         print("x and y are not centered or scaled")
        
    return x, y, obs_df, var_df

# def load_noramlized_data(main_dir):
#     tmp_dir = os.path.join(main_dir, "data")
#     obs_df = pd.read_csv(os.path.join(tmp_dir, "obs_info.csv"), index_col=0)
#     var_df = pd.read_csv(os.path.join(tmp_dir, "var_info.csv"), index_col=0)
#     x_scaled = np.load(os.path.join(tmp_dir, "x_data.npy"))
#     y_scaled = np.load(os.path.join(tmp_dir, "y_data.npy"))
#     print("Loaded: x {} and y {} from {}".format(
#         x_scaled.shape, y_scaled.shape, tmp_dir))
#     return x_scaled, y_scaled, obs_df, var_df

    
def get_projection(proj_type):
    results_file = os.path.join(dat_dir,'{}_2k.h5ad'.format(sample_name))
    proj = read_h5ad(results_file).obsm[proj_type]
    return proj