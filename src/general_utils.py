import numpy as np
import pandas as pd
import json
import os
import pickle
import logging
from ast import literal_eval

logger = logging.getLogger("feat_viz")

def rank_and_bin(z, n_bins=None, linspace=False):
    if n_bins is None:
        n_bins = len(z)
    n_items = len(z)
    T = [(z[i], i) for i in range(len(z))]
    T.sort(key=lambda x: x[0])
    rank = [0] * n_items
    for i, tup in enumerate(T):
        rank[tup[1]] = i
    if n_items <= n_bins:
        return rank

    curr_grp = n_bins - 1
    n_per_grp = n_items // n_bins
    T = [[t[0], t[1], 0] for t in T]
    
    if linspace:
        pts = np.linspace(min(z), max(z), num=2 * n_bins - 1)
        hist_centers = pts[0::2]
        avg_pts = pts[1::2]
        avg_pts = np.append(avg_pts, hist_centers[-1])
    curr_grp = 0
    for i, tup in enumerate(T):
        if linspace:
            if tup[0] > avg_pts[curr_grp]:
                curr_grp += 1
            tup[2] = hist_centers[curr_grp]   
#             print("{} : {}".format(curr_grp, tup))
        else:
            if (curr_grp < (n_bins - 1)) and (i == n_per_grp * (1 + curr_grp)):
                curr_grp += 1
            tup[2] = curr_grp
    for i, tup in enumerate(T):
        rank[tup[1]] = tup[2]
    return rank

def extended_bounds(x, f=0.5):
    min_val = min(x)
    max_val = max(x)
    diff = max_val - min_val
    pad = diff * f
    return [min_val - pad, max_val + pad]


def mask_vec(inlist, whitelist, mark="other"):
    outlist = []
    for g in inlist:
        if g in whitelist:
            outlist.append(g) 
        else:
            outlist.append(mark)
    return outlist

def norm_mtx(x, center=True, scale=True, verbose=True):
    if center:
        x = x - x.mean(axis=0)
        if scale:
            x = x / x.std(axis=0)
#             logger.debug("Data is centered and scaled")
#         else:
#             logger.debug("Data is centered but not scaled")
    return x


def evaluate_rejections(reject_set, nonnull_set):
    n_true_pos = len(reject_set.intersection(nonnull_set))
    n_false_pos = len(reject_set - nonnull_set)
    out_dict = {}
    out_dict["FDP"] = 1.0 * n_false_pos / max(len(reject_set), 1)
    out_dict["Power"] = 1.0 * n_true_pos / max(len(nonnull_set), 1)
    return out_dict


def read_spatial_expression(file,sep='\s+',num_exp_genes=0.01, num_exp_spots=0.01, min_expression=1):
    
    '''
    Read raw data and returns pandas data frame of spatial gene express
    and numpy ndarray for single cell location coordinates; 
    Meanwhile processing raw data.
    
    :param file: csv file for spatial gene expression; 
    :rtype: coord (spatial coordinates) shape (n, 2); data: shape (n, m); 
    '''
    counts = pd.read_csv(file, sep=sep, index_col = 0)
    print('raw data dim: {}'.format(counts.shape))

    num_spots = len(counts.index)
    num_genes = len(counts.columns)
    min_genes_spot_exp = round((counts != 0).sum(axis=1).quantile(num_exp_genes))
#     print("Number of expressed genes a spot must have to be kept " \
#     "({}% of total expressed genes) {}".format(num_exp_genes, min_genes_spot_exp))
    counts = counts[(counts != 0).sum(axis=1) >= min_genes_spot_exp]
#     print("Dropped {} spots".format(num_spots - len(counts.index)))
          
    # Spots are columns and genes are rows
    counts = counts.transpose()
  
    # Remove noisy genes
    min_features_gene = round(len(counts.columns) * num_exp_spots) 
#     print("Removing genes that are expressed in less than {} " \
#     "spots with a count of at least {}".format(min_features_gene, min_expression))
    counts = counts[(counts >= min_expression).sum(axis=1) >= min_features_gene]
#     print("Dropped {} genes".format(num_genes - len(counts.index)))
    
    data=counts.transpose()
    temp = [val.split('x') for val in data.index.values]
    coord = np.array([[float(a[0]), float(a[1])] for a in temp])
    
    return coord,data




def normalize_count_cellranger(data):
    '''
    normalize count as in cellranger
    
    :param file: data: A dataframe of shape (m, n);
    :rtype: data shape (m, n);
    '''
    normalizing_factor = np.sum(data, axis = 1)/np.median(np.sum(data, axis = 1))
    data = pd.DataFrame(data.values/normalizing_factor[:,np.newaxis], columns=data.columns, index=data.index)
    return data

def read_result_to_dataframe(fileName,sep=','):
    
    '''
    Read and use scGCO output file cross-platform .
    More detail can see **write_result_to_csv()**.
    '''
    converters={"p_value":converter,"nodes":converter}
    df=pd.read_csv(fileName,converters=converters,index_col=0,sep=sep)
    df.nodes=df.nodes.apply(conver_array)
    return df

def conver_list(x):
    return [list(xx) for xx in x ]

def conver_array(x):
    return [np.array(xx) for xx in x] 

def converter(x):
    #define format of datetime
    return literal_eval(x)

