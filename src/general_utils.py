import numpy as np
import pandas as pd
import json
import os
import pickle
import logging

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
