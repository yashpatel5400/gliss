import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

from matplotlib_venn import venn2, venn3

import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import eigh
import os
import logging
import matplotlib.patches as patches
from sim_utils import generate_spike_mtx
import itertools

from sklearn.metrics import pairwise_distances


from sim_utils import add_gaussian_noise, add_uniform_noise, generate_spike_mtx
from sim_utils import get_sim_params, generate_x_mtx
from graph_utils import get_laplacian, laplacian_score
from general_utils import extended_bounds, rank_and_bin
from main_methods import setup_cmp_df
from general_utils import evaluate_rejections



logger = logging.getLogger("feat_viz")

def check_spike_groups(sim_id):
    sim_params = get_sim_params(sim_id)
    plot_spike_grps(sim_params["z_param"]["spike_grp"], 
                sim_params["z_param"]["rel_noise_list"] + [1e-2, 1e-9])
    plot_spike_grps(sim_params["x_param"]["spike_grp"], 
                sim_params["noise_levs"] + [1e-2, 1e-9])
    
def check_example_noise_matrix(sim_id, sparsity, fn=None):
    sim_params = get_sim_params(sim_id)
    np.random.seed(0)
    # need to pre-specify the varying variables
    x_param = sim_params["x_param"]
    x_param["rel_noise_list"] = [0.5]
    x_param["n_repetitions"] = int(sparsity * sim_params["target_vars"] /len(x_param["spike_grp"]))
    x_param["seed_offset"] = 100
    logger.info(x_param)
    lam = np.random.uniform(size=sim_params["n_samps"])
    null_struct = sim_params["null_struct"]
    n_samps = sim_params["n_samps"]
    x_mtx = generate_x_mtx(lam, x_param, sim_params)
    plot_ground_truth_heatmap(lam, x_mtx[:,:2500], fn=fn)
    
def add_margin(ax, axis, min_val, max_val):
    # This will, by default, add 5% to the x and y margins. You 
    # can customise this using the x and y arguments when you call it.
    if axis == "x":
        lim = ax.get_xlim()
    if axis == "y":
        lim = ax.get_ylim()

    min_plus = (lim[1]-lim[0])*min_val
    max_plus = (lim[1]-lim[0])*max_val
    update = [lim[0]-min_plus,lim[1]+max_plus]
    if axis == "x":
        ax.set_xlim(*update)
    if axis == "y":
        ax.set_ylim(*update)
        
        

def get_col(key):
    col_params = dict(my_red='#FFEAEA')
    return col_params[key]

def use_alias(string):
    aliases = {"n_samps": "sample size",
           "graph": "graph-based",
           "pca": "pca-based"}
    if string in aliases:
        return aliases[string]
    else:
        return string
    
def add_circle_legend(ax, x_pos, y_pos, text, col, size=10):
    ax.text(x_pos, y_pos, text, 
            size=size, ha="center", va="center",
            bbox=dict(boxstyle="circle", 
                      edgecolor="black", 
                      facecolor=col))

def plot_base_heatmap(plt_mtx, ax, cmap="viridis"):
    plt_mtx[plt_mtx > 1] = 1
    plt_mtx[plt_mtx < 0] = 0
    plt_mtx = pd.DataFrame(plt_mtx)
    sns.heatmap(plt_mtx, 
                ax=ax,  
                xticklabels=False, 
                yticklabels=False,
                linewidths=1, 
                linecolor="grey",
                cmap=cmap,
                cbar=False)
    return ax

def add_horitonal_line(ax, vpos=0.5):
    line = mlines.Line2D([0,1], [vpos,vpos], color="black", linewidth=3)
    ax.add_line(line)
    ax.set_ylim(0,1)

def plot_spikes_examples(ax, spike_t, nlist, 
                         n_samps=500, 
                         scat_col="purple",
                         curve_col="#23006F",
                         size=10,
                         alpha=0.5):
    lam = np.linspace(0, 1, n_samps)
    x_lim = (0, 1)
    y_lim = (-0.05,1.56)
    spike_mtx, _ = generate_spike_mtx(lam, 
                                      spike_grp=[spike_t], 
                                      n_repetitions=1,
                                      rel_noise_list=nlist) 
    # noisy curve
    ax.plot(lam, spike_mtx[:, 1], color=curve_col, linewidth=3)
    ax.scatter(lam, spike_mtx[:, 0], s=size, color=scat_col, alpha=alpha)
    # noiseless curve
    ax.set_ylim(*y_lim)
    ax.set_xlim(*x_lim)
    ax.tick_params(labelsize=20)
    ax.locator_params(axis='y', nbins=4)
    ax.locator_params(axis='x', nbins=3)

def add_circle_text(ax, how="top_plot"):
    ypos = 0.5
    size = 18
    circle_col = "#E1D3FF"
    if how == "top_plot":
        add_circle_legend(ax, 0.25, ypos, str(1), circle_col, size=size)
        add_circle_legend(ax, 0.75, ypos, str(2), circle_col, size=size)
        ax.axis('off')
    if how == "plot_1":
        add_circle_legend(ax, 0.09, 1.38, str(1), circle_col, size=size)
    if how == "plot_2":
        add_circle_legend(ax, 0.09, 1.38, str(2), circle_col, size=size)
    
    
def add_symbol_text(ax, text, pos="up", 
                    fontsize=28, 
                    col=None, 
                    fontname='sans-serif'):
    xpos = 0.5
    rotation = 0
    if pos == "left":
        rotation=90
        ypos=0.5
        col = "black"
    elif pos == "low":
        ypos=0.1
        if col is None:
            col = "black"   
    else:
        ypos=0.5
        if col is None:
            col = "royalblue"       
    ax.text(xpos, ypos, text, rotation=rotation, fontsize=fontsize, color=col,
            ha="center", va="center",  fontname=fontname)
    ax.axis('off')

def draw_liver_lobule(ax, single=False):
    if single:
        centers = [[0,0]]
    else:
        centers = [[-1, 0], 
                   [1, 0], 
                   [0, np.sqrt(3)], 
                   [2, np.sqrt(3)],
                   [0, -np.sqrt(3)],
                   [2, -np.sqrt(3)],
                   [-1, -2*np.sqrt(3)], 
                   [1,  -2*np.sqrt(3)],]
    df = pd.DataFrame()
    hex_patches = []
    cir_patches = []
    for center in centers:
        df = pd.concat([generate_unit_hex(center, single=single), df], axis=0)
        polygon = mpatches.RegularPolygon(center, 6, 2/np.sqrt(3))
        hex_patches.append(polygon)
        circle = mpatches.Circle(center, 0.15, ec="none")
        cir_patches.append(circle)
    if single:
        df.plot.scatter(x="x1", y="x2", c="r", cmap="OrRd", s=400, edgecolors="black",
                        linewidth=2,
                         ax=ax, colorbar=None)
    else:
        df.plot.scatter(x="x1", y="x2", c="r", cmap="OrRd", s=40,
                    ax=ax, colorbar=None)
    if single:
        linewidth=8
        linestyle="-"
    else:
        linewidth=2
        linestyle=":"
        collection = PatchCollection(hex_patches, 
                                     cmap=plt.cm.hsv, 
                                     edgecolor="grey",
                                     linestyle=":",
                                     linewidth=linewidth,
                                 facecolor="None")
    
        ax.add_collection(collection)
    collection = PatchCollection(cir_patches, 
                                 cmap=plt.cm.hsv, 
                                 alpha=0.3,
                                 facecolor="black")
    ax.add_collection(collection)
#     ax.axis("off")
    return ax

def generate_unit_hex(center=[0,0], single=False):
    rad = 1
    sfac = 2/np.sqrt(3)
    fnx = [(1, -rad, 0), 
           (-1, -rad, 0),
           (-1*sfac/2, -rad*sfac,  1),
           ( 1*sfac/2, -rad*sfac,  1),
           ( 1*sfac/2, -rad*sfac, -1),
           (-1*sfac/2, -rad*sfac, -1)]
    # coord = np.vstack([np.linspace(0, 1, n_init),
    #                    np.linspace(0, 1, n_init)])
    if single:
        n_pts_per_dim = 15
    else:
        n_pts_per_dim = 30
    dat = np.linspace(-sfac, sfac, n_pts_per_dim)
    coord = np.array(list(itertools.product(dat, dat)))
    n_init = coord.shape[0]
    coord = coord.T
    # reject points outside of the hexagon
    idx = np.repeat(True, n_init)
    for a, b, c in fnx:
        cond = ((a * coord[0,:] + b + c * coord[1,:]) < 0)
        idx = cond & idx
    coord = coord[:, idx]
    r = np.sqrt((sfac-np.sqrt(coord[0,:]**2 + coord[1,:]**2)))
    # add offset
    coord = coord + np.expand_dims(center, axis=1)
    df = pd.DataFrame(coord.T, columns=["x1", "x2"])
    df["r"] = r
    return df

def add_heatmap_grids(gs00, l_mtx, x_mtx, y_mtx, n_mtx,
                      top_row_idx,
                      bot_row_idx,
                      top_label_idx,
                      bot_label_idx,
                      first_mtx_idx,
                      main_font_size):
    ax = plt.subplot(gs00[(top_row_idx+1):bot_row_idx, first_mtx_idx])
    ax = plot_base_heatmap(l_mtx, ax=ax, cmap="OrRd")
    ax = plt.subplot(gs00[(top_row_idx+1):bot_row_idx, first_mtx_idx + 2])
    ax = plot_base_heatmap(x_mtx, ax=ax) 
    ax = plt.subplot(gs00[(top_row_idx+1):bot_row_idx, first_mtx_idx + 3])
    ax = plot_base_heatmap(y_mtx, ax=ax) 
    ax = plt.subplot(gs00[(top_row_idx+1):bot_row_idx, first_mtx_idx + 4])
    ax = plot_base_heatmap(n_mtx, ax=ax) 
    # text annotation on top of heat map
#     ax = plt.subplot(gs00[top_row_idx, first_mtx_idx])
#     add_symbol_text(ax, r'$\xi$', pos="up", fontsize=main_font_size)
#     ax = plt.subplot(gs00[top_row_idx, first_mtx_idx + 3])
#     add_symbol_text(ax, r'$\mathbf{X}^{1}$', pos="up", fontsize=main_font_size)
#     ax = plt.subplot(gs00[top_row_idx, first_mtx_idx + 4])
#     add_symbol_text(ax, r'$\mathbf{X}^{0}$', pos="up", fontsize=main_font_size)
#     ax = plt.subplot(gs00[top_row_idx, first_mtx_idx + 2])
#     add_circle_text(ax, how="top_plot")
    # text annotation at the bottom of heat map
#     ax = plt.subplot(gs00[bot_row_idx, first_mtx_idx + 2])
#     add_horitonal_line(ax, vpos=1)
#     add_symbol_text(ax, r'$\mathbf{X}^{\ast}$', pos="low", fontsize=main_font_size)
#     ax =plt.subplot(gs00[bot_row_idx, (first_mtx_idx + 3):(first_mtx_idx +5)])
#     add_horitonal_line(ax, vpos=1)
#     add_symbol_text(ax, r'$\mathbf{X}$', pos="low", fontsize=main_font_size)
    # more text titles
#     ax = plt.subplot(gs00[top_label_idx, 1:])
#     add_symbol_text(ax, 'unobserved information to be inferred', pos="up")
#     ax = plt.subplot(gs00[bot_label_idx, 1:])
#     add_symbol_text(ax, 'observed features with partial knowledge', pos="low")
#     ax = plt.subplot(gs00[(top_row_idx+1):bot_row_idx, 0])
#     add_symbol_text(ax, 'samples (ordered by latent variable)', pos="left")

def plot_main_example_figure(sim_params, l_mtx, x_mtx, y_mtx, n_mtx, fn):

    nlist =  sim_params["x_param"]['rel_noise_list'] + [0]
    xspikes = sim_params["x_param"]['spike_grp']
    main_font_size = 30

    width_ratios = [l_mtx.shape[1], 
                    l_mtx.shape[1],
                    l_mtx.shape[1]*0.5,
                    x_mtx.shape[1], 
                    y_mtx.shape[1], 
                    n_mtx.shape[1]]
    height_ratios = [0.5, 0.5, 7, 0.5, 7, 1.1, 0.5]

    top_row_idx = 1
    bot_row_idx = -2
    top_label_idx = 0
    bot_label_idx = -1
    first_mtx_idx = 1

    fig = plt.figure(figsize=(25, 11))
    # main grid 1
    gs0 = gridspec.GridSpec(1, 3, width_ratios=[1.2, 5, 1.2], 
                            figure=fig, wspace=0.3, hspace=0)
    
    livplot = gridspec.GridSpecFromSubplotSpec(len(height_ratios), 1, 
                                            subplot_spec=gs0[0],
                                            wspace=0.08, hspace=0.2,
                                            height_ratios=height_ratios)
    ax = plt.subplot(livplot[top_row_idx:(bot_row_idx+1), 0])
    draw_liver_lobule(ax)
    hmaps = gridspec.GridSpecFromSubplotSpec(len(height_ratios), len(width_ratios), 
                                            subplot_spec=gs0[1],
                                            width_ratios=width_ratios,
                                            height_ratios=height_ratios,
                                            wspace=0.08, hspace=0.2)
    add_heatmap_grids(hmaps, l_mtx, x_mtx, y_mtx, n_mtx,
                      top_row_idx,
                      bot_row_idx,
                      top_label_idx,
                      bot_label_idx,
                      first_mtx_idx,
                      main_font_size)
    
    gs01 = gridspec.GridSpecFromSubplotSpec(len(height_ratios), 1, 
                                            subplot_spec=gs0[2],
                                            wspace=0.08, hspace=0.2,
                                            height_ratios=height_ratios)
    # heatmaps
    
    # the predominant variable plots 
    ax = plt.subplot(gs01[top_row_idx+1, 0])
    plot_spikes_examples(ax, xspikes[0], nlist)
    ax.set_ylabel(r'$f_1(\xi_i) + \epsilon_{i, 1}$', fontsize=main_font_size)
    add_circle_text(ax, how="plot_1")
    ax = plt.subplot(gs01[top_row_idx+3, 0])
    plot_spikes_examples(ax, xspikes[1], nlist)
    ax.set_ylabel(r'$f_2(\xi_i) + \epsilon_{i, 2}$', fontsize=main_font_size)
    ax.set_xlabel(r'$\xi_i$', fontsize=main_font_size)
    add_circle_text(ax, how="plot_2")
    # more text information
    ax = plt.subplot(gs01[top_label_idx, :])
    add_symbol_text(ax, 'example templates for', pos="up", col="#23006F")
    ax = plt.subplot(gs01[top_row_idx, :])
    add_symbol_text(ax, 'predominant variables', pos="up", col="#23006F")
    if fn:
        plt.savefig(fn, bbox_inches='tight') 
        logger.info("Saved figure to: {}".format(fn))
    plt.show()
    
def plot_main_strict_order_figure(df_result, params, fn=None):
    
    # Create outer and inner grid
    tot_height = 5
    fig1_width = 7
    fig2_width = 7
    fig12_wspace = 0.05
    sub_row_height = 5
    hspace = 0.3
    shared_params = {
        "main_fontsize": 12,
        "fig_nnull": {
            "wspace": 0.2,
            "hspace": hspace,
            "width_ratios": [0.1, 10],
            "height_ratios": [1, sub_row_height, sub_row_height, 0.5], 
            "size": (fig1_width, tot_height),
        },
        "fig_plot": {
            "wspace": 0.2,
            "hspace": hspace*0.8,
            "height_ratios":  [1, 2*sub_row_height, 0.5], 
            "width_ratios": [0.5, 8],
            "size": (fig2_width, tot_height),
        }
    } 
    tot_width = fig1_width + fig2_width + fig12_wspace
    fig = plt.figure(figsize=(tot_width, tot_height))
    outerGrid = gridspec.GridSpec(1, 2, figure = fig,
                                  width_ratios=[fig1_width, fig2_width],
                                  wspace=fig12_wspace)

    plot_main_sim_strict_nonnull_plots(params, shared_params, grid_id=outerGrid[0])
    plot_main_sim_strict_plot(df_result, params, shared_params, grid_id=outerGrid[1])
    if fn:
        plt.savefig(fn, bbox_inches='tight') 
        logger.info("Saved figure to: {}".format(fn))
    plt.show()
    
def plot_main_sim_strict_nonnull_plots(params, shared_params, grid_id=None, fn=None):
    main_fontsize = shared_params["main_fontsize"]
    n_samps = 1500
    grps = params['y_param']['spike_grp']
    n_rows = 2
    x_lim = (-0.05, 1.05)
    y_lim = (-0.1,1.8)
    n_fig_cols = int(len(grps)/n_rows)
    hspace = shared_params["fig_nnull"]["hspace"]
    wspace = shared_params["fig_nnull"]["wspace"]
    width_ratios = shared_params["fig_nnull"]["height_ratios"] 
    width_ratios = ([width_ratios[0] - (n_fig_cols-1) * wspace] +
                    n_fig_cols * [(width_ratios[1]-width_ratios[0]) / n_fig_cols])
    height_ratios = shared_params["fig_nnull"]["height_ratios"]
    np.random.seed(1)
    lamdas = np.linspace(0, 1, n_samps)
    
    if grid_id is None:
        fig, ax = plt.subplots(figsize=shared_params["fig_nnull"]["size"])
    else:
        grid = gridspec.GridSpecFromSubplotSpec(len(height_ratios), len(width_ratios),
           width_ratios=width_ratios, height_ratios=height_ratios,
           subplot_spec=grid_id, hspace=hspace, wspace=wspace)
        
    col = get_col('my_red')
    row_os = 1
    col_os = 1
    for i, spike_t in enumerate(grps):
        i_row = i % n_rows
        i_col = i // n_rows
        if grid_id:
            ax = plt.subplot(grid[i_row+row_os, i_col+col_os])
        add_circle_legend(ax, 0.11, y_lim[1]*0.88, str(i+1), col)
        nlist = params['y_param']['rel_noise_list'] + [0]
        spike_mtx, _ = generate_spike_mtx(lamdas, 
                                          spike_grp=[spike_t], 
                                          n_repetitions=1,
                                          rel_noise_list=nlist) 
        # noisy curve
        ax.scatter(lamdas, spike_mtx[:, 0], s=1, color="red", alpha=0.3)
        # noiseless curve
        ax.plot(lamdas, spike_mtx[:, 1], color="darkred")
        ax.set_ylim(*y_lim)
        ax.set_xlim(*x_lim)
        if i_row < (n_rows -1):
            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        if i_col > 0:
            ax.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=False) # labels along the bottom edge are off
    if grid_id:
        ax = plt.subplot(grid[-1, col_os:])
        add_symbol_text(ax, 'true latent dimension (shared)', 
                        pos="low", fontsize=main_fontsize)
        ax = plt.subplot(grid[row_os:-1, 0])
        ax.tick_params(axis='y', which='both',left=False, labelleft=False)
        add_symbol_text(ax, 'simulated non-null expression)', 
                        pos="left", fontsize=main_fontsize)
    if fn:
        plt.savefig(fn, bbox_inches='tight') 
        logger.info("Saved figure to: {}".format(fn))
    if grid_id is None:
        plt.show()


def plot_main_sim_strict_plot(df_result, params, shared_params, 
                              grid_id=None, bottom_legend=False,
                              hide_ann=False, fn=None):
    ss_order = [5000, 1500, 1000, 500][::-1]
    meth_order = ["graph", "pca"]
    plt_df = df_result.copy()
    plt_df = plt_df.loc[plt_df["n_samps"].isin(ss_order)]
    n_var_list = params['n_var_list']
    my_red = get_col("my_red")
    x_lim = [min(n_var_list), max(n_var_list)]
    y_lim = [0.84, 0.99]
    nn_grp = len(params['y_param']['spike_grp']) 
    n_reps = params['y_param']['n_repetitions']
    nn_num = (nn_grp * n_reps)
    hue_var = "n_samps"
    style_var = "method"
    width_ratios = shared_params["fig_plot"]["width_ratios"]
    height_ratios = shared_params["fig_plot"]["height_ratios"]
    main_fontsize = shared_params["main_fontsize"]
    hspace = shared_params["fig_plot"]["hspace"]
    wspace = shared_params["fig_plot"]["wspace"]
    if grid_id is None:
        fig = plt.figure(figsize=shared_params["fig_plot"]["size"])
        grid = gridspec.GridSpec(len(height_ratios), len(width_ratios), 
           width_ratios=width_ratios, height_ratios=height_ratios,
           figure=fig, hspace=hspace, wspace=wspace)
    else:
        grid = gridspec.GridSpecFromSubplotSpec(len(height_ratios), len(width_ratios),
           width_ratios=width_ratios, height_ratios=height_ratios,
           subplot_spec=grid_id, hspace=hspace, wspace=wspace)
    row_os = 1
    col_os = 1
    if not hide_ann:
        ax = plt.subplot(grid[-1, col_os])
        add_symbol_text(ax, 'number of new variables included', 
                        pos="low", fontsize=main_fontsize)
        ax = plt.subplot(grid[row_os, 0])
        add_symbol_text(ax, 'latent variable recovery', 
                        pos="left", fontsize=main_fontsize)
        # add circle annotation
        ax = plt.subplot(grid[0, col_os])
        ax.set_xlim(*x_lim)
        ax.set_ylim(0, 1)
        offset = n_reps / 2
        for i_grp in range(nn_grp):
             add_circle_legend(ax, n_reps*i_grp + offset, -0.3, 
                               str(i_grp+1),my_red)
        ax.axis('off')
    # main plot
    ax = plt.subplot(grid[row_os, col_os])
    palette = sns.color_palette("mako_r", len(ss_order))
    sns.lineplot(x="n_cand", 
                 y="corr", 
                 hue=hue_var,
                 style=style_var, 
                 hue_order=ss_order,
                 style_order=meth_order,
                 legend="full",
                 data=plt_df,
                 err_style="bars", 
                 palette=palette,
                 ax=ax)
    # add null non-null blocks
    rect_nonnull = patches.Rectangle(
                    (0, y_lim[0]), nn_num,
                    y_lim[1] - y_lim[0], 
                    facecolor=my_red)
    rect_null = patches.Rectangle(
                    (nn_num, y_lim[0]), x_lim[1] - nn_num, 
                    y_lim[1] - y_lim[0], 
                    facecolor='grey', alpha=0.05)
    # add background annotation
    ax.add_patch(rect_nonnull)
    ax.add_patch(rect_null)
    ax.set_xticks(n_var_list)
    nv_cp = n_var_list.copy()
    for i in range(len(n_var_list)):
        if i % 2 == 1:
            nv_cp[i] = ""
    ax.set_xticklabels(nv_cp)
    ax.tick_params(labelsize=main_fontsize)
    ax.grid(linestyle="dotted")
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    # handle legnd here
    handles, labels = ax.get_legend_handles_labels()
    leg_grps = {hue_var: [[], []], style_var: [[], []]}
    new_labs = []
    curr_lab = ""
    for i, lab in enumerate(labels):
        if lab in leg_grps:
            curr_lab = lab
            continue
        else:
            leg_grps[curr_lab][0].append(handles[i])
            leg_grps[curr_lab][1].append(use_alias(labels[i]))
    if bottom_legend:
        var = "n_samps"
        leg = plt.legend(*leg_grps[var], 
#                       title=use_alias(var), 
                      title_fontsize=main_fontsize,
                      fontsize=main_fontsize,
                      ncol=2,
                      loc='upper center',
                      bbox_to_anchor=[0.65, 0.25],
                      frameon=False)
#         ax.add_artist(leg)
    else:
        curr_leg = 0
        for var in leg_grps:
            bbox_to_anchor=[1.03, 0.95]
            if var == hue_var:
                bbox_to_anchor[1] = 0.9
            if var == style_var:
                bbox_to_anchor[1] = 0.4
            leg = plt.legend(*leg_grps[var], 
                      title=use_alias(var), 
                      title_fontsize=main_fontsize,
                      bbox_to_anchor=bbox_to_anchor,
                      frameon=False)
            leg._legend_box.align = "left"
            if curr_leg < (len(leg_grps.keys())-1):
                ax.add_artist(leg)
                curr_leg += 1
    if fn:
        plt.savefig(fn, bbox_inches='tight') 
        logger.info("Saved figure to: {}".format(fn))
    if grid_id is None:
        plt.show()  

def update_boxplot_cols(ax, alpha = 0.7):
    l = ax.legend(loc="upper left", frameon=False)
    l.set_title(None)
    # https://stackoverflow.com/questions/36874697/how-to-edit-properties-of-whiskers-fliers-caps-etc-in-seaborn-boxplot
    for i,artist in enumerate(ax.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = artist.get_facecolor()
        artist.set_edgecolor(col)
        artist.set_alpha(alpha)
        for j in range(i*6,i*6+6):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)
        # Also fix the legend
    for legpatch in ax.get_legend().get_patches():
        col = legpatch.get_facecolor()
        legpatch.set_edgecolor(col)
        legpatch.set_alpha(alpha)
#         legpatch.set_facecolor('None')
        
    
def plot_sub_regime(df, ax, met, noise_lev, pal, alpha, plain="False", q_lim=[]):
    sns.boxplot(x="noise_lev", y=met, hue="Method", 
                    order=noise_lev, linewidth=1,
                    data=df, ax=ax, palette=pal,
                    fliersize=1) 
    if met == "FDP" and alpha:
            ax.axhline(y=alpha, color='black', linestyle='--', linewidth=1)
    if met == "FDP":
        if plain:
#                 ax.set_ylim(-0.03,0.30) 
            if np.mean(df[met]) < alpha:
                ax.set_ylim(-0.03,0.25) 
            else:
                ax.set_ylim(-0.05,1.03)
#                 ax.set_ylim(-0.05,1.03)
            ax.set_ylabel(None)
            ax.set_xlabel(None)
        else:
            ax.set_ylabel("feature selection FDR")
    elif met == "Corr":
        if plain:
#                 ax.set_ylim(0.86, 1.001) 
#                 ax.set_ylim(-0.05,1.03)
            if q_lim:
                ax.set_ylim(*q_lim)
            ax.set_ylabel(None)
            ax.set_xlabel(None)
        else:
            ax.set_ylabel("latent variable recovery")
    elif met == "Power":
        if plain:
            ax.set_ylim(-0.05,1.03)
            ax.set_ylabel(None)
            ax.set_xlabel(None)
        else:
            ax.set_ylabel("feature selection power")
    else:
        assert 0, "Metric: {} not found".format(met)
    
    update_boxplot_cols(ax)
    ax.yaxis.grid(True) # Hide the horizontal gridlines 
    ax.xaxis.grid(True) # Show the vertical gridlines
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if plain:
        ax.legend().remove()
        ax.set_xlabel(None)
    else:
        ax.set_xlabel("noise level")
    if plain:
        plt.subplots_adjust(wspace=0.3)
    else:
        plt.subplots_adjust(wspace=0.45)
        
def plot_regime_method_comparison(df, 
                                  params, 
                                  df_grps=False,
                                  fn=None, 
                                  plain=False, 
                                  q_lim=[],
                                  pal="Set1"):
    
    metrics = params["plot_metrics"]
    noise_lev = params["plot_noise_lev"]
    alpha = params['alpha']
    sns.set_style("ticks")
    fig, axes = plt.subplots(1, len(metrics), 
                             figsize=[3.3*len(metrics), 2.7])
    for i_ax, met in enumerate(metrics):
        ax = axes[i_ax]
        plot_sub_regime(df, ax, met, noise_lev, pal, alpha, plain=plain, q_lim=q_lim)
    if fn:
        plt.savefig(fn, bbox_inches='tight', transparent=True) 
        logger.info("Saved figure to: {}".format(fn))
    plt.show()
    
def plot_feature_histogram(in_data, 
                           ax=None,
                           var_names=None, 
                           var_cols=None,
                           loc='upper right',
                           xy_lim=[-6, 6]):
    n, p = in_data.shape
    bins_scaled = np.linspace(xy_lim[0], xy_lim[1], 100)
    
    for i in range(p):
        if var_names:
            label = var_names[i]
        else:
            label = None
        if var_names and var_cols:
            color = None
        else:
            color = var_cols[var_names[i]]
        sns.distplot(in_data[:,i], 
                     bins_scaled, 
                     label=label, 
                     color=color,
                     ax=ax)
    if var_names:
        ax.legend(loc=loc)
    return ax
    
def clear_sns_dendogram(grid, cax_visible=False):
    # controls the heatmap position after hiding the colorbar legend
    grid.cax.set_visible(cax_visible)
    col = grid.ax_col_dendrogram.get_position()
    row = grid.ax_row_dendrogram.get_position()
    grid.ax_col_dendrogram.set_position([col.x0, col.y0, col.width, col.height*0.1])
    grid.ax_row_dendrogram.set_position([row.x0*4, row.y0, row.width, row.height])
    
def plot_ground_truth_heatmap(lam, plt_mtx, n_grps=9, fn=None):
    # plot ground truth heatmap
    # sort by lambdas and as well as the spike matrices
    sort_idx = np.argsort(lam)
    plt_mtx = pd.DataFrame(plt_mtx[sort_idx,])
    # add some clipping for values to range from 0-1
    plt_mtx[plt_mtx > 1] = 1
    plt_mtx[plt_mtx < 0] = 0
    grp_ids = np.repeat(n_grps-1, plt_mtx.shape[0])
    npg = int(np.floor(plt_mtx.shape[0]/n_grps))
    grp_ids[:npg*n_grps] = np.repeat(np.arange(n_grps), npg)
    lut = create_color_map(grp_ids, "Spectral")
    grp_cols = pd.Series(grp_ids).map(lut)
    pad_white = pd.Series(np.repeat("#ffffff", len(grp_ids)))
    grid = sns.clustermap(plt_mtx, xticklabels=False, yticklabels = False,
                    row_cluster=False, col_cluster=False, 
                    row_colors=[pad_white]*3 + [grp_cols],
                    figsize=(13, 5))
    
    clear_sns_dendogram(grid)
    if fn:
        plt.savefig(fn, bbox_inches='tight', dpi=100) 
        logger.info("Saved figure to: {}".format(fn))
    plt.show()
    
def plot_ground_truth_graph(lam, graph, labels=None, fn=None):
#     if color_val is None:
#         color_val = np.ones(graph.shape[0])
    sort_idx = np.argsort(lam)
    graph = graph.toarray()
    graph = graph[:, sort_idx][sort_idx,:]
    
    L = get_laplacian(graph)
    eigvals, eigvecs = eigh(L) # sorts the eigs in ascending order
   
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(graph, xticklabels=False, yticklabels=False, ax=axes[0])
    
    
    ax = axes[1]
    if labels is None:
        n_grps = 10
        labels = np.repeat(np.arange(n_grps), graph.shape[0]/n_grps)
        colmap = "Spectral"
        labs, c = np.unique(labels, return_inverse=True)
        cmap = plt.cm.get_cmap(colmap)
        bounds = np.linspace(0,len(labs),len(labs)+1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        color_map = create_color_map(labels, colmap)
        cols = [color_map[lab] for lab in labels]
        # scat = ax.scatter(x, y, c=c, s=ms, cmap=cmap, norm=norm)
        scat = ax.scatter(*eigvecs[:, 1:3].T, s=3, c=c, cmap=cmap, norm=norm)
#         pt = ax[1].scatter(, c=color_val, s=2)
#     plt.colorbar(pt, ax=ax[1])
        cb = plt.colorbar(scat, spacing='proportional',ticks=bounds+0.5, ax=ax)
        cb.set_ticklabels(labs)
    ax.set_xlabel("eigen vector 1")
    ax.set_ylabel("eigen vector 2")
                
    if fn:
        plt.savefig(fn, bbox_inches='tight') 
        logger.info("Saved figure to: {}".format(fn))
    plt.show()

def plot_spike_grps(grp, nlist, n_samps=3000, fn=None):
    np.random.seed(1)
    lamdas = np.random.uniform(size=n_samps)
    colors = cm.Blues(np.linspace(0.1, 1, len(nlist)))[::-1]
    fig, axes = plt.subplots(1,len(grp), figsize=(4.5*len(grp), 2))
    for i_spike, spike_t in enumerate(grp):
        spike_mtx, _ = generate_spike_mtx(lamdas, 
                                          spike_grp=[spike_t], 
                                          n_repetitions=1,
                                          rel_noise_list=nlist)

        ax = axes[i_spike]
        for i, noise in enumerate(nlist):
            ax.scatter(lamdas, spike_mtx[:, i], 
                       s=1, color=colors[i],
                       label=noise)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xlabel("$\lambda$")
        ax.set_ylabel("Expression")
        ax.set_title("Spike Type: {}".format(spike_t))
    if fn:
        plt.savefig(fn, bbox_inches='tight') 
        logger.info("Saved figure to: {}".format(fn))
    plt.show()
        
def label_graph_heatmap(mat_v, t_vals, lut=None):
    # plotting
    labels = np.unique(t_vals)
    if lut is None:
        lut = dict(zip(labels, sns.color_palette("tab10", len(labels))))
    else:
        # double check if the lut dict matches or not
        for label in labels:
            assert label in lut, "label color is not definited in color map dict input"
    group_cols = pd.Series(t_vals).map(lut)
    padding_white = pd.Series(np.repeat("#ffffff", len(t_vals)))
    grid = sns.clustermap(pd.DataFrame(mat_v), xticklabels=False, yticklabels = False,
                        row_cluster=False, col_cluster=False, 
                        row_colors=[padding_white,padding_white, group_cols],
                        figsize=(10, 8))
    clear_sns_dendogram(grid)

def plot_simple_graph(A, 
                      coord, 
                      cols=None, 
                      node_size=20, 
                      linewidths=1.0,
                      edge_alpha=0.3,
                      ax=None):
    """
    Plots a network using NetworkX for simple examples
    Args:
         A: an (n, n) np.array or scipy sparse matrix (positive symmetric)
         coord: an (n, 2) np.array for each point coordinate
    """
    assert A.shape[0] == coord.shape[0], "mismatch num_nodes"
    n_nodes = A.shape[0]
    node_ids = np.arange(n_nodes)
    node_pos = dict(zip(np.arange(n_nodes), coord.tolist()))
    if scipy.sparse.issparse(A):
        G = nx.from_scipy_sparse_matrix(A)
    else:
        G = nx.from_numpy_matrix(A)

    if ax is None:
        plt.figure(figsize=(6, 6))
        edges = nx.draw_networkx_edges(G, node_pos, alpha=edge_alpha)
        nodes = nx.draw_networkx_nodes(G, node_pos, node_size=node_size, 
                                       linewidths=linewidths, node_color=cols)
    else:
        edges = nx.draw_networkx_edges(G, node_pos, alpha=edge_alpha, ax=ax)
        nodes = nx.draw_networkx_nodes(G, node_pos, node_size=node_size, ax=ax,
                                       linewidths=linewidths, node_color=cols)
    nodes.set_edgecolor('k')
#     plt.axis('off')
#     plt.show()
    return nodes

def check_nonull_dimensions(mat_truth, 
                            graph_func, 
                            noise_vals = [0, 1e-2, 1e-1, 1],
                            color_val=None):
    if color_val is None:
        color_val = np.ones(mat_truth.shape[0])
    fig, ax = plt.subplots(3,len(noise_vals), 
                           figsize=[5*len(noise_vals),12])

    cols = ["$\sigma^2$ = {}".format(val) for val in noise_vals ]
    rows = ["non-null dimensions", "", "2nd vs. 3rd eigenvector"]
    for i, noise in enumerate(noise_vals):
        # construct the graph based on the noisy observation in only 2-dimensions
        nmat_s = add_gaussian_noise(mat_truth, noise)
        dmat_s = graph_func(nmat_s)
        L = get_laplacian(dmat_s.todense())
        lap_score = laplacian_score(nmat_s, dmat_s)
        eigvals, eigvecs = eigh(L) # sorts the eigs in ascending order
        
        # plot the 2-dimensional ground truth
        pt = ax[0, i].scatter(*nmat_s.T, c=color_val, s=2)
        plt.colorbar(pt, ax=ax[0, i])
        ax[0, i].set_title(cols[i])
        # plot the adjacency matrix
        sns.heatmap(dmat_s.toarray(), xticklabels=False, yticklabels=False, ax=ax[1,i])
        ax[1, i].set_title("lap_score: {}".format(lap_score))
        
        # plot the first two eigen values
        pt = ax[2, i].scatter(*eigvecs[:, 1:3].T, c=color_val, s=2)
        plt.colorbar(pt, ax=ax[2, i])
        ax[2, i].set_title("eig_vals: {}".format(eigvals[1:3]))

    for axe, row in zip(ax[:,0], rows):
        axe.set_ylabel(row)

    plt.show()


def check_noise_distribution(mat_truth, 
                            graph_func, 
                            n_null = 2000,
                            noise="uniform",
                            noise_vals = [0, 1e-2, 1e-1, 1]):

    fig, ax = plt.subplots(1,1, figsize=(5,4))
    n_samp, n_nonnull = mat_truth.shape
    np.random.seed(0)
    if noise == "uniform": 
        null_mat = add_uniform_noise(np.zeros((n_samp, n_null)), 1)
    else:
        null_mat = add_gaussian_noise(np.zeros((n_samp, n_null)), 1)
    for i, noise in enumerate(noise_vals):
        # construct the graph based on the noisy observation in only 2-dimensions
        # null_mat = add_gaussian_noise(np.zeros((n_samp, n_null)), noise)
        nmat_s = add_gaussian_noise(mat_truth, noise)
        dmat_s = graph_func(nmat_s)
        l = get_laplacian(dmat_s.todense())
        scores = laplacian_score(null_mat, dmat_s)
        sns.distplot(scores, ax=ax, label="$\sigma^2$ = {}".format(noise))
        # sns.plt.legend()

    ax.legend()
#     ax.set_xlim([0.8, 1.0])
    plt.show()

    return ax

def check_basic_null_case(n_samp,
                          graph_func, 
                          null_dims=[10, 20, 150, 500]): 
    
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    np.random.seed(0)
    for i, n_null in enumerate(null_dims):
        # construct the graph based on the noisy observation in only 2-dimensions
        # null_mat = add_gaussian_noise(np.zeros((n_samp, n_null)), noise)
        null_mat = add_gaussian_noise(np.zeros((n_samp, n_null)), 1)
        dmat_s = graph_func(null_mat)
        l = get_laplacian(dmat_s.todense())
        scores = laplacian_score(null_mat, dmat_s)
        sns.distplot(scores, ax=ax, label="dim = {}".format(n_null))
        # sns.plt.legend()

    ax.legend()
    # ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.8, 1.0])
    plt.show()

    return ax


# Plotting initially developed for liver data analysis

def plot_scatter_continuous(projected, values, ax, logscale=True, ms=5, sortvals=True):
    projected = np.array(projected)
    df = pd.DataFrame(projected, columns=["dim1", "dim2"])
    df["val"] = np.array(values)
    # print(df.head())
    # optional log scaling
    if logscale:
        df["val"] = np.log(1 + df["val"])
    # optional data point sorting
    if sortvals:
        df = df.sort_values(by="val")
    # matplotlib plotting
#     if np.sum(df["val"]) == 0:
#         vmin = 0
#         vmax = 1
#     else:
    vmin = np.min(df["val"])
    vmax = np.max(df["val"])
    scat = ax.scatter(df["dim1"], df["dim2"], c=df["val"], 
                # edgecolor="#383838", 
                vmin=vmin, vmax=vmax,
                alpha=1.0, s=ms, cmap="inferno")
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.colorbar(scat, ax=ax, format=FormatStrFormatter('%.1f'))
    return scat 

def plot_scatter_discrete(projected, labels, ax, ms=5, cols="Spectral"):
    projected = np.array(projected)
    x = projected[:, 0]
    y = projected[:, 1]
    labs, c = np.unique(labels, return_inverse=True)
    # create the new map
    if isinstance(cols, str):
        cmap = plt.cm.get_cmap(cols)
    elif isinstance(cols, dict):
        cols = [cols[lab] for lab in labs]
        cmap = mpl.colors.ListedColormap(cols)
    else:
        assert len(cols) == len(labs), "number of label mismatch"
        cmap = mpl.colors.ListedColormap(cols)
#     cmaplist = [cmap(i) for i in range(cmap.N)]
#     cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0,len(labs),len(labs)+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    scat = ax.scatter(x, y, c=c, s=ms, cmap=cmap, norm=norm)
    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds+0.5, ax=ax)
    cb.set_ticklabels(labs)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return scat


# TODO: remove plot multiple features in the future
def plot_multiple_features(proj, df, numerical=True, num_cols=5):
    feat_list = df.columns.tolist()
    num_feats = len(feat_list)
    num_rows = int(np.ceil(num_feats/num_cols))
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(4.5*num_cols, 3.9*num_rows-1))
    for i, feat in enumerate(feat_list):
        i_row = i // num_cols
        j_col = i % num_cols
        vals = df[feat]
        if (num_rows == 1):
            ax_i_j =  ax[j_col]  
        else:
            ax_i_j =  ax[i_row][j_col]
        if numerical:
            plot_scatter_continuous(proj, vals, ax_i_j, logscale=False)
        else:
            plot_scatter_discrete(proj, vals, ax_i_j, cols=['#0033cc', '#ff0000', 'grey'])
        ax_i_j.set_title(feat)
    for i in range(num_rows):
        for j in range(num_cols):
            fig_idx = i * (num_cols) + j
            if fig_idx >= len(feat_list):
                if (num_rows == 1):
                    ax_i_j =  ax[j]  
                else:
                    ax_i_j =  ax[i][j]
                ax_i_j.remove()
 

def adjust_xy_labels(ax, xy_labels=("t-SNE 1", "t-SNE 2")):
    ax.set_xlabel(xy_labels[0])
    ax.set_ylabel(xy_labels[1])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def plot_multi_scatter_discrete(input_df, proj, ms=1, logscale=True):
    genes = list(input_df.columns)
    n_cols = 8
    n_cols = min(len(genes),n_cols)
    n_rows = int(np.ceil(len(genes) / n_cols))
    # logger.info("Plotting genes: {}".format(genes))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3*n_cols, 0.3+1.9*n_rows))
    for i_row in range(n_rows):
        for i_col in range(n_cols):
            if n_rows == 1:
                if n_cols == 1:
                    ax = axes
                else:
                    ax = axes[i_col]
            else:
                ax = axes[i_row, i_col]
            i_gene = n_cols * i_row + i_col
            if i_gene >= len(genes):
                ax.axis('off')
            else:
                gene = genes[i_gene]
                vals = input_df[gene]
                plot_scatter_continuous(proj, vals, ax, ms=ms, logscale=logscale)
                ax.set_title(gene)
                adjust_xy_labels(ax)
    plt.tight_layout()
    plt.show()
    

# plot for principle curves
# plot the curve and the data projection on the curve
def plot_pcurve_2d(x, lams, curve_funcs, ax=None):
    # x: data points (N x 2)
    # lams: parameter of points (N, )
    # curve_funcs: univariate function curve in each dimension
    
    assert x.shape[1]==2, "Input data must be in 2-D"
    # parameters
    cm_name = "rainbow"
    # plane_col = 'grey'
    
    bounds = extended_bounds(lams, f=0)
    plt_lams = np.linspace(*bounds, 500)
    plt_curv = curve_funcs(plt_lams)
    z = curve_funcs(lams)
    assert z.shape[1]==2, "Projected data must be in 2-D"
    # create the dashed distance lines
    norm = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[1], clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(cm_name))
    line_cols = [mapper.to_rgba(v) for v in lams] 
    lc = mc.LineCollection(list(zip(x,z)), 
                           colors=line_cols, 
                           linewidths=1, 
                           linestyle="--")
    # for each segment, do a linear interpolation of color changes
    abs_lim = np.max(np.abs(x)) * 1.2
    # fig, ax = plt.subplots(1,1,figsize=[5.4,4.5])
    ax.scatter(plt_curv[:,0], plt_curv[:, 1], c=plt_lams, s=2, cmap=cm_name)
    colscat = ax.scatter(z[:,0], z[:,1], c=lams, s=10, cmap=cm_name)
    # plot distances to curve
    ax.add_collection(lc)
    # plot the data
    ax.scatter(x[:,0], x[:,1], facecolors='none', edgecolors='k', s=16)
    # scaling and aesthetics
    ax.set_xlim([-abs_lim, abs_lim])
    ax.set_ylim([-abs_lim, abs_lim])
    plt.colorbar(colscat, ax=ax)
    return ax

    
def create_color_map(labels, colmap, start=0, end=1):
    labs, c = np.unique(labels, return_inverse=True)
    cmap = plt.cm.get_cmap(colmap)
    scale_vals = np.linspace(start, end, len(labs))
    rgb_values = [cmap(x) for x in scale_vals]# sns.color_palette(colmap, len(labs))
    color_map = dict(zip(labs, rgb_values))
    return color_map

def plot_pcurve_fits(x, 
                     lambs, 
                     funcs, 
                     all_iterations=True, 
                     n_cases=4, 
                     var_names=None,
                     labels=None,
                     save_dir=None,
                     suffix=""):
    if not all_iterations:
        lambs = np.expand_dims(lambs, axis=1)
        funcs = [funcs]
    else:
        assert len(funcs) == lambs.shape[1], "Number of iterations mismatch"
    n_cases = min(n_cases, len(funcs))
    n_iterations = len(funcs)
    iter_vals = np.round(np.linspace(0, n_iterations-1, n_cases)).astype(int)
    iter_vals[0] = 0
    iter_vals[-1] = n_iterations-1
    if len(iter_vals) == 1:
        n_cols = x.shape[1]
        n_rows = 1
        if labels:
            col_w = 3
        else:
            col_w = 2.5
    else:
        n_cols = len(iter_vals)
        n_rows = x.shape[1]
        if labels:
            col_w = 4
        else:
            col_w = 2.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*col_w, n_rows*2.5), 
                             sharex="col", sharey="row")
    for i_col, i_iter in enumerate(iter_vals):
        lams = lambs[:, i_iter]
        curve_funcs = funcs[i_iter]
        bounds = extended_bounds(lams, f=0)
        plt_lams = np.linspace(*bounds, 500)
        plt_curv = curve_funcs(plt_lams)
        z = curve_funcs(lams)
        for i_row in range(x.shape[1]):
            if len(iter_vals) == 1:
                ax = axes[i_row]
            else:
                ax = axes[i_row, i_col]
                
            ax.plot(plt_lams, plt_curv[:, i_row], c="k")
            if labels:
                colmap = "Spectral"
                labs, c = np.unique(labels, return_inverse=True)
                cmap = plt.cm.get_cmap(colmap)
                bounds = np.linspace(0,len(labs),len(labs)+1)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                color_map = create_color_map(labels, colmap)
                cols = [color_map[lab] for lab in labels]
                # scat = ax.scatter(x, y, c=c, s=ms, cmap=cmap, norm=norm)
                scat = ax.scatter(lams, x[:,i_row], s=3, c=c, cmap=cmap, norm=norm)
                cb = plt.colorbar(scat, spacing='proportional',ticks=bounds+0.5, ax=ax)
                cb.set_ticklabels(labs)
            else:
                cols = "darkgrey"
                ax.scatter(lams, x[:,i_row], facecolors='none', edgecolors=cols, s=16)
            # ax.scatter(lams, z[:,i_row], s=10)
            sns.rugplot(lams, ax=ax, c=cols)
            if len(iter_vals) == 1:
                if var_names is not None:
                    ax.set_ylabel(var_names[i_row])
                ax.set_xlabel("$\lambda$")
            else:
                if i_row == 0:
                    ss = np.sum( (x - z)**2)
                    ax.set_title("Iter {} (ss={:.3f})".format(i_iter, ss))

                if i_row == n_rows - 1:
                    ax.set_xlabel("$\lambda$")
                if i_col == 0:
                    if var_names is None:
                        ax.set_ylabel("V {}".format(i_row))
                    else:
                        assert len(var_names) == n_rows, "var name length mismatch"
                        ax.set_ylabel(var_names[i_row])
    plt.tight_layout()
    if save_dir:
        assert os.path.exists(save_dir), "{} does not exist".format(save_dir)
        fn = "scat_fit_{}.png".format(suffix)
        fn = os.path.join(save_dir, fn)
        plt.savefig(fn, bbox_inches='tight')
        print("Saved: {}".format(fn))
    
def plot_nstruct_result(df, params, col_map=None, fn=None):
    rel_noise_list = params["noise_levs"]
    n_var_list = params["n_struct_vars"]
    if col_map is None:
        col_map = create_color_map(n_var_list, "GnBu", start=0.5)
    df['nvar'] = df['nvar'].astype(int)
    df['nlev'] = df['nlev'].astype('category')
    fig, ax = plt.subplots(figsize=(5.5,4))
    sns.barplot(x="nlev", y="corr", hue="nvar", palette=col_map,
                order=rel_noise_list, data=df, ax =ax)
    leg = ax.legend(loc='upper center', ncol=len(n_var_list), bbox_to_anchor=(0.5, 1.2))
    leg.set_title("number of structural variables per function class")
    ax.set_xlabel("noise level at each non-null variable")
    ax.set_ylabel("Spearman's correlation between $\lambda$ and $\hat{\lambda}$")
    ax.set_ylim(0,1)
    if fn:
        plt.savefig(fn, bbox_inches='tight') 
        logger.info("Saved figure to: {}".format(fn))
    plt.show()
    
def plot_nnvars_result(df, params, col_map=None, fn=None):
    rel_noise_list = params["noise_levs"]
    n_var_list = params["n_struct_vars"]
    if col_map is None:
        col_map = create_color_map(n_var_list, "GnBu", start=0.5)
    df['nvar'] = df['nvar'].astype(int)
    df['nnum'] = df['nnum'].astype(int)
    fig, ax = plt.subplots(figsize=(5.5,4))
    grouped = df.groupby("nvar")
    for key,group in grouped:
        group.plot(x="nnum", y="corr", label=key, color=col_map[key],
                   marker="o", linestyle='dashed', ax=ax)
    leg = ax.legend(loc='upper center', ncol=len(n_var_list), bbox_to_anchor=(0.5, 1.2))
    leg.set_title("number of structural variables per function class")
    ax.set_xlabel("number of null variables")
    ax.set_ylabel("Spearman's correlation between $\lambda$ and $\hat{\lambda}$")
    ax.set_xlim(0,max(df["nnum"]))
    ax.set_ylim(0,1)
    if fn:
        plt.savefig(fn, bbox_inches='tight') 
        logger.info("Saved figure to: {}".format(fn))
    plt.show()
    
    
def plot_and_compare_results(res_dict, plot_pvals=False):
    pval_df, pval_thres, cmp_sets, mets = setup_cmp_df(res_dict)
    fig, curr_ax = plt.subplots(1,1, figsize=(4,4))
    venn2(cmp_sets,
          set_labels=mets,
          set_colors=['red', 'dodgerblue'],
          ax=curr_ax)
    plt.show()
    if plot_pvals:
        g = sns.jointplot(mets[0], mets[1], data=pval_df,
                   height=4, 
                   color="k",
                   joint_kws=dict(alpha=0.5, s=1),
                   marginal_kws=dict(bins=15, rug=False),
                   annot_kws=dict(stat="r"))
        cols = ["r", "b"] # vertical and horiztonal in order
        line_opts = dict(alpha=0.5, linestyle="dashed")
        g.ax_joint.axvline(pval_thres[0], color=cols[0], **line_opts) 
        g.ax_joint.axhline(pval_thres[1], color=cols[1], **line_opts) 
        plt.show()
        fig, axes = plt.subplots(1, 2, figsize=(6.5,3))
        # q-q plot for each set of p-vals
        for i in range(2):
            ax = axes[i]
            met = mets[i]
            arg_idx = np.argsort(pval_df[met])
            order = np.zeros(len(arg_idx))
            for i, rank in enumerate(arg_idx):
                order[rank] = i
            exp_pvals = -np.log10((1+order)/len(order))
            obs_pvals = -np.log10(pval_df[met])
            m_val = max(exp_pvals)
            x_space = np.linspace(0, m_val, 100)
            ax.scatter(exp_pvals, obs_pvals, alpha=0.5, s=1, c="k")
            ax.plot(x_space, x_space, "--r")
            ax.set_xlabel("Observed ($-\log_{10}$ p-value)")
            ax.set_ylabel("Expected ($-\log_{10}$ p-value)")
            ax.set_title(met)
        plt.show()
        
def plot_corr_mtx(df, plain=False, fn=None, vmin=0, vmax=1):
    corr_df = df.corr('spearman')
    if plain:
        fig, ax = plt.subplots(1, 1, figsize=(3.2,3.2))
        show_cbar = False
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        show_cbar = True
    sns.heatmap(corr_df, annot=True, fmt=".2f", square=True, ax=ax,
                cbar=show_cbar, cbar_kws={"shrink": .8}, 
                vmin=vmin, vmax=vmax)
    if plain:
        ax.set_xticks([])
        ax.set_yticks([])
    if fn:
        plt.savefig(fn, bbox_inches='tight') 
        logger.info("Saved figure to: {}".format(fn))
    plt.show()

def plot_full_pattern(ax, lam_in, 
                      gene_expr, 
                      scatter=True, 
                      fit=True, 
                      n_bins=9,
                      order_only=True,
                      add_color_bar=False,
                      labels=None):
    if order_only: # only consider the rank ordering
        lam_in = rank_and_bin(lam_in, n_bins=len(lam_in))
    grp = rank_and_bin(lam_in, n_bins=n_bins, linspace=True)  
    df = pd.DataFrame({"lam": lam_in, "gene_expr": gene_expr, "grp": grp})
    if scatter:
        if labels:
            colmap = "Spectral"
            labs, c = np.unique(labels, return_inverse=True)
            cmap = plt.cm.get_cmap(colmap)
            bounds = np.linspace(0,len(labs),len(labs)+1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            color_map = create_color_map(labels, colmap)
            cols = [color_map[lab] for lab in labels]
            scat = ax.scatter(lam_in, gene_expr, s=3, c=c, cmap=cmap, norm=norm)
        else:
            cols = "darkgrey"
            scat = ax.scatter(lam_in, gene_expr, s=1)
        sns.rugplot(lam_in, ax=ax, c=cols)
        if add_color_bar:   
            plt.colorbar(scat, ax=ax)
    if fit:
        sns.lineplot(x="grp", y="gene_expr", data=df, ax=ax, ci="sd",
                     color="black")
    add_margin(ax, "y", 0.01, 0)
    add_margin(ax, "x", 0.01, 0.01)    
    
def plot_gene_expr_comp(lam_df, gene_df, methods, aliases, horizontal=False, fontsize=16, fn=None):
    genes = list(gene_df.columns)
    labels = list(lam_df["smFISH"])
    
    if len(genes) == 1 and len(methods) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    else:
        if horizontal:
            fig, axes = plt.subplots(len(methods), len(genes), 
                                 figsize=(2.3*len(genes), 2*len(methods)))
        else:
            fig, axes = plt.subplots(len(genes), len(methods), 
                                 figsize=(1.65*len(methods), 1.5*len(genes)),
                                 sharex="col", sharey="row")
    
    for i, gene in enumerate(genes):
        for j, method in enumerate(methods):
            if len(genes) == 1 and len(methods) == 1:
                ax = axes
                add_cbar = True
            else:
                add_cbar = False
                if len(genes) == 1:
                    ax = axes[j]
                elif len(methods) == 1:
                    ax = axes[i]
                else:        
                    if horizontal:
                        ax = axes[j, i]
                    else:
                        ax = axes[i, j]
            order_only = (method != "smFISH")
            plot_full_pattern(ax, lam_df[method], gene_df[gene], 
                              order_only=order_only,
                              add_color_bar=add_cbar, labels=labels)    
            if horizontal:
                ax.set_xlabel(None)
                fig.subplots_adjust(hspace=0.5, wspace=0.42)
                ax.set_ylabel(gene, labelpad=0, fontsize=fontsize)
                if len(methods) > 1:
                    ax.set_title(aliases[method])    
            else:
                if j == 0:
                    ax.set_ylabel(gene, fontsize=fontsize)
                if i == (len(genes) -1):
                    ax.set_xlabel(aliases[method], fontsize=fontsize)
                fig.subplots_adjust(hspace=0, wspace=0)
            
            ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.tick_params(axis='both', which='major', pad=2, labelsize=fontsize-1)
            if method == "smFISH":
                xticks = [1,3, 5, 7, 9]
            else:
                xticks = [0, 1000]
            if gene == "Malat1":
                ax.set_ylim(-0.5, 3)
                ax.set_yticks(ticks=[0, 1, 2])  
            ax.set_xticks(ticks=xticks)    
            ax.get_legend().remove()
    if fn:
        plt.savefig(fn, bbox_inches='tight', transparent=True, dpi=150) 
        logger.info("Saved figure to: {}".format(fn))
    plt.show()
    
   
def plot_col_color_heatmap(plt_mtx, grp_ids, lut, val_min=None, val_max=None,
                           short=False, square=False):
    feat_cols = pd.Series(grp_ids).map(lut)
    if val_min:
        plt_mtx[plt_mtx < val_min] = val_min
        plt_mtx[0,0] = val_min
    if val_max:
        plt_mtx[plt_mtx > val_max] = val_max
        plt_mtx[-1,-1] = val_max
    pad_white = pd.Series(np.repeat("#ffffff", len(grp_ids)))
    if square:
        figsize=(7, 7)
        shared_cols = [feat_cols] * 10 +  [pad_white] * 1
        col_colors = shared_cols
        row_colors = shared_cols
    else:
        if short:
            figsize=(7, 1)
            col_colors = None
            row_colors = [pad_white]
        else:
            figsize=(7, 3)
            col_colors = [feat_cols] * 10 +  [pad_white] * 1
            row_colors = [pad_white]
   
    grid = sns.clustermap(plt_mtx, xticklabels=False, yticklabels = False,
                          row_cluster=False, col_cluster=False, 
                          col_colors=col_colors, row_colors = row_colors,
                          figsize=figsize)
    clear_sns_dendogram(grid,  cax_visible=True)
    

def get_sim_color_map(grp_ids, palette="tab10"):    
#     lut = create_color_map(grp_ids[grp_ids>=0], palette)
    pos_int_ids = np.unique(grp_ids[grp_ids>=0])
    cols = sns.color_palette(palette, len(pos_int_ids))
    lut = {}
    for i, val in enumerate(pos_int_ids):
        lut[val] = cols[i]
    lut[-1] = (0.5, 0.5, 0.5)
    return lut
   
def plot_by_noise_struct(var_df, lam_true, x, order_by_noise=True, num_grps = 5):
    if order_by_noise:
        df = var_df.sort_values('corr_grp')   
    else:
        df = var_df
    df = df.loc[df['corr_grp'] < num_grps]    
    grp_ids = df['nn_grp']
    lut = get_sim_color_map(grp_ids)
    # plot variables correlated with lm via noise
    new_nn_idx = var_df.loc[var_df['lm_corr']].sort_values('corr_grp')['var_id']
#     plot_ground_truth_heatmap(lam_true, x[:, new_nn_idx])
    # plot the top 10 correlation groups
    corr_cols = x[:, df['var_id']]
#     plot_ground_truth_heatmap(lam_true, corr_cols)
    dist_feat = pairwise_distances(corr_cols.T, metric="euclidean")
    plot_col_color_heatmap(dist_feat, grp_ids, lut, square=True)

    
    
def plot_venn(gset, keys, fn=None, ax=None):
    vals = [gset[k] for k in keys]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4,4))
    if len(keys) == 2:
        vd = venn2(vals, keys, ax=ax)
    if len(keys) == 3:
        vd = venn3(vals, keys, ax=ax)
    if ax is None:
        if fn:
            fn = '/scratch/users/jjzhu/tmp_fig/entero_venn.pdf'
            plt.savefig(fn, bbox_inches='tight') 
            logger.info("Saved figure to: {}".format(fn))
        plt.show()
        
        
def plot_multiple_scatter_discrete(embedding, clust_df):
    methods = clust_df.columns
    fig, axes = plt.subplots(1, len(methods), figsize=(2.9*len(methods)-0.2*(len(methods)-1),2.7))    
    for i, met in enumerate(methods):
        if len(methods) == 1:
            ax = axes
        else:
            ax = axes[i]
        lut = get_sim_color_map(clust_df[met], palette='tab10')
        plot_scatter_discrete(embedding, 
                              clust_df[met], cols=lut,
                              ax=ax, ms=1)
        ax.set_title(met)
        ax.set_xlabel('umap 1')
        ax.set_ylabel('umap 2')
    plt.tight_layout()
    plt.show()
        
def plot_multiple_scatter_continuous(embed_df, plt_df, logscale=False):
    n_plts = plt_df.shape[1]
    fig, axes = plt.subplots(1, n_plts, figsize=(n_plts*3, 2.5))
    for i, col in enumerate(plt_df.columns):
        ax = axes[i]
        plot_scatter_continuous(embed_df, plt_df[col], ax, logscale=logscale)
    plt.tight_layout()
    plt.show()
    
def plot_multi_curves(ax, x, y_mtx, **kwargs):
    assert len(x) == y_mtx.shape[0], "dimension mismatch"
    for i in range(y_mtx.shape[1]):
        ax.plot(x, y_mtx[:,i], **kwargs)