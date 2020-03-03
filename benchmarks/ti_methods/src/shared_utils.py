import scanpy as sc
import pandas as pd
import numpy as np
import palantir


def run_paga_clustering(adata):
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
    sc.tl.draw_graph(adata)
    sc.tl.louvain(adata, resolution=1.0)
    sc.tl.paga(adata, groups='louvain')
    adata.obs['louvain'].cat.categories
    adata.obs['louvain_anno'] = adata.obs['louvain']
    adata.obs['louvain_anno'].cat.categories

    return adata


def run_paga_pseudotime(adata, start_cell):
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    adata.obs['cell_id'] = adata.obs.index
    adata.uns['iroot'] = np.flatnonzero(adata.obs['cell_id']  == start_cell)[0]
    sc.tl.dpt(adata)
    adata.obs = adata.obs.rename(columns={'dpt_pseudotime': 'pseudotime'})        
    return adata

def run_palantir_pseudotime(norm_df, start_cell):
    pca_projections, _ = palantir.utils.run_pca(norm_df)
    # Run diffusion maps
    dm_res = palantir.utils.run_diffusion_maps(pca_projections)
    ms_data = palantir.utils.determine_multiscale_space(dm_res)
    imp_df = palantir.utils.run_magic_imputation(norm_df, dm_res)
    pr_res = palantir.core.run_palantir(ms_data, start_cell)
    out_df = pd.DataFrame({'pseudotime': pr_res._pseudotime,
                           'entropy': pr_res._entropy})
    return pr_res, out_df