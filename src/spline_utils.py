import os 
import json
import pickle

import numpy as np
import pandas as pd

from patsy import dmatrix
# from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf

def setup_basis(x, df=3, n_knots=2, by="linspace"):
    if by == "linspace":
        knots = np.linspace(min(x), max(x), n_knots+2)[1:-1]
    else: # percentile option
        perces = np.linspace(0, 100, n_knots+2)[1:-1]
        knots = np.percentile(x, perces)
        print(dict(zip(perces, knots)))
    knots = tuple([round(q, 3) for q in knots])
    argmt = "cr(x, knots={})".format(knots)
    return argmt
    
def spline_fit(x, y_mtx, argmt):
    assert len(x) == y_mtx.shape[0], "sample size mismatch"
    # print("Splint fit params: df={}, n_knots={}".format(df, n_knots))
    trans_x = dmatrix(argmt, {"x": x}, return_type='dataframe')
    # compute the y functions
    y_funs = [None] * y_mtx.shape[1]
    for i in range(y_mtx.shape[1]):
        sm_fit = sm.GLM(y_mtx[:, i], trans_x).fit()
        y_funs[i] = sm_fit
        if (i > 0) and (i % 1000 == 0):
            print("Fitted {} parameters".format(i))
    # extract the coefficient matrix
    coef_mtx = [None] * y_mtx.shape[1]
    for i in range(y_mtx.shape[1]):
        coef_mtx[i] = (list(y_funs[i].params))
    coef_mtx = np.array(coef_mtx)
    # extract the multivariate predictor
    def pred_func(new_x):
        out = [None] * len(y_funs)
        for i, sm_fit in enumerate(y_funs): # each function represents one dimension
            vals = sm_fit.predict(dmatrix(argmt, {"x": new_x}, return_type='dataframe'))
            out[i] = list(vals)
        return np.array(out).T
    return pred_func, coef_mtx

# file i/o for splint fitting
def get_sp_ftypes():
    save_objs = ["coeff", "param", "funcs", "lambs"]
    sufx_map = dict(zip(save_objs, ["csv", "json", "pkl", "csv"]))
    return sufx_map

def load_sp_result(main_dir, pfx="default", embedding="umap"):
    tmp_dir = os.path.join(main_dir, "spline_fit_{}".format(pfx))
    sufx_map = get_sp_ftypes()
    save_objs = list(sufx_map.keys())
    print(sufx_map)
    sp_res = {}
    for obj in save_objs:
        sufx = sufx_map[obj]
        fn = os.path.join(tmp_dir, "{}.{}".format(obj, sufx))
        if obj == "coeff":
            data = pd.read_csv(fn, index_col=0)
        if obj == "lambs":
            data = pd.read_csv(fn, index_col=0)
        if obj == "funcs": 
            continue # TODO: this does not work now
        if obj == "param":
            data = json.load(open(fn, "r"))  
        sp_res[obj] = data
        print("Loaded: {}".format(fn))
    var_df = sp_res["coeff"]
    out_dict = {}
    out_dict["features"] = var_df[sp_res["param"]["sp_vars"]]
    out_dict["embedding"] = var_df[["{}_1".format(embedding), 
                                    "{}_2".format(embedding)]]
    out_dict["var_info"] = var_df[["gene_ids", "n_cells", "mean", "std"]]
    out_dict["obs_info"] = sp_res["lambs"]
    out_dict["parameters"] = sp_res["param"]
    print("Output: {}".format(out_dict.keys()))
    return out_dict
    
def save_sp_result(main_dir, obs_df, var_df, c_mtx, p_fun, base_args, pfx="default"):
    print("Coeffient matrix shape: {}".format(c_mtx.shape))
    sp_names = ["v_{}".format(i) for i in range(c_mtx.shape[1])]
    coef_df = pd.DataFrame(c_mtx, index=var_df.index, columns=sp_names)
    coef_df = pd.concat([var_df, coef_df], axis=1)
    display(coef_df.head())
    tmp_dir = os.path.join(main_dir, "spline_fit_{}".format(pfx))
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    sufx_map = get_sp_ftypes()
    save_objs = list(sufx_map.keys())
    sufx_map = dict(zip(save_objs, ["csv", "json", "pkl", "csv"]))
    for obj in save_objs:
        sufx = sufx_map[obj]
        fn = os.path.join(tmp_dir, "{}.{}".format(obj, sufx))
        if obj == "coeff":
            coef_df.to_csv(fn)
        if obj == "lambs":
            obs_df.to_csv(fn)
        if obj == "funcs": 
            continue # TODO: this does not work now
            pickle.dump(p_fun, open(fn, "wb"))
        if obj == "param":
            params = {"bases": base_args, 
                      "sp_vars": sp_names}
            json.dump(params, open(fn, "w"))  
        print("Saved: {}".format(fn))