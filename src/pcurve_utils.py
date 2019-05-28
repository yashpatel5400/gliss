import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

from general_utils import extended_bounds

def initial_pcurve(x):
    u, s, vh = np.linalg.svd(x)
    pc1_dir = vh[0,:] # the first pc directio
    proj_mtx = np.outer(pc1_dir, pc1_dir)
    z = np.matmul(x, proj_mtx)
    lams = u[:,0] * s[0]
    def pc_curve_map(lams):
        return np.outer(lams, pc1_dir)
    return lams, pc_curve_map

def run_base_pcurve(x, 
                    min_diff=0.01, 
                    max_iter=30,
                    ret_all_its=True):
    n = x.shape[0]
    p = x.shape[1]
    
    if ret_all_its:
        lamb_out = np.zeros((n, max_iter))
        func_out = [None] * max_iter
    else:
        lamb_out = np.zeros(n)
        func_out = None
        
    lambdas, pc_curve_map = initial_pcurve(x)
    for iteration in range(max_iter-1):
        if ret_all_its:
            lamb_out[:, iteration] = lambdas
            func_out[iteration] = pc_curve_map
        # Step 1: refit smooth function on the lambda parameters
        orders = np.argsort(lambdas)
        z = pc_curve_map(lambdas)
        prev_square = np.sum( (x - z)**2)
        univ_funcs = [None] * p
        for i in range(p):
            univ_funcs[i] = UnivariateSpline(lambdas[orders], x[orders, i], k=3)
        def pc_curve_map(lams): # the new smooth function
            vals = [func(lams) for func in univ_funcs]
            return np.array(vals).T

        # Step 2: project the points onto the new curve to get the new lambdas
        def obj_func(lam, dat_pnt, univ_funcs): # objective function to minimize
            # projected point on the curve
            proj = pc_curve_map(lam)
            # compute the euclidean distance between the data and this point
            return np.sum((proj - dat_pnt)**2)
        bounds = extended_bounds(lambdas)
        new_lams = np.zeros(x.shape[0])
        new_projs = np.zeros(x.shape)
        for i_dat in range(x.shape[0]):
            other_args = (x[i_dat,:], univ_funcs)
            min_res = minimize(obj_func, lambdas[i_dat], args=other_args, bounds=[bounds])
            new_lams[i_dat] = min_res.x
            new_projs[i_dat, :] = pc_curve_map(min_res.x)

        # Update (or TODO: save) parameters and check for convergence
        z = new_projs
        lambdas = new_lams
        orders = np.argsort(lambdas)
        dist_square = np.sum( (x - z)**2)
        print("Sum of squared distances: {}".format(dist_square))
        if (np.abs(dist_square - prev_square) < 0.01):
            print("Convereged at iteration: {}".format(iteration))
            break
    if ret_all_its:
        lamb_out[:, iteration+1] = lambdas
        lamb_out = lamb_out[:, :(iteration+1)]
        func_out[iteration+1] = pc_curve_map
        func_out = [func_out[i] for i in range(iteration+1)]
        
    else:
        lamb_out = lambdas
        func_out = pc_curve_map
        
    return lamb_out, func_out