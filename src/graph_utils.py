# graph utils
import numpy as np
import logging

from scipy.sparse import csr_matrix, coo_matrix, diags

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from scipy.linalg import eigh

logger = logging.getLogger("feat_viz")

def dist_corr(X, Y, metric="euclidean"):
    """ Compare the distance correlation between two variables
        Produces same result as: 
        https://gist.github.com/satra/aa3d19a12b74e9ab7941
    
    Args:
        - X: an n x p numpy array matrix
        - Y: an n x q numpy array matrix
    Returns:
        a single value that is the distance correlation
    """
    # TODO: make more efficient perhaps
    if X.ndim == 1: # fill up to a vector
        x_mtx = np.expand_dims(X, axis=1)
    else:
        x_mtx = X
    if Y.ndim == 1:
        y_mtx = np.expand_dims(Y, axis=1)
    else:
        y_mtx = Y   
    assert x_mtx.ndim == 2, "error in input X dimensions"
    assert y_mtx.ndim == 2, "error in input Y dimensions"
    assert x_mtx.shape[0] == y_mtx.shape[0], "mismatch dimensions"
    n = x_mtx.shape[0] 
    A = pairwise_distances(x_mtx, metric=metric)
    B = pairwise_distances(y_mtx, metric=metric)

    A = A - np.expand_dims(A.mean(axis=0), axis=0) - \
            np.expand_dims(A.mean(axis=1), axis=1) + \
            A.mean()
    B = B - np.expand_dims(B.mean(axis=0), axis=0) - \
            np.expand_dims(B.mean(axis=1), axis=1) + \
            B.mean()
    dcov = np.multiply(A, B).sum() / (n * n )
    Xcor = np.multiply(A, A).sum() / (n * n )
    Ycor = np.multiply(B, B).sum() / (n * n )
    dcorr = np.sqrt(dcov) / np.sqrt((np.sqrt(Xcor) * np.sqrt(Ycor)))
    return dcorr

def kernel_dist(X, centered=True, sigma=1, kernel="euclidean"):
    # compute the kernel distance based on:
    # https://arxiv.org/pdf/1205.0411.pdf
    
    if X.ndim == 1: # fill up to a vector
        x_mtx = np.expand_dims(X, axis=1)
    else:
        x_mtx = X
    assert x_mtx.ndim == 2, "error in input X dimensions"
    if kernel == "dist_gauss":
        dist = pairwise_distances(x_mtx, metric="euclidean") ** 2
        dist = np.exp(- sigma * dist) 
        if centered:
            vec_ss = np.exp(- sigma * np.sum(x_mtx**2, axis=1))
            S = dist + 1 - np.expand_dims(vec_ss, axis=0) - np.expand_dims(vec_ss, axis=1)
        else:
            S = 2 * (1 - dist)
    elif kernel == "gauss":
        dist = pairwise_distances(x_mtx, metric="euclidean") ** 2
        S = np.exp(- sigma * dist) 
    else:
        S = pairwise_distances(x_mtx, metric="euclidean")
    # the centering based on HSIC 
    S = S - np.expand_dims(S.mean(axis=0), axis=0) - \
            np.expand_dims(S.mean(axis=1), axis=1) + \
            S.mean()
    return S
    
           
def dist_kernel_corr(Y, S, k=15, symmetric=True, **kwargs):
    # S is the variable to construct the matrix graph (treated as a whole)
    # Y is the response variable matrix (treated column by column)
   
    A = kernel_dist(S, **kwargs)
    n = A.shape[0] 
    B = kernel_dist(Y, **kwargs)
#     else:
#         assert 0, "This function needs to be investigated"
#         # The following is a very heuristic assymetric correlation measure
#         # it may not result in [0,1] values
#         # A = gauss_kernel_dist(S, centered=centered)
#         if Y.ndim == 1:
#             y_mtx = np.expand_dims(Y, axis=1)
#         else:
#             y_mtx = Y   
#         B = pairwise_distances(y_mtx, metric="euclidean")
#         # doing centering just to make things a little bit better behaved?
#         B = B - np.expand_dims(B.mean(axis=0), axis=0) - \
#             np.expand_dims(B.mean(axis=1), axis=1) + \
#             B.mean()
    # TODO: double-check if centering is important here or not
    dcov = np.multiply(A, B).sum() / (n * n )
    Xcor = np.multiply(A, A).sum() / (n * n )
    Ycor = np.multiply(B, B).sum() / (n * n )
    dcorr = np.sqrt(dcov) / np.sqrt((np.sqrt(Xcor) * np.sqrt(Ycor)))
    return dcorr


def construct_kernal_graph(X, sigma2=None):
    if X.ndim == 1: # fill up to a vector
        x_mtx = np.expand_dims(X, axis=1)
    else:
        x_mtx = X
    assert x_mtx.ndim == 2, "error in input X dimensions"
    dist = pairwise_distances(x_mtx, metric="euclidean") ** 2
    if sigma2 is None:
        sigma2 = np.median(X.var(axis=0))
    dist = np.exp(-  dist / sigma2) 
    return coo_matrix(dist)
       
def construct_knn_naive(nmat_s, k=15):
    # Input:
    #   nmat_s: an (N x P) numpy array
    # Returns:
    #   dmat_s: an (N x N) nearest neighbor graph

    dmat_s = kneighbors_graph(nmat_s, k, 
                            mode='connectivity', 
                            metric="euclidean", 
                            include_self=True)
    dmat_s = (dmat_s + dmat_s.T) / 2 # make symmetric? TODO: fix later

    return dmat_s


#########
# metrics
#########

def get_laplacian(S):
    D_vec = np.array(S.sum(axis=0))
    D_sum = np.sum(D_vec)
    D = diags(D_vec.flatten())
    L = D - S # the graph laplacian
    return L

def get_graph_spectrum(graph, num=1):
    L = get_laplacian(graph.toarray())
    eigvals, eigvecs = eigh(L) 
    for i, v in enumerate(eigvals):
        if v > 1e-10:
            break
    # eig_pass = eigvals[:(i+1)]
    logger.debug("eig gap {}-> {}*: ({:.4f}, {:.4f}*)".format(
            i-1, i,  eigvals[i-1], eigvals[i]))
    if num == 1:
        return eigvecs[:, i]
    else:
        return eigvecs[:, i:(i+1+num)]
    
def laplacian_score(X, S):
    
    # for sparse representation, refer to:
    # https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/similarity_based/lap_score.py

    n_samp, n_feat = X.shape
    # handle graph S which is a csr_matrix
    assert S.shape[0] == n_samp, "sample mismatch"
    D_vec = np.array(S.sum(axis=0))
    D_sum = np.sum(D_vec)
    D = diags(D_vec.flatten())
    L = D - S # the graph laplacian
    F = X - (np.matmul(X.T, D_vec.T)).T / D_sum
    
    # TODO: handle X for sparse matrices
    # TODO: handle zero divisions

    # compute the mean-centered feature matrix (broad-casting)
    # np.mean(F, axis=0) is close to zero when uniform
    FLF = np.multiply(np.matmul(F.T, L.todense()).T, F).sum(axis=0)
    FDF = np.multiply(np.matmul(F.T, D.todense()).T, F).sum(axis=0)
    scores = np.array(np.divide(FLF, FDF)).flatten()

    return scores
