import numpy as np
import time

from sklearn.utils import check_random_state, check_array
import scipy.sparse
import scipy.sparse.csgraph

from sklearn.metrics import pairwise_distances
from sklearn.externals import joblib

from umap.umap_ import UMAP
from umap.umap_ import (
    fuzzy_simplicial_set, 
    categorical_simplicial_set_intersection,
    simplicial_set_embedding,
    nearest_neighbors,
)
import umap.distances as dist
from umap.nndescent import (
    make_nn_descent,
    make_initialisations,
    make_initialized_nnd_search,
    initialise_search,
)
import logging

logger = logging.getLogger()

class REGUMAP(UMAP):
    def __init__(self, small_data=False, **kw):
        super().__init__(**kw)
        self.small_data = small_data
        # add more attributes if needed
        # logger.info("Initializing regularized UMAP...")
        
    def _check_input_format(self, X):
        """ 
        """
        X = check_array(X, dtype=np.float32, accept_sparse="csr")
        # needed to sklearn comparability
        if X.shape[0] == 1: 
            logger.warning(
                "only one observation - returning self for embedding"
            )
            return X
        # handle the number of nearest neighbor parameter
        if X.shape[0] <= self.n_neighbors:
            logger.warning(
                "n_neighbors is larger than the dataset size;" + 
                "truncating to X.shape[0] - 1"
            )
            self._n_neighbors = X.shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors
        # handle sparse matrix format
        if scipy.sparse.isspmatrix_csr(X):
            if not X.has_sorted_indices:
                X.sort_indices()
            self._sparse_data = True
        else:
            self._sparse_data = False
        logger.info("X.shape: {}; sparse: {}".format(X.shape, self._sparse_data))
        return X
    
    def _construct_base_graph(self, X):
        """ Constructs the self.graph_ object from fuzzy_simplicial_set()
        """
        random_state = check_random_state(self.random_state)
        # Handle small cases efficiently by computing all distances
        if self.small_data and X.shape[0] < 4096:
            self._small_data = True
            dmat = pairwise_distances(X, metric=self.metric, **self._metric_kwds)
            self.graph_ = fuzzy_simplicial_set(
                dmat,
                self._n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                None,
                None,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.verbose,
            )
        else:
            self._small_data = False
            # Standard case
            (self._knn_indices, self._knn_dists, self._rp_forest) = nearest_neighbors(
                X,
                self._n_neighbors,
                self.metric,
                self._metric_kwds,
                self.angular_rp_forest,
                random_state,
                self.verbose,
            )

            self.graph_ = fuzzy_simplicial_set(
                X,
                self.n_neighbors,
                random_state,
                self.metric,
                self._metric_kwds,
                self._knn_indices,
                self._knn_dists,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                self.verbose,
            )

            self._transform_available = True
            self._search_graph = scipy.sparse.lil_matrix(
                (X.shape[0], X.shape[0]), dtype=np.int8
            )
            self._search_graph.rows = self._knn_indices
            self._search_graph.data = (self._knn_dists != 0).astype(np.int8)
            self._search_graph = self._search_graph.maximum(
                self._search_graph.transpose()
            ).tocsr()

            if callable(self.metric):
                self._distance_func = self.metric
            elif self.metric in dist.named_distances:
                self._distance_func = dist.named_distances[self.metric]
            elif self.metric == 'precomputed':
                warn('Using precomputed metric; transform will be unavailable for new data')
            else:
                raise ValueError(
                    "Metric is neither callable, " + "nor a recognised string"
                )

            if self.metric != 'precomputed':
                self._dist_args = tuple(self._metric_kwds.values())

                self._random_init, self._tree_init = make_initialisations(
                    self._distance_func, self._dist_args
                )
                self._search = make_initialized_nnd_search(
                    self._distance_func, self._dist_args
                )
            
    def _augment_base_graph(self, y):
        random_state = check_random_state(self.random_state)
        
        if self.target_metric == "categorical":
            if self.target_weight < 1.0:
                far_dist = 2.5 * (1.0 / (1.0 - self.target_weight)) 
            else:
                far_dist = 1.0e12
            self.graph_ = categorical_simplicial_set_intersection(
                self.graph_, y, far_dist=far_dist
            )
        else:
            if self.target_n_neighbors == -1:
                target_n_neighbors = self._n_neighbors
            else:
                target_n_neighbors = self.target_n_neighbors

            # Handle the small case as precomputed as before
            if self.small_data and y.shape[0] < 4096:
                ydmat = pairwise_distances(y[np.newaxis, :].T,
                                           metric=self.target_metric,
                                           **self._target_metric_kwds)
                target_graph = fuzzy_simplicial_set(
                    ydmat,
                    target_n_neighbors,
                    random_state,
                    "precomputed",
                    self._target_metric_kwds,
                    None,
                    None,
                    False,
                    1.0,
                    1.0,
                    False
                )
            else:
                # Standard case
                target_graph = fuzzy_simplicial_set(
                    y[np.newaxis, :].T,
                    target_n_neighbors,
                    random_state,
                    self.target_metric,
                    self._target_metric_kwds,
                    None,
                    None,
                    False,
                    1.0,
                    1.0,
                    False,
                )
            # product = self.graph_.multiply(target_graph)
            # # self.graph_ = 0.99 * product + 0.01 * (self.graph_ +
            # #                                        target_graph -
            # #                                        product)
            # self.graph_ = product
            self.graph_ = general_simplicial_set_intersection(
                self.graph_, target_graph, self.target_weight
            )
            self.graph_ = reset_local_connectivity(self.graph_)
            
    def _construct_embedding(self):
        if self.n_epochs is None:
            n_epochs = 0
        else:
            n_epochs = self.n_epochs

        random_state = check_random_state(self.random_state)
        self.embedding_ = simplicial_set_embedding(
            self._raw_data,
            self.graph_,
            self.n_components,
            self.initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            self.init,
            random_state,
            self.metric,
            self._metric_kwds,
            self.verbose,
        )
        

    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.

        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        """

        X = self._check_input_format(X)
        self._raw_data = X
        # used for skipping recomputation in transform()
        self._input_hash = joblib.hash(self._raw_data) 
        
        if X.shape[0] == 1: # needed to sklearn comparability
            self.embedding_ = np.zeros((1, self.n_components))  
            return self
        
        t = time.process_time()
        self._construct_base_graph(X)
        et = time.process_time() - t
        logger.info("Constructed fuzzy simplicial set ({:.2f}s)".format(et))
        
        t = time.process_time()
        self._augment_base_graph(y)
        et = time.process_time() - t
        logger.info("Augmented fuzzy simplicial set ({:.2f}s)".format(et))
        
        t = time.process_time()
        self._construct_embedding()
        et = time.process_time() - t
        logger.info("Constructed embedding ({:.2f}s)".format(et))
       
        return self
    