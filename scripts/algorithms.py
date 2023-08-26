import numpy as np
import openensembles as oe
import pandas as pd
from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from kmodes.kmodes import KModes


class EnsembleClustering(ABC):
    
    @abstractmethod
    def __init__(self, clusters, num_of_partitions=100):
        self.clusters = clusters
        self.num_of_partitions = num_of_partitions
        self.X = None
        self.labels_ = None
        
    
    @abstractmethod
    def fit(self):
        pass
    

class SpectralEnsemble(EnsembleClustering):
    
    def __init__(self, clusters, base_estimator_k, num_of_partitions=100, base_estimator="kmeans"):
        super().__init__(clusters, num_of_partitions)
        self.base_estimator = base_estimator
        self.base_estimator_k = base_estimator_k
        
    def fit(self, X):
        self.X = pd.DataFrame(X)
        data = oe.data(self.X, self.X.columns)
        cluster = oe.cluster(data)
        for i in range(self.num_of_partitions):
            cluster.cluster("parent", self.base_estimator, self.base_estimator + str(i), K=self.base_estimator_k, n_init=1, init="random")
        co_occurence_matrix = cluster.co_occurrence_matrix().co_matrix
        spectral_clustering = SpectralClustering(n_clusters=self.clusters, affinity="precomputed").fit(co_occurence_matrix)
        self.labels_ = spectral_clustering.labels_
        return self
    

class HierarchyEnsemble(EnsembleClustering):
    
    def __init__(self, clusters, base_estimator_k, num_of_partitions=100, base_estimator="kmeans", linkage="average"):
        super().__init__(clusters, num_of_partitions)
        self.base_estimator = base_estimator
        self.base_estimator_k = base_estimator_k
        self.linkage = linkage
        
    def fit(self, X):
        self.X = pd.DataFrame(X)
        data = oe.data(self.X, self.X.columns)
        cluster = oe.cluster(data)
        for i in range(self.num_of_partitions):
            cluster.cluster("parent", self.base_estimator, self.base_estimator + str(i), K=self.base_estimator_k, n_init=1, init="random")
        co_occurrence_matrix = cluster.co_occurrence_matrix().co_matrix
        hc = AgglomerativeClustering(n_clusters=self.clusters, affinity="precomputed", linkage=self.linkage).fit(1 - co_occurrence_matrix)
        self.labels_ = hc.labels_
        return self
    

class KModesEnsemble(EnsembleClustering):
    
    def __init__(self, clusters, base_estimator_k, num_of_partitions=100, base_estimator="kmeans"):
        super().__init__(clusters, num_of_partitions)
        self.base_estimator = base_estimator
        self.base_estimator_k = base_estimator_k
        
    def fit(self, X):
        self.X = pd.DataFrame(X)
        data = oe.data(self.X, self.X.columns)
        cluster = oe.cluster(data)
        Y = np.zeros((self.X.shape[0], self.num_of_partitions))
        for i in range(self.num_of_partitions):
            name = self.base_estimator + str(i)
            cluster.cluster("parent", self.base_estimator, name, K=self.base_estimator_k, n_init=1, init="random")
            Y[:, i] = cluster.labels[name]
        kmodes_ = KModes(n_clusters=self.clusters)
        self.labels_ = kmodes_.fit_predict(Y)
        return self


agg_algorithms = {"agg": AgglomerativeClustering}

class BaggedEnsemble(EnsembleClustering):
    
    def __init__(self, clusters, num_of_partitions=100, base_centers=20, base_estimator=KMeans, hclust="agg", linkage="average"):
        super().__init__(clusters, num_of_partitions)
        self.base_centers = base_centers
        self.base_estimator = base_estimator
        self.hclust = agg_algorithms[hclust]
        self.linkage = linkage
        
    def fit(self, X):
        n_row, n_col = X.shape
        all_centers = np.zeros((self.num_of_partitions * self.base_centers, n_col))
        for i in range(self.num_of_partitions):
            random_indices = np.random.randint(0, n_row, n_row)
            x = X[random_indices, :]
            all_centers[(i*self.base_centers):((i+1) * self.base_centers), :] = self.base_estimator(n_clusters=self.base_centers,
                                                                                                    n_init=1, init="random").fit(x).cluster_centers_
        clustered_centers = self.hclust(n_clusters=self.clusters, linkage=self.linkage).fit(all_centers)
        membership = clustered_centers.labels_
        function = NearestNeighbors(n_neighbors=1).fit(all_centers)
        neighborship = function.kneighbors(X, return_distance=False)
        self.labels_ = membership[neighborship].flatten()
        return self


class BaggedMajority(EnsembleClustering):
    
    def __init__(self, clusters, params, num_of_partitions=100, base_estimator=KMeans):
        super().__init__(clusters, num_of_partitions)
        self.base_estimator = base_estimator
        self.params = params
        
    @staticmethod
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

        
    def fit(self, X):
        nrow, ncol = X.shape
        indices = np.arange(0, nrow)
        initial_labels = self.base_estimator(**self.params).fit(X).labels_
        Y = np.zeros((nrow, self.clusters))
        for b in range(self.num_of_partitions):
            sample_indices = np.random.randint(0, nrow, nrow)
            intersect = np.intersect1d(indices, sample_indices)
            new_X = X[sample_indices]
            new_labels = self.base_estimator(**self.params).fit(new_X).labels_
            relabel = -np.ones(nrow, dtype=int)
            relabel[sample_indices] = new_labels
            common_labels_1, common_labels_2 = initial_labels[intersect], relabel[intersect]
            cm = confusion_matrix(common_labels_1, common_labels_2)
            a, b = list(map(list, linear_sum_assignment(self._make_cost_m(cm))))
            after_assignment = [a[b.index(i)] for i in common_labels_2]
            Y[intersect, after_assignment] += 1
        problematic = (Y.mean(axis=1) == 0)
        final_result = Y.argsort(axis=1)[:, ::-1][:, 0]
        final_result[problematic] = initial_labels[problematic]
        self.labels_ = final_result
        return self


class CoAssocEnsemble(EnsembleClustering):
    
    def __init__(self, clusters, base_estimator_k, threshold=0.8, num_of_partitions=100, base_estimator="kmeans"):
        super().__init__(clusters, num_of_partitions)
        self.base_estimator = base_estimator
        self.base_estimator_k = base_estimator_k
        self.threshold = threshold
    
    def fit(self, X):
        df = pd.DataFrame(X)
        data_obj = oe.data(df, df.columns)
        cluster = oe.cluster(data_obj)
        for i in range(self.num_of_partitions):
            cluster.cluster("parent", self.base_estimator, self.base_estimator + str(i), K=self.base_estimator_k, n_init=1, init="random")
        finishing = cluster.finish_majority_vote(threshold=self.threshold)
        self.labels_ = list(finishing.labels.values())[0]
        return self
    
    
class GraphConsensus(EnsembleClustering):
    
    def __init__(self, clusters, base_estimator_k, threshold=0.8, num_of_partitions=100, base_estimator="kmeans"):
        super().__init__(clusters, num_of_partitions)
        self.base_estimator = base_estimator
        self.base_estimator_k = base_estimator_k
        self.threshold = threshold
    
    def fit(self, X):
        df = pd.DataFrame(X)
        data_obj = oe.data(df, df.columns)
        cluster = oe.cluster(data_obj)
        for i in range(self.num_of_partitions):
            cluster.cluster("parent", self.base_estimator, self.base_estimator + str(i), K=self.base_estimator_k, n_init=1, init="random")
        finishing = cluster.finish_graph_closure(threshold=self.threshold)
        self.labels_ = list(finishing.labels.values())[0]
        return self
