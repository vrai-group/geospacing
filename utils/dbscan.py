"""code to cluster data using DBSCAN with custom distance function and in particular dip and dip direction values obtained from TLS"""

import numpy as np
from sklearn.cluster import DBSCAN
from torch import cumsum
from utils.metrics import haversine_metric

def cluster_data(data, eps_v=0.001, min_samples_v=10, custom_metric = False):
    """
    Cluster the data with a given number of classes (default = 4); stop when reaching the
    maximum number of iterations (default 50) or eps (default 0.001)
    """
    if custom_metric:
        cluster_data = DBSCAN(eps=eps_v, min_samples=min_samples_v,metric=haversine_metric, algorithm="brute").fit(data)
    else:
        cluster_data = DBSCAN(eps=eps_v, min_samples=min_samples_v).fit(data)        
    return cluster_data.labels_

    