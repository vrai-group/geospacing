"""code to cluster data and in particular dip and dip direction values obtained from TLS"""
import numpy as np
from utils.metrics import cylinder, haversine, VALUE_PI2

def cluster_data(data, number_of_classes=4, maxiter=100, eps=0.0001, init_centroids=None, convert=False):
    """
    Cluster the data with a given number of classes (default = 4); stop when reaching the
    maximum number of iterations (default 50) or eps (default 0.001)
    """
    if(convert):
        data = np.radians(data)
        if(isinstance(init_centroids, (np.ndarray, np.generic) )):
            init_centroids = np.radians(init_centroids)

    if init_centroids is None:
    # Initialize our centroids by picking random data points
        centroids = initialize_clusters(data, number_of_classes)
    else:
        centroids = init_centroids
    # Initialize the vectors in which we will store the
    # assigned classes of each data point and the
    # calculated distances from each centroid

    classes = np.zeros(data.shape[0], dtype=np.float64)
    distances = np.zeros([data.shape[0], number_of_classes], dtype=np.float64)
    # Loop for the maximum number of iterations
    prev_centroid = np.array(centroids)
    for curr_iter in range(maxiter):
        # Assign all points to the nearest centroid
        for i_value, c_value in enumerate(centroids):
            distances[:, i_value] = get_distances_over_disk(c_value, data)
            #distances[:, i_value] = get_distances_over_shere(c_value, data)
            #distances[:, i_value] = get_distances(c_value, data)
        # Determine class membership of each point
        # by picking the closest centroid
        classes = np.argmin(distances, axis=1)
        # Update centroid location using the newly
        # assigned data point classes
        for c_idx in range(number_of_classes):
            centroids[c_idx] = np.mean(data[classes == c_idx], 0)
        delta = np.min(np.abs(prev_centroid - centroids)%VALUE_PI2)
        prev_centroid = np.array(centroids)
        print(f'#of classes: {number_of_classes:02d} current iter: {curr_iter:2d} \
         with eps: {delta:8f}')
        if delta < eps:
            break
    return (centroids, classes)

def evaluate_sse(data, centroids, labels):
    """Return the Sum of Squared Error starting from clustered data"""
    sse = 0.0
    for i_value, c_value in enumerate(labels):
        sse += (data[i_value] - centroids[c_value])**2
    return np.sum(sse)

def initialize_clusters(points, k):
    """Initializes clusters as k randomly selected points from points."""
    return points[np.random.randint(points.shape[0], size=k)]

# Function for calculating the distance between centroids
def get_distances_over_shere(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    return np.array([haversine(centroid[1], centroid[0], item[1], item[0]) for item in points])

# Function for calculating the distance between centroids using disk metric
def get_distances_over_disk(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    return np.array([cylinder(centroid[0], centroid[1], item[0], item[1]) for item in points])

# Function for calculating the distance between centroids for angular data using Haversine formula
def get_distances(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    return np.linalg.norm(points - centroid, axis=1)
    