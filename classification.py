"""code to cluster data and in particular dip and dip direction values obtained from TLS"""
import matplotlib.pyplot as plt
import numpy as np
from utils.colors import GROUP_COLORS, convert_color_to_rgb
import utils.kmeans as cls
import utils.pointcloud as pc

np.random.seed(1)

def main():
    """Entry point"""
    #pre-process data
    prefix = "manual"
    full_dataset = np.genfromtxt('dataset\\datasetFalesia.txt', delimiter=',')
    data = np.array(full_dataset[:, 6:8])
    #cluster data
    n_classes = 5
    max_iter = 100
    single = True
    if single:
        centroids_user_defined = np.genfromtxt(f'dataset\\init_cluster.txt', delimiter=',')
        centroids, classes = cls.cluster_data(data, n_classes, max_iter, \
            init_centroids=centroids_user_defined, convert=True)
        # Visualize
        colors = np.array([GROUP_COLORS[j] for j in classes])
        _, ax_value = plt.subplots(figsize=(8, 8))
        centroids_angles = np.zeros((n_classes, 2))
        for item in np.unique(classes):
            idx = np.where(classes == item)
            centroids_angles[item, :] = np.mean(data[idx, :], 1)
            ax_value.scatter(data[idx, 0], data[idx, 1], color=colors[idx], \
                alpha=0.5, label=f'Class {item:d}')
        ax_value.legend()
        ax_value.scatter(centroids_angles[:, 0], centroids_angles[:, 1], \
            color=['black']*n_classes, marker='o', lw=2)
        plt.xlabel("dip [degrees]")
        plt.ylabel("dip direction [degrees]")
        plt.savefig(f'kmeans_modified_{prefix}_{n_classes:02d}K.png')
        plt.show()
        #saving data
        expanded_data = np.insert(full_dataset, full_dataset.shape[1], classes, axis=1)
        np.savetxt(f'clustered_data_{prefix}_{n_classes:02d}.txt', expanded_data, fmt='%10.5f')
        np.savetxt(f'centroids_clustered_data_{prefix}_{n_classes:02d}.txt', \
            centroids_angles, fmt='%10.5f')
        #export to ply with colored vertexes according to classes and their colors
        pc.write_pointcloud(f'kmeans_modified_{prefix}_{n_classes:02d}K.ply', \
            full_dataset[:, 0:3], convert_color_to_rgb(classes))
    else:
        number_of_replicas = 5
        sse_values = []
        for i in range(2, n_classes+1):
            sse = []
            np.random.seed(1)
            for j in range(number_of_replicas):
                centroids, labels = cls.cluster_data(data, i, max_iter, \
            init_centroids=None)
                print(f'*** Current replica {j:02d} for {i:02d} classes ***')
                sse = cls.evaluate_sse(data, centroids, labels)
            sse_values.append(np.average(sse))
        print(sse_values)
        plt.plot(np.arange(2, n_classes+1), sse_values, lw=2)
        plt.title("Trend of SSE over the number of classes", fontsize=15)
        plt.xlabel("Number of classes", fontsize=15)
        plt.ylabel("SSE", fontsize=15)
        plt.grid()
        plt.savefig("reduced_elbow_kmeans_modified.png")
        np.savetxt("elbow_reduced.txt", sse_values)
        plt.show()

main()
