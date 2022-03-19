"""Evaluation of clustered data to estimate the statistics for each class"""
import numpy as np
import matplotlib.pyplot as plt
import utils.conversion as conv

def evaluate_width(distances, value):
    """function to aggregate values with small distances..."""
    break_points = []
    break_points_values = []
    cum_sum = 0.0
    for i in range(distances.shape[0]):
        cum_sum += distances[i]
        if cum_sum > value:
            break_points.append(cum_sum)
            break_points_values.append(i)
            cum_sum = 0.0
    return (np.array(break_points), np.array(break_points_values))

MIN_DISTANCE = 0.015
AGGREGATION_VALUE = 0.01

def main():
    """Main script to process clustered data"""
    #chnge the path / file-name
    full_dataset = np.genfromtxt('clustered_data_manual_05.txt')
    labels = np.array(full_dataset[:, -1], dtype=np.uint8)
    points = full_dataset[:, 0:3]
    normals = full_dataset[:, 3:6]

    distances = np.zeros(labels.shape)
    v_distances = np.zeros(labels.shape)
    point_zero = np.zeros((3))
    out_data = np.zeros((np.unique(labels).shape[0], 4))
    count = 0

    for class_id in np.unique(labels):
        idx = np.where(labels == class_id)
        base_point = points[idx[0][0], :]
        mean_normal = np.mean(normals[idx[0], :], 0)
        for i in range(1, idx[0].shape[0]):
            distances[idx[0][i]] = conv.distance_point_plane(mean_normal, \
                 base_point - points[idx[0][i], :], point_zero)
            v_distances[idx[0][i]] = np.linalg.norm(base_point - points[idx[0][i], :])
        sorted_distances = np.sort(distances[idx[0]])
        sorted_distances_d = np.gradient(sorted_distances, 1)
        break_points, _ = evaluate_width(sorted_distances_d, AGGREGATION_VALUE)
        filtered_break_points = break_points[np.where(break_points > MIN_DISTANCE)]
        #basic stats
        mean_value = np.mean(filtered_break_points)
        std_value = np.std(filtered_break_points)
        max_value = np.max(filtered_break_points)
        out_data[count, 0] = class_id
        out_data[count, 1] = mean_value
        out_data[count, 2] = std_value
        out_data[count, 3] = max_value
        count += 1

        print(f'cls: {class_id:3d} mean: {mean_value:7.4f} max: {max_value:7.4f} \
            std: {std_value:7.4f}')
        plt.plot(sorted_distances_d)
        plt.title(f'distances for class {class_id:2d}')
        plt.xlabel("sample")
        plt.ylabel("distances between neighbours")
        plt.show()
        #export data
        np.savetxt(f'stats_clustered_data_manual_05.png', out_data, fmt='%10.5f')
main()
