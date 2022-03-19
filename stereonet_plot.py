"""Module to plot stereonet from clustered data"""
import matplotlib.pyplot as plt
import mplstereonet
import numpy as np
from utils.colors import GROUP_COLORS

def plot_save_stereonet(dip, dip_direction, labels, filename="stereonet.png"):
    """function to plot and save stereonet"""

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='stereonet')
    count = 0
    for item in np.unique(labels):
        idx = np.where(labels == item)
        tmp_dip = dip[idx]
        tmp_dip_direction = dip_direction[idx] - 90.0
        ax.pole(tmp_dip_direction, tmp_dip, color=f'{GROUP_COLORS[count]}', \
             marker="^", markersize=2, label=f'Class {item:d}')
        count += 1
    ax.legend()
    ax.grid()   
    plt.savefig(filename)
    plt.show()

def main():
    """Main script to plot stereonet on clustered data"""
    #change the path / file name    
    full_dataset = np.genfromtxt('clustered_data_manual_05.txt')
    file_name = "stereonet_clustered_data.png"
    labels = np.array(full_dataset[:, -1], dtype=np.uint8)
    dip = full_dataset[:, 6]
    dip_direction = full_dataset[:, 7]
    plot_save_stereonet(dip, dip_direction, labels, file_name)

main()
