"""colors to be used for plot"""
import numpy as np
from matplotlib import colors
GROUP_COLORS = ['skyblue', 'coral', 'lightgreen', 'red', 'orange', 'purple',\
 'pink', 'green', 'blue', 'darkblue', 'gray', 'brown']

def convert_color_to_rgb(data, normalize=True):
    """convert color (string) to rgb"""
    if normalize:
        out_data = np.zeros((data.shape[0], 3), dtype=np.uint8)
        gain_v = 255.0
    else:
        out_data = np.zeros((data.shape[0], 3))
        gain_v = 1.0
    for i in range(data.shape[0]):
        out_data[i] = gain_v * np.array(colors.to_rgba(GROUP_COLORS[data[i]])[0:3])
    return out_data
