"""this module contains functions to implement custom metrics"""
from math import radians, cos, sin, asin, sqrt, pi
import numpy as np

VALUE_PI2 = 2*pi

def haversine(lon1, lat1, lon2, lat2, radius=1):
    """
    Calculate the great circle distance between two points
    on a spere with a given radius (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a_value = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c_value = 2 * asin(sqrt(a_value))
    return c_value * radius

def haversine_metric(x_value, y_value):
    """custom implementation of haversine metric"""
    return haversine(x_value[1], x_value[0], y_value[1], y_value[0])

def cylinder(dip_a, dip_dir_a, dip_b, dip_dir_b, radius=1, convert=False):
    """
    Calculate the distance between two points
    on a circle with a given radius (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    if(convert):
        dip_a_rad, dip_dir_a_rad, dip_b_rad, dip_dir_b_rad = map(radians, [dip_a, dip_dir_a, dip_b, dip_dir_b])
    else:
        dip_a_rad, dip_dir_a_rad, dip_b_rad, dip_dir_b_rad = [dip_a, dip_dir_a, dip_b, dip_dir_b]

    #disk #1
    theta1 = np.abs(dip_a_rad-dip_b_rad)

    delta_theta1 = radius * min(VALUE_PI2 - theta1, theta1)
    #disk #2
    theta2 = np.abs(dip_dir_a_rad-dip_dir_b_rad)
    delta_theta2 = radius * min(VALUE_PI2 - theta2, theta2)
    #return 0.2*delta_theta1 + 0.8*delta_theta2
    return sqrt(delta_theta1**2 + delta_theta2**2)

def cylinder_metric(x_value, y_value):
    """custom implementation of cylinder metric"""
    return cylinder(x_value[0], x_value[1], y_value[0], y_value[1])

def lonlat2xyz(lon, lat, radius=1):
    """conver lon / lat to x/y/z"""
    z_value = radius* np.sin(lat)
    rcoselev = radius * np.cos(lat)
    x_value = rcoselev * np.cos(lon)
    y_value = rcoselev * np.sin(lon)
    return np.transpose([x_value, y_value, z_value])

if __name__ == "__main__":
    print(cylinder(80, 350, 88, 10,radius=1, convert=True))
    print(haversine(20, 88, 70, 88))
