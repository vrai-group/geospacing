## Dataset
The dataset folder contains the reference data acquired using Terrestrial Laser Scanner (TLS). In particular the txt file contains the data with the following header:

    X,Y,Z,nx,ny,nz,dip, dip direction

where X,Y,Z is the coordinate linked to a given point, nx,ny,nz are the normals.
In that folder there are also the centroids for dip and dip direction angles (in degrees) derived from *in situ* observations.
## Classification
The classification algorithm is able to cluster data acquired with the TLS; classification takes into account the dip and dip direction values.
User could change:

 - the number of classes;
 - cluster initializations (random or manual);
 - the number of iterations.

User could change the path of fileName. To run the classification:

    python classification.py

The main outputs are:

 - centroid
 - clustered data (data + labels)
 - ply of point cloud
 - figures of classified data
## Spacing Calculation
To run the spacing calculation you need to provide the output of classification (clustered data); user could change params or file path / name. To run the code:

    python distances.py

The main output is a file with the statics for distances of each class.
## Stereonet

It is possible also to create a stereonet starting from the classified data. To generate the stereonet run the code in the following way:

    python stereo_plot.py

## Requirements
Python 3.8+
matplotlib
numpy
mplstereonet



