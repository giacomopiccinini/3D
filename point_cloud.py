import numpy as np
import pyvista as pv

def create_point_cloud(image):

    """ Create point cloud from depth image"""

    # Retrieve coordinates from image (i.e. y, x)
    coordinates = np.argwhere(image)

    # Retrieve depths (greyscale values)
    depths = image.flatten().astype("float32")

    # Construct points
    points = np.column_stack((coordinates, depths))

    # Construct point cloud
    point_cloud = pv.PolyData(points)

    # Assign scalar value named "Height" at the third column (greyscale)
    point_cloud['Height'] = point_cloud.points[:, 2]

    return point_cloud


def downsample_point_cloud(point_cloud, ratio):

    """ Reduce number of PointCloud points to ratio-%. Ratio should be a percentage, 
        i.e. 0 < ratio < 1. """

    # Extract points from point cloud
    points = point_cloud.points

    # Extract total number of points in the original point cloud
    tot_points = points.shape[0]

    # Downsample PointCloud points number
    down_points = int(ratio*tot_points) 

    # Extract random positions (array indices) to downsample
    sample_positions = np.random.default_rng(42).integers(low=0, high=tot_points, size=down_points)

    # Extract the sample points
    sample_points = points[sample_positions]

    # Create downsampled PointCloud
    downsampled = pv.PolyData(sample_points)

    # Assign scalar value named "Height" at the third column (greyscale)
    downsampled['Height'] = downsampled.points[:, 2]

    return downsampled