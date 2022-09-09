import numpy as np


def points_to_image(points, shape):

    """ Reconstruct depth map given points coming from 3D analysis"""

    # Initialise image to be filled 
    image  = np.zeros(shape)

    # Extract y, x, z values from points
    y, x, z = points

    # Construct the grid in Numpy
    x, y = np.meshgrid(x, y)

    # Fill image with depth
    image[y, x] = z

    return image