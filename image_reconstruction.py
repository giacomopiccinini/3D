import numpy as np


def points_to_image(points, shape):

    """ Reconstruct depth map given points coming from 3D analysis"""

    # Extract depth
    z = points[:, 2]

    # Recreate image
    image = np.full(shape, z.reshape(shape))

    return image