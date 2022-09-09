import numpy as np
import pyvista as pv


def create_smooth_surface(image, n_iter = 1000):

    """ Create smooth surface out of greyscale image. 
        n_iter corresponds to the number of smoothing operations,
        the higher the more smoothing is applied. Computation time
        grows linearly with n_iter. """

    # Extract x,y coordinates
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])

    # Construct the grid in Numpy
    x, y = np.meshgrid(x, y)

    # Extract depth from image
    z = image[y, x]

    # Construct grid in pyVista (necessary for surface creation)
    grid = pv.StructuredGrid(y, x, z)

    # Create surface 
    surface = grid.extract_surface()

    # Smoothen surface
    smooth_surface = surface.smooth(n_iter=n_iter)

    # Add elevation scalar, i.e. points are colored based on their height
    smooth_surface = smooth_surface.elevation()

    return smooth_surface