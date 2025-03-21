import numpy as np
from pyest.sensors import PolygonalFieldOfView


def default_poly_fov():
    verts = np.array([[14, 4], [16, 4], [16, 5], [14, 5]])
    return PolygonalFieldOfView(verts)
