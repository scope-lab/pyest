import pyest.sensors as sensors
import numpy as np
import numpy.testing as npt
import pytest

from pyest.sensors.FieldOfView import EllipticalFieldOfView, ConvexPolyhedralFieldOfView


def rot_matrix(angle):
    """ rotation matrix """
    return np.array([[np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]])

def rot_matrix_3d(angle):
    """ rotation matrix about z-axis """
    return np.array([[np.cos(angle), np.sin(angle), 0],
                     [-np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

def test_PolygonalFieldOfView():
    """ test polygonal field of view """
    verts = 50*np.array([[np.cos(i*2/3*np.pi),np.sin(i*2/3*np.pi)]for i in range(3)])
    fov = sensors.PolygonalFieldOfView(verts)
    # test single point case
    assert(fov.contains([0, 0]))
    # test multiple point case
    npt.assert_array_equal(
        fov.contains([[0, 0], [-1001, 1]]),
        [True, False]
    )

    # test rotation
    angle = 2*np.pi/3
    A = rot_matrix(angle)
    new_fov = fov.apply_linear_transformation(A)
    npt.assert_array_almost_equal(new_fov.verts, verts[(2, 0, 1), :])

def test_CircularFieldOfView():
    """ test circular field of view"""
    center = np.array([4, -20])
    radius = 10

    fov = sensors.CircularFieldOfView(center, radius)
    # create a unit vector and test points along the direction
    u = np.array([np.cos(np.pi/4), np.sin(np.pi/4)])
    in_pt = center + 0.9*radius*u
    out_pt = center + 1.1*radius*u
    assert(fov.contains(in_pt))
    assert(~fov.contains(out_pt))
    npt.assert_array_equal(fov.contains([in_pt, out_pt]), [True, False])


def test_CircularFieldOfView_ApplyLinearTransformation():
    """ test the ApplyLinearTransformation method """
    center = np.array([4, -20])
    radius = 10
    fov = sensors.CircularFieldOfView(center, radius)

    A = np.eye(2)
    new_fov = fov.apply_linear_transformation(A, -center, center)
    assert(isinstance(new_fov, sensors.CircularFieldOfView))
    npt.assert_array_equal(new_fov.center, fov.center)
    assert(new_fov.radius == fov.radius)

    angle = np.pi/3
    A = rot_matrix(angle)
    new_fov = fov.apply_linear_transformation(A, -center, center)
    assert(isinstance(new_fov, sensors.CircularFieldOfView))
    npt.assert_array_equal(new_fov.center, fov.center)
    assert(new_fov.radius == fov.radius)

    angle = np.pi/3
    x_scale = 2
    y_scale = 1
    A = np.array([[x_scale*np.cos(angle), y_scale*np.sin(angle)],
                  [x_scale*-np.sin(angle), y_scale*np.cos(angle)]])
    new_fov = fov.apply_linear_transformation(A, -center, center)
    assert(isinstance(new_fov, sensors.EllipticalFieldOfView))
    npt.assert_array_equal(new_fov.center, fov.center)
    npt.assert_almost_equal(new_fov.width, 2*x_scale*fov.radius)
    npt.assert_almost_equal(new_fov.height, 2*y_scale*fov.radius)


def test_EllipticalFieldOfView():
    center = np.array([4, -20])
    width = 8
    height = 5
    angle = 0

    fov = EllipticalFieldOfView(center, width, height, angle)

    # some test points
    right = center + [0.5*width, 0]
    left = center + [-0.5*width, 0]
    top = center + [0, 0.5*height]
    bottom = center + [0, -0.5*height]
    assert(fov.contains(center))
    assert(fov.contains(right))
    assert(all(fov.contains([left, top, bottom])))
    assert(~fov.contains(center+[width, height]))

    # test a rotated ellipse
    center = np.array([4, -20])
    width = 8
    height = 5
    angle = np.pi

    fov = EllipticalFieldOfView(center, width, height, angle)

    # some test points
    right = center + [0.5*width, 0]
    left = center + [-0.5*width, 0]
    top = center + [0, 0.5*height]
    bottom = center + [0, -0.5*height]
    assert(fov.contains(center))
    assert(fov.contains(right))
    assert(all(fov.contains([left, top, bottom])))
    assert(~fov.contains(center+[width, height]))

    # test a rotated ellipse
    center = np.array([4, -20])
    width = 8
    height = 5
    angle = np.pi/2

    fov = EllipticalFieldOfView(center, width, height, angle)

    # some test points
    right = center + [0.5*height, 0]
    left = center + [-0.5*height, 0]
    top = center + [0, 0.5*width]
    bottom = center + [0, -0.5*width]
    assert(fov.contains(center))
    assert(fov.contains(right))
    assert(all(fov.contains([left, top, bottom])))
    assert(~fov.contains(center+[width, height]))

def test_ConvexPolyhedralFieldOfView():
    """ test convex polyhedral field of view """
    # create a rectangular prism
    verts = [(x,y,z) for x in (0,1) for y in (0,2) for z in (0,3)]
    fov = ConvexPolyhedralFieldOfView(verts)
    # test single point case
    assert(fov.contains([0, 0, 0]))
    # test multiple point case
    npt.assert_array_equal(
        fov.contains([[0.1, 1.4, 2.6], [1.6, 0.1, 0.1]]),
        [True, False]
    )

    # test transformation
    A = np.diag([1,2,3])

    des_verts = np.array([[ 0,  1,  2],
       [ 0,  1, 11],
       [ 0,  5,  2],
       [ 0,  5, 11],
       [ 1,  1,  2],
       [ 1,  1, 11],
       [ 1,  5,  2],
       [ 1,  5, 11]])

    new_fov = fov.apply_linear_transformation(A, post_shift=np.arange(3))
    npt.assert_array_equal(new_fov.verts, des_verts)


if __name__ == "__main__":
    pytest.main([__file__])
