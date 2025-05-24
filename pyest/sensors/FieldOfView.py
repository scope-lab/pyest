import numpy as np

from abc import ABC, abstractmethod
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.vectorized import contains as shapelycontains
from scipy.spatial import Delaunay

class FieldOfView(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def contains(self, pts):
        """ check if field of view contains point(s)

        Parameters
        ----------
        pts : arraylike
            (n,2) list of test points

        Returns
        -------
        list
            (n,) list of Booleans

        Written by Keith LeGrand, June 2020
        """
        pass

    @abstractmethod
    def apply_linear_transformation(self, A, pre_shift=None, post_shift=None):
        """ applies the linear transformation A to the field-of-view

        Parameters
        ----------
        A : ndarray
            (n,n) transformation matrix, where n is the dimension of the
            field-of-view

        Returns
        -------
        FieldOfView

        Written by Keith LeGrand, October 2022
        """


class PolygonalFieldOfView(FieldOfView):
    """ 2D polygonal field of view

    Parameters
    ----------
    verts : ndarray
        (n,2) vertices of polygon

    Written by Keith LeGrand, June 2020

    """
    def __init__(self, verts):
        self.verts = verts
        self.polygon = Polygon(verts)
        self._center = None

    def contains(self, pts):
        """ check if field of view contains point(s)

        Paramaters
        ----------
        pts : arraylike
            (n,2) list of test points

        Returns
        -------
        list
            (n,) list of Booleans

        Written by Keith LeGrand, June 2020
        """

        pts = np.atleast_2d(pts)
        if pts.shape[0] == 1:
            return self.polygon.contains(Point(pts[0]))
        else:
            #return [self.polygon.contains(Point(p)) for p in pts]
            return shapelycontains(self.polygon, pts[:, 0], pts[:, 1])

    def apply_linear_transformation(self, A, pre_shift=None, post_shift=None):
        """ apply linear transformation to vertices

        Parameters
        ----------
        A : ndarray
            (n,n) transformation matrix, where n is the dimension of the
            field-of-view

        Returns
        -------
        PolygonalFieldOfView

        Written by Keith LeGrand, October 2022
        """
        #if about_center:
        #    return PolygonalFieldOfView(
        #        (A@(self.verts - self.center).T).T+self.center)
        #else:
        if pre_shift is None:
            pre_shift = np.zeros(2)
        if post_shift is None:
            post_shift = np.zeros(2)

        return PolygonalFieldOfView((A@(self.verts + pre_shift).T).T + post_shift)

    def get_center(self):
        if self._center is None:
            self._center = np.array(self.polygon.centroid.coords)
        return self._center

    def set_center(self, center):
        old_center = self.center
        # compute the change in center
        delta = center - old_center
        # store the new center location
        self._center = center
        # update the vertices and polygon
        self.verts += delta
        self.polygon = Polygon(self.verts)

    @property
    def lb(self):
        return np.min(self.verts, axis=0)

    @property
    def ub(self):
        return np.max(self.verts, axis=0)

    center = property(get_center, set_center)


class CircularFieldOfView(FieldOfView):
    """ 2D circular field of view

    Parameters
    ----------
    center : ndarray
        (2,) center of circle
    radius : float
        radius of circle
    res : int, optional
        when represented as a polygon, the number of vertices to use. Default
        is res=25

    Written by Keith LeGrand, December 2020

    """
    def __init__(self, center, radius, res=25):
        self.center = center
        th = np.linspace(0, 2*np.pi, res)
        x = center[0] + radius*np.cos(th)
        y = center[1] + radius*np.sin(th)
        self.verts = np.vstack((x, y)).T
        self.polygon = Polygon(self.verts)
        self.radius = radius
        self._radius_squared = radius**2
        self._ub = self.center + self.radius
        self._lb = self.center - self.radius

    def contains(self, pts):
        """ check if field-of-view contains point(s)

        Paramaters
        ----------
        pts : arraylike
            (n,2) list of test points

        Returns
        -------
        list
            (n,) list of Booleans

        Written by Keith LeGrand, December 2020
        """

        pts = np.atleast_2d(pts)
        if pts.shape[0] == 1:
            return np.sum((pts - self.center)**2) <= self._radius_squared
        else:
            return np.sum((pts - self.center)**2, axis=1) <= self._radius_squared

    def apply_linear_transformation(self, A, pre_shift=None, post_shift=None):

        if pre_shift is None:
            pre_shift = np.zeros(2)
        if post_shift is None:
            post_shift = np.zeros(2)

        xv = A@np.array([self.radius, 0])
        yv = A@np.array([0, self.radius])

        width = 2*np.linalg.norm(xv)
        height = 2*np.linalg.norm(yv)

        new_center = A@(self.center + pre_shift) + post_shift

        if width == height:
            return CircularFieldOfView(new_center, radius=0.5*width)

        else:
            # result is an ellipse
            angle = np.arctan2(xv[1], xv[0])
            return EllipticalFieldOfView(new_center, width, height, angle)

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub


class EllipticalFieldOfView(FieldOfView):
    """ 2D elliptical field of view

    Parameters
    ----------
    center : ndarray
        (2,) center of circle
    width : float
        width of ellipse. If the ellipse is wider than it is tall, the width
        is equal to twice the semi-major axis
    height : float
        height of ellsipse. If the ellipse is wider than it is tall, the height
        is equal to twice the semi-minor axis
    angle : float
        orientation angle of the ellipse in radians, measured counter-clockwise
        from the horizontal axis to the ellipse line of apsides.

    Written by Keith LeGrand, October 2022

    """

    def __init__(self, center, width, height, angle):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle

        self._rx = self.width/2
        self._ry = self.height/2
        self._rx_sq = self._rx*self._rx
        self._ry_sq = self._ry*self._ry
        self._ca = np.cos(angle)
        self._sa = np.sin(angle)

        rot = np.array([[self._ca, self._sa], [-self._sa, self._ca]])
        self._xv = rot@np.array([self._rx, 0])
        self._yv = rot@np.array([0, self._ry])

        # compute the (non-axis-aligned) bounding box coordinates
        sa_sq = self._sa*self._sa
        ca_sq = self._ca*self._ca
        xmax = np.sqrt(self._rx_sq*ca_sq + self._ry_sq*sa_sq)
        ymax = np.sqrt(self._rx_sq*sa_sq + self._ry_sq*ca_sq)
        self._lb = self.center - np.array([xmax, ymax])
        self._ub = self.center + np.array([xmax, ymax])

    def contains(self, pts):
        """ check if field-of-view contains point(s)

        Paramaters
        ----------
        pts : arraylike
            (n,2) list of test points

        Returns
        -------
        list
            (n,) list of Booleans

        Written by Keith LeGrand, October 2022
        """
        pts = np.atleast_2d(pts)
        x_m_xc = pts[:, 0] - self.center[0]
        y_m_yc = pts[:, 1] - self.center[1]
        inshape = (
                self._ca*x_m_xc + self._sa*y_m_yc
            )**2*self._ry_sq + (
                -self._sa*x_m_xc + self._ca*y_m_yc
            )**2*self._rx_sq <= self._rx_sq*self._ry_sq
        return inshape

    def apply_linear_transformation(self, A, pre_shift=None, post_shift=None):

        if pre_shift is None:
            pre_shift = np.zeros(2)
        if post_shift is None:
            post_shift = np.zeros(2)

        # compute axes of new transformed ellipse
        xv = A@self._xv
        yv = A@self._yv

        width = 2*np.linalg.norm(xv)
        height = 2*np.linalg.norm(yv)
        angle = np.arctan2(xv[1], xv[0])

        new_center = A@(self.center + pre_shift) + post_shift

        return EllipticalFieldOfView(new_center, width, height, angle)

    def ub(self):
        return self._ub

    def lb(self):
        return self._lb

class ConvexPolyhedralFieldOfView(FieldOfView):
    """ 3D convex polyhedral field-of-view

    Parameters
    ----------
    verts : ndarray
        (n,3) vertices of polyhedron

    """
    def __init__(self, verts):
        self.verts = verts
        self._center = None
        self._hull = Delaunay(verts)
        self._lb = np.min(verts,axis=0)
        self._ub = np.max(verts,axis=0)

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub

    def contains(self, pts):
        return self._hull.find_simplex(pts) >= 0

    def apply_linear_transformation(self, A, pre_shift=None, post_shift=None):

        if pre_shift is None:
            pre_shift = np.zeros(3)
        if post_shift is None:
            post_shift = np.zeros(3)

        new_verts = (self.verts + pre_shift)@A.T + post_shift

        return ConvexPolyhedralFieldOfView(new_verts)

