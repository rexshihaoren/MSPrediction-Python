from __future__ import division, print_function, absolute_import
import numpy as np
from scipy import interpolate
from scipy.interpolate.interpnd import LinearNDInterpolator, NDInterpolatorBase, \
     CloughTocher2DInterpolator, _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from scipy.misc import factorial

# __all__ = ['griddata', 'NearestNDInterpolator', 'LinearNDInterpolator',
#            'CloughTocher2DInterpolator']
def griddata(points, values, xi, method='linear', fill_value=np.nan, tol = 1e-6):
    """modified griddata plus tol arg"""
    points = _ndim_coords_from_arrays(points)

    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]

    if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
        from interpolate import interp1d
        points = points.ravel()
        if isinstance(xi, tuple):
            if len(xi) != 1:
                raise ValueError("invalid number of dimensions in xi")
            xi, = xi
        # Sort points/values together, necessary as input for interp1d
        idx = np.argsort(points)
        points = points[idx]
        values = values[idx]
        ip = interp1d(points, values, kind=method, axis=0, bounds_error=False,
                      fill_value=fill_value)
        return ip(xi)
    elif method == 'nearest':
        ip = NearestNDInterpolator(points, values)
        return ip(xi)
    elif method == 'linear':
        ip = LinearNDInterpolator(points, values, fill_value=fill_value)
        return ip(xi)
    elif method == 'cubic' and ndim == 2:
        ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value, tol = tol)
        return ip(xi)
    else:
        raise ValueError("Unknown interpolation method %r for "
                         "%d dimensional data" % (method, ndim))