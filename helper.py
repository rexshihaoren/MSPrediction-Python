from __future__ import division, print_function, absolute_import
import numpy as np
from scipy import interpolate
from scipy.interpolate.interpnd import LinearNDInterpolator, NDInterpolatorBase, \
     CloughTocher2DInterpolator, _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from scipy.misc import factorial
from scipy.stats import chisquare, itemfreq
from scipy.optimize import fmin, fmin_bfgs, fminbound
import EnjoyLifePred as ELP
from statsmodels.discrete.discrete_model import Poisson as pois
from naive_bayes import BernoulliNB, GaussianNB, GaussianNB2, MultinomialNB, PoissonNB, MixNB
import os
os.environ['R_HOME'] = "/opt/local/Library/Frameworks/R.framework/Resources"
from rpy2.robjects.packages import importr
MASS = importr('MASS')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

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

def argmax(fucntion):
    """a function with only one variable, we out put the argmax"""
    x = fmin(- function)
    return x

def funcadd(a, b):
    return lambda lamb: a(lamb)+ b(lamb)

def testargmax():
    data_path = '../MSPrediction-R/Data Scripts/data/predData.h5'
    obj = 'fam2_bin'
    target = 'EnjoyLife'
    # X, y, featureNames = ELP.pred_prep(data_path, obj, target)
    X = np.random.randint(5, size= (10,6))
    # loglike for Poisson
    func = lambda lamb, x: x* np.log(lamb)- lamb - np.log(factorial(x))
    # func = lambda lamb, x: facto(x) - lamb + x
    vecfunc = np.vectorize(func)
    X = X.astype(int)
    # X = np.random.poisson(5,100)
    [nrow, ncol] = X.shape
    negloglikemat = np.zeros(X.shape, dtype = object)
    negloglikecolsum = np.zeros(ncol, dtype = object)
    la = np.zeros(X.shape, dtype = object)
    # lambinit = np.zeros(ncol)
    for j in range(0, ncol):
        negloglikemat[:,j] = map(lambda lamb: lambda x: -vecfunc(lamb,x), X[:,j])
        temp = lambda lamb: 0
        for i, item in enumerate(negloglikemat[:,j]):
            temp = funcadd(item, temp)
        negloglikecolsum[j] = temp
        lambopt = fmin(func = negloglikecolsum[j], xtol=0.0001, ftol=0.0001, x0 = 0.0)
        la[:, j] = lambopt
    return X, negloglikemat, negloglikecolsum, la


def testMixNB():
    import EnjoyLifePred as ELP
    data_path = '../MSPrediction-R/Data Scripts/data/predData.h5'
    obj = 'EDSS'
    target = 'ModEDSS'
    X, y, featureNames = ELP.pred_prep(data_path, obj, target)
    from naive_bayes import BernoulliNB, GaussianNB, GaussianNB2, MultinomialNB, PoissonNB, MixNB
    clf = GaussianNB2()
    clf.fit(X,y)
    clf = {}
    clf0 = GaussianNB()
    clf1 = GaussianNB2()
    clf2 = PoissonNB()
    clf3 = MixNB()
    clf[0] = clf0
    clf[1] = clf1
    clf[2] = clf2
    clf[3] = clf3
    for k in clf.keys():
        clf[k].fit(X,y)
    return clf
def plotCoeff(X, y, obj, featureNames, whichReg):
    """ Plot Regression's Coeff
    """
    clf = classifier[whichReg]
    clf.fit(X,y)
    coeff = clf.coef_
    indices = np.argsort(coeff)[::-1]
    featureList = []
    # num_features = len(featureNames)
    print("Feature ranking:")
    for f in range(num_features):
        featureList.append(featureNames[indices[f]])
        print("%d. feature %s (%.2f)" % (f, featureNames[indices[f]], coeff[indices[f]]))
    fig = pl.figure(figsize=(8,6),dpi=150)
    pl.title("Feature importances",fontsize=30)
    pl.bar(range(num_features), coeff[indices],
            yerr = std_importance[indices], color=paired[0], align="center",
            edgecolor=paired[0],ecolor=paired[1])
    pl.xticks(range(num_features), featureList, size=15,rotation=90)
    pl.ylabel("Importance",size=30)
    pl.yticks(size=20)
    pl.xlim([-1, num_features])
    # fix_axes()
    pl.tight_layout()
    save_path = 'plots/'+obj+'/'+WhichReg+'_feature_importances.pdf'
    fig.savefig(save_path)



