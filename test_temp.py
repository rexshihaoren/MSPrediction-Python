from __future__ import division, print_function, absolute_import
import numpy as np
from scipy import interpolate
from scipy.interpolate.interpnd import LinearNDInterpolator, NDInterpolatorBase, \
     CloughTocher2DInterpolator, _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from scipy.misc import factorial
from scipy.optimize import fmin, fmin_bfgs, fminbound
from scipy.stats import chisquare, itemfreq
import EnjoyLifePred as ELP
from statsmodels.discrete.discrete_model import Poisson as pois
from sklearn.naive_bayes import BernoulliNB, GaussianNB, GaussianNB2, MultinomialNB, PoissonNB, MixNB
# import helper
# clf = helper.testMixNB()
# clf3 = clf[3]

data_path = '../MSPrediction-R/Data Scripts/data/predData.h5'
obj = 'diagnostatic'
target = 'ModEDSS'
X, y, featureNames = ELP.pred_prep(data_path, obj, target)
ELP.plotMixNB(X, y, obj, featureNames)