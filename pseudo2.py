import pylab as pl
import h5py as hp
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib.mlab import rec_drop_fields
import itertools
from inspect import getargspec
import random
from sklearn.grid_search import RandomizedSearchCV
import pdb

# That's almost perfect. Your code will do the job, but it doesn't really impement any modularity, and there is still some redudancy (See RandomizedSearchCV used two times). 
# Here is a proposition: ( I use declarations in the interfaces to make it clearer)

# 1. Evaluation pipeline: 

#     List:final_params = fitAlgo(Classifier:clf, Matrix:Xtrain, Vector:Ytrain, Dictionnary: optimization_params, Scorer:optimization_metric, Boolean:opt_bool) This only fits the algorithm and return the parameters
#     Vector:Y_pred = evalAlgo(Classifier:clf, Matrix:X, List:final_params) NO CV explicitely done here ! 

#  Called by fitAlgo:
#     RandomizedSearchCV
#     fit



def fitAlgo(clf, Xtrain, Ytrain, opt = False, param_dict = None, opt_metric = 'roc_auc', n_iter = 5):
    '''Return the fitted classifier
    Keyword arguments:
    clf - - base classifier
    Xtrain - - training feature matrix
    Ytrain - - training target array
    param_dict - - the parameter distribution of param, grids space, if opt == False, every element should have length 1
    opt_metric - - optimization metric
    opt - - whether to do optimization or not
    '''
    if opt:
        assert(map(lambda x: isinstance(param_dict[x],list), param_dict))
        rs = RandomizedSearchCV(estimator = clf, n_iter = 10,
                                param_distributions = param_dict,
                                scoring = opt_metric,
                                refit = True,
                                n_jobs=-1, cv = 5, verbose = 3)

        rs.fit(Xtrain, Ytrain)
        return rs.best_estimator_
    else:
        assert(map(lambda x: not isinstance(param_dict[x], list), param_dict))
        # for k in param_dict.keys():
        #     clf.set_params(k = param_dict[k])
        clf.fit(Xtrain, Ytrain)
        return clf

# 2. Testing Pipeline:
    
#     List:results = testAlgo(Integer:folds, Integer:times, String:method="CV" , kargs[Anything that goes into fitAlgo] ) # Here, you call the testing wrapper that execute your folds-fold cross-validation averaged over times time. The list objects "results" will encapsulate anything you will need later (the performance averaged from all the calls, but also the different results for each folds, etc....)
    
#     plot = plot(List:results, String:plot_type) # Here you take your validation results and you plot them accordingly (performance-recall, ROC, ...)

def testAlgo(clf, X, y, opt = False, param_dict = None, opt_metric = 'roc_auc', n_iter = 5, folds = 10, times = 10):
    '''
    What is this method?
    '''
    y_true = []
    y_pred = []
    # score = []
    for i in range(0, times):
        rs = random.randint(1,1000)
        cv = KFold(len(y), n_folds = folds, shuffle = True, random_state = rs)
        for train_index, test_index in cv:
            impr_clf  = fitAlgo(clf, X[train_index], y[train_index], opt, param_dict, opt_metric, n_iter)
            proba = impr_clf.predict_proba(X[test_index])
            y_pred0 = proba[:,1]
            y_true0 = y[test_index]
            y_pred.append(y_pred0)
            y_true.append(y_true0)
            # score.append(metric(y_pred0, y_true0))
    return y_pred, y_true

def plot(y_pred, y_true, plot_type):
    fig = pl.figure(figsize=(8,6),dpi=150)
    mean_y= 0.0
    mean_x = np.linspace(0, 1, 100)
    if len(y_pred)==1:
            folds = zip([y_pred],[y_true])
    else:
            folds = zip(y_pred,y_true)
    for i, (pred,true) in enumerate(folds):
        # pred & true represent each of the experiment folds
        try:
            if plot_type == 'roc':
                x, y, thresholds = roc_curve(true, pred)
                roc_auc = auc(x, y)
                pl.plot(x, y, color='grey', alpha = 0.15, lw=1.2)
                mean_y += np.interp(mean_x, x, y)
            else:
                #precision-recall 'pr', y is prec, x is rec. rec is a decreasing array
                y, x, thresholds = precision_recall_curve(true, pred)
                # numpy.interp(x, xp, fp, left=None, right=None)
                # xp must be increasing, so reverse x array, which means the corresponding y has to reverse order as well.
                mean_y += np.interp(mean_x, x[::-1], y[::-1])
        except ValueError:
            print metric +" is currently not available"
        # mean_y[0] = 0.0
    mean_y/= len(folds)
    mean_area = auc(mean_x,mean_y)
    print("ROC AUC: %0.2f" % mean_area)
    pl.plot(mean_x, mean_y, 'k--',
                             label='Mean ROC (area = %0.2f)' % mean_area, lw=2)
    pl.xlim([0.0, 1.00])
    pl.ylim([0.0, 1.00])
    pl.xlabel('False Positive Rate',size=30)
    pl.ylabel('True Positive Rate',size=30)
    pl.title('Receiver operating characteristic',size=25)
    pl.legend(loc="lower right")
    pl.tight_layout()
    # fig.savefig('plots/'+obj+'/'+clfName+'_roc_auc.pdf')
    pl.show()


# 3. Meta-functions
# You can then wrap the different stuff above with scripts that actually ask you for the kind of classifier you wnat, if you want to compare them, etc... But it's a piece of cake if you have buit the other bricks before !


def simpleTest():
    data_path = '../MSPrediction-R/Data Scripts/data/predData.h5'
    obj = 'fam2'
    target = 'EnjoyLife'
    ########## Can use raw_input instead as well#######################
    X, y, featureNames = pred_prep(data_path, obj, target)
    clf = RandomForestClassifier()
    RFParamDist = {"n_estimators": 8,
              "max_features": 6,
              "min_samples_split": 15,
              "min_samples_leaf": 15,
              "bootstrap": True,
              "criterion": "entropy"}
    y_pred, y_true = testAlgo(clf, X, y, opt = False, param_dict = RFParamDist)
    plot(y_pred, y_true, 'roc')

def optTest():
    data_path = '../MSPrediction-R/Data Scripts/data/predData.h5'
    obj = 'fam2'
    target = 'EnjoyLife'
    ########## Can use raw_input instead as well#######################
    X, y, featureNames = pred_prep(data_path, obj, target)
    clf = RandomForestClassifier()
    num_features = X.shape[1]
    RFParamDist = {"n_estimators": range(1,30),
                  "max_features": range(1, num_features + 1),
                  "min_samples_split": range(1, 30),
                  "min_samples_leaf": range(1, 30),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    y_pred, y_true = testAlgo(clf, X, y, opt = True, param_dict = RFParamDist, opt_metric = 'roc_auc', n_iter = 10, folds = 5, times = 2)
    # y_pred, y_true = testAlgo(clf, X, y, True, RFParamDist)
    plot(y_pred, y_true, 'roc')

def pred_prep(data_path, obj, target):
    '''A generalized method that could return the desired X and y, based on the file path of the data, the name of the obj, and the target column we are trying to predict.'''
    f=hp.File(data_path, 'r+')
    dataset = f[obj].value
    # Convert Everything to float for easier calculation
    dataset = dataset.astype([(k,float) for k in dataset.dtype.names])
    featureNames = dataset.dtype.fields.keys()
    featureNames.remove(target)
    y = dataset[target]
    newdataset = dataset[featureNames]
    X = newdataset.view((np.float64, len(newdataset.dtype.names)))
    y = y.view((np.float64, 1))
    return X, y, featureNames
