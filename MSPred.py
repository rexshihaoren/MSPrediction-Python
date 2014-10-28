import pylab as pl
import math
import h5py as hp
import numpy as np
import math as M
from termcolor import colored
from scipy.interpolate import griddata
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from sklearn import metrics, preprocessing
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from naive_bayes import BernoulliNB, GaussianNB, GaussianNB2, MultinomialNB, PoissonNB, MixNB, MixNB2
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib.mlab import rec_drop_fields
from matplotlib import cm
from inspect import getargspec
import sklearn.grid_search as gd
import os
import re
from sklearn import preprocessing
import brewer2mpl
from scipy.stats import itemfreq
from pprint import pprint
paired = brewer2mpl.get_map('Paired', 'qualitative', 10).mpl_colors

# Testing Pipeline:
def testAlgo(clf, X, y, clfName, featureNames, opt = False, param_dict = None, opt_metric = 'roc_auc', n_iter = 5, folds = 10, times = 10,  rs = 0):
    '''An algorithm that output the perdicted y and real y'''
    y_true = []
    y_pred = []
    grids_score = []
    imp = []
    rs = [np.random.randint(1,1000) for i in xrange(times)] if rs == 0 else rs
    for i_CV in range(0, times):
        print "\n###### \CV of testAlgo number " + str(i_CV+1) + " for " + clfName+ "\n###"
        cv = StratifiedKFold(y, n_folds = folds, shuffle = True, random_state = rs[i_CV])
        i_fold = 0
        for train_index, test_index in cv:
            impr_clf, grids_score0, imp0 = fitAlgo(clf, X[train_index], y[train_index], opt, param_dict, opt_metric, n_iter)
            grids_score += [[i_CV, i_fold, grids_score0]]
            imp.append(imp0)
            if (clfName != "LinearRegression"):
                proba = impr_clf.predict_proba(X[test_index])
                y_pred0 = proba[:,1]
            else:
                proba = impr_clf.predict(X[test_index])
                y_pred0 = proba
            y_true0 = y[test_index]
            y_pred.append(y_pred0)
            y_true.append(y_true0)
            i_fold += 1
    # Only rearange format if grids is not []
    grid_score_final = []
    if grids_score[0][2] !=[]:
        fields = grids_score[0][2][0].parameters.keys() + list(['mean_validation_score'])
        fields.append('std')
        l_i = len(grids_score[0][2])
        for grids_score_i in grids_score:
            i_CV = grids_score_i[0]
            i_fold = grids_score_i[1]
            grids2_i = map(lambda x: tuple([i_CV, i_fold] + x.parameters.values()+[x.mean_validation_score,x.cv_validation_scores.std()]),grids_score_i[2])
            datatype = [((['i_CV', 'i_fold'] + fields)[i], np.result_type(grids2_i[0][i]) if not isinstance(grids2_i[0][i], str) else '|S14') for i in range(0, len(fields)+2)]
            grid_score_final_i = np.array(grids2_i, dtype = datatype)
            grid_score_final.append(grid_score_final_i)
        grid_score_final = np.concatenate(grid_score_final)
    else:
        grid_score_final = np.array([])
    if (clfName == 'RandomForest') & opt:
        imp = np.vstack(imp)
        imp = imp.view(dtype=[(i, 'float64') for i in featureNames]).reshape(len(imp),)
    return y_pred, y_true, grid_score_final, imp

# Evaluation pipeline:
def fitAlgo(clf, Xtrain, Ytrain, opt = False, param_dict = None, opt_metric = 'roc_auc', n_iter = 5, n_optFolds = 3):
    '''Return the fitted classifier
    Keyword arguments:
    clf - - base classifier
    Xtrain - - training feature matrix
    Ytrain - - training target array
    param_dict - - the parameter distribution of param, grids space, if opt == False, every element should have length 1
    opt_metric - - optimization metric
    opt - - whether to do optimization or not
    '''

    if opt & (param_dict != None):
        assert(map(lambda x: isinstance(param_dict[x],list), param_dict))
        prod_feature_05 =np.prod([math.pow(len(v),0.5) for x, v in param_dict.iteritems()])
        prod_feature =np.prod([len(v) for x, v in param_dict.iteritems()])
        N_iter = int(np.ceil(prod_feature_05* n_iter / 5 * 1.5))
        N_iter = N_iter if N_iter < prod_feature else prod_feature
        print("Using N_iter = " + str(N_iter))
        if n_iter != 0:
            rs = gd.RandomizedSearchCV(estimator = clf, n_iter = N_iter,
                                    param_distributions = param_dict,
                                    scoring = opt_metric,
                                    refit = True,
                                    n_jobs=-1, cv = n_optFolds, verbose = 1)
        else:
            rs = gd.GridSearchCV(estimator = clf,
                                    param_grid = param_dict,
                                    scoring = opt_metric,
                                    refit = True,
                                    n_jobs=-1, cv = n_optFolds, verbose = 1)
        print "Simulation with num_features=", num_features
        print "max_features=",  param_dict
        rs.fit(Xtrain, Ytrain)
        print "\n### Optimal parameters: ###"
        pprint(rs.best_params_)
        print "####################### \n"

        imp = []
        if clf.__class__.__name__ == "RandomForestClassifier":
            imp = rs.best_estimator_.feature_importances_
        return rs.best_estimator_, rs.grid_scores_, imp
    else:
        if param_dict != None:
            assert(map(lambda x: not isinstance(param_dict[x], list), param_dict))
            for k in param_dict.keys():
                print k
                print opt
                print param_dict
                clf.set_params(k = param_dict[k])
        clf.fit(Xtrain, Ytrain)
        return clf, [], []

###### Meta-functions #######

### Prepare data
def pred_prep(h5_path, obj, target):
    '''A generalized method that could return the desired X and y, based on the file path of the data, the name of the obj, and the target column we are trying to predict.
    Keyword Arguments:
        h5_path: the data path for h5 file
        obj: name of the dataset
        target: the target column name
    '''
    # Make sure "data/obj" and "plots/obj" exist
    if not os.path.exists(data_path+obj):
        os.makedirs(data_path+obj)
    if not os.path.exists(plot_path+obj):
        os.makedirs(plot_path+obj)
    f=hp.File(h5_path, 'r+')
    dataset = f[obj].value
    f.close()
    # Convert Everything to float for easier calculation
    dataset = dataset.astype([(k,float) for k in dataset.dtype.names])
    featureNames = dataset.dtype.fields.keys()
    featureNames.remove(target)
    y = dataset[target]
    newdataset = dataset[featureNames]
    X = newdataset.view((np.float64, len(newdataset.dtype.names)))
    y = y.view((np.float64, 1))
    return X, y, featureNames

### Save data

def fill_2d(X, fill = np.nan):
    '''Function to fill list of array with a certain fill to make it a 2d_array with shape m X n, where m is the number of arrays in the list, n is the maxinum length of array in the list.
    '''
    maxlen = max([len(x) for x in X])
    newX = [np.append(x, np.array([fill] * (maxlen-len(x)))) for x in X]
    return np.array(newX)


def save_output(obj, X, y, featureNames, opt = True, n_CV = 10, n_iter = 2, scaling = False):
    '''Save Output (y_pred, y_true, grids_score, and imp) for this dataframe
    Keyword arguments:
    obj - - dataframe name
    X - - feature matrix
    y - - training target array
    opt - - whether to use parameter optimization, default is True
    '''
    rs = [np.random.randint(1,1000) for i in xrange(n_CV)]

    if scaling:
        X = preprocessing.scale(X)

    for clfName in classifiers1.keys():
        clf = classifiers1[clfName]
        if opt:
            param_dict = param_dist_dict[clfName]
            # print param_dict
        else:
            param_dict = None
        # grids = grid_score_list
        y_pred, y_true, grids_score, imp = testAlgo(clf, X, y, clfName, featureNames, opt, param_dict, times = n_CV, rs=rs, n_iter = n_iter)
        y_pred = fill_2d(y_pred)
        y_true = fill_2d(y_true)
        res_table = getTable(y_pred, y_true, n_CV, n_folds = 10)
        optString = '_opt' if opt else '_noopt'
        scalingString = '_scaled' if scaling else ''
        f = hp.File(data_path + obj + '/' + clfName + optString + scalingString + '.h5', 'w')
        print("Saving output to file for " + clfName)
        f.create_dataset('y_true', data = y_true)
        f.create_dataset('y_pred', data = y_pred)
        f.create_dataset('grids_score', data = grids_score)
        f.create_dataset('imp', data = imp)
        f.create_dataset('fullPredictionTable', data = res_table)
        f.close()

def getTableOneLinerForFun(y_pred, y_true, n_CV, n_folds = 10):
    return np.vstack([np.column_stack((y_pred[i_CV*n_folds + i_fold], y_true[i_CV*n_folds + i_fold], [i_CV]*len(y_pred[i_CV*n_folds + i_fold]), [i_fold]*len(y_pred[i_CV*n_folds + i_fold]))) for i_CV in range(n_CV) for i_fold in range(n_folds)])

def getTable(y_pred, y_true, n_CV, n_folds = 10): #Cleaner
    res = []
    for i_CV in range(n_CV) :
        for i_fold in range(n_folds):
            index_i = i_CV*n_folds + i_fold
            y_pred_i = y_pred[index_i]
            y_true_i = y_true[index_i]
            l_i = len(y_pred_i)
            res.append(np.column_stack((y_pred_i, y_true_i, [i_CV]*l_i, [i_fold]*l_i)))

    return np.vstack(res)

### Plot results

def open_output(clfName, obj, opt):
    ''' Open the ouput file and transform the data into desired format
    Keyword arguments:
        clfName: name of the classifier
        obj: dataframe name
        opt: whether use optimization
    Returns:
        y_true
        y_pred
        grids_score
        imp
    '''
    data_path0 = data_path +obj + '/'
    print("Open output for " + clfName)
    if opt:
        data_path1 = data_path0 + clfName + '_opt.h5'
    else:
        data_path1 = data_path0 + clfName + '_noopt.h5'
    f = hp.File(data_path1, 'r')
    y_pred = f['y_pred'].value
    y_pred = map(lambda x: x[~np.isnan(x)], y_pred)
    y_true = f['y_true'].value
    y_true = map(lambda x: x[~np.isnan(x)], y_true)
    grids_score = f['grids_score'].value
    imp = f['imp'].value
    f.close()
    return y_pred, y_true, grids_score, imp


def compare_clf(clfs, obj, metric = 'roc_auc', opt = False, n_iter=4, folds=4, times=4):
    '''Compare classifiers with mean roc_auc'''
    mean_everything= {}
    mean_everything1 = {}
    for clfName in clfs.keys():
        print clfName
        y_pred, y_true, grids_score, imp = open_output(clfName, obj, opt)
        # Need to check imp's shape maybe
        if (len(imp[0])!= 1) & opt & (clfName == "RandomForest"):
            plot_importances(imp,clfName, obj)
        # Because if opt = Flase, grids_score should be []
        if len(grids_score)>0:
            plotGridPref(grids_score, clfName, obj, n_iter, metric)
        # output roc results and plot folds
        mean_fpr, mean_tpr, mean_auc = plot_roc(y_pred, y_true, clfName, obj, opt)
        mean_everything[clfName] = [mean_fpr, mean_tpr, mean_auc]
        # out pr results and plot folds
        mean_rec, mean_prec, mean_auc1 = plot_pr(y_pred, y_true, clfName, obj, opt)
        mean_everything1[clfName] = [mean_rec, mean_prec, mean_auc1]
    # Compare mean roc score of all clfs
    fig = pl.figure(figsize=(8,6),dpi=150)
    for clfName in  mean_everything:
        [mean_fpr, mean_tpr, mean_auc] = mean_everything[clfName]
        pl.plot(mean_fpr, mean_tpr, lw=3, label = clfName + ' (area = %0.2f)' %mean_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate',fontsize=30)
    pl.ylabel('True Positive Rate',fontsize=30)
    pl.title('Receiver Operating Characteristic',fontsize=25)
    pl.legend(loc='lower right')
    pl.tight_layout()
    if opt:
        save_path = plot_path +obj+'/'+'clf_comparison_'+ 'roc_auc' +'_opt.pdf'
    else:
        save_path = plot_path +obj+'/'+'clf_comparison_'+ 'roc_auc' +'_noopt.pdf'
    fig.savefig(save_path)
    # Compare pr score of all clfs
    fig1 = pl.figure(figsize=(8,6),dpi=150)
    for clfName in  mean_everything1:
        [mean_rec, mean_prec, mean_auc1] = mean_everything1[clfName]
        pl.plot(mean_rec, mean_prec, lw=3, label = clfName + ' (area = %0.2f)' %mean_auc1)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('Recall',fontsize=30)
    pl.ylabel('Precision',fontsize=30)
    pl.title('Precision-Recall',fontsize=25)
    pl.legend(loc='lower right')
    pl.tight_layout()
    if opt:
        save_path = plot_path +obj+'/'+'clf_comparison_'+ 'pr' +'_opt.pdf'
    else:
        save_path = plot_path +obj+'/'+'clf_comparison_'+ 'pr' +'_noopt.pdf'
    fig1.savefig(save_path)

def clf_plot(obj, clfName, opt, featureNames):
    '''
    Plot experiment results
    Keyword Arguments:
        dfName: dataframe name
        clfName: classifier name
    '''
    if opt:
        datapath = data_path +obj+'/'+clfName+'_opt.h5'
    else:
        datapath = data_path +obj+'/'+clfName+'_noopt.h5'
    f=hp.File(datapath, 'r+')
    y_pred = f['y_pred'].value
    y_true = f['y_true'].value
    imp = f['imp'].value
    f.close()
    # Plotting auc_roc and precision_recall
    plot_roc(y_pred, y_true, clfName, obj, opt)
    # Plotting precision_recall
    plot_pr(y_pred, y_true, clfName, obj, opt)
    # Plotting feature_importances
    if opt & (clfName == "RandomForest")& (X.shape[1] != 1):
        plot_importances(imp,clfName, obj)

def plot_roc(y_pred, y_true, clfName, obj, opt, save_sub = True):
    '''Plots the ROC Curve'''
    fig = pl.figure(figsize=(8,6),dpi=150)
    mean_fpr, mean_tpr, mean_auc = plot_unit_prep(y_pred, y_true, 'roc_auc', plotfold = True)
    mean_tpr[-1] = 1.0
    pl.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7),lw=3,label='Random')
    print("ROC AUC: %0.2f" % mean_auc)
    print(clfName)
    pl.plot(mean_fpr, mean_tpr, 'k--',
                             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    pl.xlim([0.0, 1.00])
    pl.ylim([0.0, 1.00])
    pl.xlabel('False Positive Rate',size=30)
    pl.ylabel('True Positive Rate',size=30)
    pl.title('Receiver operating characteristic',size=25)
    pl.legend(loc="lower right")
    pl.tight_layout()
    if opt:
        save_path = plot_path +obj+'/'+clfName+'_roc_opt.pdf'
    else:
        save_path = plot_path +obj+'/'+clfName+'_roc_noopt.pdf'
    if save_sub:
        fig.savefig(save_path)
    # pl.show()
    return mean_fpr, mean_tpr, mean_auc

def plot_pr(y_pred, y_true,clfName, obj, opt, save_sub = True):
    '''Plot the Precision-Recall Curve'''
    fig = pl.figure(figsize=(8,6),dpi=150)
    mean_rec, mean_prec, mean_auc = plot_unit_prep(y_pred, y_true, 'pr')
    print("Precision Recall AUC: %0.2f" % mean_auc)
    pl.clf()
    pl.plot(mean_rec, mean_prec, label='Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall: AUC=%0.2f' % mean_auc)
    pl.legend(loc="lower left")
    if opt:
        save_path = plot_path +obj+'/'+clfName+'_pr_opt.pdf'
    else:
        save_path = plot_path +obj+'/'+clfName+'_pr_noopt.pdf'
    if save_sub:
        fig.savefig(save_path)
    # pl.show()
    return mean_rec, mean_prec, mean_auc

def plot_unit_prep(y_pred, y_true, metric, plotfold = False):
    ''' Prepare mean_x, mean_y array for classifier evaludation, from predicted y and real y.
    Keyword arguments:
    y_pred - - predicted y array
    y_true - - true y target array
    metric - - the metric in use to evaludate classifier
    plotfold - - whether to plot indiviudal fold or not'''
    mean_y= 0.0
    mean_x = np.linspace(0, 1, 1000)
    if len(y_pred)==1:
            print "y_pred length 1"
            folds = zip([y_pred],[y_true])
    else:
            folds = zip(y_pred,y_true)
    for i, (pred,true) in enumerate(folds):
        # pred & true represent each of the experiment folds
        try:
            if metric == 'roc_auc':
                x, y, thresholds = roc_curve(true, pred)
                roc_auc = auc(x, y)
                if plotfold:
                    pl.plot(x, y, color='grey', alpha = 0.15, lw=1.2)
                mean_y += np.interp(mean_x, x, y)
            else:
                #precision-recall 'pr', y is prec, x is rec. rec is a decreasing array
                y, x, thresholds = precision_recall_curve(true, pred)
                # numpy.interp(x, xp, fp, left=None, right=None)
                # xp must be increasing, so reverse x array, which means the corresponding y has to reverse order as well.
                mean_y += np.interp(mean_x, x[::-1], y[::-1])
        except ValueError:
            print true, pred
            print metric +" is currently not available"
        # mean_y[0] = 0.0
    mean_y
    mean_y/= len(folds)
    mean_area = auc(mean_x,mean_y)
    return mean_x, mean_y, mean_area

def plotGridPref(gridscore, clfName, obj , n_iter, metric = 'roc_auc'):
    ''' Plot Grid Performance
    '''
    # data_path = data_path+obj+'/'+clfName+'_opt.h5'
    # f=hp.File(data_path, 'r')
    # gridscore = f['grids_score'].value
    # Get numblocks
    CV = np.unique(gridscore["i_CV"])
    folds = np.unique(gridscore["i_fold"])
    numblocks = len(CV) * len(folds)
    paramNames = gridscore.dtype.fields.keys()
    paramNames.remove("mean_validation_score")
    paramNames.remove("std")
    paramNames.remove("i_CV")
    paramNames.remove("i_fold")
    score = gridscore["mean_validation_score"]
    std = gridscore["std"]
    newgridscore = gridscore[paramNames]
    num_params = len(paramNames)
    ### get index of hit ###
    hitindex = []
    n_iter = len(score)/numblocks
    for k in range(numblocks):
        hit0index = np.argmax(score[k*n_iter: (k+1)*n_iter])
        hitindex.append(k*n_iter+hit0index )

    for m in range(num_params-1):
        i = paramNames[m]
        x = newgridscore[i]
        for n in range(m+1, num_params):
        # for j in list(set(paramNames)- set([i])):
            j = paramNames[n]
            y = newgridscore[j]
            compound = [x,y]
            # Only plot heat map if dtype of all elements of x, y are int or float
            if [True]* len(compound)== map(lambda t: np.issubdtype(t.dtype,  np.float) or np.issubdtype(t.dtype, np.int), compound):
                gridsize = 50
                fig = pl.figure()
                points = np.vstack([x,y]).T
                #####Construct MeshGrids##########
                xnew = np.linspace(max(x), min(x), gridsize)
                ynew = np.linspace(max(y), min(y), gridsize)
                X, Y = np.meshgrid(xnew, ynew)
                #####Interpolate Z on top of MeshGrids#######
                Z = griddata(points, score, (X, Y), method = "cubic")
                z_min = min(score)
                z_max = max(score)
                pl.pcolormesh(X,Y,Z, cmap='RdBu', vmin=z_min, vmax=z_max)
                pl.axis([x.min(), x.max(), y.min(), y.max()])
                pl.xlabel(i, fontsize = 30)
                pl.ylabel(j, fontsize = 30)
                cb = pl.colorbar()
                cb.set_label(metric, fontsize = 30)
                ##### Mark the "hit" points #######
                hitx = x[hitindex]
                hity = y[hitindex]
                pl.plot(hitx, hity, 'rx')
                # Save the plot
                save_path = plot_path +obj+'/'+ clfName +'_' +metric+'_'+ i +'_'+ j+'.pdf'
                fig.savefig(save_path)

def compare_obj_sd(clfName, obj, y_pred, y_true, metric = 'roc_auc',folds = 2, times = 10, opt = True):
    '''Compare different classifiers on single obj, plot mean and sd, based on times and folds'''
    mean_metric = []
    for i in range(times):
        y_true0 = y_true[i*folds: (i+1)*folds]
        y_pred0 = y_pred[i*folds: (i+1)*folds]
        if metric == 'roc_auc':
            _, _, mean_metric0 = plot_roc(y_pred0, y_true0, clfName, obj, opt, save_sub = False)
        else:
            _, _, mean_metric0 = plot_roc(y_pred0, y_true0, clfName, obj, opt, save_sub = False)
        mean_metric.append(mean_metric0)
    return mean_metric

def compare_obj(datasets = [], models = [], opt = True):
    ''' A function that takes a list of datasets and clfNames, so that it compare the model performance (roc_auc, and pr)
    '''
    dsls = ''
    for i in datasets:
        dsls += (i+'_')
    mean_sd_roc_auc = {}
    mean_sd_pr = {}
    for clfName in models:
        # Make sure "plots/clfName" exists
        if not os.path.exists(plot_path + clfName):
            os.makedirs(plot_path + clfName)
        mean_everything= {}
        mean_everything1 = {}
        roc_list = []
        pr_list = []
        clf = classifiers1[clfName]
        param_dict = param_dist_dict[clfName]
        for obj in datasets:
            y_pred, y_true, _, _ = open_output(clfName, obj, opt)
            mean_fpr, mean_tpr, mean_auc = plot_roc(y_pred, y_true, clfName, obj, opt, save_sub = False)
            mean_everything[obj] = [mean_fpr, mean_tpr, mean_auc]

            # out pr results and plot folds
            mean_rec, mean_prec, mean_auc1 = plot_pr(y_pred, y_true, clfName, obj, opt, save_sub = False)
            mean_everything1[obj] = [mean_rec, mean_prec, mean_auc1]

            # sd list
            roc_list0 = compare_obj_sd(clfName, obj, y_pred, y_true, metric = 'roc_auc', opt= opt)
            pr_list0 = compare_obj_sd(clfName, obj, y_pred, y_true, metric = 'pr', opt = opt)
            roc_list.append(roc_list0)
            pr_list.append(pr_list0)

        # Compare mean roc score of all datasets with clf
        fig = pl.figure(figsize=(8,6),dpi=150)
        for obj in  mean_everything:
            [mean_fpr, mean_tpr, mean_auc] = mean_everything[obj]
            pl.plot(mean_fpr, mean_tpr, lw=3, label = obj + ' (area = %0.2f)' %mean_auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate',fontsize=30)
        pl.ylabel('True Positive Rate',fontsize=30)
        pl.title('Receiver Operating Characteristic',fontsize=25)
        pl.legend(loc='lower right')
        pl.tight_layout()
        if opt:
            save_path = plot_path +clfName+'/'+'dataset_comparison_'+ dsls + 'roc_auc' +'_opt.pdf'
        else:
            save_path = plot_path +clfName+'/'+'dataset_comparison_'+ dsls + 'roc_auc' +'_noopt.pdf'
        fig.savefig(save_path)

        # Compare pr score of all clfs
        fig1 = pl.figure(figsize=(8,6),dpi=150)
        for obj in  mean_everything1:
            [mean_rec, mean_prec, mean_auc1] = mean_everything1[obj]
            pl.plot(mean_rec, mean_prec, lw=3, label = obj + ' (area = %0.2f)' %mean_auc1)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('Recall',fontsize=30)
        pl.ylabel('Precision',fontsize=30)
        pl.title('Precision-Recall',fontsize=25)
        pl.legend(loc='lower right')
        pl.tight_layout()
        if opt:
            save_path = plot_path +clfName+'/'+'dataset_comparison_'+ dsls + 'pr' +'_opt.pdf'
        else:
            save_path = plot_path +clfName+'/'+'dataset_comparison_'+ dsls + 'pr' +'_noopt.pdf'
        fig1.savefig(save_path)
        # store sd score of all roc_auc of all clfs
        mean_sd_roc_auc[clfName] = roc_list
        # store sd score of all prs of all clfs
        mean_sd_pr[clfName] = pr_list
    plot_sd(mean_sd_roc_auc, datasets, 'roc_auc', opt)
    plot_sd(mean_sd_pr, datasets, 'pr', opt)


def plot_sd(mean_sd, datasets, metric, opt):
    ''' Plot sd plot with every clfs of different color, comparing performance of different objs
    '''
    dsls = ''
    for i in datasets:
        dsls += (i+'_')
    # number of dataframes in question
    num_df = len(datasets)
    fig = pl.figure(figsize=(8,6),dpi=150)
    for clfName in mean_sd:
        metric_list = mean_sd[clfName]
        metric_list = np.array(metric_list).T
        mean_metric = np.mean(metric_list, axis = 0)
        print "mean_"+metric, mean_metric
        metric_sterr = np.std(metric_list, axis = 0)/np.sqrt(len(metric_list))
        indices = np.argsort(mean_metric)[::-1]
        print "indices", indices
        dfList = []
        for i in range(num_df):
            print i
            dfList.append(datasets[indices[i]])
            print("%d. dataset %s (%.2f)" % (i, datasets[indices[i]], mean_metric[indices[i]]))
        pl.title(metric.upper() + "SD",fontsize=30)
        pl.errorbar(range(num_df), mean_metric[indices], yerr = metric_sterr[indices], label = clfName)
    pl.xticks(range(num_df), dfList, size=15,rotation=90)
    pl.ylabel(metric.upper(),size=30)
    pl.legend(loc='lower right')
    pl.yticks(size=20)
    pl.xlim([-1, num_df])
    # fix_axes()
    pl.tight_layout()
    if opt:
        save_path = plot_path +'dataset_sd_comp_'+ dsls + metric +'_opt.pdf'
    else:
        save_path = plot_path +'dataset_sd_comp_'+ dsls + metric +'_noopt.pdf'
    fig.savefig(save_path)


### Functions to analyze different models, plot importances for random forest, coefficients for logistic and linear regressions, and fit pdf plot for Bayes

def plot_importances(imp, clfName, obj):
    featureNames = list(imp.dtype.names)
    # imp=np.vstack(imp)
    imp = imp.view(np.float64).reshape(imp.shape + (-1,))
    mean_importance = np.mean(imp,axis=0)
    std_importance = np.std(imp,axis=0)
    indices = np.argsort(mean_importance)[::-1]
    print indices
    print featureNames
    featureList = []
    num_features = len(featureNames)
    print("Feature ranking:")
    for f in range(num_features):
        featureList.append(featureNames[indices[f]])
        print("%d. feature %s (%.2f)" % (f, featureNames[indices[f]], mean_importance[indices[f]]))
    fig = pl.figure(figsize=(8,6),dpi=150)
    pl.title("Feature importances",fontsize=30)
    pl.bar(range(num_features), mean_importance[indices],
            yerr = std_importance[indices], color=paired[0], align="center",
            edgecolor=paired[0],ecolor=paired[1])
    pl.xticks(range(num_features), featureList, size=15,rotation=90)
    pl.ylabel("Importance",size=30)
    pl.yticks(size=20)
    pl.xlim([-1, num_features])
    # fix_axes()
    pl.tight_layout()
    save_path = plot_path +obj+'/'+clfName+'_feature_importances.pdf'
    fig.savefig(save_path)

def plotGaussian(X, y, obj, featureNames):
    """Plot Gausian fit on top of X.
    """
    save_path = plot_path +obj+'/'+'BayesGaussian2'
    clf = classifiers["BayesGaussian2"]
    clf,_,_ = fitAlgo(clf, X,y, opt= True, param_dict = param_dist_dict["BayesGaussian2"])
    unique_y = np.unique(y)
    theta = clf.theta_
    sigma = clf.sigma_
    class_prior = clf.class_prior_
    norm_func = lambda x, sigma, theta: 1 if np.isnan(x) else -0.5 * np.log(2 * np.pi*sigma) - 0.5 * ((x - theta)**2/sigma)
    norm_func = np.vectorize(norm_func)
    n_samples = X.shape[0]
    for j in range(X.shape[1]):
        fcol = X[:,j]
        jfeature = featureNames[j]
        jpath = save_path +'_'+jfeature+'.pdf'
        fig = pl.figure(figsize=(8,6),dpi=150)
        for i, y_i in enumerate(unique_y):
            fcoli = fcol[y == y_i]
            itfreq = itemfreq(fcoli)
            uniqueVars = itfreq[:,0]
            freq = itfreq[:,1]
            freq = freq/sum(freq)
            the = theta[i, j]
            sig = sigma[i,j]
            pred = np.exp(norm_func(uniqueVars, sig, the))
            pl.plot(uniqueVars, pred, label= str(y_i)+'_model')
            pl.plot(uniqueVars, freq, label= str(y_i) +'_true')
        pl.xlabel(jfeature)
        pl.ylabel("density")
        pl.legend(loc='best')
        pl.tight_layout()
        # pl.show()
        fig.savefig(jpath)

def plotMixNB(X, y, obj, featureNames, whichMix):
    """Plot MixNB fit on top of X.
    """
    save_path = plot_path +obj+'/'+whichMix
    clf = classifiers[whichMix]
    clf,_,_ = fitAlgo(clf, X,y, opt= True, param_dict = param_dist_dict[whichMix])
    unique_y = np.unique(y)
    # norm_func = lambda x, sigma, theta: 1 if np.isnan(x) else -0.5 * np.log(2 * np.pi*sigma) - 0.5 * ((x - theta)**2/sigma)
    # norm_func = np.vectorize(norm_func)
    n_samples = X.shape[0]
    for j in range(X.shape[1]):
        fcol = X[:,j]
        optmodel = clf.optmodels[:,j]
        distname = clf.distnames[j]
        jfeature = featureNames[j]
        jpath = save_path +'_'+jfeature+'.pdf'
        fig = pl.figure(figsize=(8,6),dpi=150)
        for i, y_i in enumerate(unique_y):
            fcoli = fcol[y == y_i]
            itfreq = itemfreq(fcoli)
            uniqueVars = itfreq[:,0]
            freq = itfreq[:,1]
            freq = freq/sum(freq)
            pred = np.exp(optmodel[i](uniqueVars))
            pl.plot(uniqueVars, pred, label= str(y_i)+'_model')
            pl.plot(uniqueVars, freq, label= str(y_i) +'_true')
        pl.xlabel(jfeature)
        pl.ylabel("density")
        pl.title(distname)
        pl.legend(loc='best')
        pl.tight_layout()
        fig.savefig(jpath)

def plotCoeff(X, y, obj, featureNames, whichReg):
    """ Plot Regression's Coeff
    """
    clf = classifiers[whichReg]
    clf,_,_ = fitAlgo(clf, X,y, opt= True, param_dict = param_dist_dict[whichReg])
    if whichReg == "LogisticRegression":
        coeff = np.absolute(clf.coef_[0])
    else:
        coeff = np.absolute(clf.coef_)
    print coeff
    indices = np.argsort(coeff)[::-1]
    print indices
    print featureNames
    featureList = []
    num_features = len(featureNames)
    print("Feature ranking:")
    for f in range(num_features):
        featureList.append(featureNames[indices[f]])
        print("%d. feature %s (%.2f)" % (f, featureNames[indices[f]], coeff[indices[f]]))
    fig = pl.figure(figsize=(8,6),dpi=150)
    pl.title("Feature importances",fontsize=30)
    pl.bar(range(num_features), coeff[indices], color=paired[0], align="center",
            edgecolor=paired[0],ecolor=paired[1])
    pl.xticks(range(num_features), featureList, size=15,rotation=90)
    pl.ylabel("Importance",size=30)
    pl.yticks(size=20)
    pl.xlim([-1, num_features])
    pl.tight_layout()
    save_path = plot_path + obj+'/'+whichReg+'_feature_importances.pdf'
    fig.savefig(save_path)

def param_sweeping(clf, obj, X, y, param_dist, metric, param, clfName):
    '''Plot a parameter sweeping (ps) curve with the param_dist as a axis, and the scoring based on metric as y axis.
    Keyword arguments:
    clf - - classifier
    X - - feature matrix
    y - - target array
    param - - a parameter of the classifier
    param_dist - - the parameter distribution of param
    clfName - - the name of the classifier
    metric - - the metric we use to evaluate the performance of the classifiers
    obj - - the name of the dataset we are using'''
    scores = []
    for i in param_dist:
        y_true = []
        y_pred = []
        # new classifer each iteration
        newclf = eval("clf.set_params("+ param + "= i)")
        y_pred, y_true, grids_score, amp = testAlgo(newclf, X, y, clfName, featureNames)
        mean_fpr, mean_tpr, mean_auc = plot_unit_prep(y_pred, y_true, metric)
        scores.append(mean_auc)
        print("Area under the ROC curve : %f" % mean_auc)
    fig = pl.figure(figsize=(8,6),dpi=150)
    paramdist_len = len(param_dist)
    pl.plot(range(paramdist_len), scores, label = 'Parameter Sweeping Curve')
    pl.xticks(range(paramdist_len), param_dist, size = 15, rotation = 45)
    pl.xlabel(param.upper(),fontsize=30)
    pl.ylabel(metric.upper(),fontsize=30)
    pl.title('Parameter Sweeping Curve',fontsize=25)
    pl.legend(loc='lower right')
    pl.tight_layout()
    fig.savefig(plot_path + obj+'/'+ clfName +'_' + param +'_'+'ps.pdf')
    pl.show()

### Question Helpers ###
#

def param_sweep_select(clf):
    '''Asking user the specifics about parameter sweeping'''
    arglist = getargspec(clf.__init__).args
    arglist.remove('self')
    param = raw_input("What parameters would you choose?\n" + str(arglist)+": ")
    s = raw_input("Define the range of parameter you would like to sweep?\n")
    param_dist = eval(s)
    metric = raw_input("What metric would you like to use to evaluate the classifier?\n")
    return param, param_dist, metric

def choose_clf(classifiers):
    print ("Which Classifer would you like to use?")
    print ("Options:")
    clfName = raw_input(str(classifiers.keys())+"\n")
    clf = classifiers[clfName]
    return clf, clfName

def comp_obj_select():
    print("Which following datasets do you wanna use?")
    for i in objs:
        print i
    s= raw_input("Please input in a list format, e.g. [\"Core\", \"Core_Imp\"]")
    datasets = eval(s)
    print("Which following classifiers do you wanna use?")
    for i in classifiers1.keys():
        print i
    s = raw_input("Please input in a list format, e.g. [\"RandomForest\", \"LogisticRegression\"]")
    models = eval(s)
    comp_obj_opt = raw_input("With optimization (Y\N) ? ")
    return datasets, models, (comp_obj_opt == 'Y')


def save_output_select():
    '''
    Choose dataset and parameters to generate y_pred, y_true, grids_score and imp
    '''
    # Choose the datasets
    e = True
    while e:
        e = False
        print("Choose datasets from the following in a list format. e.g. ['Core', 'Core_Imp']")
        for obj in objs:
            # Check whether the ouput data for obj has been generated
            if os.path.exists(general_path + 'data/' + obj):
                print(obj + "(related output has already been generated)")
            else:
                print(obj)
        cc = raw_input("--->")
        choices = objs if cc == "" else eval(cc)
        for obj in choices:
            if obj not in objs:
                print (obj + "No such dataset exists. Please type again the list of datasets... \n")
                e = True
    # launch the computations after a last switch:
    cp = "bloup"
    while cp not in ["", "Complicated"]:
        cp = raw_input("Do you want to answer a lot of - useless ? - different questions for each dataset that forces you to stare at your terminal the whole time " +
        "or would you prefer just to go on with the fitting of all the models for the sected datasets? \n (Answer 'Complicated' for the first option or just press return for the simple option)\n -->")
    if cp == "Complicated":
        for obj in choices:
            save_output_single(obj)
    else:
        print("Last couple of questions:")
        opt = raw_input("Do you want optimisation on all of the model fittings? (return or 'Y' for Yes) \n -->") in ["", "Yes", "Y"]
        n_CV = raw_input("How many Cross-Validations should be done for the validation of the algorithms? (return for default = 10) \n -->")
        n_CV = 10 if n_CV == "" else int(n_CV)
        b_scaling = raw_input("Do you want to scale the imput data? (return or 'Y' for Yes) \n -->") in ["", "Yes", "Y"]
        # if opt: #Not really relevant since the n_iter is now precomputed.
        #     n_iter = raw_input("How many iteration should be done when optimizing the algorithms? (return for default = 5) \n -->")
        #     n_iter = 5 if n_iter == "" else int(n_iter)

        for obj in choices:
            print ("Saving output for " + obj)
            target = 'ModEDSS'
            # global featureNames
            X, y, featureNames = pred_prep(h5_path, obj, target)
            # global num_features
            try:
                num_features = X.shape[1]
            except IndexError:
                X = X.reshape(X.shape[0], 1)
                num_features = 1
            random_forest_params["max_features"] = range(2, num_features + 1)
            save_output(obj, X, y, featureNames, opt = opt, n_CV=n_CV, scaling = b_scaling)

def save_output_single(obj):
    print ("Saving output for " + obj)
    target = 'ModEDSS'
    # global featureNames
    X, y, featureNames = pred_prep(h5_path, obj, target)
    # global num_features
    try:
        num_features = X.shape[1]
    except IndexError:
        X = X.reshape(X.shape[0], 1)
        num_features = 1
    random_forest_params["max_features"] = range(2, num_features + 1)
    ### Importances/ Coefficient of different params
    plot_gaussian = raw_input("Plot Gaussian2 Fit? (Y/N)")
    if plot_gaussian == "Y":
        plotGaussian(X, y, obj, featureNames)
    plot_MixNB = raw_input("Plot MixNB Fit? (Y/N)")
    if plot_MixNB == "Y":
        whichMix = raw_input("MixNB or MixNB2? (BayesMixed/BayesMixed2/Both)")
        if (whichMix == "Both") or (whichMix == "both"):
            plotMixNB(X, y, obj, featureNames, whichMix = "BayesMixed")
            plotMixNB(X, y, obj, featureNames, whichMix = "BayesMixed2")
        else:
            plotMixNB(X, y, obj, featureNames, whichMix)
    reg_Imp = raw_input("Plot Regression's Importance? (Y/N)")
    if reg_Imp == "Y":
        whichReg = raw_input("LogisticRegression/LinearRegression/Both? ")
        if (whichReg == "Both") or (whichReg == "both"):
            plotCoeff(X, y, obj, featureNames, whichReg = "LogisticRegression")
            plotCoeff(X, y, obj, featureNames, whichReg = "LinearRegression")
        else:
            plotCoeff(X, y, obj, featureNames, whichReg)
    # output y_pred, y_true, grids_score and imp
    saveoutput = raw_input("Do you want to save the output (y_pred, y_true, gridscore (plus importance for RandomForest)) for " + obj + "? (Y/N)")
    if saveoutput == "Y":
        output_opt = raw_input("Save output with parameter optimization? (Yes/ No/ Both)")
        if (output_opt == 'Yes'):
            save_output(obj, X, y, featureNames, opt = True)
        elif (output_opt == 'No'):
            save_output(obj, X, y, featureNames, opt = False)
        else:
            save_output(obj, X, y, featureNames, opt = True)
            save_output(obj, X, y, featureNames, opt = False)
    # Single clf analysis for obj
    sin_ana = raw_input("Single clf analysis for "+ obj + " ? (Y\N)")
    if (sin_ana == 'Y'):
        clf, clfName = choose_clf(classifiers1)
        param_sweep = raw_input("Parameter Sweeping? (Y/N) ")
        if param_sweep == "Y" or param_sweep == "y":
            param, param_dist, metric = param_sweep_select(clf)
            param_sweeping(clf, obj, X, y, param_dist, metric, param, clfName)
        else:
            print ("Your only choice now is to plot ROC and PR curves for "+clfName+" classifier")
            # Asking whether to optimize
            opt = raw_input("Optimization? (Y/N)")
            opt = (opt== "Y" or opt == "y")
            if opt:
                param_dist = param_dist_dict[clfName]
            else:
                param_dist = None
            clf_plot(obj, clfName, opt, featureNames)

def com_clf_select():
    existobjs = []
    print("Here are the existing datasets with output saved: \n")
    for obj in objs:
        # Check whether the ouput data for obj has been generated
        if os.path.exists("./data/"+obj):
        # if os.path.exists(data_path + "/" + obj):
            existobjs.append(obj)
            print(obj)
    obj = raw_input('Which dataset would you choose from above list?')
    while obj not in existobjs:
        obj = raw_input('Which dataset would you choose from above list?')
    com_clf_opt = raw_input ("With optimization? (Y/N)")
    compare_clf(classifiers1, obj, metric = 'roc_auc', opt = (com_clf_opt == 'Y'))


def path_finder():
    h5name = " "
    while h5name not in ["PredData", "PredData_Impr0-4"]:
        h5name = raw_input("Which h5 file? do you want to use (PredData or PredData_Impr0-4)")
    global general_path, h5_path, data_path, plot_path
    general_path = './' + h5name + '/'
    h5_path = './' + h5name + '/' + h5name + '.h5'
    data_path = general_path + 'data/'
    plot_path = general_path + 'plots/'
    f = hp.File(h5_path, 'r')
    global objs
    objs = [str(i) for i in f.keys()]
    f.close()


######## Global Parameters #######

# Possible Classifiers
classifiers = {"LogisticRegression": LogisticRegression(),
                    "KNN": KNeighborsClassifier(),
                    "BayesBernoulli": BernoulliNB(),
                    "BayesMultinomial": MultinomialNB(),
                    "BayesGaussian": GaussianNB(),
                    "BayesPoisson": PoissonNB(),
                    "BayesGaussian2":GaussianNB2(),
                    "SVM": SVC(probability = True),
                    "RandomForest": RandomForestClassifier(),
                    "LinearRegression": LinearRegression(),
                    "BayesMixed": MixNB(),
                    "BayesMixed2": MixNB2()
                    }
# Classifiers actually considered
# classifiers1 = {"LogisticRegression": LogisticRegression(),
#                     "BayesBernoulli": BernoulliNB(),
#                     "BayesGaussian": GaussianNB(),
#                     "BayesGaussian2":GaussianNB2(),
#                     "RandomForest": RandomForestClassifier(),
#                     "LinearRegression": LinearRegression(),
#                     "BayesMixed": MixNB(),
#                     "BayesMixed2": MixNB2()
#                     }
# Only for local testing at Rex's machine
classifiers1 = {"LogisticRegression": LogisticRegression(),
                    "RandomForest": RandomForestClassifier(),
                    "BayesMixed2": MixNB2()
                    }
# dictionaries of different classifiers, these can be eyeballed from my parameter sweeping curve
num_features = 6
random_forest_params = {"n_estimators": [50,100,200,300],
              "max_features": range(2, num_features + 1),
              # "min_samples_split": [2, 3,4,6,8,10],
              # "min_samples_leaf": [5,10,15],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
# ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling', 'class_weight', 'random_state']
logistic_regression_params = {"penalty":['l1','l2'],
                    "C": np.linspace(.1, 1, 11),
                    "fit_intercept":[True],#, False],
                    "intercept_scaling":np.linspace(.1, 1, 11),
                    "tol":[1e-4, 1e-5]}#, 1e-6]}
# ['n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p', 'metric']
knn_params= {"n_neighbors":range(1,6),
                "algorithm":['auto', 'ball_tree', 'kd_tree'],
                "leaf_size":range(25,30),
                "p":range(1,3)}
# ['alpha', 'binarize', 'fit_prior', 'class_prior']
bayesian_bernoulli_params= {"alpha": np.linspace(.1, 1, 11),
                "binarize": np.linspace(.1, 1, 11)}
# ['alpha', 'binarize', 'fit_prior', 'class_prior']
bayesian_multi_params= {"alpha": np.linspace(.1, 1, 10)}
# ['alpha', 'binarize', 'fit_prior', 'class_prior']
bayesian_gaussian_params= None
bayesian_gaussian2_params= None
bayesian_poisson_params = None
bayesian_mixed_params = None
bayesian_mixed2_params = None
# ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking', 'probability', 'tol', 'cache_size', 'class_weight', 'verbose', 'max_iter', 'random_state']
svm_params = {
    "C" : [0.1,0.3,1,3,10000],
    "kernel": ["linear", "poly"] #'rbf'
}
# ['fit_intercept', 'normalize', 'copy_X']
linear_regression_params = {#"fit_intercept":[True, False], # False doesn't make sense here.
                    "normalize": [True, False]}
# a dictionary storing the param_dist for different classifiers
param_dist_dict = {"LogisticRegression": logistic_regression_params,
                "KNN":knn_params,
                "BayesBernoulli": bayesian_bernoulli_params,
                "BayesMultinomial": bayesian_multi_params,
                "BayesGaussian": bayesian_gaussian_params,
                "BayesGaussian2":bayesian_gaussian2_params,
                "BayesPoisson": bayesian_poisson_params,
                "SVM":svm_params,
                "RandomForest":random_forest_params,
                "LinearRegression":linear_regression_params,
                "BayesMixed": bayesian_mixed_params,
                "BayesMixed2": bayesian_mixed2_params
                }

def main():
    '''Some basic setup for prediction'''
    #########QUESTIONS################################################
    # Ask which h5 file to use (PredData or PredData_Impr0-4)
    path_finder()
    option = raw_input("Launch Computation (a) or display result (b)? (a/b) ")
    if (option == 'a'):
        save_output_select()
    else:
    # Compre datasets
        comp_obj = raw_input("Do you want to compare different datasets? (Y/N)")
        if (comp_obj == 'Y'):
            datasets, models, opt = comp_obj_select()
            compare_obj(datasets, models, opt)
        # Compare classifiers
        com_clf = raw_input("Compare classifiers? (Y/N) ")
        if (com_clf == "Y"):
            com_clf_select()

if __name__ == "__main__":
    main()
