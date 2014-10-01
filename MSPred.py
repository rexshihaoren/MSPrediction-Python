import pylab as pl
import h5py as hp
import numpy as np
import math as M
from termcolor import colored
from scipy.interpolate import griddata
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from naive_bayes import BernoulliNB, GaussianNB, GaussianNB2, MultinomialNB, PoissonNB, MixNB, MixNB2
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib.mlab import rec_drop_fields
from matplotlib import cm
from inspect import getargspec
from sklearn.grid_search import RandomizedSearchCV
import os
import re
from sklearn import preprocessing
import brewer2mpl
from scipy.stats import itemfreq
paired = brewer2mpl.get_map('Paired', 'qualitative', 10).mpl_colors

# Testing Pipeline:
def testAlgo(clf, X, y, clfName, opt = False, param_dict = None, opt_metric = 'roc_auc', n_iter = 5, folds = 10, times = 10):
    '''An algorithm that output the perdicted y and real y'''
    y_true = []
    y_pred = []
    if opt:
    	param_dict = param_dist_dict[clfName]
    grids_score = []
    imp = []
    for i in range(0, times):
    	print str(i) +" iteration of testAlgo"
        rs = np.random.randint(1,1000)
        cv = KFold(len(y), n_folds = folds, shuffle = True, random_state = rs)
        for train_index, test_index in cv:
            impr_clf, grids_score0, imp0 = fitAlgo(clf, X[train_index], y[train_index], opt, param_dict, opt_metric, n_iter)
            grids_score += grids_score0
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
    return y_pred, y_true, grids_score, imp

# Evaluation pipeline:
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
    if opt & (param_dict != None):
        assert(map(lambda x: isinstance(param_dict[x],list), param_dict))
        rs = RandomizedSearchCV(estimator = clf, n_iter = n_iter,
                                param_distributions = param_dict,
                                scoring = opt_metric,
                                refit = True,
                                n_jobs=-1, cv = 3, verbose = 3)

        rs.fit(Xtrain, Ytrain)
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
def pred_prep(data_path, obj, target):
	'''A generalized method that could return the desired X and y, based on the file path of the data, the name of the obj, and the target column we are trying to predict.
	Keyword Arguments:
		data_path: the data path
		obj: name of the dataset
		target: the target column name
	'''
	# Make sure "data/obj" and "plots/obj" exist
	if not os.path.exists('data/'+obj):
		os.makedirs('data/'+obj)
	if not os.path.exists('plots/'+obj):
		os.makedirs('plots/'+obj)
	f=hp.File(data_path, 'r+')
	dataset = f[obj].value
	f.close()
	# Convert Everything to float for easier calculation
	dataset = dataset.astype([(k,float) for k in dataset.dtype.names])
	featureNames = dataset.dtype.fields.keys()
	featureNames.remove(target)
	y = dataset[target]
	newdataset = dataset[featureNames]
	X = newdataset.view((np.float64, len(newdataset.dtype.names)))
	# X = preprocessing.scale(X)
	y = y.view((np.float64, 1))
	return X, y, featureNames

### Save data

def fill_2d(X, fill = np.nan):
	'''Function to fill list of array with a certain fill to make it a 2d_array with shape m X n, where m is the number of arrays in the list, n is the maxinum length of array in the list.
	'''
	maxlen = max([len(x) for x in X])
	newX = [np.append(x, np.array([fill] * (maxlen-len(x)))) for x in X]
	return np.array(newX)


def save_output(obj, X, y, opt = True):
	'''Save Output (y_pred, y_true, grids_score, and imp) for this dataframe
    Keyword arguments:
    obj - - dataframe name
    X - - feature matrix
    y - - training target array
    opt - - whether to use parameter optimization, default is True
	'''
	for clfName in classifiers1.keys():
		clf = classifiers1[clfName]
		if opt:
			param_dist = param_dist_dict[clfName]
		else:
			param_dist = None
		# grids = grid_score_list
		y_pred, y_true, grids, imp = testAlgo(clf, X, y, clfName, opt, param_dist)
		y_pred = fill_2d(y_pred)
		y_true = fill_2d(y_true)
		# Only rearange format if grids is not []
		if grids !=[]:
			grids2 = grids
			# stds = map(lambda x: x.__repr__().split(',')[1], grids)
			fields = grids[0][0].keys()+list(grids[0]._fields)
			fields.remove('parameters')
			fields.remove('cv_validation_scores')
			fields.append('std')
			grids2 = map(lambda x: tuple(x[0].values()+[x[2].mean(),x[2].std()]),grids2)
			datatype = [(fields[i], np.result_type(grids2[0][i]) if not isinstance(grids2[0][i], str) else '|S14') for i in range(0, len(fields))]
			grids_score = np.array(grids2, datatype)
		else:
			grids_score = np.array([])
		imp = np.array(imp)
		if opt:
			f = hp.File('./data/'+ obj + '/' + clfName + '_opt.h5', 'w')
		else:
			f = hp.File('./data/'+ obj + '/' + clfName + '_noopt.h5', 'w')
		print("Saving output for " + clfName)
		print y_true
		f.create_dataset('y_true', data = y_true)
		f.create_dataset('y_pred', data = y_pred)
		f.create_dataset('grids_score', data = grids_score)
		f.create_dataset('imp', data = imp)
		f.close()

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
	data_path = './data/'+obj + '/'
	print("Open output for " + clfName)
	if opt:
		data_path1 = data_path + clfName + '_opt.h5'
	else:
		data_path1 = data_path + clfName + '_noopt.h5'
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
	# Data path for this dataframe
	data_path = './data/'+obj + '/'
	for clfName in clfs.keys():
		print clfName
		y_pred, y_true, grids_score, imp = open_output(clfName, obj, opt)
		# Need to check imp's shape maybe
		if (imp.shape[1]!= 1) & opt & (clfName == "RandomForest"):
			imp.shape[1]
			print clfName
			plot_importances(imp,clfName, obj)
		# Because if opt = Flase, grids_score should be []
		if len(grids_score)>0:
			plotGridPref(grids_score, clfName, metric, obj)
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
		save_path = 'plots/'+obj+'/'+'clf_comparison_'+ 'roc_auc' +'_opt.pdf'
	else:
		save_path = 'plots/'+obj+'/'+'clf_comparison_'+ 'roc_auc' +'_noopt.pdf'
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
		save_path = 'plots/'+obj+'/'+'clf_comparison_'+ 'pr' +'_opt.pdf'
	else:
		save_path = 'plots/'+obj+'/'+'clf_comparison_'+ 'pr' +'_noopt.pdf'
	fig1.savefig(save_path)

def clf_plot(obj, clfName, opt):
	'''
	Plot experiment results
	Keyword Arguments:
		dfName: dataframe name
		clfName: classifier name
	'''
	if opt:
		datapath = 'data/'+obj+'/'+clfName+'_opt.h5'
	else:
		datapath = 'data/'+obj+'/'+clfName+'_noopt.h5'
	f=hp.File(data_path, 'r+')
	y_pred = f['y_pred'].value
	y_true = f['y_true'].value
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
		save_path = 'plots/'+obj+'/'+clfName+'_roc_opt.pdf'
	else:
		save_path = 'plots/'+obj+'/'+clfName+'_roc_noopt.pdf'
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
		save_path = 'plots/'+obj+'/'+clfName+'_pr_opt.pdf'
	else:
		save_path = 'plots/'+obj+'/'+clfName+'_pr_noopt.pdf'
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

def plotGridPref(gridscore, clfName, metric, obj):
	''' Plot Grid Performance
	'''
	# data_path = 'data/'+obj+'/'+clfName+'_opt.h5'
	# f=hp.File(data_path, 'r')
	# gridscore = f['grids_score'].value
	paramNames = gridscore.dtype.fields.keys()
	paramNames.remove("mean_validation_score")
	paramNames.remove("std")
	score = gridscore["mean_validation_score"]
	std = gridscore["std"]
	newgridscore = gridscore[paramNames]
	# for i in paramNames:
	num_params = len(paramNames)
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
				save_path = '../MSPrediction-Python/plots/'+obj+'/'+ clfName +'_' +metric+'_'+ i +'_'+ j+'.pdf'
				fig.savefig(save_path)

def compare_obj(datasets = [], models = [], opt = True):
	''' A function that takes a list of datasets and clfNames, so that it compare the model performance (roc_auc, and pr)
	'''
	for clfName in models:
		# Make sure "plots/clfName" exists
		if not os.path.exists('plots/'+clfName):
			os.makedirs('plots/'+clfName)
		mean_everything= {}
		mean_everything1 = {}
		clf = classifiers1[clfName]
		param_dict = param_dist_dict[clfName]
		for obj in datasets:
			y_pred, y_true, _, _ = open_output(clfName, obj, opt)
			mean_fpr, mean_tpr, mean_auc = plot_roc(y_pred, y_true, clfName, obj, opt, save_sub = False)
			mean_everything[obj] = [mean_fpr, mean_tpr, mean_auc]

			# out pr results and plot folds
			mean_rec, mean_prec, mean_auc1 = plot_pr(y_pred, y_true, clfName, obj, opt, save_sub = False)
			mean_everything1[obj] = [mean_rec, mean_prec, mean_auc1]

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
			save_path = 'plots/'+clfName+'/'+'dataset_comparison_'+ 'roc_auc' +'_opt.pdf'
		else:
			save_path = 'plots/'+clfName+'/'+'dataset_comparison_'+ 'roc_auc' +'_noopt.pdf'
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
			save_path = 'plots/'+clfName+'/'+'dataset_comparison_'+ 'pr' +'_opt.pdf'
		else:
			save_path = 'plots/'+clfName+'/'+'dataset_comparison_'+ 'pr' +'_noopt.pdf'
		fig1.savefig(save_path)

### Functions to analyze different models, plot importances for random forest, coefficients for logistic and linear regressions, and fit pdf plot for Bayes

def plot_importances(imp, clfName, obj):
    imp=np.vstack(imp)
    print imp
    mean_importance = np.mean(imp,axis=0)
    std_importance = np.std(imp,axis=0)
    indices = np.argsort(mean_importance)[::-1]
    print indices
    print featureNames
    featureList = []
    # num_features = len(featureNames)
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
    save_path = 'plots/'+obj+'/'+clfName+'_feature_importances.pdf'
    fig.savefig(save_path)

def plotGaussian(X, y, obj, featureNames):
	"""Plot Gausian fit on top of X.
	"""
	save_path = '../MSPrediction-Python/plots/'+obj+'/'+'BayesGaussian2'
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
	save_path = '../MSPrediction-Python/plots/'+obj+'/'+whichMix
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
    # num_features = len(featureNames)
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
    save_path = 'plots/'+obj+'/'+whichReg+'_feature_importances.pdf'
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
		y_pred, y_true, grids_score, amp = testAlgo(newclf, X, y, clfName)
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
	fig.savefig('plots/'+obj+'/'+ clfName +'_' + param +'_'+'ps.pdf')
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
	for i in classifiers1.keys():
		print i
	s= raw_input("Please input in a list format, e.g. [\"Core\", \"Core_Imp\"]")
	datasets = eval(s)
	print("Which following classifiers do you wanna use?")
	for i in classifiers1.keys():
		print i
	s = raw_input("Please input in a list format, e.g. [\"RandomForest\", \"LogisticRegression\"]")
	models = eval(s)
	comp_obj_opt = raw_input("With optimization?")
	return datasets, models, (comp_obj_opt == 'Y')
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
classifiers1 = {"LogisticRegression": LogisticRegression(),
					"BayesBernoulli": BernoulliNB(),
					"BayesGaussian": GaussianNB(),
					"BayesGaussian2":GaussianNB2(),
					"RandomForest": RandomForestClassifier(),
					"LinearRegression": LinearRegression(),
					"BayesMixed": MixNB(),
					"BayesMixed2": MixNB2()
					}

# dictionaries of different classifiers, these can be eyeballed from my parameter sweeping curve
num_features = 6
random_forest_params = {"n_estimators": range(50,300,50),
			  "max_features": range(2, num_features + 1),
			  "min_samples_split": range(2, 10),
			  "min_samples_leaf": range(2, 10),
			  "bootstrap": [True, False],
			  "criterion": ["gini", "entropy"]}
# ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling', 'class_weight', 'random_state']
logistic_regression_params = {"penalty":['l1','l2'],
					"C": np.linspace(.1, 1, 10),
					"fit_intercept":[True, False],
					"intercept_scaling":np.linspace(.1, 1, 10),
					"tol":[1e-4, 1e-5, 1e-6]}
# ['n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p', 'metric']
knn_params= {"n_neighbors":range(1,6),
				"algorithm":['auto', 'ball_tree', 'kd_tree'],
				"leaf_size":range(25,30),
				"p":range(1,3)}
# ['alpha', 'binarize', 'fit_prior', 'class_prior']
bayesian_bernoulli_params= {"alpha": np.linspace(.1, 1, 10),
				"binarize": np.linspace(.1, 1, 10)}
# ['alpha', 'binarize', 'fit_prior', 'class_prior']
bayesian_multi_params= {"alpha": np.linspace(.1, 1, 10)}
# ['alpha', 'binarize', 'fit_prior', 'class_prior']
bayesian_gaussian_params= None
bayesian_gaussian2_params= None
bayesian_poisson_params = None
bayesian_mixed_params = None
bayesian_mixed2_params = None
# ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking', 'probability', 'tol', 'cache_size', 'class_weight', 'verbose', 'max_iter', 'random_state']
svm_params = {"C": np.linspace(.1, 1, 10),
				"kernel":['linear','poly','rbf'],
				"shrinking":[True, False],
				"tol":[1e-3,1e-4]}
# ['fit_intercept', 'normalize', 'copy_X']
linear_regression_params = {"fit_intercept":[True, False],
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
	####### This part can be modified to fulfill different needs #####
	data_path = './data/predData.h5'
	f = hp.File(data_path, 'r')
	objs = [str(i) for i in f.keys()]
	f.close()
	#########QUESTIONS################################################
	#
	# Compre datasets
	comp_obj = raw_input("Do you want to compare different datasets? (Y/N)")
	if (comp_obj == 'Y'):
		datasets, models, opt = comp_obj_select()
		print datasets
		print models
		print opt
		compare_obj(datasets, models, opt)
	# single_ana = raw_input("Do you want to do single dataset analysis?")
	print("Now let's do single dataset analysis. \n")
	print("Choose one dataset from the following list?")
	for i in objs:
		print i
	obj = raw_input()
	if obj not in objs:
		obj = raw_input("No such dataset exists. Please enter a different dataset name. \n")
	target = 'ModEDSS'
	########## Can use raw_input instead as well######################
	global featureNames
	X, y, featureNames = pred_prep(data_path, obj, target)
	global num_features
	try:
		num_features = X.shape[1]
	except IndexError:
		X = X.reshape(X.shape[0], 1)
		num_features = 1
	random_forest_params["max_features"] = range(1, num_features + 1)
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

	### Output Data
	# Check whether the ouput data for obj has been generated
	if os.listdir("./data/"+obj) != []:
		print("Output for " + obj + " has already been generated")
	else:
		print("There is no output data for " + obj)
	saveoutput = raw_input("Do you want to save the output (y_pred, y_true, gridscore (plus importance for RandomForest)) for " + obj + "? (Y/N)")
	if saveoutput == "Y":
		output_opt = raw_input("Save output with parameter optimization? (Yes/ No/ Both)")
		if (output_opt == 'Yes'):
			save_output(obj, X, y, opt = True)
		elif (output_opt == 'No'):
			save_output(obj, X, y, opt = False)
		else:
			save_output(obj, X, y, opt = True)
			save_output(obj, X, y, opt = False)

	### Plot results
	com_clf = raw_input("Compare classifiers? (Y/N) ")
	# com_clf = "Y"
	if com_clf == "Y":
		com_clf_opt = raw_input ("With optimization? (Y/N)")
		# com_clf_opt = "Y"
		com_clf_opt = (com_clf_opt == 'Y')
		compare_clf(classifiers1, obj, metric = 'roc_auc', opt = com_clf_opt)
	else:
		clf, clfName = choose_clf(classifiers)
		param_sweep = raw_input("Parameter Sweeping? (Y/N) ")
		# param_sweep ="Y"
		if param_sweep == "Y" or param_sweep == "y":
			param, param_dist, metric = param_sweep_select(clf)
			param_sweeping(clf, obj, X, y, param_dist, metric, param, clfName)
		else:
			print ("Your only choice now is to plot ROC and PR curves for "+clfName+" classifier")
			# Asking whether to optimize
			opt = raw_input("Optimization? (Y/N)")
			opt = (opt== "Y" or opt == "y")
			# opt = True
			if opt:
				param_dist = param_dist_dict[clfName]
			else:
				param_dist = None
			clf_plot(obj, clfName, opt)

if __name__ == "__main__":
	main()
