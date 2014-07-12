import pylab as pl
import h5py as hp
import numpy as np
import math as M
from termcolor import colored
# import helper
from scipy.interpolate import griddata
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
####### Modify your naive_bayes source code############
import sklearn.naive_bayes as SNB
fullpath = SNB.__file__
path, filename = fullpath.rsplit('/', 1)
dst = path+"/naive_bayes.py"
scr = "./naive_bayes.py"
from shutil import copyfile
try:
    copyfile(scr, dst)
    print "Success!"
except IOError as e:
	print e
	print colored("TIPS: MUST HAVE ADMINISTRATOR PRIVILEGE...!", 'red')
#######################################################
from sklearn.naive_bayes import BernoulliNB, GaussianNB, GaussianNB2, MultinomialNB, PoissonNB, MixNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib.mlab import rec_drop_fields
from matplotlib import cm
# import itertools
from inspect import getargspec
from sklearn.grid_search import RandomizedSearchCV
import os

# Testing Pipeline:
def testAlgo(clf, X, y, clfName, opt = False, param_dict = None, opt_metric = 'roc_auc', n_iter = 10, folds = 10, times =  10):
    '''An algorithm that output the perdicted y and real y'''
    y_true = []
    y_pred = []
    if opt:
    	param_dict = param_dist_dict[clfName]
    gs_score_list = []
    for i in range(0, times):
    	print str(i) +" iteration of testAlgo"
        rs = np.random.randint(1,1000)
        cv = KFold(len(y), n_folds = folds, shuffle = True, random_state = rs)
        for train_index, test_index in cv:
            impr_clf, gs_score = fitAlgo(clf, X[train_index], y[train_index], opt, param_dict, opt_metric, n_iter)
            gs_score_list += gs_score
            if (clfName != "LinearRegression"):
                proba = impr_clf.predict_proba(X[test_index])
                y_pred0 = proba[:,1]
            else:
                proba = impr_clf.predict(X[test_index])
                y_pred0 = proba
            y_true0 = y[test_index]
            y_pred.append(y_pred0)
            y_true.append(y_true0)
    return y_pred, y_true, gs_score_list

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
        return rs.best_estimator_, rs.grid_scores_
    else:
        if param_dict != None:
            assert(map(lambda x: not isinstance(param_dict[x], list), param_dict))
            for k in param_dict.keys():
	            clf.set_params(k = param_dict[k])
        clf.fit(Xtrain, Ytrain)
        return clf, []

# Meta-functions
def clf_plot(clf, X, y, clfName, obj, opt, param_dist, metric = 'roc_auc'):
	'''Plot experiment results'''
	# Produce data for plotting
	y_pred, y_true, gs_score_list = testAlgo(clf, X, y, clfName, opt, param_dist)
	if len(gs_score_list)>0:
		saveGridPref(obj, clfName, metric, gs_score_list)
		plotGridPrefTest(obj, clfName, metric)
	# Plotting auc_roc and precision_recall
	plot_roc(y_pred, y_true, clfName, obj, opt)
	# Plotting precision_recall
	plot_pr(y_pred, y_true, clfName, obj, opt)

def pred_prep(data_path, obj, target):
	'''A generalized method that could return the desired X and y, based on the file path of the data, the name of the obj, and the target column we are trying to predict.
	Keyword Arguments:
		data_path: the data path
		obj: name of the dataset
		target: the target column name
	'''
	# Make sure "data/obj" and "plot/obj" exist
	if not os.path.exists('data/'+obj):
		os.makedirs('data/'+obj)
	if not os.path.exists('plots/'+obj):
		os.makedirs('plots/'+obj)
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

def compare_clf(X, y, clfs, obj, metric = 'roc_auc', opt = False, n_iter=4, folds=4, times=4):
	'''Compare classifiers with mean roc_auc'''
	mean_everything= {}
	mean_everything1 = {}
	for clfName in clfs.keys():
		clf = clfs[clfName]
		y_pred, y_true, gs_score_list= testAlgo(clf, X, y, clfName, opt, opt_metric = metric, n_iter=n_iter, folds=folds, times=times)
		if len(gs_score_list)>0:
			saveGridPref(obj, clfName, metric, gs_score_list)
			plotGridPrefTest(obj, clfName, metric)
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
	# pl.show()

def plot_roc(y_pred, y_true, clfName, obj, opt):
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
	fig.savefig(save_path)
	# pl.show()
	return mean_fpr, mean_tpr, mean_auc

def plot_pr(y_pred, y_true,clfName, obj, opt):
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
			print metric +" is currently not available"
		# mean_y[0] = 0.0
	mean_y/= len(folds)
	mean_area = auc(mean_x,mean_y)
	return mean_x, mean_y, mean_area

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
		y_pred, y_true, gs_score_list = testAlgo(newclf, X, y, clfName)
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

def testGrid():
	data_path = '../MSPrediction-R/Data Scripts/data/predData.h5'
	obj = 'fam2_bin'
	target = 'EnjoyLife'
	X, y, featureNames = pred_prep(data_path, obj, target)
	num_features = X.shape[1]
	random_forest_params["max_features"] = range(1, num_features + 1)
	clfName = 'RandomForest'
	opt_metric = 'roc_auc'
	clf = classifiers[clfName]
	opt = True
	param_dist = param_dist_dict[clfName]
	y_pred, y_true, gs_score_list = testAlgo(clf, X, y, clfName,opt, param_dist)
	saveGridPref(obj, clfName, opt_metric, gs_score_list)
	# return gs_score_list
	# 

def testDiagnoStatic():
	"""sklearn's Naive Bayes couldn't handle missing value"""
	data_path = '../MSPrediction-R/Data Scripts/data/predData.h5'
	obj = 'diagnoeffstatic'
	target = 'ModEDSS'
	X, y, featureNames = pred_prep(data_path, obj, target)
	clfName = 'BayesGaussian2'
	opt_metric = 'roc_auc'
	clf = classifiers[clfName]
	opt = True
	param_dist = None
	clf_plot(clf, X, y, clfName, obj, opt, param_dist)




def saveGridPref(obj, clfName, metric, grids):
	# Transfer grids to list of numetuples to numpy structured array
	grids2 = grids
	# stds = map(lambda x: x.__repr__().split(',')[1], grids)
	fields = grids[0][0].keys()+list(grids[0]._fields)
	fields.remove('parameters')
	fields.remove('cv_validation_scores')
	fields.append('std')
	grids2 = map(lambda x: tuple(x[0].values()+[x[2].mean(),x[2].std()]),grids2)
	datatype = [(fields[i], np.result_type(grids2[0][i]) if not isinstance(grids2[0][i], str) else '|S14') for i in range(0, len(fields))]
	dataset = np.array(grids2, datatype)
	f = hp.File('../MSPrediction-Python/data/'+obj+'/'+clfName+'_grids_'+metric+'.h5', 'w')
	dset = f.create_dataset(clfName, data = dataset)
	f.close()

def plotGridPrefTest(obj, clfName, metric):
	data_path = '../MSPrediction-Python/data/'+obj+'/'+clfName+'_grids_'+metric+'.h5'
	target = 'EnjoyLife'
	f=hp.File(data_path, 'r')
	dataset = f[clfName].value
	paramNames = dataset.dtype.fields.keys()
	paramNames.remove("mean_validation_score")
	paramNames.remove("std")
	score = dataset["mean_validation_score"]
	std = dataset["std"]
	newdataset = dataset[paramNames]
	for i in paramNames:
		x = newdataset[i]
		for j in list(set(paramNames)- set([i])):
			y = newdataset[j]
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


classifiers = {"LogisticRegression": LogisticRegression(),
					"KNN": KNeighborsClassifier(),
					"BayesBernouilli": BernoulliNB(),
					"BayesMultinomial": MultinomialNB(),
					"BayesGaussian": GaussianNB(),
					"BayesPoisson": PoissonNB(),
					"BayesGaussian2":GaussianNB2(),
					"SVM": SVC(probability = True),
					"RandomForest": RandomForestClassifier(),
					"LinearRegression": LinearRegression(),
					"BayesMixed": MixNB()
					}
# dictionaries of different classifiers, these can be eyeballed from my parameter sweeping curve
num_features = 4
random_forest_params = {"n_estimators": range(25,100),
			  "max_features": range(1, num_features + 1),
			  "min_samples_split": range(1, 8),
			  "min_samples_leaf": range(1, 8),
			  "bootstrap": [True, False],
			  "criterion": ["gini", "entropy"]}
# ['penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling', 'class_weight', 'random_state']
logistic_regression_params = {"penalty":['l1','l2'],
					"C": np.linspace(.1, 1, 10),
					"fit_intercept":[True, False],
					"intercept_scaling":(.1, 1, 10),
					"tol":[1e-2, 1e-3,1e-4, 1e-5]}
# ['n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p', 'metric']
knn_params= {"n_neighbors":range(5,7),
				"algorithm":['auto', 'ball_tree', 'kd_tree', 'brute'],
				"leaf_size":range(25,30),
				"p":range(1,3)}
# ['alpha', 'binarize', 'fit_prior', 'class_prior']
bayesian_bernouilli_params= {"alpha": np.linspace(.1, 1, 10),
				"binarize": np.linspace(.1, 1, 10).tolist()}
# ['alpha', 'binarize', 'fit_prior', 'class_prior']
bayesian_multi_params= {"alpha": np.linspace(.1, 1, 10)}
# ['alpha', 'binarize', 'fit_prior', 'class_prior']
bayesian_gaussian_params= None
bayesian_gaussian2_params= None
bayesian_poisson_params = None
bayesian_mixed_params = None

# ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking', 'probability', 'tol', 'cache_size', 'class_weight', 'verbose', 'max_iter', 'random_state']
svm_params = {"C": np.linspace(.1, 1, 10),
				"kernel":['linear','poly','rbf'],
				"shrinking":[True, False],
				"tol":[1e-2, 1e-3,1e-4, 1e-5]}
# ['fit_intercept', 'normalize', 'copy_X']
linear_regression_params = {"fit_intercept":[True, False],
					"normalize": [True, False]}
# a dictionary storing the param_dist for different classifiers
param_dist_dict = {"LogisticRegression": logistic_regression_params,
				"KNN":knn_params,
				"BayesBernouilli": bayesian_bernouilli_params,
				"BayesMultinomial": bayesian_multi_params,
				"BayesGaussian": bayesian_gaussian_params,
				"BayesGaussian2":bayesian_gaussian2_params,
				"BayesPoisson": bayesian_poisson_params,
				"SVM":svm_params,
				"RandomForest":random_forest_params,
				"LinearRegression":linear_regression_params,
				"BayesMixed": bayesian_mixed_params
				}


def main():
	'''Some basic setup for prediction'''
	####### This part can be modified to fulfill different needs #####
	data_path = './data/predData.h5'
	obj = 'fam2_bin'
	target = 'EnjoyLife'
	########## Can use raw_input instead as well######################
	X, y, featureNames = pred_prep(data_path, obj, target)
	global num_features
	num_features = X.shape[1]
	random_forest_params["max_features"] = range(1, num_features + 1)
	#########QUESTIONS################################################
	com_clf = raw_input("Compare classifiers? (Y/N) ")
	# com_clf = "Y"
	if com_clf == "Y":
		com_clf_opt = raw_input ("With optimization? (Y/N)")
		# com_clf_opt = "Y"
		com_clf_opt = (com_clf_opt == 'Y')
		compare_clf(X, y, classifiers, obj, metric = 'roc_auc', opt = com_clf_opt, n_iter=4, folds=4, times=4)
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
			clf_plot(clf, X, y, clfName, obj, opt, param_dist)



if __name__ == "__main__":
	main()
