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

def compare_clf(X, y, clfs, obj, metric = 'roc'):
	'''Compare classifiers with mean roc_auc'''
	fig = pl.figure(figsize=(8,6),dpi=150)
	# cv = KFold(len(y), n_folds = 10, shuffle = True)
	# cv = StratifiedKFold(y, n_folds = 10)
	for clfName in clfs.keys():
		clf = clfs[clfName]
		y_pred, y_true = run_clf(clf, X, y, clfName)
		mean_fpr, mean_tpr, mean_auc = plot_unit_prep(y_pred, y_true, metric)
		print("Area under the ROC curve : %f" % mean_auc)
		# Plot ROC curve
		pl.plot(mean_fpr, mean_tpr, lw=3, label = clfName + ' (area = %0.2f)' %mean_auc)
	pl.plot([0, 1], [0, 1], 'k--')
	pl.xlim([0.0, 1.0])
	pl.ylim([0.0, 1.0])
	pl.xlabel('False Positive Rate',fontsize=30)
	pl.ylabel('True Positive Rate',fontsize=30)
	pl.title('Receiver operating characteristic',fontsize=25)
	pl.legend(loc='lower right')
	pl.tight_layout()
	fig.savefig('plots/'+obj+'/'+'clf_comparison.pdf')
	pl.show()

def run_clf(clf, X, y, clfName):
	'''Run a single classifier, return y_pred and y_truefor producing plots'''
	# initialize predicted y, real y
	y_true = []
	y_pred = []
	# Use 10-folds cross validation
	cv = StratifiedKFold(y, n_folds = 10)
	for train_index, test_index in cv:
		clf.fit(X[train_index], y[train_index])
		if (clfName != "LinearRegression"):
			proba = clf.predict_proba(X[test_index])
			y_pred.append(proba[:,1])
		else:
			proba = clf.predict(X[test_index])
			y_pred.append(proba)
		y_true.append(y[test_index])
	return y_pred, y_true

def clf_plot(clf, X, y, clfName, obj):
	# Produce data for plotting
	y_pred, y_true = run_clf(clf,X, y,clfName)
	# Plotting auc_roc and precision_recall
	plot_roc(y_pred, y_true, clfName, obj)
	plot_pr(y_pred, y_true, clfName, obj)

def plot_roc(y_pred, y_true, clfName, obj):
	'''Plots the ROC Curve'''
	fig = pl.figure(figsize=(8,6),dpi=150)
	mean_fpr, mean_tpr, mean_auc = plot_unit_prep(y_pred, y_true, 'roc', plotfold = True)
	mean_tpr[-1] = 1.0
	pl.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7),lw=3,label='Random')
	print("ROC AUC: %0.2f" % mean_auc)
	pl.plot(mean_fpr, mean_tpr, 'k--',
	             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
	pl.xlim([0.0, 1.00])
	pl.ylim([0.0, 1.00])
	pl.xlabel('False Positive Rate',size=30)
	pl.ylabel('True Positive Rate',size=30)
	pl.title('Receiver operating characteristic',size=25)
	pl.legend(loc="lower right")
	pl.tight_layout()
	fig.savefig('plots/'+obj+'/'+clfName+'_roc_auc.pdf')
	pl.show()

def plot_pr(y_pred, y_true,clfName, obj):
	'''Plot the Precision-Recall Curve'''
	fig = pl.figure(figsize=(8,6),dpi=150)
	mean_rec, mean_prec, mean_area = plot_unit_prep(y_pred, y_true, 'pr')
	print("Precision Recall AUC: %0.2f" % mean_area)
	pl.clf()
	pl.plot(mean_rec, mean_prec, label='Precision-Recall curve')
	pl.xlabel('Recall')
	pl.ylabel('Precision')
	pl.ylim([0.0, 1.05])
	pl.xlim([0.0, 1.0])
	pl.title('Precision-Recall: AUC=%0.2f' % mean_area)
	pl.legend(loc="lower left")
	fig.savefig('plots/'+obj+'/'+clfName+'_pr.pdf')
	pl.show()

def plot_unit_prep(y_pred, y_true, metric, plotfold = False):
	''' Prepare mean_x, mean_y array for classifier evaludation, from predicted y and real y.
	Keyword arguments:
	y_pred - - predicted y array
	y_true - - true y target array
	metric - - the metric in use to evaludate classifier
	plotfold - - whether to plot indiviudal fold or not'''
	mean_y= 0.0
	mean_x = np.linspace(0, 1, 100)
	if len(y_pred)==1:
	    folds = zip([y_pred],[y_true])
	else:
	    folds = zip(y_pred,y_true)
	for i, (pred,true) in enumerate(folds):
		# pred & true represent each of the experiment folds
		try:
			if metric == 'roc':
				x, y, thresholds = roc_curve(true, pred)
				roc_auc = auc(x, y)
				if plotfold:
					pl.plot(x, y, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
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
		y_pred, y_true = run_clf(newclf, X, y, clfName)
		mean_fpr, mean_tpr, mean_auc = plot_unit_prep(y_pred, y_true, metric)
		print mean_auc
		scores.append(mean_auc)
		print("Area under the ROC curve : %f" % mean_auc)
	fig = pl.figure(figsize=(8,6),dpi=150)
	pl.plot(param_dist, scores, label = 'Parameter Sweeping Curve')
	pl.xlabel('Parameter Distribution',fontsize=30)
	pl.ylabel('AUC_ROC Score',fontsize=30)
	pl.title('Parameter Sweeping Curve',fontsize=25)
	pl.legend(loc='lower right')
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


if __name__ == "__main__":
	'''Some basic setup for prediction'''
	####### This part can be modified to fulfill different needs ######
	data_path = '../MSPrediction-R/Data Scripts/data/predData.h5'
	obj = 'fam2'
	target = 'EnjoyLife'
	########## Can use raw_input instead as well#######################
	X, y, featureNames = pred_prep(data_path, obj, target)
	classifiers = {"LogisticRegression": LogisticRegression(), 
					"KNN": KNeighborsClassifier(),
					"Bayes": BernoulliNB(),
					"SVM": SVC(probability = True),
					"RandomForest": RandomForestClassifier(),
					"LinearRegression": LinearRegression()
					}
	com_clf = raw_input("Compare classifiers? (Y/N) ")
	if com_clf == "Y":
		compare_clf(X, y, classifiers, obj)
	else:
		clf, clfName = choose_clf(classifiers)
		param_sweep = raw_input("Parameter Sweeping? (Y/N) ")
		if param_sweep == "Y":
			param, param_dist, metric = param_sweep_select(clf)
			param_sweeping(clf, obj, X, y, param_dist, metric, param, clfName)
		else:
			print ("Your only choice now is to plot ROC and PR curves for "+clfName+" classifier")
			clf_plot(clf, X, y, clfName, obj)



