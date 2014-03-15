import pylab as pl
import h5py as hp
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import uniform,expon
from matplotlib.mlab import rec_drop_fields

def pred_prep(data_path, obj, target):
	'''A generalized method that could return the desired X and y, based on the file path of the data, the name of the obj, and the target column we are trying to predict.'''
	f=hp.File(data_path, 'r+')
	dataset = f[obj].value
	featureNames = dataset.dtype.fields.keys()
	featureNames.remove(target)
	y = dataset[target]
	newdataset = dataset[featureNames]
	X = newdataset.view((np.float64, len(newdataset.dtype.names)))
	y = y.view((np.float64, 1))
	return X, y, featureNames



def run_clf(clf, X, y):
	'''Run a single classifier, return Y_pred and Y_truefor producing plots'''
	# initialize predicted y, real y
	Y_true = []
	Y_pred = []
	# Use 10-folds cross validation
	cv = StratifiedKFold(y, n_folds = 10)
	for train_index, test_index in cv:
		clf.fit(X[train_index], y[train_index])
		proba = clf.predict_proba(X[test_index])
		Y_pred.append(proba[:,1])
		Y_true.append(y[test_index])
	return Y_pred, Y_true

def clf_plot(clf, X, y, clfName):
	# Produce data for plotting
	Y_pred, Y_true = run_clf(clf, X, y)
	# Plotting auc_roc and precision_recall
	plot_auc_roc(Y_pred, Y_true, clfName)
	# plot_pr(Y_pred, Y_true)

def plot_auc_roc(y_pred, y_true, clfName):
	'''Plots the ROC Curves'''
	fig = pl.figure(figsize=(8,6),dpi=150)
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	if len(y_pred)==1:
	    folds = zip([y_pred],[y_true])
	else:
	    folds = zip(y_pred,y_true)
	mean_roc_auc=0
	total_yt =[]
	total_yp = []
	for i, (pred,true) in enumerate(folds):
	    # pred & true represent each of the experiment folds
		fpr, tpr, thresholds = roc_curve(true, pred)
		mean_tpr += np.interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc (fpr, tpr)
		pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
	pl.plot([0, 1], [0, 1], '--', color=(0.7, 0.7, 0.7),lw=3,label='Random')
	mean_tpr /= len(folds)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr,mean_tpr)
	pl.plot(mean_fpr, mean_tpr, 'k--',
	             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
	pl.xlim([-0.05, 1.05])
	pl.ylim([-0.05, 1.05])
	pl.xlabel('False Positive Rate',size=30)
	pl.ylabel('True Positive Rate',size=30)
	pl.title('Receiver operating characteristic',size=25)
	# fix_axes()
	# fix_legend(loc="lower right")
	pl.legend(loc="lower right")
	pl.tight_layout()
	fig.savefig('plots/'+clfName+'_roc_auc.pdf')
	fig.show()




if __name__ == "__main__":
	'''Some basic setup for prediction'''
	####### This part can be modified to fulfill different needs ######
	data_path = '../MSPrediction-R/Data Scripts/data/predData.h5'
	obj = 'modfam2'
	target = 'EnjoyLife'
	###################################################################
	X, y, featureNames = pred_prep(data_path, obj, target)
	classifiers = {"LogisticRegression": LogisticRegression(), 
					"KNN": KNeighborsClassifier(),
					"Naive_Bayes": GaussianNB(),
					"SVM": SVC(probability = True)
					}
	print ("Which Classifer would you like to use?")
	print ("Options:")
	print (classifiers.keys())
	clfName = raw_input()
	clf = classifiers[clfName]
	clf_plot(clf, X, y, clfName)
	# mean_tpr = 0.0
	# mean_fpr = np.linspace(0, 1, 100)
	# all_tpr = []
	# fig = pl.figure(figsize = (8,6), dpi = 150)
	# for i, (train, test) in enumerate(cv):
	#     probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
	#     # Compute ROC curve and area the curve
	#     fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
	#     mean_tpr += np.interp(mean_fpr, fpr, tpr)
	#     mean_tpr[0] = 0.0
	#     roc_auc = auc(fpr, tpr)
	#     pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

	# pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
	# mean_tpr /= len(cv)
	# mean_tpr[-1] = 1.0
	# mean_auc = auc(mean_fpr, mean_tpr)
	# pl.plot(mean_fpr, mean_tpr, 'k--',
	#         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
	# pl.xlim([-0.05, 1.05])
	# pl.ylim([-0.05, 1.05])
	# pl.xlabel('False Positive Rate')
	# pl.ylabel('True Positive Rate')
	# pl.title('Receiver operating characteristic example')
	# pl.legend(loc="lower right")
	# fig.savefig("plots/"+clfName+"_roc_auc.pdf")
	# fig.show()