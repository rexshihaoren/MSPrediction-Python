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

if __name__ == "__main__":
	'''Some basic setup for prediction'''
	# Retrieve data frame
	f=hp.File('../MSPrediction-R/Data Scripts/data/predData.h5','r+')
	fam2 = f['fam2'].value
	modfam2 = f['modfam2'].value
	# The names of all the columns
	colNames = modfam2.dtype.fields.keys()
	# Convert record array to a regular Numpy array for better matrix manipulation
	modfam2 = modfam2.view((np.float64, len(modfam2.dtype.names)))
	# Generate Feature Matrix X and target array y
	X = modfam2[:,0:4]
	y = modfam2[:,4]
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
	cv = StratifiedKFold(y, n_folds = 10)
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	all_tpr = []
	fig = pl.figure(figsize = (8,6), dpi = 150)
	for i, (train, test) in enumerate(cv):
	    probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
	    # Compute ROC curve and area the curve
	    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
	    mean_tpr += np.interp(mean_fpr, fpr, tpr)
	    mean_tpr[0] = 0.0
	    roc_auc = auc(fpr, tpr)
	    pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

	pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
	mean_tpr /= len(cv)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	pl.plot(mean_fpr, mean_tpr, 'k--',
	        label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
	pl.xlim([-0.05, 1.05])
	pl.ylim([-0.05, 1.05])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('Receiver operating characteristic example')
	pl.legend(loc="lower right")
	fig.savefig("plots/"+clfName+"_roc_auc.pdf")
	fig.show()





