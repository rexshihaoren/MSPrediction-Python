'''fitAlgo(Classifier:clf, Matrix:Xtrain, Vector:Ytrain, Dictionnary: optimization_params, Scorer:optimization_metric, Boolean:opt_bool) This only fits the algorithm and return the parameters'''

def 

def clf_eval(clf, param_dist = None, X, y, metric, opt = False):
	'''Evaluate the performance of a classifier using specific metric
	Keyword arguments:
	clf - - classifier
	X - - feature matrix
	y - - target array
	opt - - whether to use parameter optimization
	param_dist - - the parameter distribution of param, grids space
	metric - - the metric we use to evaluate the performance of the classifiers, it could be a callable'''
	cv = KFold(len(y), n_folds = 10, shuffle = True, random_state = rs)
	score = 0
	if opt:
		rs = RandomizedSearchCV(clf, n_iter = 20,
									param_distributions = param_dist,
									scoring = metric,
									refit = True,
									n_jobs=-1)
	for train_index, test_index in cv:
		if opt:
			rs.fit(X[train_index], y[train_index])
			imprv_clf = rs.best_estimator_
			y_pred = imprv_clf.predict_proba(X[test_index])[:,1]

		else:
			clf.fit(X[train_index], y[train_index])
			clf.pred(X[test_index])
			proba = clf.predict_proba(X[test_index])
			y_pred = proba[:,1]
		y_true = y[test_index]
		# eval() evaluate clf's performance using specified metric. 
		score += eval(metric, y_pred, y_true)
	return score/10

def get_opt_clf(clf, param_dist = None, X, y, metric):
	'''Return the optimized classifier
	Keyword arguments:
	clf - - base classifier
	X - - feature matrix
	y - - target array
	param_dist - - the parameter distribution of param, grids space
	metric - - the metric we use to evaluate the performance of the classifiers, it could be a callable'''
	rs = RandomizedSearchCV(clf, n_iter = 20,
							param_distributions = param_dist,
							scoring = metric,
							refit = True,
							n_jobs=-1)

	rs.fit(X,y)
	imprv_clf = rs.best_estimator_
	return improve_clf


def scorer(metric, greater_is_better = True, needs_proba = True, needs_threshold = False, folds = 10, times = 10, X,y):
	
	scorer.make_score()

