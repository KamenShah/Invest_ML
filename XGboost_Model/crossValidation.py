import xgboost as xgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from dataRetrieval import *
from getDataset import *
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score, make_scorer



def crossValidationHyperTuning(train, optomization):
	target_col = train.columns.get_loc('Beat Estimate')
	col_start = train.columns.get_loc('EPS Estimate')
	col_end = train.columns.get_loc('Asset Growth')

	clf_xgb = xgb.XGBClassifier()
	param_dist = {'n_estimators': stats.randint(150, 500),
	              'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
	              'subsample': stats.uniform(0.3, 0.7),
	              'max_depth': list(range(1,20,2)),
	              'colsample_bytree': stats.uniform(0.5, 0.45),
	              'min_child_weight': [1, 2, 3]
	               }

	clf = RandomizedSearchCV(clf_xgb, param_distributions = param_dist, n_iter = 25, scoring = optomization, error_score = 0, verbose = 3, n_jobs = -1)

	numFolds = 5
	folds = KFold(n_splits = numFolds, shuffle = True)

	estimators = []
	results = np.zeros(len(train))
	score = 0.0
	for train_index, test_index in folds.split(train):
		X_train, X_test = train.iloc[train_index, col_start:col_end].values.astype(float), train.iloc[test_index, col_start:col_end].values.astype(float)
		y_train, y_test = train.iloc[train_index, target_col].values.astype(float), train.iloc[test_index, target_col].values.astype(float)

		clf.fit(X_train, y_train)

		estimators.append(clf.best_estimator_)
		results[test_index] = clf.predict(X_test)
		score += recall_score(y_test, results[test_index])
	score /= numFolds


	best_score = clf.best_score_
	best_params = clf.best_params_
	print("Averaged score:",score)
	print("Best score: {}".format(best_score))
	print("Best params: ")
	for param_name in sorted(best_params.keys()):
	    print('%s: %r' % (param_name, best_params[param_name]))




train, test = getData(filename='stockData.csv', rebalance=True)

crossValidationHyperTuning(train, 'recall')

#Optomize f1
# Best score: 0.8243093292016175
# Best params: 
# colsample_bytree: 0.6293598129274831
# learning_rate: 0.001
# max_depth: 17
# min_child_weight: 2
# n_estimators: 218
# subsample: 0.3115037346846945

#Optomize Recall
# colsample_bytree: 0.7453193972758985
# learning_rate: 3
# max_depth: 13
# min_child_weight: 3
# n_estimators: 476
# subsample: 0.3252849104357345


# colsample_bytree: 0.7854433530717853
# learning_rate: 3
# max_depth: 9
# min_child_weight: 3
# n_estimators: 425
# subsample: 0.6344895093430756

#Optomize Kappa
# Best score: 0.3691041089102482
# Best params: 
# colsample_bytree: 0.5423528582173227
# learning_rate: 0.2
# max_depth: 9
# min_child_weight: 1
# n_estimators: 459
# subsample: 0.6221676098203561
