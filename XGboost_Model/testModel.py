import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from dataRetrieval import *
from getDataset import *
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score, make_scorer


def testModel(clf, dataset, nfolds):
	target_col = dataset.columns.get_loc('Beat Estimate')
	col_start = dataset.columns.get_loc('EPS Estimate')
	col_end = dataset.columns.get_loc('Asset Growth')

	folds = KFold(n_splits = nfolds, shuffle = True)

	acc = 0
	pres = 0
	recall = 0
	f1 = 0
	kappa = 0
	for train_index, test_index in folds.split(dataset):
		X_train, X_test = dataset.iloc[train_index, col_start:col_end].values.astype(float), dataset.iloc[test_index, col_start:col_end].values.astype(float)
		y_train, y_test = dataset.iloc[train_index, target_col].values.astype(float), dataset.iloc[test_index, target_col].values.astype(float)


		clf.fit(X_train, y_train)

		pred = clf.predict(X_test)
	
		acc += accuracy_score(y_test, pred)
		pres += precision_score(y_test, pred)
		recall += recall_score(y_test, pred)
		f1 += f1_score(y_test, pred)
		kappa += cohen_kappa_score(y_test, pred)

	print("Accuracy:",acc/nfolds)
	print("Precision:",pres/nfolds)
	print("Recall",recall/nfolds)
	print("F1:",f1/nfolds)
	print("Cohen Kappa:",kappa/nfolds)




train, test = getData(filename='stockData.csv', rebalance=True)
clf = pickle.load(open("model.pkl", 'rb'))
dataset = train.append(test, ignore_index = True, sort=False) 
testModel(clf, dataset, 5)
