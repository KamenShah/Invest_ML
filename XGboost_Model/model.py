import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataRetrieval import *
from getDataset import *
import pickle


def trainXGBoost(train):
	negCount = np.sum(train.loc[:, 'Beat Estimate'].values.astype(float) == 0)
	posCount = np.sum(train.loc[:, 'Beat Estimate'].values.astype(float) == 1)

	X_train = train.loc[:, 'EPS Estimate':'Asset Growth'].values.astype(float)
	y_train = train.loc[:, 'Beat Estimate'].values.astype(float)

	clf = xgb.XGBClassifier()
	param_dist = {'n_estimators': 425,
	              'learning_rate': 3,
	              'subsample': 0.6344895093430756,
	              'max_depth': 9,
	              'colsample_bytree': 0.7854433530717853,
	              'min_child_weight': 3,
	              'scale_pos_weight' : negCount / posCount
	              }
	clf.fit(X_train, y_train)

	return clf





train, test = getData(filename='stockData.csv', rebalance=True)
clf = trainXGBoost(train)
pickle.dump(clf, open('model.pkl','wb'))

