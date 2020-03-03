import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN

def getData(filename, rebalance):

	df = pd.read_csv(filename)
	df = df.replace('None', 0)
	df = df.loc[(df!='Stock').all(1)]
	df = df.dropna(axis=1,how='all')
	df = df.drop_duplicates()
	df = df.reset_index(drop=True)
	df = df[(df != 0).all(1)]


	df = df.drop('Debt Equity Ratio', 1)
	df = df.drop('Dividend Yield', 1)
	df = df.drop('Dividend Payout Ratio', 1)
	df = df.drop('Inventory Turnover', 1)
	df = df.drop('Receivables Growth', 1)
	df = df.drop('Debt Growth', 1)
	df = df.drop('R&D Growth', 1)
	df = df.sample(frac=1).reset_index(drop=True)

	df['Beat Estimate'] = df['Beat Estimate'].map({'-1': 0, '1':1})

	trainSize = round(0.90 * len(df))
	train = df[:trainSize]
	test = df[trainSize:]

	if(rebalance):
		x, y = SMOTE().fit_resample(train.loc[:, 'EPS Estimate':'Asset Growth'], train.loc[:, 'Beat Estimate'])

		x = pd.DataFrame(x)
		y = pd.DataFrame(y)
		train = pd.concat([y,x], axis= 1)

	return train, test