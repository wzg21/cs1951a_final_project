import pandas as pd
import random
import csv
from scipy import stats
import statsmodels.api as sm
from statsmodels.tools import eval_measures

def split_data(data, prob):
	"""input: 
	 data: a list of pairs of x,y values
	 prob: the fraction of the dataset that will be testing data, typically prob=0.2
	 output:
	 two lists with training data pairs and testing data pairs 
	"""

	random.shuffle(data)
	i = int(len(data)*(1 - prob))
	return data[:i+1], data[i+1:]

def train_test_split(x, y, test_pct):
	"""input:
	x: list of x values, y: list of independent values, test_pct: percentage of the data that is testing data=0.2.

	output: x_train, x_test, y_train, y_test lists
	"""
	
	data = list(zip(x, y))
	train, test = split_data(data, test_pct)
	x_train, y_train = list(zip(*train))
	x_test, y_test = list(zip(*test))
	return x_train, x_test, y_train, y_test



if __name__=='__main__':

	# Setting p to 0.2 allows for a 80% training and 20% test split
	p = 0.2

	def load_file(file_path):
		"""input: file_path: the path to the data file
		   output: X: array of independent variables values, y: array of the dependent variable values
		"""
		data = pd.read_csv(file_path)
		cols = ['latitude', 'longitude', 'delivery', 'pickup', 'restaurant_reservation', 'price',
		'italian', 'breakfast_brunch', 'newamerican', 'pizza', 'tradamerican', 'sandwiches', 'chinese', 'bars', 'japanese', 'mexican',
		'seafood', 'delis', 'coffee', 'salad', 'mediterranean', 'french', 'indpak', 'thai', 'asianfusion', 'korean', 'latin', 'desserts', 'bakeries']
		d = [data[col].values.tolist() for col in cols]
		for i, col in enumerate(d):
			mi, ma = min(col), max(col)
			for j, v in enumerate(col):
				d[i][j] = (v - mi) / (ma - mi)
		X = list(zip(*d))
		y = data['rating'].values.tolist()
		return X, y



	X, y = load_file("businesses_nyc_small.csv")

	x_train, x_test, y_train, y_test = train_test_split(X, y, p)

	x_train = sm.add_constant(x_train)
	res = sm.OLS(y_train, x_train).fit()

	pred_train = res.predict(x_train)
	x_test = sm.add_constant(x_test)
	pred_test = res.predict(x_test)

	# Prints out the Report
	# TODO: print R-squared, test MSE & train MSE
	print(res.summary())
	print('training R-squared: ' + str(res.rsquared))
	print('training MSE: ' + str(eval_measures.mse(y_train, pred_train)))
	print('testing MSE: ' + str(eval_measures.mse(y_test, pred_test)))