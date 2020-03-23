import pandas as pd 
from sklearn import svm, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

def isEven(n): 
	training_numbers = pd.DataFrame([x for x in range(0,100)], columns = ['numbers'])
	training_numbers['remainder'] = training_numbers['numbers'] % 2
	training_numbers['label'] = [1 if y == 0 else 0 for y in training_numbers['remainder']]
	
	X_train = training_numbers[['numbers', 'remainder']][:round(len(training_numbers.values) * 0.7)]
	Y_train = training_numbers['label'][:round(len(training_numbers.values) * 0.7)] 

	X_test = training_numbers['numbers'][round(len(training_numbers.values) * 0.7):]
	Y_test = training_numbers['numbers'][round(len(training_numbers.values) * 0.7):]

	training_numbers[['numbers', 'remainder']] = preprocessing.MinMaxScaler().fit_transform(training_numbers[['numbers', 'remainder']])

	candidate_parameters = [{'C': [0.01, 0.1, 1], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['linear']},
							{'C': [0.01, 0.1, 1], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']},
							{'C': [0.01, 0.1, 1], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['poly']}]

	clf = GridSearchCV(estimator = svm.SVC(decision_function_shape = 'ovr'), param_grid = candidate_parameters, n_jobs = -1)

	clf.fit(X_train, Y_train)
	print('Best Test Score:', clf.best_score_)
	print('Optimised C value:', clf.best_estimator_.C)
	print('Optimised gamma value:', clf.best_estimator_.gamma)
	print('Optimised kernel:', clf.best_estimator_.kernel)

	n_remainder = n % 2

	return bool(int(clf.predict([[n, n_remainder]])))

