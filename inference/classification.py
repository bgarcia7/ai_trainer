#=====[ Import preprocessing tools  ]=====
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics

#=====[ Import Inference models  ]=====
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn import linear_model

#=====[ Import util tools  ]=====
import random
import pandas as pd
import numpy as np
from collections import defaultdict

#=====[ Tests each classifier and chooses the one with highest accuracy  ]=====
def get_classifiers(trainer):

	classifiers = [linear_model.LogisticRegression, svm.SVC, ensemble.RandomForestClassifier]
	X, Y, file_names = trainer.extract_all_advanced_features(toIgnore=[])

	selected_classifiers = {}
	
	#=====[ Select best classifier for each form component  ]=====
	for key in Y:

		score = 0

		#=====[ Tests each classifier to get the one with the highest score  ]=====
		for classifier in classifiers:
			
			try:
				
				accuracy = rnd_prediction(X, Y[key], file_names, classifier)
				
				#=====[ If accuracy of current classifier higher than score we've seen so far, update selected_classifier  ]=====
				print key, accuracy, score
				if accuracy > score:
					score = accuracy
					selected_classifiers[key] = classifier()
				
			except Exception as e:
				print key, e

	#=====[ Train each classifier  ]=====
	for key in selected_classifiers:
		selected_classifiers[key].fit(X,Y[key])

	return selected_classifiers


#=====[ Returns the probability for a specified set of features, class and an X, y to fit to  ]=====
def predictProbs(X,y,X_test, clf_class, **kwargs):
	clf = clf_class(**kwargs)
	clf.fit(X ,y)
	return clf.predict_proba(X_test)

#=====[ Returns predictions for a specified set of features, class and an X, y to fit to  ]=====
def predict_labels(X, y, X_test, clf_class, **kwargs):
	clf = clf_class(**kwargs)
	clf.fit(X ,y)
	# print "Labels ", clf.classes_
	return clf.predict(X_test)


#=====[ Tests accuracy with hold out sets  ]=====
def stratified_cv(X, y, clf_class, shuffle=True, n_folds=10, **kwargs):
	stratified_k_fold = cross_validation.StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
	y_pred = y.copy()

	#=====[ Predict for each held out fold  ]=====
	for i, j in stratified_k_fold:
		X_train, X_test = X[i], X[j]
		y_train = y[i]
		clf = clf_class(**kwargs)
		clf.fit(X_train,y_train)
		y_pred[j] = clf.predict(X_test)
	return y_pred


#=====[ Tests prediction while holding out entire files to ensure that we don't test against a squat 
#=====[ from a user we've trained on  ]=====
def rnd_prediction(training_data, labels, file_names, clf_class, toIgnore=None, num_iters=10, **kwargs):
	
	#=====[ Instantiate our personal trainer for feature extraction ]=====
	accuracy = 0

	#=====[ Randomly leave out one of the files and test on it num_iter times ]======
	names = list(set(file_names))
	for name in names:
		 
		toIgnore = name
		squats = []
		
		to_ignore = name
	
		#=====[ Creates training examples and labels without the file to ignore  ]=====
		X = [x for index, x in enumerate(training_data) if file_names[index] != toIgnore]
		Y = [y for index, y in enumerate(labels) if file_names[index] != toIgnore]
		
		#=====[ Creates test examples and labels without the file to ignore  ]=====
		X_test = [x for index, x in enumerate(training_data) if file_names[index] == toIgnore]
		y_test = [y for index, y in enumerate(labels) if file_names[index] == toIgnore]
		
		local_accuracy = metrics.accuracy_score(y_test, predict_labels(X, Y, X_test, clf_class, **kwargs))
	
		accuracy += local_accuracy
		
		# print predictProbs(X,Y,X_test,clf_class,**kwargs)
		# print(toIgnore, local_accuracy)

	return accuracy/len(names)