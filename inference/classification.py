#=====[ Import preprocessing tools  ]=====
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics

#=====[ Import Inference models  ]=====
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree

#=====[ Import util tools  ]=====
import random
import pandas as pd
import numpy as np
from collections import defaultdict

#=====[ Tests each classifier and chooses the one with highest accuracy  ]=====
def train_classifiers(trainer):

	#=====[ Instantiates classifiers for each component of the squat  ]=====
	classifiers = {'bend_hips_knees': tree.DecisionTreeClassifier(max_depth=3, criterion="entropy"), 'stance_width': linear_model.LogisticRegression(penalty='l1'),'squat_depth': linear_model.LogisticRegression(penalty='l1'),'knees_over_toes': tree.DecisionTreeClassifier(max_depth=3, criterion="entropy"),'back_hip_angle': linear_model.LogisticRegression()}

	#=====[ Retreives relevant training data for each classifier  ]=====
	X0, Y, file_names = trainer.extract_advanced_features(multiples=[0.5])
	X1, Y, file_names = trainer.extract_advanced_features(multiples=[0.2, 0.4, 0.6, 0.8])
	X3, Y, file_names  = trainer.extract_advanced_features(multiples=[0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

	#=====[ Trains each classifier  ]=====
	classifiers['bend_hips_knees'].fit(X3['bend_hips_knees'], Y['bend_hips_knees'])
	classifiers['stance_width'].fit(X1['stance_width'], Y['stance_width'])
	classifiers['squat_depth'].fit(X0['squat_depth'], Y['squat_depth'])
	
	X30 = np.concatenate([X3[x] for x in X3],axis=1)
	X00 = np.concatenate([X0[x] for x in X0],axis=1)
	coalesced_y = replace_label(Y['knees_over_toes'],2,1)
	classifiers['knees_over_toes'].fit(X30, coalesced_y)
	classifiers['back_hip_angle'].fit(X00, Y['back_hip_angle'])

	return classifiers

#=====[ Replace a label with another from an array of labels  ]=====
def replace_label(Y, to_replace, new_val):
	
	coalesced_y = []
	
	for y in Y:
		if y == to_replace:
			coalesced_y.append(new_val)
		else:
			coalesced_y.append(y)

	return coalesced_y

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