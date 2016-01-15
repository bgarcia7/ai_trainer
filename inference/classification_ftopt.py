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
import pickle
import os
import pandas as pd
import numpy as np
from collections import defaultdict

#=====[ Tests each classifier and chooses the one with highest accuracy  ]=====
def train_squat_classifiers(trainer):

	#=====[ Load feature indicies  ]=====
	feature_indices = pickle.load(open(os.path.join('../inference/','squat_feature_indices.p'),'rb'))

	#=====[ Instantiates classifiers for each component of the squat  ]=====
	classifiers = {'bend_hips_knees': tree.DecisionTreeClassifier(max_depth=3, criterion="entropy"), 'stance_width': linear_model.LogisticRegression(penalty='l1'),'squat_depth': linear_model.LogisticRegression(penalty='l1'),'knees_over_toes': tree.DecisionTreeClassifier(max_depth=3, criterion="entropy"),'back_hip_angle': linear_model.LogisticRegression()}

	#=====[ Retreives relevant training data for each classifier  ]=====
	X3, Y, file_names = trainer.extract_advanced_features(multiples=[float(x)/20 for x in range(1,20)])
	X30 = np.concatenate([X3[x] for x in X3],axis=1)

	#=====[ Trains each classifier  ]=====
	classifiers['bend_hips_knees'].fit(X30[:,feature_indices['bend_hips_knees']], Y['bend_hips_knees'])
	classifiers['stance_width'].fit(X30[:,feature_indices['stance_width']], Y['stance_width'])
	classifiers['squat_depth'].fit(X30[:,feature_indices['squat_depth']], Y['squat_depth'])
	coalesced_y = replace_label(Y['knees_over_toes'],2,1)
	classifiers['knees_over_toes'].fit(X30[:,feature_indices['knees_over_toes']], coalesced_y)
	classifiers['back_hip_angle'].fit(X30[:,feature_indices['back_hip_angle']], Y['back_hip_angle'])

	return classifiers

def train_pushup_classifiers(trainer):

	#=====[Load feature indicies  ]=====
	feature_indices = pickle.load(open(os.path.join('../inference/','pushup_feature_indices.p'),'rb'))

	#=====[ Instantiates classifiers for each component of the squat  ]=====
	classifiers = {'head_back': linear_model.LogisticRegression(penalty='l1',C=5), 'knees_straight': linear_model.LogisticRegression(penalty='l1', C=8),'elbow_angle': linear_model.LogisticRegression(penalty='l1', C=8)}

	#=====[ Retreives relevant training data for each classifier  ]=====
	X3, Y, file_names  = trainer.extract_pu_features(multiples=[0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
	X30 = np.concatenate([X3[x] for x in X3],axis=1)

	#=====[ Trains each classifier  ]=====
	classifiers['head_back'].fit(X30[:,feature_indices['head_back']], Y['head_back'])
	classifiers['knees_straight'].fit(X30[:,feature_indices['knees_straight']], Y['knees_straight'])
	classifiers['elbow_angle'].fit(X30[:,feature_indices['elbow_angle']], Y['elbow_angle'])

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