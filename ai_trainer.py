import pandas as pd
import numpy as np
import pickle
import sys
from sklearn import preprocessing
from pprint import pprint
from collections import defaultdict

sys.path.append('data')
sys.path.append('inference')

#=====[ Import our utils  ]=====
import squat_separation as ss
import featurizer as fz
import pu_featurizer as pfz

#=====[ Import Data ]=====
import coordKeysZ as keysXYZ
import label_columns as feature_names

class Personal_Trainer:

	def __init__(self, key):
		self.key = key
		self.squats = []

	#=====[ Loads a pickled file and stores squat values  ]=====
	def load_squats(self,file):
		data = pickle.load(open(file,"rb"))
		self.squats = data['X']
		self.labels = data['Y']
		self.file_names = data['file_names']


	#=====[ Does basic preprocessing for squats from data source: squat separation, normalization, etc. ]=====
	def analyze_squats(self, data_file, labels=None, epsilon=0.15, gamma=20, delta=0.5, beta=1):

		return [squat for squat in ss.separate_squats(data_file, self.key, keysXYZ.columns)]

	#=====[ Adds squats to personal trainer's list of squats  ]=====
	def add_squats(self, squats):
		self.squats.extend(squats)

	#=====[ Provides the client with an array of squat DataFrames  ]=====
	def get_squats(self):
		return self.squats

	#=====[ Provides the client with a DataFrame of squat labels  ]=====
	def get_labels(self):
		return self.labels

	#=====[ Provides the client with an array of squat DataFrames  ]=====
	def get_file_names(self):
		return self.file_names

	#=====[ Sets classifiers for the personal trainer  ]=====
	def set_classifiers(self, classifiers):
		self.classifiers = classifiers

	#=====[ Classies an example based on a specified key  ]=====
	def classify(self, key, X):
		try:
			return self.classifiers[key].predict(X)
		except:
			return None
		
		#=====[ THIS SHOULD BE A RETURN self.classifiers... etc  ]=====

	def get_classifiers(self):
		return self.classifiers

	#=====[ Extracts features from squats and prepares X, an mxn matrix with m squats and n features per squat  ]=====
	def extract_features(self, squats=None):

		#=====[ If no set of squats passed in to extract features from, extracts features from self.squats  ]=====
		if squats is None:
			squats = self.squats

		feature_vectors = []
		labels = []
		
		#=====[ Extract basic features for each squat  ]=====
		for squat in squats:
			feature_vectors.append(fz.extract_basic(squat[0], self.key))
			labels.append(squat[1])

		#=====[ Return X, and y ]=====
		X = np.concatenate(feature_vectors,axis=0)
		X = preprocessing.StandardScaler().fit_transform(X)
		y = np.array(labels)

		return X, y

	#=====[ Extracts advanced features from squats and prepares X, a dictionary of mxn matrices with m squats and n features per squat for each of various keys  ]=====
	def extract_advanced_features(self, multiples=[0.5], squats=None, labels=None, toIgnore=[]):

		#=====[ If no set of squats passed in to extract features from, extracts features from self.squats  ]=====
		if squats is None:
			squats = self.squats
			labels = self.labels

		#=====[ Get Feature Vector ]=====
		advanced_feature_vector = fz.get_advanced_feature_vector(squats,self.key,multiples)

		#=====[ Return X, and y ]=====
		X = {}
		Y = {}

		for feature in advanced_feature_vector:
			training_data = np.array([training_example for training_example in advanced_feature_vector[feature]])
		
			#=====[ Try to fit_transform data, print feature name if fail  ]=====
			try:
				if feature not in toIgnore:
					X[feature] = preprocessing.StandardScaler().fit_transform(training_data)
					Y[feature] = labels[feature]	    
			except Exception as e:
				print e, feature
		
		return X, Y, self.file_names	

	#=====[ Extracts features from squats and prepares X, an mxn matrix with m squats and n features per squat  ]=====
	def extract_all_advanced_features(self, multiples=[0.5], squats=None, labels=None, toIgnore=[]):
		
		X, Y, file_names = self.extract_advanced_features(squats=squats, labels=labels, toIgnore=toIgnore,multiples=multiples)

		return np.concatenate([X[x] for x in X],axis=1), Y, file_names

	# def get_prediction_features(self, squats, multiples=[0.5],toIgnore=[]):
	# 	#=====[ Get feature vectors ]=====
	# 	advanced_feature_vector = fz.get_advanced_feature_vector(squats,self.key,multiples)

	# 	#=====[ Build m x n feature matrix  ]=====
	# 	X = {}
	# 	for index, feature in enumerate(advanced_feature_vector):
			
	# 		training_data = np.array([training_example for training_example in advanced_feature_vector[feature]])
	# 		#=====[ Try to fit_transform data, print feature name if fail  ]=====
	# 		try:
	# 			if feature not in toIgnore:
	# 				X[feature] = preprocessing.StandardScaler().fit_transform(training_data)

	# 		except Exception as e:
	# 			print e, feature

	# 	return X, X.keys()

	#=====[ Gets feature vectors for prediction of data  ]=====
	def get_prediction_features(self,squats):
			
		#=====[ Retreives relevant training data for each classifier  ]=====
		X0, Y, file_names = self.extract_advanced_features(squats=squats, multiples=[0.5])
		X1, Y, file_names = self.extract_advanced_features(squats=squats, multiples=[0.2, 0.4, 0.6, 0.8])
		X3, Y, file_names  = self.extract_advanced_features(squats=squats, multiples=[0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

		#=====[ Sets up dictionary of feature vectors  ]=====
		X= {}
		X['bend_hips_knees'] = X3['bend_hips_knees']
		X['stance_width'] = X1['stance_width']
		X['squat_depth'] = X0['squat_depth']
		X['knees_over_toes'] = np.concatenate([X3[x] for x in X3],axis=1)
		X['back_hip_angle'] = np.concatenate([X0[x] for x in X0],axis=1)

		return X


	def extract_pu_features(self, multiples=[0.5], reps=None, labels=None, toIgnore=[]):

		#=====[ If no set of squats passed in to extract features from, extracts features from self.squats  ]=====
		if reps is None:
			reps = self.squats
			labels = self.labels

		#=====[ Get Feature Vector ]=====
		advanced_feature_vector = pfz.get_advanced_feature_vector(reps,self.key,multiples)

		#=====[ Return X, and y ]=====
		X = {}
		Y = {}

		for feature in advanced_feature_vector:
			training_data = np.array([training_example for training_example in advanced_feature_vector[feature]])
		
			#=====[ Try to fit_transform data, print feature name if fail  ]=====
			try:
				if feature not in toIgnore:
					X[feature] = preprocessing.StandardScaler().fit_transform(training_data)
					Y[feature] = labels[feature]	    
			except Exception as e:
				print e, feature
		
		return X, Y, self.file_names	
