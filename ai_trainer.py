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
import utils as ut

#=====[ Import Data ]=====
import coordKeysZ as keysXYZ
import label_columns as feature_names

class Personal_Trainer:

	def __init__(self, key):
		self.key = key
		self.squats = []
		self.file_names = None

	#=====[ Loads a pickled file and stores squat values  ]=====
	def load_reps(self,file):
		data = pickle.load(open(file,"rb"))
		self.reps = data['X']
		self.labels = data['Y']
		self.file_names = data['file_names']


	#=====[ Does basic preprocessing for squats from data source: squat separation, normalization, etc. ]=====
	def analyze_reps(self, data_file, labels=None, epsilon=0.15, gamma=20, delta=0.5, beta=1):

		reps = [rep for rep in ss.separate_squats(data_file, self.key, keysXYZ.columns)]
		ut.print_success('Reps segmented and normalized')

		return reps

	#=====[ Adds squats to personal trainer's list of squats  ]=====
	def add_reps(self, reps):
		self.reps.extend(reps)

	#=====[ Provides the client with an array of squat DataFrames  ]=====
	def get_reps(self):
		return self.reps

	#=====[ Provides the client with a DataFrame of squat labels  ]=====
	def get_labels(self):
		return self.labels

	#=====[ Provides the client with an array of squat DataFrames  ]=====
	def get_file_names(self):
		return self.file_names

	#=====[ Sets classifiers for the personal trainer  ]=====
	def set_classifiers(self, classifiers):
		self.classifiers = classifiers
		ut.print_success("Classifiers stored")


	#=====[ Classies an example based on a specified key  ]=====
	def classify(self, key, X):
		try:
			prediction = self.classifiers[key].predict(X)
			ut.print_success(key + ': reps classified')
			return prediction
			
		except:
			ut.print_failure(key + ': reps not classified')
			return None
		
	def get_classifiers(self):
		return self.classifiers

	#=====[ Extracts advanced features from squats and prepares X, a dictionary of mxn matrices with m squats and n features per squat for each of various keys  ]=====
	def extract_advanced_features(self, multiples=[0.5], reps=None, labels=None, toIgnore=[], predict=False):

		#=====[ If no set of squats passed in to extract features from, extracts features from self.squats  ]=====
		if reps is None:
			reps = self.reps
			labels = self.labels

		#=====[ Get Feature Vector ]=====
		advanced_feature_vector = fz.get_advanced_feature_vector(reps,self.key,multiples)

		#=====[ Set data to have 0 mean and unit variance  ]=====
		X, Y = fz.transform_data(advanced_feature_vector, labels, toIgnore, predict)		
		
		return X, Y, self.file_names	

	#=====[ Extracts features from squats and prepares X, an mxn matrix with m squats and n features per squat  ]=====
	def extract_all_advanced_features(self, multiples=[0.5], reps=None, labels=None, toIgnore=[], predict=False):
		
		X, Y, file_names = self.extract_advanced_features(reps=reps, labels=labels, toIgnore=toIgnore,multiples=multiples, predict=predict)

		return np.concatenate([X[x] for x in X],axis=1), Y, file_names

	#=====[ Gets feature vectors for prediction of data  ]=====
	def get_prediction_features(self,reps):
			
		#=====[ Retreives relevant training data for each classifier  ]=====
		X0, Y, file_names = self.extract_advanced_features(reps=reps, multiples=[0.5], predict=True)
		X1, Y, file_names = self.extract_advanced_features(reps=reps, multiples=[0.2, 0.4, 0.6, 0.8], predict=True)
		X3, Y, file_names = self.extract_advanced_features(reps=reps, multiples=[0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], predict=True)

		#=====[ Sets up dictionary of feature vectors  ]=====
		X= {}
		X['bend_hips_knees'] = preprocessing.StandardScaler().fit_transform(X3['bend_hips_knees'])
		X['stance_width'] = preprocessing.StandardScaler().fit_transform(X1['stance_width'])
		X['squat_depth'] = preprocessing.StandardScaler().fit_transform(X0['squat_depth'])
		X['knees_over_toes'] = preprocessing.StandardScaler().fit_transform(np.concatenate([X3[x] for x in X3],axis=1))
		X['back_hip_angle'] = preprocessing.StandardScaler().fit_transform(np.concatenate([X0[x] for x in X0],axis=1))

		ut.print_success('Features extracted')

		return X

	#=====[ Extracts advanced features from pushups and prepares X, a dictionary of mxn matrices with m squats and n features per squat for each of various keys  ]=====
	def extract_pu_features(self, multiples=[0.5], reps=None, labels=None, toIgnore=[]):

		#=====[ If no set of squats passed in to extract features from, extracts features from self.squats  ]=====
		if reps is None:
			reps = self.reps
			labels = self.labels

		#=====[ Get Feature Vector ]=====
		advanced_feature_vector = pfz.get_advanced_feature_vector(reps,self.key,multiples)

		#=====[ Set data to have 0 mean and unit variance  ]=====
		X, Y = fz.transform_data(advanced_feature_vector, labels, toIgnore)		
		
		return X, Y, self.file_names	
