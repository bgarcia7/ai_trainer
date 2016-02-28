import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn import preprocessing
from pprint import pprint
from collections import defaultdict

sys.path.append('data')
sys.path.append('inference')
sys.path.append('feedback')

#=====[Preppend ../ for notebook]=====
sys.path.append('inference')
sys.path.append('data')
sys.path.append('feedback')

#=====[ Import our utils  ]=====
import rep_separation as rs
import featurizer as fz
import pu_featurizer as pfz
import utils as ut
import result_interpretation as advice

#=====[ Import Data ]=====
import coordKeysZ as keysXYZ
import label_columns as feature_names

class Personal_Trainer:

	def __init__(self, keys, auto_start=False):
		self.keys = keys
		self.reps = defaultdict(list)
		self.labels = defaultdict(list)
		self.file_names = defaultdict(list)
		self.classifiers = {}

		#=====[ Rehydrate classifiers if auto_start enabled  ]=====
		if auto_start:
			if 'squat' in keys:
				self.classifiers['squat'] = pickle.load(open(os.path.join('inference/','squat_classifiers_ftopt.p'),'rb'))
			if 'pushup' in keys:
				self.classifiers['pushup'] = pickle.load(open(os.path.join('inference/','pushup_classifiers_ftopt.p'),'rb'))

	#=====[ Loads a pickled file and stores squat values  ]=====
	def load_reps(self, exercise, file):
		data = pickle.load(open(file,"rb"))
		self.reps[exercise] = data['X']
		self.labels[exercise] = data['Y']
		self.file_names[exercise] = data['file_names']


	#=====[ Does basic preprocessing for squats from data source: squat separation, normalization, etc. ]=====
	def analyze_reps(self, exercise, data_file, labels=None, epsilon=0.15, gamma=20, delta=0.5, beta=1, auto_analyze=False, verbose=False):

		reps = [rep for rep in rs.separate_reps(data_file, exercise, self.keys[exercise], keysXYZ.columns)]
		
		if verbose:
			ut.print_success('Reps segmented and normalized for ' + exercise)

		if not auto_analyze:
			return reps
		
		#=====[ Get feature vector  ]=====
		feature_vectors = self.get_prediction_features_opt(exercise, reps, verbose)
		
		#=====[ Get results for classifications and populate dictionary  ]=====
		results = {}

		if verbose:
			print "\n\n###################################################################"
			print "######################## Classification ###########################"
			print "###################################################################\n\n"

		for key in feature_vectors:
		    X = feature_vectors[key]
		    classification = self.classify(exercise, key, X, verbose)
		    results[key] = classification
		    if verbose:
		    	print '\n\n', key ,':\n', classification, '\n'

		#=====[ Print advice based on results  ]=====
		print "\n\n###################################################################"
		print "########################### Feedback ##############################"
		print "###################################################################\n\n"
		return self.get_advice(exercise, results)

	#=====[ Adds reps to personal trainer's list of squats  ]=====
	def add_reps(self, exercise, reps):
		self.reps[exercise].extend(reps)

	#=====[ Provides the client with an array of squat DataFrames  ]=====
	def get_reps(self, exercise):
		return self.reps[exercise]

	#=====[ Provides the client with a DataFrame of squat labels  ]=====
	def get_labels(self, exercise):
		return self.labels[exercise]

	#=====[ Provides the client with an array of squat DataFrames  ]=====
	def get_file_names(self, exercise):
		return self.file_names[exercise]

	#=====[ Sets classifiers for the personal trainer  ]=====
	def set_classifiers(self, exercise, classifiers):
		self.classifiers[exercise] = classifiers
		ut.print_success("Classifiers stored for " + exercise)


	#=====[ Classies an example based on a specified key  ]=====
	def classify(self, exercise, key, X, verbose=False):
		
		try:
			prediction = self.classifiers[exercise][key].predict(X)
			if verbose:
				ut.print_success(key + ': reps classified')
			return prediction
			
		except Exception as e:
			print e
			ut.print_failure(key + ': reps not classified')
			return None
		
	def get_classifiers(self, exercise):
		return self.classifiers[exercise]

	def get_advice(self, exercise, results):
		to_return = ""
		for message in advice.advice(exercise, results):
			print message
			to_return += message + '\n'
		return to_return

	#=====[ Gets feature vectors for prediction of data  ]=====
	def get_prediction_features(self, exercise, reps):
			
		if exercise is 'squat':
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

		elif exercise is 'pushup':
			#=====[ Retreives relevant training data for each classifier  ]=====
			X3, Y, file_names  = self.extract_pu_features(reps=reps, multiples=[0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], predict=True)
			X4, Y, file_names = self.extract_pu_features(reps=reps, multiples=[float(x)/100 for x in range(100)], predict=True)

			X30 = np.concatenate([X3[x] for x in X3],axis=1)
			X40 = np.concatenate([X4[x] for x in X4],axis=1)

			#=====[ Sets up dictionary of feature vectors  ]=====
			X = {}
			X['head_back'] = preprocessing.StandardScaler().fit_transform(X40)
			X['knees_straight'] = preprocessing.StandardScaler().fit_transform(X30)
			X['elbow_angle'] = preprocessing.StandardScaler().fit_transform(X3['elbow_angle'])


		ut.print_success('Features extracted for ' + exercise)

		return X

	def get_prediction_features_opt(self, exercise, reps, verbose=False):
			
		if exercise is 'squat':

			#=====[ Load feature indicies  ]=====
			feature_indices = pickle.load(open(os.path.join('inference/','squat_feature_indices.p'),'rb'))

			#=====[ Retreives relevant training data for each classifier  ]=====
			X3, Y, file_names = self.extract_advanced_features(reps=reps, multiples=[float(x)/20 for x in range(1,20)],predict=True)
			X30 = np.concatenate([X3[x] for x in X3],axis=1)

			#=====[ Sets up dictionary of feature vectors  ]=====
			X= {}
			X['bend_hips_knees'] = X30[:,feature_indices['bend_hips_knees']]
			X['stance_width'] = X30[:,feature_indices['stance_width']]
			X['squat_depth'] = X30[:,feature_indices['squat_depth']]
			X['knees_over_toes'] = X30[:,feature_indices['knees_over_toes']]
			X['back_hip_angle'] = X30[:,feature_indices['back_hip_angle']]

		elif exercise is 'pushup':

			#=====[ Load feature indicies  ]=====
			feature_indices = pickle.load(open(os.path.join('inference/','pushup_feature_indices.p'),'rb'))

			#=====[ Retreives relevant training data for each classifier  ]=====
			X3, Y, file_names  = self.extract_pu_features(reps=reps, multiples=[0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], predict=True)
			X30 = np.concatenate([X3[x] for x in X3],axis=1)

			#=====[ Sets up dictionary of feature vectors  ]=====
			X = {}
			X['head_back'] = X30[:,feature_indices['head_back']]
			X['knees_straight'] = X30[:,feature_indices['knees_straight']]
			X['elbow_angle'] = X30[:,feature_indices['elbow_angle']]


		if verbose:
			ut.print_success('Features extracted for ' + exercise)

		return X



	#=====[ Extracts advanced features from pushups and prepares X, a dictionary of mxn matrices with m squats and n features per squat for each of various keys  ]=====
	def extract_pu_features(self, multiples=[0.5], reps=None, labels=None, toIgnore=[], predict=False):

		#=====[ If no set of squats passed in to extract features from, extracts features from self.reps  ]=====
		if reps is None:
			reps = self.reps['pushup']
			labels = self.labels['pushup']

		#=====[ Get Feature Vector ]=====
		advanced_feature_vector = pfz.get_advanced_feature_vector(reps,self.keys['pushup'],multiples)

		#=====[ Set data to have 0 mean and unit variance  ]=====
		X, Y = fz.transform_data(advanced_feature_vector, labels, toIgnore, predict)		
		
		return X, Y, self.file_names['pushup']	

	#=====[ Extracts advanced features from squats and prepares X, a dictionary of mxn matrices with m squats and n features per squat for each of various keys  ]=====
	def extract_advanced_features(self, multiples=[0.5], reps=None, labels=None, toIgnore=[], predict=False):

		#=====[ If no set of squats passed in to extract features from, extracts features from self.reps  ]=====
		if reps is None:
			reps = self.reps['squat']
			labels = self.labels['squat']

		#=====[ Get Feature Vector ]=====
		advanced_feature_vector = fz.get_advanced_feature_vector(reps,self.keys['squat'],multiples)

		#=====[ Set data to have 0 mean and unit variance  ]=====
		X, Y = fz.transform_data(advanced_feature_vector, labels, toIgnore, predict)		
		
		return X, Y, self.file_names['squat']	


