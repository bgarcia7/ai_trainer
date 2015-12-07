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

#=====[ Import Data ]=====
import coordKeys as keysXY
import coordKeysZ as keysXYZ
import label_columns as feature_names

class Personal_Trainer:

	def __init__(self, key):
		self.key = key
		self.squats = []


	def load_squats(self,file):
		data = pickle.load(open(file,"rb"))
		self.squats = data['X']
		self.labels = data['Y']
		self.file_names = data['file_names']


	#=====[ Does basic preprocessing for squats from data source: squat separation, normalization, etc. ]=====
	def analyze_squats(self, data, labels, z_coords=False, epsilon=0.05, gamma=20, delta=0.5, beta=1):

		#=====[ Get data from python file and place in DataFrame ]=====
		if not z_coords:
			df = pd.DataFrame(data,columns=keysXY.columns)
			return [(squat, labels) for squat in ss.separate_squats(df, self.key)]
		else:
			df = pd.DataFrame(data,columns=keysXYZ.columns)
			return [(squat, labels) for squat in ss.separate_squats(df, self.key, z_coords)]

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
		return self.classifiers[key].predict(X)

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
	def extract_advanced_features(self, squats=None, labels=None, toIgnore=None):

		#=====[ If no set of squats passed in to extract features from, extracts features from self.squats  ]=====
		if squats is None:
			squats = self.squats
			labels = [self.labels[name] for name in self.labels]

		#=====[ Initialize dict  ]=====
		advanced_feature_vector = defaultdict(list)
		
		#=====[ Extract advanced features for each squat  ]=====
		for squat in squats:
			squat = fz.get_states(squat,self.key)
			advanced_feature_vector['stance_width'].append(fz.stance_shoulder_width(squat))
			advanced_feature_vector['stance_alignment'].append(fz.stance_straightness(squat))
			advanced_feature_vector['knees_over_toes'].append(fz.knees_over_toes(squat))
			advanced_feature_vector['bend_hips_knees'].append(fz.bend_hips_knees(squat))
			advanced_feature_vector['back_straight'].append(fz.back_straight(squat))
			advanced_feature_vector['head_alignment'].append(fz.head_aligned_back(squat))
			advanced_feature_vector['squat_depth'].append(fz.depth(squat))
			advanced_feature_vector['back_hip_angle'].append(fz.back_hip_angle(squat))

		#=====[ Return X, and y ]=====
		X = {}
		Y = {}

		for index, feature in enumerate(advanced_feature_vector):
			training_data = np.array([training_example for training_example in advanced_feature_vector[feature]])
		
			#=====[ Try to fit_transform data, print feature name if fail  ]=====
			try:
				if feature not in toIgnore:
					X[feature] = preprocessing.StandardScaler().fit_transform(training_data)
					Y[feature] = labels[index]	    
			except Exception as e:
				print e, feature
		
		return X, Y, self.file_names	

	#=====[ Extracts features from squats and prepares X, an mxn matrix with m squats and n features per squat  ]=====
	def extract_all_advanced_features(self, squats=None, labels=None, toIgnore=None):
		
		X, Y, file_names = self.extract_advanced_features(squats, labels, toIgnore)

		return np.concatenate([X[x] for x in X],axis=1), Y, file_names


