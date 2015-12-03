import pandas as pd
import numpy as np
import pickle
import sys
from sklearn import preprocessing
from pprint import pprint
from collections import defaultdict

sys.path.append('data')

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
		self.keys_to_indices = {}
		for i, key in enumerate(keysXYZ.columns):
			self.keys_to_indices[key] = i

	def load_squats(self, file):
		self.squats = pickle.load(open(file,"rb"))

	def load_advanced_squats(self,file):
		data = pickle.load(open(file,"rb"))
		self.squats = data['X']
		self.labels = data['Y']
		self.file_names = data['file_names']

	#=====[ Does basic preprocessing for squats from data source: squat separation, normalization, etc. ]=====
	#=====[ Takes data(an array of frames) and a label to be applied to each  ]=====
	#=====[ THIS IS TO BE ALTERED TO ACCEPT AN ARRAY OF LABELS THAT IS OF EQUAL LENGTH TO DATA  ]=====
	def analyze_squats(self, data, label, z_coords=False, epsilon=0.05, gamma=20, delta=0.5, beta=1):

		#=====[ Get data from python file and place in DataFrame ]=====
		if not z_coords:
			df = pd.DataFrame(data,columns=keysXY.columns)
			return [(squat, label) for squat in ss.separate_squats(df, self.key)]
		else:
			df = pd.DataFrame(data,columns=keysXYZ.columns)
			return [(squat, label) for squat in ss.separate_squats(df, self.key, z_coords)]


	def add_squats(self, squats):
		self.squats.extend(squats)

	#=====[ Provides the client with an array of squat DataFrames  ]=====
	def get_squats(self):
		return self.squats

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

	def extract_advanced_features(self, squats=None, labels=None):

		#=====[ If no set of squats passed in to extract features from, extracts features from self.squats  ]=====
		if squats is None:
			squats = self.squats
			labels = [self.labels[name] for name in self.labels]

		#=====[ Initialize dict  ]=====
		advanced_feature_vector = defaultdict(list)
		
		#=====[ Extract advanced features for each squat  ]=====
		for squat in squats:
			advanced_feature_vector['stance_width'].append(fz.extract_stance_shoulder_width(squat, self.key, self.keys_to_indices))
			advanced_feature_vector['stance_alignment'].append(fz.extract_stance_straightness(squat, self.key, self.keys_to_indices))
			advanced_feature_vector['knees_over_toes'].append(fz.extract_feet(squat, self.key, self.keys_to_indices))
			advanced_feature_vector['bend_hips_knees'].append(fz.bend_hips_knees(squat, self.key, self.keys_to_indices))
			advanced_feature_vector['back_straight'].append(fz.back_straight(squat, self.key, self.keys_to_indices))
			advanced_feature_vector['head_alignment'].append(fz.head_aligned_back(squat, self.key, self.keys_to_indices))
			advanced_feature_vector['squat_depth'].append(fz.depth(squat, self.key, self.keys_to_indices))
			advanced_feature_vector['back_hip_angle'].append(fz.back_hip_angle(squat, self.key, self.keys_to_indices))


		#=====[ Return X, and y ]=====
		X = {}
		Y = {}

		#=====[ Specify keys to ignore  ]=====
		toIgnore = ['stance_width', 'back_straight','back_hip_angle']

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



	#=====[ Returns set of squats and extracted features  ]=====
	def get_X(self):
		return self.X

	#=====[ Returns labels for squats  ]=====
	def get_Y(self):
		return self.Y


