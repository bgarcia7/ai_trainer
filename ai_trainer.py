import pandas as pd
import numpy as np
import pickle
import sys

sys.path.append('data')

#=====[ Import our utils  ]=====
import squat_separation as ss
import featurizer as fz

#=====[ Import Data ]=====
import coordKeys as keys


class Personal_Trainer:

	def __init__(self, key):
		self.key = key
		self.squats = []

	def load_squats(self, file):
		self.squats = pickle.load(open(file,"rb"))

	#=====[ Does basic preprocessing for squats from data source: squat separation, normalization, etc. ]=====
	#=====[ Takes data(an array of frames) and a label to be applied to each  ]=====
	#=====[ THIS IS TO BE ALTERED TO ACCEPT AN ARRAY OF LABELS THAT IS OF EQUAL LENGTH TO DATA  ]=====
	def analyze_squats(self, data, label, epsilon=0.05, gamma=20, delta=0.5, beta=1):

		#=====[ Get data from python file and place in DataFrame ]=====
		df = pd.DataFrame(data,columns=keys.columns)
		self.squats.extend([(squat, label) for squat in ss.separate_squats(df, self.key)])

	#=====[ Provides the client with an array of squat DataFrames  ]=====
	def get_squats(self):
		return self.squats

	#=====[ Extracts features from squats and prepares X, an mxn matrix with m squats and n features per squat  ]=====
	def extract_features(self):
		
		feature_vectors = []
		labels = []
		
		#=====[ Extract features for each squat  ]=====
		for squat in self.squats:
			feature_vectors.append(fz.extract_basic(squat[0], self.key))
			labels.append(squat[1])

		#=====[ Create training set X ]=====
		self.X = np.concatenate(feature_vectors,axis=0)
		self.Y = np.array(labels)

	#=====[ Returns set of squats and extracted features  ]=====
	def get_X(self):
		return self.X

	#=====[ Returns labels for squats  ]=====
	def get_Y(self):
		return self.Y


