import pandas as pd
import numpy as np
import sys

sys.path.append('data')

#=====[ Import our utils  ]=====
import squat_separation as ss
import featurizer as fz

#=====[ Import Data ]=====
import coordData3 as cd
import coordKeys as keys


class Personal_Trainer:

	def __init__(self, key):
		self.key = key


	#=====[ Does basic preprocessing for squats from data source: squat separation, normalization, etc. ]=====
	def analyze_squats(self):

		#=====[ Get data from python file and place in DataFrame ]=====
		data = cd.data
		df = pd.DataFrame(data,columns=keys.columns)
		self.squats = ss.separate_squats(df, self.key)

	#=====[ Provides the client with an array of squat DataFrames  ]=====
	def get_squats(self):
		return self.squats

	#=====[ Extracts features from squats and prepares X, an mxn matrix with m squats and n features per squat  ]=====
	def extract_features(self):
		
		feature_vectors = []
		
		#=====[ Extract features for each squat  ]=====
		for squat in self.squats:
			feature_vectors.append(fz.extract_basic(squat, self.key))

		#=====[ Create training set X ]=====
		self.X = np.concatenate(feature_vectors,axis=0)

	#=====[ Returns set of squats and extracted features  ]=====
	def get_X(self):
		return self.X


