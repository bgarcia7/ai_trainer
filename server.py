import os
import sys
import pickle

#=====[ Import inference tools  ]=====
sys.path.append('inference')
from ai_trainer import Personal_Trainer
import classification
import utils as ut

#=====[ Flask ]=====
from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory
from flask import abort
from flask import make_response
from flask import redirect
from flask import flash

#=====[ WebApp Setup ]=====
base_dir = os.path.split(os.path.realpath(__file__))[0]
static_dir = os.path.join(base_dir, 'static')
app = Flask(__name__, static_folder=static_dir)

#=====[ Squat Inference Setup ]=====
pt = Personal_Trainer('NeckY')

try:
	pt.load_squats(os.path.join('data/data_sets','multipleClass4.p'))
	ut.print_success('Training data loaded')
	try:
		classifiers = pickle.load(open(os.path.join('inference/','trained_classifiers.p'),'rb'))
		pt.set_classifiers(classifiers)
		ut.print_success('Classifiers trained')
	except Exception as e:
		ut.print_failure('Could not train classifiers' + str(e))
except Exception as e:
	ut.print_failure('Could not load training data:' + str(e))



################################################################################
####################[ HANDLING REQUESTS ]#######################################
################################################################################

@app.route("/")
def home():
	return pt.get_classifiers()

@app.route("/analyze/<file_name>")
def analyze(file_name):
	
	print file_name
	
	#=====[ Analyze squat data to pull out normalized, spliced reps ]=====
	squats = pt.analyze_squats(file_name)

	#=====[ Extract feature vectors from squats for each exercise componenet  ]=====
	feature_vectors = pt.get_prediction_features(squats)
	print feature_vectors
	
	#=====[ Run classification on each squat component and report results  ]=====
	for key in feature_vectors:
		X = feature_vectors[key]
		print key,':', pt.classify(key, X),'\n'

if __name__ == '__main__':
	app.run()
