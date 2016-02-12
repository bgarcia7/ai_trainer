import os
import sys
import pickle

#=====[ Import inference tools  ]=====
sys.path.append('inference')
from ai_trainer import Personal_Trainer
import classification_ftopt
import utils as ut

#=====[ Flask ]=====
from flask import Flask
from flask import request
from flask import render_template
from flask import send_from_directory
from flask import abort
from flask import make_response
from flask import redirect
from flask import render_template
from flask import url_for
from flask import flash

#=====[ WebApp Setup ]=====
base_dir = os.path.split(os.path.realpath(__file__))[0]
static_dir = os.path.join(base_dir, 'static')
app = Flask(__name__, static_folder=static_dir)
results = []

#=====[ Squat Inference Setup ]=====
pt = Personal_Trainer({'squat':'NeckY','pushup':'NeckY'})

try:
	#=====[ Get classifiers from pickled file ]=====
	squat_classifiers = pickle.load(open(os.path.join('inference/','squat_classifiers_ftopt.p'),'rb'))
	pushup_classifiers = pickle.load(open(os.path.join('inference/','pushup_classifiers_ftopt.p'),'rb'))
	
	ut.print_success('Loaded trained classifiers')
	try:
		#=====[ Set classifiers for our trainer ]=====
		pt.set_classifiers('squat',squat_classifiers)
		pt.set_classifiers('pushup',pushup_classifiers)

		ut.print_success('Saved classifiers')
	
	except Exception as e:
		ut.print_failure('Could not save classifiers: ' + str(e))
except Exception as e:
	ut.print_failure('Could not load classifiers:' + str(e))

################################################################################
####################[ HANDLING REQUESTS ]#######################################
################################################################################

@app.route("/")
def home():
	return pt.get_classifiers()

@app.route("/analyze/<file_name>")
def analyze(file_name):
	
	
	#=====[ Analyze squat data to pull out normalized, spliced reps ]=====
	squats = pt.analyze_reps('squat', file_name)
	
	#=====[ Extract feature vectors from squats for each exercise componenet  ]=====
	squat_feature_vectors = pt.get_prediction_features_opt('squat',squats)
	
	results = {}
	#=====[ Run classification on each squat component and report results  ]=====
	for key in squat_feature_vectors:
	    X = squat_feature_vectors[key]
	    classification = pt.classify('squat', key, X)
	    results[key] = classification
		
	return results

def advice():
	results = analyze('squatData.txt')
	output_advice = pt.get_advice('squat',results)
	ut.print_success('Feedback retrieved')
	advice_file = open('advice_file.txt','wb')
	advice_file.write(output_advice)

@app.route("/analyze_raw", methods=['POST'])
def analyze_raw():
	# to_write = open('squatData.txt','wb')
	# to_write.write(request.data)
	# ut.print_success('Data written to file')
	# advice()
	# ut.print_success('Advice file populated')
	return 'OK'

if __name__ == '__main__':
	app.run(host='0.0.0.0')
