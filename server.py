import os
import sys
import pickle

#=====[ Import inference tools  ]=====
sys.path.append('inference')
from ai_trainer import Personal_Trainer
import classification_ftopt
import utils as ut
import json

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
pt = Personal_Trainer({'squat':'NeckY','pushup':'NeckY'}, auto_start=True)

################################################################################
####################[ HANDLING REQUESTS ]#######################################
################################################################################

@app.route("/", methods=['GET'])
def home():
	return "Hi"

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
	advice_file.close()


@app.route("/get_advice", methods=['GET'])
def get_advice():
	advice_file = open('advice_file.txt','wb')
	return advice_file.read()

@app.route("/analyze_raw", methods=['POST'])
def analyze_raw():
	to_write = open('squatData.txt','wb')
	data = json.loads(request.data)

	to_write.write(data['data'])
	to_write.close()
	# ut.print_success('Data written to file')
	# advice()
	# ut.print_success('Advice file populated')
	return 'OK'

if __name__ == '__main__':
	app.run(host='0.0.0.0')
