import os
import sys

#=====[ Import inference tools  ]=====
sys.path.append('inference')
from ai_trainer import Personal_Trainer

import classification

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
pt.load_squats(os.path.join('data/data_sets','multipleClass3.p'))
pt.set_classifiers(classification.get_classifiers(pt))

################################################################################
####################[ HANDLING REQUESTS ]#######################################
################################################################################

@app.route("/")
def home():
	return pt.get_classifiers()

@app.route("/analyze/<file_name>")
def analyze(file_name):
	print file_name
	squats = pt.analyze_squats(file_name)
	X, keys = pt.get_prediction_features(squats)
	for key in keys:
		print key,':', pt.classify(key, X),'\n'

if __name__ == '__main__':
	app.run()