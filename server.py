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
results = None

#=====[ Squat Inference Setup ]=====
pt = Personal_Trainer({'squat':'NeckY','pushup':'NeckY'}, auto_start=True)

################################################################################
####################[ HANDLING REQUESTS ]#######################################
################################################################################

@app.route("/get_results/", methods=['GET'])
def get_results():
	return results

@app.route("/analyze_raw/<file_name>", methods=['POST'])
def analyze_raw(file_name):

	#=====[ Load json data ]=====
	data = request.get_data()

	#=====[ Write coordinate data to file ]=====
	to_write = open(file_name,'wb')
	to_write.write(data)
	to_write.close()

	#=====[ Analyze Reps ]=====
	try:
		results = pt.analyze_reps('squat',file_name,auto_analyze=True,verbose=False)
	except Exception as e:
		print e 
		results = ''

	return results

if __name__ == '__main__':
	app.run(host='0.0.0.0')
