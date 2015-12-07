import os
import sys

#=====[ Import inference tools  ]=====
sys.path.append('inference')

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

#=====[ Inference Setup ]=====
pt = Personal_Trainer('NeckY')
pt.load_squats(os.path.join('data/data_sets','multipleClass3.p'))
pt.set_classifiers(classification.get_classifiers(pt))


################################################################################
####################[ HANDLING REQUESTS ]#######################################
################################################################################

@app.route("/")
def home():
	return 'hi'