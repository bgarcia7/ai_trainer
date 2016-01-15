import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import normalization as nz
import pu_normalization as pnz
import os


#=====[ Checks if the point being observed qualifies as a minimum  ]=====
def is_min(y_coords, height, gradient, index, epsilon, beta):
	if np.abs(y_coords[index] - height)/height < epsilon:
		for i in range(1,beta):
			if gradient[index - i] > 0 or gradient[index + i - 1] < 0:
				return False
		return True
		
#=====[ Checks if we're evaluating a new squat  ]=====
def in_new_squat(y_coords, height, index, delta):
	return abs((y_coords[index] - height)/height) > delta
	
#=====[ Gets local maxes within accepted epsilon of global max and with max len(y_coors)/gamma maxes ]=====
def get_local_mins(y_coords, epsilon=0.25, gamma=20, delta=0.5, beta=1):
		
	local_mins = []
	height = np.min(y_coords[len(y_coords)/3:len(y_coords)*2/3])
	gradient = np.gradient(y_coords)
		
	#=====[ Checks gradients to make sure  ]=====
	min_located = False
	for index, dy in enumerate(gradient[2:]):
		if(min_located):
			if in_new_squat(y_coords, height, index, delta):
				min_located = False       
			else:
				continue
					
		if  is_min(y_coords, height, gradient, index, epsilon, beta + 1):
			local_mins.append(index)
			min_located = True
				
	return sorted(local_mins)


#=====[ Separates reps in a given dataframe based on changes in y-coord of specified column "key"  ]=====
def separate_reps(data_file, exercise, key, column_labels, epsilon=0.15, gamma=20, delta=0.5, beta=1):

	front_cut_values = [0, 0, 0, 25, 0, 50, 0, 25, 50, 100, 0, 100]
	back_cut_values = [0, 0, 0, 0, 25, 0, 50, 25, 50, 0, 100, 100]
	epsilon_values = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
	
	if exercise is 'pushup':
		delta = 0.2
		beta = 2

	for iteration in range(0,len(front_cut_values)):
		
		front_cut = front_cut_values[iteration]
		back_cut = back_cut_values[iteration]
		epsilon = epsilon_values[iteration]

		data = []

		#=====[ Format each line of data  ]=====
		# os.path.join('data/raw_data/squat_pushupData_10to20',
		with open(data_file) as f:
			for line in f:
				try:
					if 'Infinity' in line or 'NaN' in line:
						continue
					line = [float(x.replace('\r\n','')) for x in line.split(',')]
					data.append(line)
				except Exception as e:
					continue
					# print e


		#=====[ Make dataframe and readjust indices to take into account front and back cuts  ]=====
		df = pd.DataFrame(data, columns=column_labels)
		df = df[front_cut:df.shape[0]-back_cut]
		df = df.set_index([range(0,df.shape[0])])
			
		y_coords = np.array(df.get(key))
		mins = get_local_mins(y_coords, epsilon, gamma, delta, beta)
		reps = []

		#=====[ Get points from DF between each max found -- constitutes a single rep ]=====
		for index,x in enumerate(mins):
			if(index == len(mins) -1 ):
				continue
			rep = (df.loc[x:mins[index+1]-1]).copy(True)
			reps.append(rep.set_index([range(rep.shape[0])]))

		if len(reps) > 1:
			break

	if exercise is 'squat':
		return nz.normalize(df, reps)
	elif exercise is 'pushup': 
		return pnz.normalize(df, reps)
