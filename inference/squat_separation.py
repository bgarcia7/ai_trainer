import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import normalization as nz

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
def get_local_mins(y_coords, epsilon=0.05, gamma=20, delta=0.5, beta=1):
		
	local_mins = []
	height = np.min(y_coords)
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


#=====[ Separates squats in a given dataframe based on changes in y-coord of specified column "key"  ]=====
def separate_squats(df, key, z_coords=False, epsilon=0.05, gamma=20, delta=0.5, beta=1):
		
	y_coords = df.get(key)
	mins = get_local_mins(y_coords, epsilon, gamma, delta, beta)
	squats = []

	#=====[ Get points from DF between each max found -- constitutes a single squat ]=====
	for index,x in enumerate(mins):
		if(index == len(mins) -1 ):
			continue
		squat = (df.loc[x:mins[index+1]-1]).copy(True)
		squats.append(squat.set_index([range(squat.shape[0])]))

	return nz.normalize(df, squats, z_coords)
