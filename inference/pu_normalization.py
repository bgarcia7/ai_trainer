import numpy as np
import pandas as pd

def pushup_is_right(df):
	return np.median(df['NeckX']) > np.median(df['SpineBaseX'])

def direction_substring(dir_bool):
    if dir_bool:
        return 'Right'
    else:
        return 'Left'

#=====[ Provides max y-coord of head  ]=====
def y_upper_bound(df):
	return np.min(df['HeadY'][df.shape[0]/3:df.shape[0]*2/3])

#=====[ Provides median y-coord of feet  ]=====
def y_lower_bound(df, rightward):
	return np.max(df['Hand' + direction_substring(rightward) + 'Y'][df.shape[0]/3:df.shape[0]*2/3])

#=====[ Provides centered x-coord for pushup  ]=====
def x_zero(df, rightward):
	return np.median(df['Foot' + direction_substring(rightward) + 'X'])

#=====[ Provides centered z-coord for pushup  ]=====
def z_zero(df):
	return np.median(df['SpineMidZ'])

#=====[ Provides factor to scale height to 1  ]=====
def scaling_factor(df, rightward):
	return np.abs(y_upper_bound(df) - y_lower_bound(df, rightward))

#=====[ Normalizes pushups to keep feet --> head = 0 --> 1  ]=====
def normalize(df, pushups):
	
	#=====[ Normalizing constants for the entire set of exercises ]=====
	rightward = pushup_is_right(df)
	y_head = y_upper_bound(df)
	scale = scaling_factor(df, rightward)
	x_start = x_zero(df, rightward)
	z_midpoint = z_zero(df)
	
	for pushup in pushups:
		
		#=====[ Even columns are x-coordinates, odd columns are y-coordinates -- normalize respectively ]=====
		for index, col in enumerate(pushup.columns):
			if index % 3 == 2:
				pushup[col] = pushup[col].apply((lambda z: ((z - z_midpoint)/scale)))
			elif index % 3 == 1:
				pushup[col] = pushup[col].apply((lambda y: ((y - y_head)/scale)))
			else:
				pushup[col] = pushup[col].apply(lambda x: (abs((x - x_start)/scale)))        

	return pushups
