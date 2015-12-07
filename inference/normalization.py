import numpy as np
import pandas as pd

#=====[ Provides max y-coord of head  ]=====
def y_upper_bound(df):
	return np.min(df['HeadY'])

#=====[ Provides median y-coord of feet  ]=====
def y_lower_bound(df):
	return np.median(pd.concat([df['FootRightY'],df['FootLeftY']],axis =0))

#=====[ Provides centered x-coord for squat  ]=====
def x_zero(df):
	return np.median(df['SpineMidX'])

#=====[ Provides centered z-coord for squat  ]=====
def z_zero(df):
	return np.median(np.concatenate([df.get('FootLeftZ'),df.get('FootRightZ')],axis=0))

#=====[ Provides factor to scale height to 1  ]=====
def scaling_factor(df):
	return np.abs(y_upper_bound(df) - y_lower_bound(df))

#=====[ Normalizes squats to keep feet --> head = -1 --> 0  ]=====
def normalize(df, squats, z_coords=False):
	
	#=====[ Normalizing constants for the entire set of exercises ]=====
	y_head = y_upper_bound(df)
	scale = scaling_factor(df)
	x_midpoint = x_zero(df)
	if(z_coords):
		z_midpoint = z_zero(df)
	
	for squat in squats:
		
		#=====[ Even columns are x-coordinates, odd columns are y-coordinates -- normalize respectively ]=====
		if(not z_coords):
			for index, col in enumerate(squat.columns):
				if (index % 2) == 1:
					squat[col] = squat[col].apply((lambda y: ((y - y_head)/scale)))
				else:
					squat[col] = squat[col].apply((lambda x: ((x - x_midpoint)/scale)))
		else:
			for index, col in enumerate(squat.columns):
				if index % 3 == 2:
					squat[col] = squat[col].apply((lambda z: ((z - z_midpoint)/scale)))
				elif index % 3 == 1:
					squat[col] = squat[col].apply((lambda y: ((y - y_head)/scale)))
				else:
					squat[col] = squat[col].apply(lambda x: ((x - x_midpoint)/scale))        

	return squats
