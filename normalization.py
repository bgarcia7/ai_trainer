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

#=====[ Provides factor to scale height to 1  ]=====
def scaling_factor(df):
    return np.abs(y_upper_bound(df) - y_lower_bound(df))

#=====[ Normalizes squats to keep feet --> head = -1 --> 0  ]=====
def normalize(squats):
	
	for squat in squats:
    
	    #=====[ Get normalizing constants ]=====
	    y_head = y_upper_bound(squat)
	    scale = scaling_factor(squat)
	    x_midpoint = x_zero(squat)
	    
	    #=====[ Even columns are x-coordinates, odd columns are y-coordinates -- normalize respectively ]=====
	    for index, col in enumerate(squat.columns):
	        if index % 2 == 1:
	            squat[col] = squat[col].apply(lambda y: ((y - y_head)/scale))
	        else:
	            squat[col] = squat[col].apply(lambda x: (x - x_midpoint))

	return squats