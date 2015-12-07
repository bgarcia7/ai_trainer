import pandas as pd
import numpy as np
import math

#======[ Returns index to frame with minimum y-coord for specified key ]=====
def get_min(squat,key):   
    
    #=====[ Return max because of inverse frame of reference of kinect ]=====
    return max([(coord,index) for index, coord in enumerate(squat[key])])[1]

#=====[ Returns index to frame with y-coord closes to the midpoint between start/end and squat position for specified key ]=====
def get_midpoint(squat,start,key):
    
    #=====[ Decide whether getting midpoint between start and squat or squat and end ]=====
    if start:
        start = 1
        end = get_min(squat,key)
    else:
        start = get_min(squat,key)
        end = squat.shape[0] - 1
        
    #=====[ Uses the 'true_mid' as an evaluation metric to find optimal index  ]=====
    true_mid = (squat.iloc[end][key] - squat.iloc[start][key])/2
    deltas = [(np.abs(true_mid - (squat.iloc[end][key] - squat.iloc[index][key])), index) for index in range(start,end)]
    return min(deltas)[1]

#=====[ Returns squat at the first position ]=====
def starting_position(squat):
    return squat.iloc[[1]]

#=====[ Returns index to frame with y-coord closes to the midpoint between start and squat position for specified key ]=====
def start_to_squat(squat,key):
    return squat.iloc[[get_midpoint(squat,start=1,key=key)]]

#=====[ Returns frame with minimum y-coord for specified key ]=====
def squat_position(squat,key):
    return squat.iloc[[get_min(squat,key)]]

#=====[ Returns index to frame with y-coord closes to the midpoint between squat position and end for specified key ]=====
def squat_to_end(squat,key):
    return squat.iloc[[get_midpoint(squat,start=0,key=key)]]

#=====[ function for plotting full set of 25 coordinates for a given frame ]=====
def plotBody(df):
    coords = np.array(df)
    xs = [coords[0][i] for i in range(0,coords.size) if i % 2 == 0]
    #=====[ Plot -1* coords because of kinect's frame of reference ]=====
    ys = [-1*coords[0][i] for i in range(0,coords.size) if i % 2 == 1]
    plt.plot(xs,ys,linestyle='None',marker='o')
    plt.axis([-60,60,-1.2,0.2])

#=====[ Returns angle between three specified joints along the two specified axes  ]=====
def get_angle(state, joint1, joint2, joint3, axis1, axis2):

    bone1 = math.sqrt(math.pow(state[joint2 + axis1] - state[joint1 + axis1], 2) + math.pow(state[joint2 + axis2] - state[joint1 + axis2], 2))
    bone2 = math.sqrt(math.pow(state[joint2 + axis1] - state[joint3 + axis1], 2) + math.pow(state[joint2 + axis2] - state[joint3 + axis2], 2))

    #=====[ Gets distance between the disconnected joints  ]=====
    distance = math.sqrt(math.pow(state[joint1 + axis1] - state[joint3 + axis1], 2) + math.pow(state[joint1 + axis2] - state[joint3 + axis2], 2))
    
    angle = math.acos((math.pow(bone1, 2) + math.pow(bone2, 2) - math.pow(distance, 2)) / (2 * bone1 * bone2))
    return angle

#=====[ Returns ratios of changes between two angles over time  ]=====
def get_angle_changes(angle1, angle2):

    assert(len(angle1) == len(angle2))
    
    #=====[ Gets max angle swept  ]=====
    full_angle1 = angle1[-1] - angle1[0]
    full_angle2 = angle2[-1] - angle2[0]
    
    ratios=[]
    
    for time in range(1,len(angle1)):
        ratios.append(abs(((angle1[time] - angle1[time-1]) / full_angle1) - (angle2[time] - angle2[time-1]) / full_angle2))
        
    return ratios

#=====[ Returns states to use for feature extraction  ]=====
def get_states(squat, key):
    
    states = []
    states.append(starting_position(squat))
    states.append(start_to_squat(squat,key))
    states.append(squat_position(squat,key))
    states.append(squat_to_end(squat,key))
    
    return states

#=====[ Extracts four basic sets of features for a given squat and concatenates them  ]=====
def extract_basic(squat, key):
    
    return np.concatenate(get_states(squat,key),axis=1)


#############################################################################################
####### Extracts advanced features for a given squat - ASSUMES Z COORDINATES INCLUDED  ######
#############################################################################################

#=====[ Extracts features for determining whether feet are shoulder width apart  ]=====
def stance_shoulder_width(states):
   
    #=====[ Checks distance between heels and shoulsers in all frames ]=====
    left_heels_shoulder_apart = [state['AnkleLeftX'] - state['ShoulderLeftX']for state in states]
    right_heels_shoulder_apart = [state['AnkleRightX'] - state['ShoulderRightX'] for state in states]
    
    return np.concatenate([left_heels_shoulder_apart, right_heels_shoulder_apart],axis=1)

#=====[ Extracts features for determining whether shoulders are directly over ankles  ]=====
def stance_straightness(states):
    
    #=====[ Checks to make sure left heels directly under shoulder in all states ]=====
    left_heels_under_shoulder =[state['AnkleLeftZ'] - state['ShoulderLeftZ'] for state in states]

    #=====[ Checks to make sure right heels directly under shoulder in all states  ]=====
    right_heels_under_shoulder = [state['AnkleRightZ'] - state['ShoulderRightZ'] for state in states]
    return np.concatenate([left_heels_under_shoulder, right_heels_under_shoulder],axis=1)


#=====[ Extracts features to determine if the knees are going past the toes (and possibly heels lifitng up)  ]=====
def knees_over_toes(states):

    #=====[ Checks to make sure knees are not pushing out over feet  ]=====
    left_feet_flat = [math.pow(state['KneeLeftZ'] - state['AnkleLeftZ'], 2) for state in states]
    right_feet_flat = [math.pow(state['KneeRightZ'] - state['AnkleRightZ'], 2) for state in states]
    
    return np.concatenate([left_feet_flat, right_feet_flat],axis=1)


#=====[ Extracts features to determine if the hips and knees are simultaneously bending  ]=====
def bend_hips_knees(states):

    #=====[ Gets angles at the knees and hips for the left and right sides of the body  ]=====
    left_bend_knees = [get_angle(state, 'AnkleLeft','KneeLeft','HipLeft','Y','Z') for state in states]
    left_bend_hips = [get_angle(state,'SpineMid','HipLeft','KneeLeft','Y','Z') for state in states]
    right_bend_knees = [get_angle(state,'AnkleRight','KneeRight','HipRight','Y','Z') for state in states]
    right_bend_hips = [get_angle(state,'SpineMid','HipRight','KneeRight','Y','Z') for state in states]

    ratios = np.concatenate([get_angle_changes(left_bend_hips,left_bend_knees),get_angle_changes(right_bend_hips,right_bend_knees)],axis=1)
    
    return np.concatenate([left_bend_knees, left_bend_hips, right_bend_knees, right_bend_hips, ratios],axis=1)


#=====[ Extracts features to determine if the back is straight throughout the squat  ]=====
def back_straight(states):
    assert len(states) > 0
    back_angles = [get_angle(state,'SpineBase','SpineMid','SpineShoulder','Y','Z') for state in states]

    #=====[ Gets average and variance  ]=====
    avg = np.average(back_angles)
    features = []
    variance = sum(map(lambda x : (x - avg)**2, back_angles)) / len(back_angles)
    features.append(variance)
    features.append(avg)

    return np.array(features)

#=====[ Extracts features to determine if the head and back are aligned  ]=====
def head_aligned_back(states):
    assert len(states) > 0
    head_angles = [get_angle(state,'Head','Neck','SpineShoulder','Y','Z') for state in states]

    #=====[ Gets average and variance  ]=====
    avg = np.average(head_angles)
    features = []
    variance = sum(map(lambda x : (x - avg)**2, head_angles)) / len(head_angles)
    features.append(variance)
    features.append(avg)

    return np.array(features)

#=====[ Extracts features to determine if the squat is deep enough  ]=====
def depth(states):

    #=====[ Gets state at bottom of the squat  ]=====
    state = max(states, key=lambda x: float(x['NeckY']))   

    depth_angle = get_angle(state, 'AnkleLeft','KneeLeft','HipLeft','Y','Z')

    return np.array([depth_angle, state['HipLeftY'],state['HipRightY']])

#=====[ Extracts features to determine if the back is appropriately angled at the hip  ]=====
def back_hip_angle(states):

    slopes = []
    
    for state in states:
        slopes.append(abs(state['NeckY'] - np.average([state['HipLeftY'], state['HipRightY']])) / (state['NeckZ'] - np.average([state['HipLeftZ'], state['HipRightZ']])))
        
    return np.array(slopes)
















