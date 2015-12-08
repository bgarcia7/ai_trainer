import pandas as pd
import numpy as np
import math

### 12/6/12 - PUSHUP FEATURES

#======[ Returns index to frame with minimum y-coord for specified key ]=====
def get_min(pushup,key):   
    
    #=====[ Return max because of inverse frame of reference of kinect ]=====
    return max([(coord,index) for index, coord in enumerate(pushup[key])])[1]

#=====[ Returns index to frame with y-coord closes to the midpoint between start/end and pushup position for specified key ]=====
def get_midpoint(pushup,start,key):
    
    #=====[ Decide whether getting midpoint between start and squat or squat and end ]=====
    if start:
        start = 1
        end = get_min(pushup,key)
    else:
        start = get_min(pushup,key)
        end = pushup.shape[0] - 1
        
    #=====[ Uses the 'true_mid' as an evaluation metric to find optimal index  ]=====
    true_mid = (pushup.iloc[end][key] - pushup.iloc[start][key])/2
    deltas = [(np.abs(true_mid - (pushup.iloc[end][key] - pushup.iloc[index][key])), index) for index in range(start,end)]
    return min(deltas)[1]

def starting_position(pushup):
    return pushup.iloc[[1]]

def start_to_pushup(pushup):
    return pushup.iloc[[get_midpoint(pushup,start=1,key=key)]]

def pushup_position(pushup):
    return pushup.iloc[[get_min(pushup,key)]]

def pushup_to_end(pushup):
    return pushup.iloc[[get_midpoint(pushup,start=0,key=key)]]

#=====[ Returns states to use for feature extraction  ]=====
def get_states(pushup, key):
    
    states = []
    states.append(starting_position(pushup))
    states.append(start_to_pushup(pushup,key))
    states.append(pushup_position(pushup,key))
    states.append(pushup_to_end(pushup,key))
    
    return states

#=====[ Extracts four basic sets of features for a given pushup and concatenates them  ]=====
def extract_basic(pushup, key):
    return np.concatenate(get_states(pushup,key),axis=1)

#=====[ Extracts advanced features for a given pushup ]=====
def avf(angles):
    avg = np.average(angles)
    flat_line_offset = abs(avg - 180)
    return [avg, sum(map(lambda x : (x - avg)**2, neck_angles)) / len(neck_angles), flat_line_offset]

def direction_substring(dir_bool):
    if dir_bool:
        return 'Right'
    else:
        return 'Left'

def head_in_line_with_back(states):
    assert len(states) > 0
    neck_angles = [get_angle(state, 'SpineMid','SpineShoulder','Neck','X','Y') for state in states]

    #=====[ Gets average, variance, and the flat line offset of the average ]=====
    features = avf(neck_angles)
    return np.array(features)

# W.R.T side facing kinect
def back_straight(states, rightward):
    assert len(states) > 0
    back_angles = [get_angle(state, 'SpineShoulder','SpineBase','Ankle' + direction_substring(rightward),'X','Y') for state in states]

    #=====[ Gets average, variance, flat line offset, and first & last angles]=====
    top_frame = 0
    bottom_frame = 2
    features = avf(back_angles)
    features.append(back_angles[top_frame])
    features.append(back_angles[bottom_frame])
    return np.array(features)

def knees_straight(states, rightward):
    assert len(states) > 0
    knee_angles = [get_angle(state, 'Hip' + direction_substring(rightward), 'Knee' + direction_substring(rightward), 'Ankle' + direction_substring(rightward), 'X', 'Y') for state in states]

    #=====[ Gets average, variance, and flat line offset ]=====
    features = avf(knee_angles)
    return np.array(features)

def elbow_angle(states, rightward):
    assert len(states) > 0
    bottom_frame = 2
    elbow_angles = [get_angle(state, 'Shoulder' + direction_substring(rightward), 'Elbow' + direction_substring(rightward), 'Hand' + direction_substring(rightward), 'Y', 'Z') for state in states]

    #=====[ Gets elbow angle at bottom frame ]=====
    features = [elbow_angles[bottom_frame]]
    return np.array(features)

def hands_aligned_chest(states, rightward):
    shoulder_hand_diffs = [state['Shoulder' + direction_substring(rightward) + 'X'] - state['Hand' + direction_substring(rightward) + 'X'] for state in states]
    frame_diffs = []
    top_frame = 0
    bottom_frame = 2
    frame_diffs.append(states[top_frame]['Shoulder' + direction_substring(rightward) + 'X'] - states[top_frame]['Hand' + direction_substring(rightward) + 'X'])
    frame_diffs.append(states[bottom_frame]['Shoulder' + direction_substring(rightward) + 'X'] - states[bottom_frame]['Hand' + direction_substring(rightward) + 'X'])
    return np.concatenate([shoulder_hand_diffs, frame_diffs],axis=1)

