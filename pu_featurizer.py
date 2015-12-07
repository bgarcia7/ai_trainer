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

def starting_position_pushup(pushup):
    return pushup.iloc[[1]]

def start_to_pushup(pushup):
    return pushup.iloc[[get_midpoint(pushup,start=1,key=key)]]

def pushup_position(pushup):
    return pushup.iloc[[get_min(pushup,key)]]

def pushup_to_end(pushup):
    return None

#=====[ Extracts four basic sets of features for a given pushup and concatenates them  ]=====
# def extract_basic_pushup(states):
#     fset1 = np.array(starting_position_pushup(squat))
#     fset2 = np.array(start_to_pushup(squat,key))
#     fset3 = np.array(pushup_position(squat,key))
#     fset4 = np.array(pushup_to_end(squat,key))
    
#     return np.concatenate([fset1,fset2,fset3,fset4],axis=1)

#=====[ Extracts advanced features for a given pushup ]=====
def direction_substring(dir_bool):
    if dir_bool:
        return 'Right'
    else:
        return 'Left'

def head_in_line_with_back(states):

    neck_angles = [get_angle(state, 'SpineMid','SpineShoulder','Neck','X','Y') for state in states]
    avg = np.average(neck_angles)
    variance = map(lambda x : (x - avg)**2, neck_angles)
    flat_line_offset = abs(avg - 180)
    variance.append(flat_line_offset)
    return np.array(variance)

def back_straight(states):
    back_angles = [get_angle(state, 'SpineBase','SpineMid','SpineShoulder','X','Y') for state in states]
    avg = np.average(back_angles)
    variance = map(lambda x : (x - avg)**2, back_angles)
    flat_line_offset = abs(avg - 180)
    variance.append(flat_line_offset)
    variance.append(back_angles[0])
    variance.append(back_angles[3])
    return np.array(variance)

# W.R.T side facing kinect
def knees_straight(states, rightward):
    knee_angles = [get_angle(state, 'Hip' + direction_substring(rightward), 'Knee' + direction_substring(rightward), 'Foot' + direction_substring(rightward), 'X', 'Y') for state in states]
    avg = np.average(knee_angles)
    variance = map(lambda x : (x - avg)**2, knee_angles)
    flat_line_offset = abs(avg - 180)
    variance.append(flat_line_offset)
    return np.array(variance)

def elbow_angle(states, rightward):
    elbow_angles = [get_angle(state, 'Shoulder' + direction_substring(rightward), 'Elbow' + direction_substring(rightward), 'Hand' + direction_substring(rightward), 'Y', 'Z') for state in states]
    avg = np.average(elbow_angles)
    variance = map(lambda x : (x - avg)**2, elbow_angles)
    flat_line_offset = abs(avg - 180)
    variance.append(flat_line_offset)
    return np.array(variance)

def hands_aligned_chest(states, rightward):
    shoulder_hand_diffs = [state['Shoulder' + direction_substring(rightward) + 'X'] - state['Hand' + direction_substring(rightward) + 'X'] for state in states]
    frame_diffs = []
    bottom_frame = 0
    top_frame = 3
    frame_diffs.append(states[bottom_frame]['Shoulder' + direction_substring(rightward) + 'X'] - states[bottom_frame]['Hand' + direction_substring(rightward) + 'X'])
    frame_diffs.append(states[top_frame]['Shoulder' + direction_substring(rightward) + 'X'] - states[top_frame]['Hand' + direction_substring(rightward) + 'X'])
    return np.concatenate([shoulder_hand_diffs, frame_diffs],axis=1)

