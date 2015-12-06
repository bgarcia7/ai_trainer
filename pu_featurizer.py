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
def head_in_line_with_back(states):

    neck_angles = []
    for state in states:
        # P12
        spine_shoulder_mid = math.sqrt(math.pow(state['SpineShoulderY'] - state['SpineMidY'], 2) + math.pow(state['SpineShoulderX'] - state['SpineMidX'], 2))
        # P13
        spine_shoulder_neck = math.sqrt(math.pow(state['SpineShoulderY'] - state['NeckY'], 2) + math.pow(state['SpineShoulderX'] - state['NeckX'], 2))
        # P23
        spine_mid_neck = math.sqrt(math.pow(state['SpineMidY'] - state['NeckY'], 2) + math.pow(state['SpineMidX'] - state['NeckX'], 2))
        spine_neck_angle = math.acos((math.pow(spine_shoulder_mid, 2) + math.pow(spine_shoulder_neck, 2) - math.pow(spine_mid_neck, 2)) / (2 * spine_shoulder_mid * spine_shoulder_neck))
        neck_angles.append(spine_neck_angle)

    avg = np.average(neck_angles)
    variance = map(lambda x : (x - avg)**2, neck_angles)
    flat_line_offset = abs(avg - 180)
    variance.append(flat_line_offset)
    return np.array(variance)

def back_straight(states):

    back_angles = []
    for state in states:
        # P12
        spine_mid_base = math.sqrt(math.pow(state['SpineMidY'] - state['SpineBaseY'], 2) + math.pow(state['SpineMidX'] - state['SpineBaseX'], 2))
        # P13
        spine_mid_shoulder = math.sqrt(math.pow(state['SpineMidY'] - state['SpineShoulderY'], 2) + math.pow(state['SpineMidX'] - state['SpineShoulderX'], 2))
        # P23
        spine_base_shoulder = math.sqrt(math.pow(state['SpineBaseY'] - state['SpineShoulderY'], 2) + math.pow(state['SpineBaseX'] - state['SpineShoulderX'], 2))
        spine_mid_angle = math.acos((math.pow(spine_mid_base, 2) + math.pow(spine_mid_shoulder, 2) - math.pow(spine_base_shoulder, 2)) / (2 * spine_mid_base * spine_mid_shoulder))
        back_angles.append(spine_mid_angle)

    avg = np.average(back_angles)
    variance = map(lambda x : (x - avg)**2, back_angles)
    flat_line_offset = abs(avg - 180)
    variance.append(flat_line_offset)
    variance.append(back_angles[0])
    variance.append(back_angles[3])
    return np.array(variance)

# W.R.T side facing kinect
def knees_straight(states, direction):
    
    
    # knee_angles = []
    # for state in states:
    #     # P12
    #     spine_shoulder_mid = math.sqrt(math.pow(state['SpineShoulderY'] - state['SpineMidY'], 2) + math.pow(state['SpineShoulderX'] - state['SpineMidX'], 2))
    #     # P13
    #     spine_shoulder_neck = math.sqrt(math.pow(state['SpineShoulderY'] - state['NeckY'], 2) + math.pow(state['SpineShoulderX'] - state['NeckX'], 2))
    #     # P23
    #     spine_mid_neck = math.sqrt(math.pow(state['SpineMidY'] - state['NeckY'], 2) + math.pow(state['SpineMidX'] - state['NeckX'], 2))
    #     spine_neck_angle = math.acos((math.pow(spine_shoulder_mid, 2) + math.pow(spine_shoulder_neck, 2) - math.pow(spine_mid_neck, 2)) / (2 * spine_shoulder_mid * spine_shoulder_neck))
    #     neck_angles.append(spine_neck_angle)

    # avg = np.average(neck_angles)
    # variance = map(lambda x : (x - avg)**2, neck_angles)
    # flat_line_offset = abs(avg - 180)
    # variance.append(flat_line_offset)
    # return np.array(variance)

