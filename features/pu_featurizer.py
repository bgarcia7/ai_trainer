import pandas as pd
import numpy as np
import math
import pu_normalization as normalizer
from collections import defaultdict

#=====[ Returns concatenated, advanced feature vector  ]=====
def get_advanced_feature_vector(reps, key, multiples):

    #=====[ Initialize dict  ]=====
    advanced_feature_vector = defaultdict(list)

    #=====[ Extract advanced features for each squat  ]=====
    for rep in reps:

        #=====[ Get direction of pushup  ]=====
        rightward = normalizer.pushup_is_right(rep)

        rep = get_states(rep,key,multiples)

        advanced_feature_vector['head_back'].append(head_in_line_with_back(rep, rightward))
        # advanced_feature_vector['back_straight'].append(back_straight(rep, rightward))
        advanced_feature_vector['knees_straight'].append(knees_straight(rep, rightward))
        advanced_feature_vector['elbow_angle'].append(elbow_angle(rep, rightward))
        # advanced_feature_vector['hands_aligned_chest'].append(hands_aligned_chest(rep, rightward))

    return advanced_feature_vector

#=====[ Returns angle between three specified joints along the two specified axes  ]=====
def get_angle(state, joint1, joint2, joint3, axis1, axis2):

    bone1 = math.sqrt(math.pow(state[joint2 + axis1] - state[joint1 + axis1], 2) + math.pow(state[joint2 + axis2] - state[joint1 + axis2], 2))
    bone2 = math.sqrt(math.pow(state[joint2 + axis1] - state[joint3 + axis1], 2) + math.pow(state[joint2 + axis2] - state[joint3 + axis2], 2))

    #=====[ Gets distance between the disconnected joints  ]=====
    distance = math.sqrt(math.pow(state[joint1 + axis1] - state[joint3 + axis1], 2) + math.pow(state[joint1 + axis2] - state[joint3 + axis2], 2))
    
    try:
        angle = math.acos((math.pow(bone1, 2) + math.pow(bone2, 2) - math.pow(distance, 2)) / (2 * bone1 * bone2))
    except Exception as e:
        print e
        return 0

    return angle

#======[ Returns index to frame with minimum y-coord for specified key ]=====
def get_min(pushup,key):   
    
    #=====[ Return max because of inverse frame of reference of kinect ]=====
    return max([(coord,index) for index, coord in enumerate(pushup[key])])[1]

#=====[ Returns index to frame with y-coord closes to the midpoint between start/end and squat position for specified key ]=====
def get_midpoint(squat,start,key, multiple):
    
    #=====[ Decide whether getting midpoint between start and squat or squat and end ]=====
    if start:
        start = 1
        end = get_min(squat,key)
    else:
        start = get_min(squat,key)
        end = squat.shape[0] - 1
        
    #=====[ Uses the 'true_mid' as an evaluation metric to find optimal index  ]=====
    true_mid = (squat.iloc[end][key] - squat.iloc[start][key])*multiple
    deltas = [(np.abs(true_mid - (squat.iloc[end][key] - squat.iloc[index][key])), index) for index in range(start,end)]
    try: 
        return min(deltas)[1]
    except:
        return start

#=====[ Returns squat at the first position ]=====
def starting_position(pushup):
    return pushup.iloc[[1]]

#=====[ Returns index to frame with y-coord closes to the midpoint between start and squat position for specified key ]=====
def start_to_pushup(pushup,key,multiple=0.5):
    return pushup.iloc[[get_midpoint(pushup,start=1,key=key,multiple=multiple)]]

#=====[ Returns frame with minimum y-coord for specified key ]=====
def pushup_position(pushup,key):
    return pushup.iloc[[get_min(pushup,key)]]

#=====[ Returns index to frame with y-coord closes to the midpoint between squat position and end for specified key ]=====
def pushup_to_end(pushup,key,multiple=0.5):
    return pushup.iloc[[get_midpoint(pushup,start=0,key=key,multiple=multiple)]]

#=====[ Returns states to use for feature extraction  ]=====
def get_states(pushup, key, multiples=[0.5]):
    
    states = []
    states.append(starting_position(pushup))

    for multiple in multiples:
        states.append(start_to_pushup(pushup,key, multiple))
        states.append(pushup_to_end(pushup,key, multiple))

    states.append(pushup_position(pushup,key))
    
    return states

#=====[ Extracts four basic sets of features for a given pushup and concatenates them  ]=====
def extract_basic(pushup, key):
    return np.concatenate(get_states(pushup,key),axis=1)

#=====[ Extracts advanced features for a given pushup ]=====
def avf(angles):
    avg = np.average(angles)
    flat_line_offset = abs(avg - 180)
    return [avg, sum(map(lambda x : (x - avg)**2, angles)) / len(angles), flat_line_offset]

def direction_substring(dir_bool):
    if dir_bool:
        return 'Right'
    else:
        return 'Left'

def head_in_line_with_back(states, rightward):
    assert len(states) > 0
    neck_angles = [get_angle(state, 'SpineMid','SpineShoulder','Neck','X','Y') for state in states]

    #=====[ Gets average, variance, and the flat line offset of the average ]=====
    features = avf(neck_angles)

    return np.array(features)

def back_straight(states, rightward):
    assert len(states) > 0
    back_angles = [get_angle(state, 'SpineShoulder','SpineBase','Ankle' + direction_substring(rightward),'X','Y') for state in states]

    #=====[ Gets average, variance, flat line offset, and first & last angles]=====
    features = avf(back_angles)
    
    return np.concatenate([features,back_angles],axis=0)

def knees_straight(states, rightward):
    
    assert len(states) > 0
   
    knee_angles = [get_angle(state, 'Hip' + direction_substring(rightward), 'Knee' + direction_substring(rightward), 'Ankle' + direction_substring(rightward), 'X', 'Y') for state in states]
    #=====[ Gets average, variance, and flat line offset ]=====
    features = avf(knee_angles)
    
    return np.array(features)

def elbow_angle(states, rightward):
    assert len(states) > 0
    elbow_angles = [get_angle(state, 'Shoulder' + direction_substring(rightward), 'Elbow' + direction_substring(rightward), 'Hand' + direction_substring(rightward), 'Y', 'Z') for state in states]

    #=====[ Gets elbow angle at bottom frame ]=====
    features = elbow_angles
    
    return np.array(features)

def hands_aligned_chest(states, rightward):
    
    shoulder_hand_diffs = [state['Shoulder' + direction_substring(rightward) + 'X'] - state['Hand' + direction_substring(rightward) + 'X'] for state in states]
 
    return shoulder_hand_diffs

