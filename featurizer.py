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


#=====[ Extracts four basic sets of features for a given squat and concatenates them  ]=====
def extract_basic(squat, key):
	fset1 = np.array(starting_position(squat))
	fset2 = np.array(start_to_squat(squat,key))
	fset3 = np.array(squat_position(squat,key))
	fset4 = np.array(squat_to_end(squat,key))
	
	return np.concatenate([fset1,fset2,fset3,fset4],axis=1)

#=====[ Extracts advanced features for a given squat - ASSUMES Z COORDINATES INCLUDED  ]=====

def extract_stance_shoulder_width(squat, key, keys_to_indices):
    fset1 = np.array(starting_position(squat))
    fset2 = np.array(start_to_squat(squat,key))
    fset3 = np.array(squat_position(squat,key))
    fset4 = np.array(squat_to_end(squat,key))

    states = [fset1, fset2, fset3, fset4]

    # W.R.T ALL 4 FRAMES
    left_heels_shoulder_apart = []
    right_heels_shoulder_apart = []
    for state in states:
        # Left heels shoulder width apart
        left_heels_shoulder_apart.append(state[keys_to_indices['AnkleLeftX']] - state[keys_to_indices['ShoulderLeftX']])

        # Right heels shoulder width apart
        right_heels_shoulder_apart.append(state[keys_to_indices['AnkleRightX']] - state[keys_to_indices['ShoulderRightX']])
    return np.concatenate([left_heels_shoulder_apart, right_heels_shoulder_apart],axis=1)

def extract_stance_straightness(squat, key, keys_to_indices):
    fset1 = np.array(starting_position(squat))
    fset2 = np.array(start_to_squat(squat,key))
    fset3 = np.array(squat_position(squat,key))
    fset4 = np.array(squat_to_end(squat,key))

    states = [fset1, fset2, fset3, fset4]

    # W.R.T ALL 4 FRAMES
    left_heels_under_shoulder = []
    right_heels_under_shoulder = []
    for state in states:
        # Left heels directly under shoulders
        left_heels_under_shoulder.append(state[keys_to_indices['AnkleLeftZ']] - state[keys_to_indices['ShoulderLeftZ']])

        # Right heels directly under shoulders
        right_heels_under_shoulder.append(state[keys_to_indices['AnkleRightZ']] - state[keys_to_indices['ShoulderRightZ']])
    return np.concatenate([left_heels_under_shoulder, right_heels_under_shoulder],axis=1)

def extract_feet(squat, key, keys_to_indices):
    fset1 = np.array(starting_position(squat))
    fset2 = np.array(start_to_squat(squat,key))
    fset3 = np.array(squat_position(squat,key))
    fset4 = np.array(squat_to_end(squat,key))

    states = [fset1, fset2, fset3, fset4]

    # W.R.T ALL 4 FRAMES    
    left_feet_flat = []
    right_feet_flat = []
    for state in states:
        # Left foot flat on floor
        left_feet_flat.append(math.pow(state[keys_to_indices['KneeLeftZ']] - state[keys_to_indices['AnkleLeftZ']], 2))

        # Right foot flat on floor
        right_feet_flat.append(math.pow(state[keys_to_indices['KneeRightZ']] - state[keys_to_indices['AnkleRightZ']], 2))
    return np.concatenate([left_feet_flat, right_feet_flat],axis=1)

def bend_hips_knees(squat, key, keys_to_indices):
    fset1 = np.array(starting_position(squat))
    fset2 = np.array(start_to_squat(squat,key))
    fset3 = np.array(squat_position(squat,key))
    fset4 = np.array(squat_to_end(squat,key))

    states = [fset1, fset2, fset3, fset4]

    # W.R.T ALL 4 FRAMES    
    left_bend_knees = []
    left_bend_hips = []
    right_bend_knees = []
    right_bend_hips = []
    for state in states:
        # Left knee angle - formed by Y,Z coords W.R.T knee
        # P12
        left_knee_ankle = math.sqrt(math.pow(state[keys_to_indices['KneeLeftY']] - state[keys_to_indices['AnkleLeftY']], 2) + math.pow(state[keys_to_indices['KneeLeftZ']] - state[keys_to_indices['AnkleLeftZ']], 2))
        # P13
        left_knee_hip = math.sqrt(math.pow(state[keys_to_indices['KneeLeftY']] - state[keys_to_indices['HipLeftY']], 2) + math.pow(state[keys_to_indices['KneeLeftZ']] - state[keys_to_indices['HipLeftZ']], 2))
        # P23
        left_ankle_hip = math.sqrt(math.pow(state[keys_to_indices['AnkleLeftY']] - state[keys_to_indices['HipLeftY']], 2) + math.pow(state[keys_to_indices['AnkleLeftZ']] - state[keys_to_indices['HipLeftZ']], 2))
        left_knee_angle = math.acos((math.pow(left_knee_ankle, 2) + math.pow(left_knee_hip, 2) - math.pow(left_ankle_hip, 2)) / (2 * left_knee_ankle * left_knee_hip))

        # Left hip angle - formed by Y,Z coords W.R.T hip
        # P12
        left_hip_spine = math.sqrt(math.pow(state[keys_to_indices['HipLeftY']] - state[keys_to_indices['SpineMidY']], 2) + math.pow(state[keys_to_indices['HipLeftZ']] - state[keys_to_indices['SpineMidZ']], 2))
        # P13
        left_hip_knee = math.sqrt(math.pow(state[keys_to_indices['HipLeftY']] - state[keys_to_indices['KneeLeftY']], 2) + math.pow(state[keys_to_indices['HipLeftZ']] - state[keys_to_indices['KneeLeftZ']], 2))
        # P23
        left_spine_knee = math.sqrt(math.pow(state[keys_to_indices['SpineMidY']] - state[keys_to_indices['KneeLeftY']], 2) + math.pow(state[keys_to_indices['SpineMidZ']] - state[keys_to_indices['KneeLeftZ']], 2))
        left_hip_angle = math.acos((math.pow(left_hip_spine, 2) + math.pow(left_hip_knee, 2) - math.pow(left_spine_knee, 2)) / (2 * left_hip_spine * left_hip_knee))

        # Right knee angle - formed by Y,Z coords W.R.T knee
        # P12
        right_knee_ankle = math.sqrt(math.pow(state[keys_to_indices['KneeRightY']] - state[keys_to_indices['AnkleRightY']], 2) + math.pow(state[keys_to_indices['KneeRightZ']] - state[keys_to_indices['AnkleRightZ']], 2))
        # P13
        right_knee_hip = math.sqrt(math.pow(state[keys_to_indices['KneeRightY']] - state[keys_to_indices['HipRightY']], 2) + math.pow(state[keys_to_indices['KneeRightZ']] - state[keys_to_indices['HipRightZ']], 2))
        # P23
        right_ankle_hip = math.sqrt(math.pow(state[keys_to_indices['AnkleRightY']] - state[keys_to_indices['HipRightY']], 2) + math.pow(state[keys_to_indices['AnkleRightZ']] - state[keys_to_indices['HipRightZ']], 2))
        right_knee_angle = math.acos((math.pow(right_knee_ankle, 2) + math.pow(right_knee_hip, 2) - math.pow(right_ankle_hip, 2)) / (2 * right_knee_ankle * right_knee_hip))

        # Right hip angle - formed by Y,Z coords W.R.T hip
        # P12
        right_hip_spine = math.sqrt(math.pow(state[keys_to_indices['HipRightY']] - state[keys_to_indices['SpineMidY']], 2) + math.pow(state[keys_to_indices['HipRightZ']] - state[keys_to_indices['SpineMidZ']], 2))
        # P13
        right_hip_knee = math.sqrt(math.pow(state[keys_to_indices['HipRightY']] - state[keys_to_indices['KneeRightY']], 2) + math.pow(state[keys_to_indices['HipRightZ']] - state[keys_to_indices['KneeRightZ']], 2))
        # P23
        right_spine_knee = math.sqrt(math.pow(state[keys_to_indices['SpineMidY']] - state[keys_to_indices['KneeRightY']], 2) + math.pow(state[keys_to_indices['SpineMidZ']] - state[keys_to_indices['KneeRightZ']], 2))
        right_hip_angle = math.acos((math.pow(right_hip_spine, 2) + math.pow(right_hip_knee, 2) - math.pow(right_spine_knee, 2)) / (2 * right_hip_spine * right_hip_knee))

        left_bend_knees.append(left_knee_angle)
        left_bend_hips.append(left_hip_angle)
        right_bend_knees.append(right_knee_angle)
        right_bend_hips.append(right_hip_angle)

    left_ratio = abs(((left_bend_hips[1] - left_bend_hips[0]) / (left_bend_hips[3] - left_bend_hips[0])) - ((left_bend_knees[1] - left_bend_knees[0]) / (left_bend_knees[3] - left_bend_knees[0]))) 
    right_ratio = abs(((right_bend_hips[1] - right_bend_hips[0]) / (right_bend_hips[3] - right_bend_hips[0])) - ((right_bend_knees[1] - right_bend_knees[0]) / (right_bend_knees[3] - right_bend_knees[0])))
    ratios = [left_ratio, right_ratio]
    return np.concatenate([left_bend_knees, left_bend_hips, right_bend_knees, right_bend_hips, ratios],axis=1)

def back_straight(squat, key, keys_to_indices):
    fset1 = np.array(starting_position(squat))
    fset2 = np.array(start_to_squat(squat,key))
    fset3 = np.array(squat_position(squat,key))
    fset4 = np.array(squat_to_end(squat,key))

    states = [fset1, fset2, fset3, fset4]

    back_angles = []
    for state in states:
        # P12
        spine_mid_base = math.sqrt(math.pow(state[keys_to_indices['SpineMidY']] - state[keys_to_indices['SpineBaseY']], 2) + math.pow(state[keys_to_indices['SpineMidZ']] - state[keys_to_indices['SpineBaseZ']], 2))
        # P13
        spine_mid_shoulder = math.sqrt(math.pow(state[keys_to_indices['SpineMidY']] - state[keys_to_indices['SpineShoulderY']], 2) + math.pow(state[keys_to_indices['SpineMidZ']] - state[keys_to_indices['SpineShoulderZ']], 2))
        # P23
        spine_base_shoulder = math.sqrt(math.pow(state[keys_to_indices['SpineShoulderY']] - state[keys_to_indices['SpineBaseY']], 2) + math.pow(state[keys_to_indices['SpineShoulderZ']] - state[keys_to_indices['SpineBaseZ']], 2))
        spine_mid_angle = math.acos((math.pow(spine_mid_base, 2) + math.pow(spine_mid_shoulder, 2) - math.pow(spine_base_shoulder, 2)) / (2 * spine_mid_base * spine_mid_shoulder))
        back_angles.append(spine_mid_angle)

    avg = np.average(back_angles)
    variance = map(lambda x : (x - avg)**2, back_angles)
    flat_line_offset = abs(avg - 180)
    return np.array([variance, flat_line_offset])

def head_aligned_back(squat, key, keys_to_indices):
    fset1 = np.array(starting_position(squat))
    fset2 = np.array(start_to_squat(squat,key))
    fset3 = np.array(squat_position(squat,key))
    fset4 = np.array(squat_to_end(squat,key))

    states = [fset1, fset2, fset3, fset4]

    head_angles = []
    for state in states:
        # P12
        neck_head = math.sqrt(math.pow(state[keys_to_indices['NeckY']] - state[keys_to_indices['HeadY']], 2) + math.pow(state[keys_to_indices['NeckZ']] - state[keys_to_indices['HeadZ']], 2))
        # P13
        neck_back = math.sqrt(math.pow(state[keys_to_indices['NeckY']] - state[keys_to_indices['SpineShoulderY']], 2) + math.pow(state[keys_to_indices['NeckZ']] - state[keys_to_indices['SpineShoulderZ']], 2))
        # P23
        head_back = math.sqrt(math.pow(state[keys_to_indices['HeadY']] - state[keys_to_indices['SpineShoulderY']], 2) + math.pow(state[keys_to_indices['HeadZ']] - state[keys_to_indices['SpineShoulderZ']], 2))
        neck_angle = math.acos((math.pow(neck_head, 2) + math.pow(neck_back, 2) - math.pow(head_back, 2)) / (2 * neck_head * neck_back))
        head_angles.append(neck_angle)

    avg = np.average(head_angles)
    variance = map(lambda x : (x - avg)**2, head_angles)
    flat_line_offset = abs(avg - 180)
    return np.array([variance, flat_line_offset])

def depth(squat, key, keys_to_indices):
    fset3 = np.array(squat_position(squat,key))
    left_depth = []
    right_depth = []
    # P12
    left_knee_ankle = math.sqrt(math.pow(fset3[keys_to_indices['KneeLeftY']] - fset3[keys_to_indices['AnkleLeftY']], 2) + math.pow(fset3[keys_to_indices['KneeLeftZ']] - fset3[keys_to_indices['AnkleLeftZ']], 2))
    # P13
    left_knee_hip = math.sqrt(math.pow(fset3[keys_to_indices['KneeLeftY']] - fset3[keys_to_indices['HipLeftY']], 2) + math.pow(fset3[keys_to_indices['KneeLeftZ']] - fset3[keys_to_indices['HipLeftZ']], 2))
    # P23
    left_ankle_hip = math.sqrt(math.pow(fset3[keys_to_indices['AnkleLeftY']] - fset3[keys_to_indices['HipLeftY']], 2) + math.pow(fset3[keys_to_indices['AnkleLeftZ']] - fset3[keys_to_indices['HipLeftZ']], 2))
    
    depth_angle = math.acos((math.pow(left_knee_ankle, 2) + math.pow(left_knee_hip, 2) - math.pow(left_ankle_hip, 2)) / (2 * left_knee_ankle * left_knee_hip))
    
    left_hip_y = fset3[keys_to_indices['HipLeftY']]
    right_hip_y = fset3[keys_to_indices['HipRightY']]
    return np.array([depth_angle, left_hip_y, right_hip_y])

def back_hip_angle(squat, key, keys_to_indices):
    fset1 = np.array(starting_position(squat))
    fset2 = np.array(start_to_squat(squat,key))
    fset3 = np.array(squat_position(squat,key))
    fset4 = np.array(squat_to_end(squat,key))

    bottom_frame = abs((fset3[keys_to_indices['NeckY']] - np.average([fset3[keys_to_indices['HipLeftY']], fset3[keys_to_indices['HipRightY']]])) / (fset3[keys_to_indices['NeckZ']] - np.average([fset3[keys_to_indices['HipLeftZ']], fset3[keys_to_indices['HipRightZ']]])))
    midpoint_frame_down = abs((fset2[keys_to_indices['NeckY']] - np.average([fset2[keys_to_indices['HipLeftY']], fset2[keys_to_indices['HipRightY']]])) / (fset2[keys_to_indices['NeckZ']] - np.average([fset2[keys_to_indices['HipLeftZ']], fset2[keys_to_indices['HipRightZ']]])))
    midpoint_frame_up = abs((fset4[keys_to_indices['NeckY']] - np.average([fset4[keys_to_indices['HipLeftY']], fset4[keys_to_indices['HipRightY']]])) / (fset4[keys_to_indices['NeckZ']] - np.average([fset4[keys_to_indices['HipLeftZ']], fset4[keys_to_indices['HipRightZ']]])))
    return np.array([bottom_frame, midpoint_frame_down, midpoint_frame_up])















