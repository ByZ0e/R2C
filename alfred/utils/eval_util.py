import os
import json
import queue
import torch
import shutil
import filelock
import numpy as np
import re
from termcolor import colored
import supervision as sv
import sys
import random
import copy
import cv2
from PIL import Image

from alfred import constants
from alfred.utils import gen_util
from alfred.utils.depth import apply_real_noise
import envs.utils.pose as pu
from keyboard_fb.chessboard_vis.infer_vis import ChessVis
from keyboard_fb.agent_obj_rel import rotate_to_target

prompt_templete_v2 = """On a 48*48 block chessboard, the rules of the game are as follows: 
Establish a coordinate system with the top left grid as (1,1). Each block can be represented by coordinates.
for instance, the block in the 5th column and 3rd row is denoted as (5,3). 
In the chessboard, there are the following explored objects: {}
The movememt is forbidden on the object block.
You can only move 1 block at a time.
### Task: {}
### Sub-goal: {}
### Postion: {}
Please select your next position and judge whether the sub-goal is finished.
"""

prompt_templete_v4 = """On a 64*64 block chessboard, the rules of the game are as follows: 
Establish a coordinate system with the top left grid as (1,1). Each block can be represented by coordinates.
for instance, the block in the 5th column and 3rd row is denoted as (5,3). 
In the chessboard, there are the following explored objects: {}.
The movement is forbidden on the object block.
You can only move 1 block at a time.
### Task: {}
### Current Position: {}
### History Trajectory: {}
### Please select your next position from {}.
### Please analyze the options and choose the best one to finish the task.
"""

prompt_templete_v5 = """On a 64*64 block chessboard, the rules of the game are as follows: 
Establish a coordinate system with the top left grid as (1,1). Each block can be represented by coordinates.
for instance, the block in the 5th row and 3rd column is denoted as (5,3). 
In the chessboard, there are the following explored objects: {}.
The movement is forbidden on the object block.
You can only move 1 block at a time.
### Task: {}
### Current Position: {}
### History Trajectory: {}
### Please select your next position from {}.
### Please analyze the options and choose the best one to finish the task.
"""


# testing zero-shot mistral
prompt_templete_v6 = """On a 64*64 block chessboard, the rules of the game are as follows: 
Establish a coordinate system with the top left grid as (1,1). Each block can be represented by coordinates.
for instance, the block in the 5th row and 3rd column is denoted as (5,3). 
In the chessboard, there are the following explored objects: {}.
The movement is forbidden on the object block.
You can only move 1 block at a time.
### Task: {}
### Current Position: {}
### History Trajectory: {}
### Please select your next position from {}.
### Please analyze the options and choose the best one to finish the task.
### MUST answer with # Next position: [x, y] in the end.
"""


task_instruct_templete = """### You are an indoor agent.
### Please divide the TASK into some sub-steps according to the DESCRIPTION.
### Each sub-step MUST be expressed in the form of (Action, [Object]), where Action is one of (GotoLocation, PickupObject, PutObject, CoolObject, HeatObject, CleanObject, SliceObject, ToggleObject, NoOp), and Object refers to one or more objects related to the Action.
### TASK: {}
### DESCRIPTION: {}
"""


def task_prompt_generator(task, desc):
    prompt = task_instruct_templete.format(task, desc)
    return prompt 


def get_subgoal_str(subgoal):
    # subgoal = "(GotoLocation, [toilet])"
    action, args = subgoal
    return "(" + action + ", [" + args[0] + "])"
    
def setup_scene(env, traj_data, reward_type='dense', test_split=False):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name, silent=True)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)
    # initialize to start position
    
    env.step(dict(traj_data['scene']['init_action']))
    # init_action = traj_data['scene']['init_action']
    # init_action["horizon"] = 0
    # env.step(dict(init_action))
    
    # setup task for reward
    if not test_split:
        env.set_task(traj_data, reward_type=reward_type)
    return

def read_task_data(task, subgoal_idx=None):
    '''
    read data from the traj_json
    '''
    # read general task info
    repeat_idx = task['repeat_idx']
    task_dict = {'repeat_idx': repeat_idx,
                 'type': task['task_type'],
                 'task': '/'.join(task['root'].split('/')[-3:-1])}
    # read subgoal info
    if subgoal_idx is not None:
        task_dict['subgoal_idx'] = subgoal_idx
        task_dict['subgoal_action'] = task['plan']['high_pddl'][
            subgoal_idx]['discrete_action']['action']
    return task_dict

def camel_to_space(input_str):
    result_str = re.sub(r'([a-z])([A-Z])', r'\1 \2', input_str)
    return result_str

def get_obj_list(subgoals):
    obj_list = copy.deepcopy(constants.LARGE_OBJ)
    obj_targets = get_obj_args(subgoals)
    for obj in obj_targets:
        if obj not in obj_list:  # 交互小物体
            obj_list.append(obj)
        else:
            obj_list.remove(obj)
            obj_list.append(obj)
    obj_list = [camel_to_space(x) for x in obj_list]
    return obj_list

def get_obj_args(subgoals):
    # task_type = traj_data['task_type']
    # task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                    #   'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                    #   'pick_and_place_with_movable_recep']
    pickup_objs = []
    put_objs = []
    task_type = ''
    sliced = False
    object_target = parent_target = mrecep_target = toggle_target = None
    for idx, subgoal in enumerate(subgoals):
        if subgoal[0] == "SliceObject": sliced = True
        elif subgoal[0] == "CoolObject": task_type = 'pick_cool_then_place_in_recep'; 
        elif subgoal[0] == "HeatObject": task_type = 'pick_heat_then_place_in_recep'
        elif subgoal[0] == "CleanObject": task_type = 'pick_clean_then_place_in_recep'
        elif subgoal[0] == "ToggleObject": task_type = 'look_at_obj_in_light'; toggle_target = subgoal[1][-1]
        elif subgoal[0] == "PickupObject": pickup_objs.append(subgoal[1])
        elif subgoal[0] == "PutObject": put_objs.append(subgoal[1])
    
    if len(pickup_objs) == 1 and task_type == 'look_at_obj_in_light': object_target = pickup_objs[0][-1]
    elif len(put_objs) == 1 and task_type == '': object_target, parent_target = put_objs[0]  # 'pick_and_place_simple'
    elif len(put_objs) == 1 and task_type != '': object_target, parent_target = put_objs[0]  # clean/cool/heat
    elif len(put_objs) == 2 and task_type != '': object_target, parent_target = put_objs[1]  # clean/cool/heat
    elif len(put_objs) == 2 and task_type == '' and put_objs[0] == put_objs[1]: 
        object_target, parent_target = put_objs[0]  # 'pick_two_obj_and_place' no put knife goal
        task_type = 'pick_two_obj_and_place'
    elif len(put_objs) == 2 and task_type == '' and put_objs[0] != put_objs[1]: # 'pick_and_place_with_movable_recep' no sliced
        object_target, parent_target = put_objs[0]
        parent_target, mrecep_target = put_objs[1]
    elif len(put_objs) == 3 and task_type == '' and put_objs[0] != put_objs[1]: # 'pick_and_place_with_movable_recep' sliced
        object_target, parent_target = put_objs[1]
        parent_target, mrecep_target = put_objs[2]
    

    # if sliced:
    #     object_target = object_target + 'Sliced'
    if parent_target == "Sink":
        parent_target = "Sink Basin"
    if parent_target == "Bathtub":
        parent_target = "Bathtub Basin"
    # if object_target == "Knife":
    #     object_target = "ButterKnife"
    # categories_in_inst = [x for x in [mrecep_target, object_target, parent_target, toggle_target] if x != None]
    categories_in_inst = [constants.OBJECTS_LOWER_TO_UPPER[x] for x in [object_target, mrecep_target, parent_target, toggle_target] if x != None]  # 按优先级顺序！！！
    if task_type == 'pick_two_obj_and_place':
        categories_in_inst = [constants.OBJECTS_LOWER_TO_UPPER[x] for x in [parent_target, mrecep_target, object_target, toggle_target] if x != None]  # 按优先级顺序！！！
    
    # TODO: 区分交互大物体和交互小物体
    if sliced:
        categories_in_inst = [object_target + 'Sliced', 'Knife', 'Butter Knife'] + categories_in_inst
    if task_type == 'pick_cool_then_place_in_recep':
        categories_in_inst += ['Fridge']
    if task_type == 'pick_heat_then_place_in_recep':
        categories_in_inst += ['Microwave']
    if task_type == 'pick_clean_then_place_in_recep':
        categories_in_inst += ['Faucet', 'Sink Basin']

    # categories_in_inst = list(set(categories_in_inst))
    categories_in_inst = [camel_to_space(x) for x in categories_in_inst]
    return categories_in_inst

def get_rgbd(env, add_depth_noise=False):
    rgb = env.last_event.frame
    bgr = env.last_event.cv2img
    depth = env.last_event.depth_frame
    # print('=======================')
    # for obj in env.last_event.metadata["objects"]:
    #     if obj["visible"] and obj["objectId"] in env.last_event.instance_masks.keys():
    #         print('obj["objectId"]', obj["objectId"])
    #         print('obj["objectType"]', obj["objectType"])
    
    if add_depth_noise:
        depth = apply_real_noise(depth_arr=depth, size=300)
        
    return rgb, bgr, depth

def has_interaction(action):
    '''
    check if low-level action is interactive
    '''
    non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>']
    if any(a in action for a in non_interact_actions):
        return False
    else:
        return True

def extract_seg_pred(class_name, class_idx, seg_model, env, verbose=False, renderInstanceSegmentation=True, obj_classes=None):
    '''
    extract a pixel mask using a pre-trained MaskRCNN
    '''
    '''
    class Detections:
    """
    Data class containing information about the detections in a video frame.

    Attributes:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        mask: (Optional[np.ndarray]): An array of shape
            `(n, H, W)` containing the segmentation masks.
        confidence (Optional[np.ndarray]): An array of shape
            `(n,)` containing the confidence scores of the detections.
        class_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the class ids of the detections.
        tracker_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the tracker ids of the detections.
    """
    '''
    # rcnn_pred = obj_predictor.predict_objects(Image.fromarray(env.last_event.frame))
    candidates = []
    # if class_name == 'Sink Basin': class_name = 'Sink'
    objectType = class_name.replace(' ', '')  # camel format
    # print('objectType: ', objectType)  # objectType:  Plate
    if renderInstanceSegmentation:
        for obj in env.last_event.metadata["objects"]:
            # print(obj["objectType"], obj["objectId"])
            # if obj["objectType"] in constants.OBJECT_MAPPING.keys():
            #     obj["objectType"] = constants.OBJECT_MAPPING[obj["objectType"]]
            
            if obj["visible"] and obj["objectId"] in env.last_event.instance_masks.keys() and obj["objectType"] == objectType:
                print('==============detected=============')
                print('obj["objectId"]', obj["objectId"])
                obj_mask = env.last_event.instance_masks[obj["objectId"]] * 1.
                if obj["objectId"].startswith('Sink') and not obj["objectId"].endswith('SinkBasin'):
                    continue
                if obj["objectId"].startswith('Bathtub') and not obj["objectId"].endswith('BathtubBasin'):
                    continue
                candidates.append(sv.Detections(xyxy=np.array([[0.,0.,0.,0.]]), confidence=np.array([1.]), mask=np.expand_dims(obj_mask, 0)))
    else:
        rcnn_pred = seg_model.segmentation(env.last_event.cv2img.astype(np.uint8), obj_classes)
        for idx, pred in enumerate(rcnn_pred.class_id):
            if pred == class_idx:
                candidates.append(rcnn_pred[idx])
    # candidates = list(filter(lambda p: p.class_id == class_idx, rcnn_pred))
    if verbose:
        visible_objs = [
            obj for obj in env.last_event.metadata['objects']
            if obj['visible'] and obj["objectType"] == objectType]
        print('Agent prediction = {}, detected {} objects (visible {})'.format(
            objectType, len(candidates), len(visible_objs)))
    if len(candidates) > 0:
        if env.last_interaction[0] == class_name:
            # last_obj['id'] and class_name + '|' in env.last_obj['id']:
            # do the association based selection
            last_center = np.array(env.last_interaction[1].nonzero()).mean(axis=1)
            cur_centers = np.array(
                [np.array(c.mask[0].nonzero()).mean(axis=1) for c in candidates])
            distances = ((cur_centers - last_center)**2).sum(axis=1)
            index = np.argmin(distances)
            mask = candidates[index].mask[0]
        else:
            if renderInstanceSegmentation:
                # do the area based selection
                areas = np.array([np.array(c.mask[0].flatten().nonzero()).sum(axis=1) for c in candidates])
                print(areas)
                index = np.argmax(areas)
                mask = candidates[index].mask[0]
            else:
                # do the confidence based selection
                index = np.argmax([p.confidence for p in candidates])
                mask = candidates[index].mask[0]  # mask: batchsize on 0 dimension
            '''
            # do the confidence based selection
            index = np.argmax([p.confidence for p in candidates])
            mask = candidates[index].mask[0]  # mask: batchsize on 0 dimension
            '''
                
    else:
        mask = None
    return mask

def extract_rcnn_pred(class_idx, obj_predictor, env, verbose=False, confidence_threshold=0.0):
    '''
    extract a pixel mask using a pre-trained MaskRCNN
    '''
    print('interact threshold', confidence_threshold)
    rcnn_pred = obj_predictor.predict_objects(Image.fromarray(env.last_event.frame), confidence_threshold=confidence_threshold)
    class_name = obj_predictor.vocab_obj.index2word(class_idx)  # Sink
    
    if 'Lamp' in class_name:
        class_idx = [32, 25]
        class_name = ['FloorLamp', 'DeskLamp']
    elif 'Knife' in class_name:
        class_idx = [40, 12]
        class_name = ['Knife', 'ButterKnife']
    else:
        class_idx = [class_idx]
        class_name = [class_name]
    print('class_name', class_name)
    candidates = list(filter(lambda p: p.label in class_name, rcnn_pred))
    if verbose:
        if len(class_name) == 1:
            visible_objs = [
                obj for obj in env.last_event.metadata['objects']
                if obj['visible'] and obj['objectId'].startswith(class_name[0] + '|')]
            print('Agent prediction = {}, detected {} objects (visible {})'.format(
                class_name[0], len(candidates), len(visible_objs)))
        else:
            visible_objs = [
                obj for obj in env.last_event.metadata['objects']
                if obj['visible'] and obj['objectId'].startswith(class_name[0] + '|')]
            print('Agent prediction = {}, detected {} objects (visible {})'.format(
                class_name[0], len(candidates), len(visible_objs)))
            visible_objs = [
                obj for obj in env.last_event.metadata['objects']
                if obj['visible'] and obj['objectId'].startswith(class_name[1] + '|')]
            print('Agent prediction = {}, detected {} objects (visible {})'.format(
                class_name[1], len(candidates), len(visible_objs)))
    if len(candidates) > 0:
        print('----------more than one candidate---------')
        print('last_interaction', env.last_interaction[0])
        if env.last_interaction[0] in class_name:
            # last_obj['id'] and class_name + '|' in env.last_obj['id']:
            # do the association based selection
            last_center = np.array(env.last_interaction[1].nonzero()).mean(axis=1)
            cur_centers = np.array(
                [np.array(c.mask[0].nonzero()).mean(axis=1) for c in candidates])
            distances = ((cur_centers - last_center)**2).sum(axis=1)
            index = np.argmin(distances)
            mask = candidates[index].mask[0]
        else:
            # do the confidence based selection
            index = np.argmax([p.score for p in candidates])
            mask = candidates[index].mask[0]
    else:
        mask = None
    return mask

def get_orientation(orientation, do):
    orientation += do * 57.29577951308232
    orientation = np.fmod(orientation-180.0, 360.0) + 180.0
    orientation = np.fmod(orientation+180.0, 360.0) - 180.0
    return round(orientation)

def postion2action(ori, last_pos, next_pos):
    # next_pos to action list (fubin)
    """
    Input:
        last_pos: (x, y)
        next_pos: (x, y)
        orientation: (0/90/180/270, int)
    Output:
        action_list(list):[action1, action2, ...]
    x-y coodinate: x is down, y is right
    """
    # print('from fubin, ori: ', ori)
    while ori < 0:
        ori += 360
    while ori >= 360:
        ori -= 360
    x, y = last_pos
    x_next, y_next = next_pos
    
    if ori == 0:
        if x_next == x and y_next > y:  # Modified here
            return ["MoveAhead"]
        elif x_next == x and y_next < y:  # Modified here
            return ["RotateRight", "RotateRight", "MoveAhead"]
        elif x_next > x and y_next == y:  # Modified here
            return ["RotateRight", "MoveAhead"]
        elif x_next < x and y_next == y:  # Modified here
            return ["RotateLeft", "MoveAhead"]
    elif ori == 90:
        if x_next == x and y_next > y:  # Modified here
            return ["RotateRight", "MoveAhead"]
        elif x_next == x and y_next < y:  # Modified here
            return ["RotateLeft", "MoveAhead"]
        elif x_next > x and y_next == y:  # Modified here
            return ["RotateRight", "RotateRight", "MoveAhead"]
        elif x_next < x and y_next == y:  # Modified here
            return ["MoveAhead"]
    elif ori == 180:
        if x_next == x and y_next > y:  # Modified here
            return ["RotateRight", "RotateRight", "MoveAhead"]
        elif x_next == x and y_next < y:  # Modified here
            return ["MoveAhead"]
        elif x_next > x and y_next == y:  # Modified here
            return ["RotateLeft", "MoveAhead"]
        elif x_next < x and y_next == y:  # Modified here
            return ["RotateRight", "MoveAhead"]
    elif ori == 270:
        if x_next == x and y_next > y:  # Modified here
            return ["RotateLeft", "MoveAhead"]
        elif x_next == x and y_next < y:  # Modified here
            return ["RotateRight", "MoveAhead"]
        elif x_next > x and y_next == y:  # Modified here
            return ["MoveAhead"]
        elif x_next < x and y_next == y:  # Modified here
            return ["RotateRight", "RotateRight", "MoveAhead"]
    else:
        print("Error: ori should be 0/90/180/270")
        return None

def postion2action_x_up(ori, last_pos, next_pos):
    """
    Input:
        last_pos: (x, y)
        next_pos: (x, y)
        ori: (0/90/180/270, int)
    Output:
        action_list(list):[action1, action2, ...]
    x-y coodinate: x is up, y is right
    """
    ori = int(ori)
    while ori < 0:
        ori += 360
    while ori >= 360:
        ori -= 360
    x, y = last_pos
    x_next, y_next = next_pos
    
    if ori == 0:
        if x_next == x and y_next > y:  # Modified here
            return ["MoveAhead"]
        elif x_next == x and y_next < y:  # Modified here
            return ["RotateRight", "RotateRight", "MoveAhead"]
        elif x_next < x and y_next == y:  # Modified here
            return ["RotateRight", "MoveAhead"]
        elif x_next > x and y_next == y:  # Modified here
            return ["RotateLeft", "MoveAhead"]
    elif ori == 90:
        if x_next == x and y_next > y:  # Modified here
            return ["RotateRight", "MoveAhead"]
        elif x_next == x and y_next < y:  # Modified here
            return ["RotateLeft", "MoveAhead"]
        elif x_next < x and y_next == y:  # Modified here
            return ["RotateRight", "RotateRight", "MoveAhead"]
        elif x_next > x and y_next == y:  # Modified here
            return ["MoveAhead"]
    elif ori == 180:
        if x_next == x and y_next > y:  # Modified here
            return ["RotateRight", "RotateRight", "MoveAhead"]
        elif x_next == x and y_next < y:  # Modified here
            return ["MoveAhead"]
        elif x_next < x and y_next == y:  # Modified here
            return ["RotateLeft", "MoveAhead"]
        elif x_next > x and y_next == y:  # Modified here
            return ["RotateRight", "MoveAhead"]
    elif ori == 270:
        if x_next == x and y_next > y:  # Modified here
            return ["RotateLeft", "MoveAhead"]
        elif x_next == x and y_next < y:  # Modified here
            return ["RotateRight", "MoveAhead"]
        elif x_next < x and y_next == y:  # Modified here
            return ["MoveAhead"]
        elif x_next > x and y_next == y:  # Modified here
            return ["RotateRight", "RotateRight", "MoveAhead"]
    else:
        print("Error: ori should be 0/90/180/270")
        return None

def update_info(env, args, last_sim_location, prev_action, info):
    curr_location = gen_util.get_sim_location(env.last_event)
    dx, dy, do = pu.get_rel_pose_change(curr_location, last_sim_location)
    last_sim_location = curr_location          
    info['sensor_pose'] = [dx, dy, do]
    
    if prev_action == 'PickupObject':  # start holidng
        info['holding_state'] = 1
        info['holding_obj'] = args.obj_list.index(env.last_interaction[0])
    elif prev_action == 'PutObject':  # end holding
        info['holding_state'] = 3
        info['holding_obj'] = -1
    elif info['holding_state'] == 1:  # is holding
        info['holding_state'] = 2
    else:
        info['holding_state'] = 0  # not holding
        info['holding_obj'] = -1
    return info

def find_available_position(obj_dict):
    agent_pos = obj_dict['agent'][0]
    agent_x, agent_y = agent_pos
    # 选择除了explore和unexplore之外的所有obj_dict中的坐标作为unvalid_block
    unvalid_block = []
    for obj_name, obj_blocks in obj_dict.items():
        if obj_name not in ['unexplore', 'explore']:
            unvalid_block.extend(obj_blocks)
    # agent四周的位置
    direction_candidate = [[agent_x-1, agent_y], [agent_x+1, agent_y], [agent_x, agent_y-1], [agent_x, agent_y+1]]
    selection_info = []
    for i in direction_candidate:
        if i not in unvalid_block:
            selection_info.append(i)
    return selection_info

def find_available_position_w_margin(obj_dict, available_boolean, collision_list):
    print('collision history', collision_list)
    agent_pos = obj_dict['Agent'][0]
    agent_x, agent_y = agent_pos
    
    # from [0, 90, 180, 270] to [90, 270, 180, 0] 上下左右
    new_available_boolean = [available_boolean[1], available_boolean[3], available_boolean[2], available_boolean[0]]
    direction_candidate = [[agent_x-1, agent_y], [agent_x+1, agent_y], [agent_x, agent_y-1], [agent_x, agent_y+1]]
    
    selection_info = []
    for i in range(4):
        if new_available_boolean[i] and direction_candidate[i] not in collision_list:
            selection_info.append(direction_candidate[i])
    return selection_info

def obj_descriptor(obj_dict):
    obj_info = ""
    # 遍历obj_dict字典，描述每个物体的位置
    for obj_name, obj_blocks in obj_dict.items():
        # 排除unexplore\explore\agent
        if obj_name not in ['unexplore', 'explore', 'agent', 'grid']:
            info = obj_name + ": " + str(obj_blocks) + "\n"
            obj_info += info

    return obj_info

def prompt_v9(chess_info, sub_goal, traj):
    """_summary_

    Args:
        chess_info (dict): 
        sub_goal (str): eg. (GotoLocation, [diningtable])
        traj (list): eg. [[20, 33], [21, 33], [22, 33], [23, 33], [24, 33]]

    Returns:
        str: prompt_v9
    """
    
    chess_info = {k.lower().replace(" ", ""): v for k, v in chess_info.items()}
    agent_position = chess_info["agent"][0]
    obj_description = obj_descriptor(chess_info)
    
    if len(traj) > 5:
        traj = traj[:5]
    
    available_position = find_available_position(chess_info)
    
    prompt = prompt_templete_v4.format(obj_description, sub_goal, agent_position, traj, available_position)
    return prompt, available_position

def prompt_v9_new(chess_info, sub_goal, available_position, traj):
    """_summary_

    Args:
        chess_info (dict): 
        sub_goal (str): eg. (GotoLocation, [diningtable])
        traj (list): eg. [[20, 33], [21, 33], [22, 33], [23, 33], [24, 33]]

    Returns:
        str: prompt_v9
    """
    
    chess_info = {k.lower().replace(" ", ""): v for k, v in chess_info.items()}
    agent_position = chess_info["agent"][0]
    obj_description = obj_descriptor(chess_info)
    
    if len(traj) > 5:
        traj = traj[:5]
    
    prompt = prompt_templete_v4.format(obj_description, sub_goal, agent_position, traj, available_position)
    return prompt


def prompt_v13(chess_info, sub_goal, available_position, traj):
    chess_info = {k.lower().replace(" ", ""): v for k, v in chess_info.items()}
    agent_position = chess_info["agent"][0]
    obj_description = obj_descriptor(chess_info)
    
    if len(traj) > 5:
        traj = traj[:5]
        
    prompt = prompt_templete_v5.format(obj_description, sub_goal, agent_position, traj, available_position)
    return prompt


def prompt_zeroshot_mistral(chess_info, sub_goal, available_position, traj):
    chess_info = {k.lower().replace(" ", ""): v for k, v in chess_info.items()}
    agent_position = chess_info["agent"][0]
    obj_description = obj_descriptor(chess_info)
    
    if len(traj) > 5:
        traj = traj[:5]
        
    prompt = prompt_templete_v6.format(obj_description, sub_goal, agent_position, traj, available_position)
    return prompt


def model_generate(model, tokenizer, prompt):
    # model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda:1")
    # model.to("cuda:1")

    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    model.to("cuda")

    # 只输出[\INST]之后的内容
    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
    result = tokenizer.batch_decode(generated_ids)[0]
    result = result.split(prompt)[-1]
    result = result.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    print(result)
    return result

def agent_step(model, tokenizer, prompt, orientation, last_pos, env, args, num_fails, seg_model):
    '''
    environment step based on model prediction
    '''

    # TODO: (hanxuan)
    '''
    # forward model
    with torch.no_grad():
        m_out = model.step(input_dict, vocab, prev_action=prev_action)
    m_pred = model_util.extract_action_preds(
        m_out, model.pad, vocab['action_low'], clean_special_tokens=False)[0]
    action = m_pred['action']
    if args.debug:
        print("Predicted action: {}".format(action))
    '''
    with torch.no_grad():
        prompt = f"[INST] {prompt} [/INST]"
        result = model_generate(model, tokenizer, prompt)
    
    # FIXME: transfer the model output into position
    # -------------------- V6 -------------------------
    # matches = re.findall(r'\[(\d+),\s*(\d+)\]', result)
    # next_pos = [(int(match[0]), int(match[1])) for match in matches]
    # -------------------- V9 -------------------------
    pos = result.find("# Next position")
    start_pos = result.find("[", pos)
    end_pos = result.find("]", pos)
    next_pos = result[start_pos : end_pos + 1]
    matches = re.findall(r'\[(\d+),\s*(\d+)\]', next_pos)
    next_pos = [(int(match[0]), int(match[1])) for match in matches]
    # -------------------------------------------------
    print('next_pos', next_pos)
    if len(next_pos) == 0:
        action_list = []
        return False, None, last_pos, num_fails, '', None, 0, False  # may get into endless loop
    else:
        next_pos = next_pos[0]
    # FIXME: x up/down
    action_list = postion2action(orientation, last_pos, next_pos)
    # action_list = postion2action_x_up(orientation, last_pos, next_pos)

    # action = "[MoveAhead]"
    # action = "[PickupObject, CD]"
    # m_pred = {'object': ['Alarm Clock', 'Desk'], 'action': 'PutObject'}
    # m_pred = {'object': None, 'action': 'MoveAhead'}
    # action = m_pred['action']
    action_count = 0
    action_success = True
    for action in action_list:
        m_pred = {'object': None, 'action': action}
        action = m_pred['action']

        mask = None
        obj = None
        if has_interaction(action):
            obj = m_pred['object'][0]
            obj_idx = args.obj_list.index(obj)
        if obj is not None:
            # get mask from a pre-trained RCNN
            assert seg_model is not None
            mask = extract_rcnn_pred(obj, obj_idx, seg_model, env, args.debug, args.renderInstanceSegmentation)
            m_pred['mask_rcnn'] = mask
        
        '''
        # remove blocking actions
        action = obstruction_detection(
            action, env, m_out, model.vocab_out, args.debug)
        m_pred['action'] = action
        '''

        # use the predicted action
        episode_end = (action == constants.STOP_TOKEN)
        api_action = None
        # constants.TERMINAL_TOKENS was originally used for subgoal evaluation
        target_instance_id = ''
        if not episode_end:
            action_count += 1
            step_success, _, target_instance_id, err, api_action = env.va_interact(
                action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            env.last_interaction = (obj, mask)
            if not step_success:
                action_success = False
                num_fails += 1
                if num_fails >= args.max_fails:
                    if args.debug:
                        print("Interact API failed {} times; latest error '{}'".format(
                            num_fails, err))
                    episode_end = True
    return episode_end, str(action), next_pos, num_fails, target_instance_id, api_action, action_count, action_success

def agent_nav_step(model, tokenizer, prompt, available_position, orientation, last_pos, env, args, num_fails, seg_model, try_times=3):
    '''
    environment step based on model prediction
    '''
    available, tries = False, 0
    while not available and tries < try_times:
        with torch.no_grad():
            prompt = f"[INST] {prompt} [/INST]"
            result = model_generate(model, tokenizer, prompt)
        tries += 1
        pos = result.find("# Next position")
        start_pos = result.find("[", pos)
        end_pos = result.find("]", pos)
        next_pos = result[start_pos : end_pos + 1]
        matches = re.findall(r'\[(\d+),\s*(\d+)\]', next_pos)
        next_pos = [(int(match[0]), int(match[1])) for match in matches]
        # -------------------------------------------------

        if len(next_pos) == 0: continue
        else: next_pos = next_pos[0]

        if list(next_pos) in available_position: available = True
    
    if tries >= try_times:
        if len(available_position) == 0:
            print('Got stuck, task failed!')
            return True, None, last_pos, num_fails, '', None, 0, False
        next_pos = random.choice(available_position)  # random choose one from available positions
        # return False, None, last_pos , num_fails, '', None, 0, False  # may get into endless loop
    
    print('next_pos', next_pos)
    # FIXME: x up/down
    action_list = postion2action(orientation, last_pos, next_pos)
    # action_list = postion2action_x_up(orientation, last_pos, next_pos)

    action_count = 0
    action_success = True
    # FIXME: return invalid orientation
    if action_list == None:
        print('Invalid action, task failed!')
        return True, None, last_pos, num_fails, '', None, 0, False
    for action in action_list:
        m_pred = {'object': None, 'action': action}
        action = m_pred['action']
        
        assert not has_interaction(action)
        
        # remove blocking actions
        action = obstruction_detection(action, env, args.debug)
        
        # use the predicted action
        episode_end = (action == constants.STOP_TOKEN)
        api_action = None
        # constants.TERMINAL_TOKENS was originally used for subgoal evaluation
        target_instance_id = ''
        if not episode_end:
            action_count += 1
            step_success, _, target_instance_id, err, api_action = env.va_interact(
                action, interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
            if not step_success:
                print('action failed.', err)
                action_success = False
                num_fails += 1
                if num_fails >= args.max_fails:
                    if args.debug:
                        print("Interact API failed {} times; latest error '{}'".format(
                            num_fails, err))
                    episode_end = True
    return episode_end, str(action), next_pos, num_fails, target_instance_id, api_action, action_count, action_success

def agent_step_auto(subgoal, env, args, num_fails, seg_model, output_dir, trial_dir, t_agent):
    '''
    environment step automatically for interaction
    '''
    objects = subgoal[1]  # [small_object, receptacle_object] lower format, alarmclock
    objects = [gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[obj]) for obj in objects]  # space format, Alarm Clock
    if subgoal[0] == "CleanObject":
        action_list = [
            {'action': "PutObject", 'object': "Sink Basin"},  # receptacle_object
            {'action': "ToggleObjectOn", 'object': "Faucet"}, 
            {'action': "ToggleObjectOff", 'object': "Faucet"},
            {'action': "PickupObject", 'object': objects[0]},
            ]
    elif subgoal[0] == "CoolObject":
        action_list = [
            {'action': "OpenObject", 'object': "Fridge"}, 
            {'action': "PutObject", 'object': "Fridge"},  # receptacle_object
            {'action': "CloseObject", 'object': "Fridge"},
            {'action': "OpenObject", 'object': "Fridge"},
            {'action': "PickupObject", 'object': objects[0]},
            {'action': "CloseObject", 'object': "Fridge"}
            ]
    elif subgoal[0] == "HeatObject":
        action_list = [
            {'action': "OpenObject", 'object': "Microwave"}, 
            {'action': "PutObject", 'object': "Microwave"},  # receptacle_object
            {'action': "CloseObject", 'object': "Microwave"},
            {'action': "ToggleObjectOn", 'object': "Microwave"},
            {'action': "ToggleObjectOff", 'object': "Microwave"},
            {'action': "OpenObject", 'object': "Microwave"},
            {'action': "PickupObject", 'object': objects[0]},
            {'action': "CloseObject", 'object': "Microwave"}
            ]
    elif subgoal[0] == "NoOp":
        action_list = [{'action': "<<stop>>", 'object': None}]
    elif subgoal[0] == "ToggleObject":
        action_list = [{'action': "ToggleObjectOn", 'object': objects[0]}]
    elif subgoal[0] == "PutObject":
        # FIXME: interact with sink basin or sink?
        if objects[-1] == 'Sink Basin':
            action_list = [
                # {'action': "LookDown", 'object': None},
                {'action': "PutObject", 'object': 'Sink Basin'},
                # {'action': "LookUp", 'object': None}
                ]
        if objects[-1] == 'Bathtub Basin':
            # action_list = [{'action': "PutObject", 'object': 'bathtub'}]
            action_list = [
                # {'action': "LookDown", 'object': None},
                {'action': "PutObject", 'object': 'Bathtub Basin'},
                # {'action': "LookUp", 'object': None}
                ]
        if objects[-1] == 'Cabinet':
            # action_list = [{'action': "PutObject", 'object': 'bathtub'}]
            action_list = [
                {'action': "LookDown", 'object': None},
                {'action': "LookDown", 'object': None},
                {'action': "OpenObject", 'object': 'Cabinet'},
                {'action': "PutObject", 'object': 'Cabinet'},
                {'action': "LookUp", 'object': None},
                {'action': "LookUp", 'object': None}
                ]
        else:
            action_list = [{'action': "PutObject", 'object': objects[-1]}]
    else:
        action_list = [{'action': subgoal[0], 'object': objects[0]}]
    
    action_count = 0
    action_success = True
    last_success_action = None
    target_object = None
    if len(objects) > 0:
        target_object = objects[-1].replace(' ', '')
    print('target_object', target_object)
    visible = False
    while not visible and action_count < 3 and target_object != None:
        for obj in env.last_event.metadata["objects"]:
            if obj["objectId"] in env.last_event.instance_masks.keys() and obj["objectType"] == target_object:
                print(obj["objectId"])
                if obj["visible"]: 
                    visible = True
                    break
                
        print('is visible?', visible)
        if not visible:
            mask = None
            action = 'RotateRight'
            _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            action_count += 1

    for m_pred in action_list:
        action = m_pred['action']
        obj = m_pred['object']
        mask = None
        if has_interaction(action):
            obj = m_pred['object']
            obj_idx = args.obj_list.index(obj)
        if obj is not None:
            mask = extract_rcnn_pred(obj, obj_idx, seg_model, env, args.debug, args.renderInstanceSegmentation)
            m_pred['mask_rcnn'] = mask

        # use the predicted action
        episode_end = (action == constants.STOP_TOKEN)
        api_action = None
        # constants.TERMINAL_TOKENS was originally used for subgoal evaluation
        target_instance_id = ''
        if not episode_end:
            action_count += 1
            step_success, _, target_instance_id, err, api_action = env.va_interact(
                action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            
            rgb = env.last_event.frame
            rgb_save_path = os.path.join(output_dir, trial_dir, 'seg_images', '{:09d}.png'.format(t_agent+action_count))
            cv2.imwrite(rgb_save_path, rgb)
            
            if not step_success:
                print('action failed.', err)
                action_success = False
                num_fails += 1
                if num_fails >= args.max_fails:
                    if args.debug:
                        print("Interact API failed {} times; latest error '{}'".format(
                            num_fails, err))
                    episode_end = True
            else:  # interact action success!
                last_success_action = action
                env.last_interaction = (obj, mask)
    return episode_end, str(last_success_action), num_fails, target_instance_id, api_action, action_count, action_success

def is_arrived_L1(curr_pos, obj_dict, target_obj, threshold_dis=6):
    """
    L1 Distance
    """
    if target_obj not in obj_dict.keys(): return False
    for position in obj_dict[target_obj]:
        dis = sum(abs(p - q) for p, q in zip(curr_pos, position))
        if dis <= threshold_dis: return True
    return False


def is_arrived(curr_pos, obj_dict, target_obj, threshold_dis=6):
    """
    L2 Distance
    """
    if target_obj not in obj_dict.keys(): return False
    min_dis = np.inf
    for position in obj_dict[target_obj]:
        dis = np.sqrt(sum(np.power(p - q, 2) for p, q in zip(curr_pos, position)))
        if dis <= min_dis:
            min_dis = dis
        if dis <= threshold_dis:
            print("&"*10, "is arrived, min dis =", dis)
            print("curr_pos", curr_pos)
            print("target position", position)
            return True
    print("&"*10, "not arrive")
    return False


def move_to_target(curr_pos, obj_dict, target_obj, best_dis=4):
    if target_obj not in obj_dict.keys(): return []
    min_dis = 0
    for position in obj_dict[target_obj]:
        dis = sum(abs(p - q) for p, q in zip(curr_pos, position))
        if dis <= min_dis: 
            min_dis = dis
    if min_dis == best_dis:
        return []
    elif min_dis < best_dis:   
        return ["RotateRight", "RotateRight"] + ["MoveAhead" for x in range(best_dis - min_dis)] + ["RotateLeft", "RotateLeft"]
    else:
        return ["MoveAhead" for x in range(min_dis - best_dis)]
    

def try_move_to_target(curr_pos, obj_dict, target_obj, best_dis=4):
    if target_obj not in obj_dict.keys(): return []
    min_dis = 0
    for position in obj_dict[target_obj]:
        dis = sum(abs(p - q) for p, q in zip(curr_pos, position))
        if dis <= min_dis: 
            min_dis = dis
    if min_dis == best_dis:
        return []
    elif min_dis < best_dis:   
        for x in range(best_dis - min_dis):
            move_action_list += move_back_action_list
    else:
        move_action_list = ["MoveAhead" for x in range(min_dis - best_dis)]
    
    return move_action_list

def move_back_action_list():
    return ["RotateRight", "RotateRight", "MoveAhead", "RotateLeft", "RotateLeft"]

def generate_return_action_list(num_move_steps):
    return ["RotateRight", "RotateRight"] + ["MoveAhead" for x in range(num_move_steps)] + ["RotateLeft", "RotateLeft"]


def look_down(env, args, sem_map_module):
    action = 'LookDown'
    mask = None
    _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
    sem_map_module.reset_view_angle(env.last_event.metadata['agent']['cameraHorizon'])
    
def look_up(env, args, sem_map_module):
    action = 'LookUp'
    mask = None
    _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
    sem_map_module.reset_view_angle(env.last_event.metadata['agent']['cameraHorizon'])

class OutputRedirector:
    def __init__(self, filename):
        """
        初始化重定向器，指定输出文件名。
        :param filename: 输出文件的名称
        """
        self.filename = filename
        self.original_stdout = sys.stdout
        self.file = None

    def start(self):
        """
        开始重定向输出到文件。
        """
        self.file = open(self.filename, 'a')
        sys.stdout = self.file

    def stop(self):
        """
        停止重定向输出并恢复标准输出。
        """
        if self.file:
            sys.stdout = self.original_stdout
            self.file.close()
            self.file = None

def obstruction_detection(action, env, verbose):
    '''
    change 'MoveAhead' action to a turn in case if it has failed previously
    '''
    if action != 'MoveAhead':
        return action
    if env.last_event.metadata['lastActionSuccess']:
        return action
    action = random.choice(['RotateLeft', 'RotateRight'])
    # action = 'LookDown'
    if verbose:
        print("Blocking action is changed to: {}".format(action))
    return action

def delet_pickup_obj(obj_dict, pickup_obj, recep_obj, remove_pos, threshold_dis=10):  # space format
    print('--------delet {} from {}--------'.format(pickup_obj, recep_obj))
    obj_dict_new = copy.deepcopy(obj_dict)
    if len(remove_pos) == 0:
        curr_pos = obj_dict["Agent"][0]    
        for position in obj_dict[pickup_obj]:
            dis = sum(abs(p - q) for p, q in zip(curr_pos, position))
            if dis <= threshold_dis:
                remove_pos.append(position)
    for position in remove_pos:
        obj_dict_new[pickup_obj].remove(position)
        if len(obj_dict_new[pickup_obj]) == 0:
            del obj_dict_new[pickup_obj]
        obj_dict_new[recep_obj].append(position)
    print('remove_pos', remove_pos)
    print('obj_dict_new', obj_dict_new[recep_obj])
    return obj_dict_new, remove_pos