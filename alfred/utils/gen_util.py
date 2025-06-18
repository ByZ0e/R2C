from alfred import constants
import os
import json
import queue
import torch
import shutil
import filelock
import re
import numpy as np
from torchvision import transforms
from PIL import Image
from termcolor import colored
import copy

def none_or_str(string):
    if string == '':
        return None
    else:
        return string

def exist_or_no(string):
    if string == '' or string == False:
        return 0
    else:
        return 1

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
    '''
    "init_action": {
            "action": "TeleportFull",
            "horizon": 30,
            "rotateOnTeleport": true,
            "rotation": 0,
            "x": 0.5,
            "y": 0.9009992,
            "z": -1.0
        }
    '''
    # return traj_data['scene']['init_action']

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

def get_task_desc(json_dict):
    # task_type = json_dict["task_type"]
    task_anns = json_dict["turk_annotations"]["anns"]
    task_desc = []
    high_descs = []
    for anno in task_anns:
        task_desc.append(anno["task_desc"])
        high_descs.append(anno["high_descs"])
    return {"high_descs": high_descs, "task_desc": task_desc}

def read_task_info(json_dict):
    task_type = json_dict["task_type"]
    task_anns = json_dict["turk_annotations"]["anns"]
    task_desc = []
    for anno in task_anns:
        task_desc.append(anno["task_desc"])
    return {"task_type": task_type, "task_desc": task_desc}

def get_subgoals(json_dict):
    sub_task_list = []
    for i in json_dict['plan']['high_pddl']:
        action = i['discrete_action']['action']
        args = i['discrete_action']['args']
        sub_task_list.append((action, args))
    return sub_task_list

# BUG heatObject also need pickup and put
def get_holding_idx(json_dict):
    hl_actions = json_dict['plan']['high_pddl']
    pickup_idx = 0
    put_idx = len(hl_actions)
    for action in hl_actions:
        if action['discrete_action']['action'] == 'PickupObject':
            pickup_idx = action['high_idx']
        elif action['discrete_action']['action'] == 'PutObject':
            put_idx = action['high_idx']
    return pickup_idx, put_idx

def camel_to_space(input_str):
    result_str = re.sub(r'([a-z])([A-Z])', r'\1 \2', input_str)
    return result_str

def convert_to_camel_case(input_str):
    words = input_str.split()
    result_str = words[0] + ''.join(word.lower() for word in words[1:])
    return result_str

def remove_space(input_str):
    result_str = input_str.replace(' ', '')
    return result_str

def get_ll_action(json_dict, step, num_steps):
    """get low-level action of current step

    Args:
        json_dict ([dic]): [description]
        step ([int]): [description]

    Returns:
        [tuple]: [action tuple]
        [bool]: [PickUpFlag]
        [bool]: [putDownFlag] 
    """
    PickUp, PutDown = 0, 0
    act = ()
    if step == num_steps:  # ending step
        act = ('STOP')
        high_idx = -1
        action = json_dict['plan']['low_actions'][step-1]["api_action"]
    else:
        action = json_dict['plan']['low_actions'][step]["api_action"]
        high_idx = json_dict['plan']['low_actions'][step]["high_idx"]
        if "objectId" not in action.keys():  # nav act
            act = (action["action"])
        elif action["action"] == "PickupObject":
            PickUp = 1
            obj = camel_to_space(action["objectId"].split('|')[0])
            act = (action["action"], obj)
        elif action["action"] == "PutObject":
            PutDown = 1
            obj = camel_to_space(action["objectId"].split('|')[0])
            r_obj = camel_to_space(action["receptacleObjectId"].split('|')[0])
            act = (action["action"], obj, r_obj)
        else:
            obj = camel_to_space(action["objectId"].split('|')[0])
            act = (action["action"], obj)
    return act, PickUp, PutDown, high_idx, action

def agent_step(cmd, env):
    cmd = {k: cmd[k] for k in [
            'action', 'objectId', 'receptacleObjectId',
            'placeStationary', 'forceAction'] if k in cmd}
    event = env.step(cmd)
    if event is None:
        return False
    if not event.metadata['lastActionSuccess']:
        print(colored("Replay Failed: %s" % (
            env.last_event.metadata['errorMessage']), 'red'))
        return False

def get_obj_args(traj_data):
    task_type = traj_data['task_type']
    sliced = exist_or_no(traj_data['pddl_params']['object_sliced'])
    mrecep_target = none_or_str(traj_data['pddl_params']['mrecep_target'])  # 装交互小物体的 move_receptacle
    object_target = none_or_str(traj_data['pddl_params']['object_target'])  # 交互小物体
    parent_target = none_or_str(traj_data['pddl_params']['parent_target'])  # 放置小物体的目标大物体
    toggle_target = none_or_str(traj_data['pddl_params']['toggle_target'])  # Lamp
    # if sliced:
    #     object_target = object_target + 'Sliced'
    if parent_target == "Sink":
        parent_target = "Sink Basin"
    if parent_target == "Bathtub":
        parent_target = "Bathtub Basin"
    # if object_target == "Knife":
    #     object_target = "ButterKnife"
    # categories_in_inst = [x for x in [mrecep_target, object_target, parent_target, toggle_target] if x != None]
    categories_in_inst = [x for x in [object_target, mrecep_target, parent_target, toggle_target] if x != None]  # 按优先级顺序！！！
    if task_type == 'pick_two_obj_and_place':
        categories_in_inst = [x for x in [parent_target, mrecep_target, object_target, toggle_target] if x != None]  # 按优先级顺序！！！
    
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

def get_obj_list(traj_data):
    obj_list = copy.deepcopy(constants.LARGE_OBJ)
    obj_targets = get_obj_args(traj_data)
    for obj in obj_targets:
        if obj not in obj_list:  # 交互小物体
            obj_list.append(obj)
        else:
            obj_list.remove(obj)
            obj_list.append(obj)
    obj_list = [camel_to_space(x) for x in obj_list]
    return obj_list


def get_obj_list_rcnn(traj_data):
    obj_list = copy.deepcopy(constants.LARGE_OBJ)
    obj_targets = get_obj_args(traj_data)
    for obj in obj_targets:
        if obj not in obj_list:  # 交互小物体
            obj_list.append(obj)
        else:
            obj_list.remove(obj)
            obj_list.append(obj)
    obj_list = [camel_to_space(x) for x in obj_list]
    
    for obj in obj_list:
        if 'Basin' in obj: obj = obj.replace(' Basin', '')
    return obj_list

def get_color_to_object(json_dict):
    return json_dict["scene"]["color_to_object_type"]

def get_sim_location(event):
    y = - event.metadata['agent']['position']['x']
    x = event.metadata['agent']['position']['z']
    o = np.deg2rad(-event.metadata['agent']['rotation']['y']) # (-np.pi, np.pi]
    if o > np.pi:
        o -= 2 * np.pi
    if o <= -np.pi:
        o += 2 * np.pi
    return x, y, o

def get_location(action):
    y = - action['x']
    x = action['z']
    o = np.deg2rad(-action['rotation']) # (-np.pi, np.pi]
    if o > np.pi:
        o -= 2 * np.pi
    if o <= -np.pi:
        o += 2 * np.pi
    return x, y, o

def read_agent_pose(pose_dic):
    x = pose_dic['position']['x']
    y = pose_dic['position']['y']
    z = pose_dic['position']['z']
    horizon = pose_dic['cameraHorizon']
    rotation = pose_dic['rotation']['y']
    standing = True
    pose = dict(
        x=x,
        y=y,
        z=z,
        horizon=horizon,
        rotation=rotation,
        standing=standing
    )
    return pose

def preprocess_obs(rgb, bgr, depth, args, seg_model=None, env=None, obj_classes=None):  
    def _get_gt_segmentation():
        gt_large_objects = [remove_space(x) for x in args.obj_list]
        # print(gt_large_objects)
        sem_seg_image = env.last_event.instance_segmentation_frame
        sem_seg_pred = np.zeros((args.env_frame_height, args.env_frame_width, args.num_sem_categories))

        for obj in env.last_event.metadata["objects"]:
            if obj["objectType"] in constants.OBJECT_MAPPING.keys():
                obj["objectType"] = constants.OBJECT_MAPPING[obj["objectType"]]
            # if obj["objectType"] == 'SinkBasin':
            #     print(obj["objectType"], obj["visible"], obj["distance"])
            # if obj["visible"] and \
            #    obj["objectId"] in self.env.last_event.instance_masks.keys() and \
            #    obj["objectType"] in gt_large_objects:
            if obj["objectId"] in env.last_event.instance_masks.keys() and \
                obj["objectType"] in gt_large_objects:
                obj_id = gt_large_objects.index(obj["objectType"])
                obj_mask = env.last_event.instance_masks[obj["objectId"]] * 1.
                sem_seg_pred[:, :, obj_id] += obj_mask

        return sem_seg_pred, sem_seg_image
    
    if args.renderInstanceSegmentation:
        # GT segmentation
        sem_seg_pred, sem_seg_image = _get_gt_segmentation()
    else:
        # Grounded SAM, 似乎要 BGR 图片, seg_seg_pred [300, 300, #Class]
        if args.seg == 'maskrcnn':
            # desklamp_idx_ = obj_classes.index("DeskLamp")
            # floorlamp_idx_ = obj_classes.index("FloorLamp")
            # none_idx = obj_classes.index("None")
            # if desklamp_idx_ > none_idx or floorlamp_idx_ > none_idx:
            #     detections, sem_seg_image = seg_model.segmentation_for_map(env, classes_needed=obj_classes, confidence_threshold=0.2)
            # else:
            #     detections, sem_seg_image = seg_model.segmentation_for_map(env, classes_needed=obj_classes, confidence_threshold=0.6)
            
            detections, sem_seg_image = seg_model.segmentation_for_map(env, classes_needed=obj_classes, confidence_threshold=0.75)
            
            sem_seg_pred = np.zeros((args.env_frame_height, args.env_frame_width, args.num_sem_categories))
            for _, mask, confidence, class_id, _ in detections:
                v = confidence * mask
                sem_seg_pred[:, :, class_id] += v.astype('float')
        else:
            sem_seg_pred, sem_seg_image = seg_model.segmentation_for_map(bgr.astype(np.uint8), classes_needed=obj_classes)

    # TODO: event.depth_frame 单位是 mm，点云地图要 cm
    depth = depth / 1000.
    mask = depth > args.max_depth
    # FIXME: 
    # depth[mask] = self.max_depth
    depth[mask] = 100.
    depth = depth * 100.
    
    ds = args.env_frame_width // args.frame_width  # Downscaling factor, args.env_frame_width=640, args.frame_width=160
    if ds != 1:
        res = transforms.Compose([transforms.ToPILImage(), transforms.Resize((args.frame_height, args.frame_width), interpolation=Image.NEAREST)])
        rgb = np.asarray(res(rgb.astype(np.uint8)))
        bgr = np.asarray(res(bgr.astype(np.uint8)))
        # sem_seg_image = np.asarray(res(sem_seg_image.astype(np.uint8)))
        depth = depth[ds//2::ds, ds//2::ds]
        sem_seg_pred = sem_seg_pred[ds//2::ds, ds//2::ds]

    
    depth = np.expand_dims(depth, axis=2)

    obs = np.concatenate((rgb, depth, sem_seg_pred),axis=2).transpose(2, 0, 1) # C H W

    return obs, sem_seg_image

    
