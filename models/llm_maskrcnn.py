import os
import json
import datetime
import cv2
import numpy as np
import supervision as sv
from PIL import Image
import re

from alfred.utils import gen_util, eval_util
from keyboard_fb.sem_map_helper import SemMapHelper
import envs.utils.pose as pu
from keyboard_fb.chessboard_vis.infer_vis import ChessVis
from keyboard_fb import agent_obj_rel
# TODO: DINO+SAM
# from segmentation_helper2 import SegmentationHelper
# from segmentation_helper import SegmentationHelper
from alfred import constants
from alfred.nn.enc_visual import FeatureExtractor
from alfred.utils.depth import apply_real_noise

def subgoal_str2list(subgoal_str):
    subgoal_str = subgoal_str.replace(" ", "")
    tuples = re.findall(r'\(([^)]+)\)', subgoal_str)
    sub_goal = []
    for step in tuples:
        step_split = step.split(",", 1)
        action = step_split[0]
        objects = step_split[1].replace("[", "").replace("]", "").split(",")
        sub_goal.append([action, objects])
    return sub_goal

# TODO: load maskrcnn model
def load_maskrcnn(args):
    if args.seg != 'maskrcnn':
        return None
    return FeatureExtractor(
        archi='maskrcnn', device='cuda',
        checkpoint=args.maskrcnn_path, load_heads=True)

class R2C:
    def __init__(self, args):
        self.args = args
        self.args.pp_folder = 'pp'
        self.json_name = 'traj_data.json'
        if not self.args.renderInstanceSegmentation:
            if self.args.seg == 'maskrcnn':
                self.seg_model = load_maskrcnn(self.args)
            else:
                pass
                self.seg_model = SegmentationHelper(self.args)
        else:
            self.seg_model = None
        
    def start_eval(self, task, split, r_idx):
        '''
        init eval
        '''
        print('====================start eval {}================='.format(task['task']))
        json_path = os.path.join(self.args.data_dir, split, task['task'], self.json_name)
        self.trial_dir = '/'.join(json_path.split('/')[-4:-1])
        self.trial_dir += '/' + str(r_idx)
        print('self.trial_dir', self.trial_dir)
        
        with open(json_path) as f:
            self.json_dict = json.load(f)
        
        obj_list = gen_util.get_obj_list(self.json_dict)
        # TODO: DINO+SAM
        if not self.args.renderInstanceSegmentation:
            if self.args.seg == 'G-DINO':
                self.seg_model.obj_classes = obj_list
        self.args.obj_list = obj_list
        # TODO: DINO+SAM
        self.obj_classes = [gen_util.remove_space(x) for x in self.args.obj_list]
        if self.args.seg == 'G-DINO':
            self.obj_classes = None
        
        print('self.obj_classes', self.obj_classes)
        self.args.num_sem_categories = len(obj_list)
        print('----------------init sem map helper---------------')
        print(self.args.num_sem_categories)
        self.sem_map_module = SemMapHelper(self.args,map_x_down=True)
        
        self.prev_action = None
        self.episode_end = False
        self.action_success = False
        self.searching = False 
        self.navigate_finished = False
        self.interact_finished = False
        
        self.subgoal_idx = -1
        self.num_fails = 0
        self.reward = 0
        
        self.sliced = False
        self.sliced_object = None
        self.slice_subgoal_idx = -1
        
        self.heat = False
        self.microwave = False
                
        self.openable_obj = ["Cabinet", "Drawer", "Safe", "Box"]
        
        if not os.path.exists(os.path.join(self.args.output_dir, self.trial_dir)):
            os.makedirs(os.path.join(self.args.output_dir, self.trial_dir))
        
        if self.args.log:
            print('logging in {}'.format(os.path.join(self.args.output_dir, self.trial_dir, 'log.txt')))    
            self.logger = eval_util.OutputRedirector(os.path.join(self.args.output_dir, self.trial_dir, 'log.txt'))
            self.logger.start()
        
        return self.json_dict
    
    def agent_state_init(self):
        self.info = {'sensor_pose': [0., 0., 0.], 'holding_state': 0, 'holding_obj': -1, 'holding_box': False, 'holding_obj_with_hole': False}
        self.init_action = self.json_dict['scene']['init_action']
        self.orientation = 0
        self.last_orientation = self.orientation
        self.do = 0
        self.history_traj = []
        
    def init_chessboard(self):
        # init sem map module
        self.sem_map_module.sem_map_builder.agent_height = 155.
        self.sem_map_module.reset_view_angle(self.init_action["horizon"])
        self.last_sim_location = gen_util.get_location(self.init_action)
        
        # init visualization
        self.chess_vis = ChessVis(x_down=True, chess_size=int(self.args.map_size_cm/self.args.agent_step_length))
        self.chess_vis.reset_history_traj()
    
    def close_log(self):
        self.logger.stop()
    
    def get_subgoals(self, r_idx, process_idx, chessboard_info_queue, next_position_queue):
        """
        
        """
        task_info = gen_util.get_task_desc(self.json_dict)
        task = task_info['task_desc'][r_idx]
        desc = task_info['high_descs'][r_idx]
        print('task_desc', task)
        print('high_descs', desc)
        
        # ----FIX----
        # prompt = eval_util.task_prompt_generator(task, desc)
        # msg = (process_idx, prompt)
        
        # chessboard_info_queue.put(msg)
        
        # llm_subgoals_output = next_position_queue.get()
        # llm_subgoals = subgoal_str2list(llm_subgoals_output)
        # ----FIX----
        
        # task_info = eval_util.read_task_data(traj_data, subgoal_idx)  # return task_dict['subgoal_idx'],['subgoal_action']
        gt_gt_subgoals = gen_util.get_subgoals(self.json_dict)  # ["GotoLocation", ["obj", "r_obj"]]
        # gt_subgoals = [["GotoLocation", ["fridge"]], ["OpenObject", ["fridge"]]]
        
        # gt_subgoals = llm_subgoals
        gt_subgoals = gt_gt_subgoals
        
        if gt_subgoals[-1][0] != "NoOp":
            gt_subgoals.append(["NoOp", []])
        for idx, subgoal in enumerate(gt_subgoals):
            if subgoal[0] == "SliceObject":
                self.sliced = True
                self.sliced_object = subgoal[1][0]  # 'lettuce'
                self.slice_subgoal_idx = idx
                print(self.sliced, self.sliced_object, self.slice_subgoal_idx)
                break
        # for idx, subgoal in enumerate(gt_subgoals):
        #     if subgoal[0] == "HeatObject":
        #         self.heat = True
        #         break
        for idx, subgoal in enumerate(gt_subgoals):
            if subgoal[1] == []:continue
            elif subgoal[1][-1] == "microwave":
                self.microwave = True
                break
        self.subgoals = gt_subgoals
        print('gt_subgoals:', gt_subgoals)
    
    def set_subgoal(self, prev_action):
        self.prev_action = prev_action
        if self.subgoal_idx == -1: self.subgoal_idx += 1
        self.curr_subgoal = self.subgoals[self.subgoal_idx]
        print(self.curr_subgoal)
        
        # TODO: strategy: get small target object!!
        if self.curr_subgoal[0] == 'GotoLocation':
            # FIXME: what is the target object?
            # target_obj = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[subgoal[1][0]])
            
            self.large_target = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[self.curr_subgoal[1][-1]])
            # small_target = 'none'
            if self.subgoal_idx + 1 < len(self.subgoals) - 1 and self.prev_action != 'PickupObject':
                self.small_target = self.subgoals[self.subgoal_idx+1][1][-1]
                self.small_target = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[self.small_target])
            else: 
                self.small_target = self.large_target
            
            # if self.sliced == True:
            #     self.correct_slice_nav()
            self.curr_subgoal = eval_util.get_subgoal_str(self.curr_subgoal)
            print('large_target', self.large_target)
            print('small_target', self.small_target) 
                
    def correct_slice_nav(self):
        if self.subgoal_idx > self.slice_subgoal_idx:
            if self.sliced_object in self.small_target.lower():
                self.small_target += ' Sliced'
            if self.sliced_object in self.large_target.lower():
                self.large_target += ' Sliced'
    
    def update_subgoal_idx(self):
        if isinstance(self.curr_subgoal, str) and self.navigate_finished:
            self.subgoal_idx += 1
            self.navigate_finished = False
            # subgoal_success = (env.get_subgoal_idx() == subgoal_idx)  # env.get_subgoal_idx() return env.finished
            subgoal_success = True
        
        # FIXME: what if interaction action failed?
        elif not isinstance(self.curr_subgoal, str) and self.interact_finished:
            self.subgoal_idx += 1
            # subgoal_success = (env.get_subgoal_idx() == subgoal_idx)  # env.get_subgoal_idx() return env.finished
            subgoal_success = True
            
        else:
            subgoal_success = False
        return subgoal_success
    
    def update_history(self, last_pos):
        self.history_traj.insert(0, last_pos)
        
    def update_collision(self, next_pos):
        self.sem_map_module.collision.append(list(next_pos))
    
    def save_bgr(self, bgr, step):
        pass
        bgr_save_path = os.path.join(self.args.output_dir, self.trial_dir, 'rgb_images', '{:09d}.png'.format(step))
        cv2.imwrite(bgr_save_path, bgr)
    
    def save_depth(self, depth, step, add_depth_noise=False):
        pass
        depth_save_path = os.path.join(self.args.output_dir, self.trial_dir, 'depth_images', '{:09d}.png'.format(step))
        if add_depth_noise:
            depth = apply_real_noise(depth_arr=depth, size=300)
        depth_image = depth * (255 / 10000)
        depth_image = depth_image.astype(np.uint8)
        cv2.imwrite(depth_save_path, depth_image)
        
    def save_seg(self, seg, step):
        pass
        seg_save_path = os.path.join(self.args.output_dir, self.trial_dir, 'seg_images', '{:09d}.png'.format(step))
        cv2.imwrite(seg_save_path, seg)
    
    def visualize_chessboard(self, t_agent, last_pos, next_pos):
        pass
        # visualize chessboard
        # if self.args.debug:
        #     chess_output_dir = os.path.join(self.args.output_dir, self.trial_dir, 'chess')
        #     # if t_agent < 4 or isinstance(self.curr_subgoal, str):
        #     #     next_chessboard_pos = last_pos
        #     self.chess_vis.infer_chessboard(t_agent, self.obj_dict, self.args.obj_list, last_pos, next_pos, self.orientation, self.curr_subgoal, output_dir=chess_output_dir)
        #     self.chess_vis.add_history_traj(last_pos)
    
    
    def save_sam_seg(self, env, t_agent):
        pass
        # model.save_bgr(bgr=env.last_event.cv2img, step=t_agent)
        rgb, bgr, depth = eval_util.get_rgbd(env, add_depth_noise=self.args.add_depth_noise)
        # obj_classes = [gen_util.remove_space(x) for x in self.args.obj_list]
        obs, sem_seg_image = gen_util.preprocess_obs(rgb, bgr, depth, self.args, self.seg_model, env, self.obj_classes)
        self.save_seg(seg=sem_seg_image, step=t_agent)
    
    # TODO:
    def step(self, env, t_agent, prev_action, next_pos):
        self.prev_action = prev_action
        # pred = {}
        # pred["action_low"] = []
        # pred["action_low_mask"] = []
        # pred["object"] = []
        
        action_low = []
        action_low_mask = []
        object = []
        pred = (action_low, action_low_mask, object)
        
        prompt = None
        interact_step = False
        available_position = None
        
        # get an observation and do an agent step
        rgb, bgr, depth = eval_util.get_rgbd(env, add_depth_noise=self.args.add_depth_noise)
        # obj_classes = [gen_util.remove_space(x) for x in self.args.obj_list]
        obs, sem_seg_image = gen_util.preprocess_obs(rgb, bgr, depth, self.args, self.seg_model, env, self.obj_classes)
        
        action_count = 0
        num_fails = 0
        
        '''     
        # save seg image
        if self.args.debug:
            seg_output_dir = os.path.join(self.args.output_dir, self.trial_dir, 'seg_images')
            if not os.path.exists(seg_output_dir):
                os.makedirs(seg_output_dir)
            seg_save_path = os.path.join(seg_output_dir, '{:09d}.png'.format(t_agent))
            cv2.imwrite(seg_save_path, sem_seg_image)
        
        if self.args.debug:
            rgb_output_dir = os.path.join(self.args.output_dir, self.trial_dir, 'rgb_images')
            if not os.path.exists(rgb_output_dir):
                os.makedirs(rgb_output_dir)
            rgb_save_path = os.path.join(rgb_output_dir, '{:09d}.png'.format(t_agent))
            cv2.imwrite(rgb_save_path, bgr)
        
        if self.args.debug:
            depth_output_dir = os.path.join(self.args.output_dir, self.trial_dir, 'depth_images')
            if not os.path.exists(depth_output_dir):
                os.makedirs(depth_output_dir)
            depth = (depth * 255 / np.max(depth)).astype('uint8')
            depthImg = np.stack((depth, depth, depth), axis=2)
            depth_save_path = os.path.join(depth_output_dir, '{:09d}.png'.format(t_agent))
            cv2.imwrite(depth_save_path, depthImg)
        ''' 
        
        # update info & orientation
        if t_agent > 0:  # step=0 | do not need to 
            self.curr_location = gen_util.get_sim_location(env.last_event)
            dx, dy, self.do = pu.get_rel_pose_change(self.curr_location, self.last_sim_location)
            self.last_sim_location = self.curr_location          
            self.info['sensor_pose'] = [dx, dy, self.do]
            
            if self.prev_action == 'PickupObject':  # start holidng
                self.info['holding_state'] = 1
                holding_obj = env.last_interaction[0]  # space format
                self.info['holding_obj'] = self.args.obj_list.index(holding_obj) + 4
                if env.last_interaction[0] in ['Box', 'Laptop']:
                    print('is holding', env.last_interaction[0])
                    if env.last_interaction[0] in ['Laptop']:
                        self.info['holding_obj_with_hole'] = True
                    self.info['holding_box'] = True
                print('start holding', env.last_interaction[0])
            elif self.prev_action == 'PutObject':  # end holding
                self.info['holding_state'] = 3
                self.info['holding_obj'] = -1
                self.info['holding_box'] = False
                self.info['holding_obj_with_hole'] = False
                print('end holding')
                # if not pickup_one:
                #     pickup_one = True
                #     pickup_obj = gt_subgoals[subgoal_idx-1][1][0]
                #     pickup_obj = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[pickup_obj])
                #     recep_obj = gt_subgoals[subgoal_idx-1][1][1]
                #     recep_obj = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[recep_obj])
            elif self.info['holding_state'] == 1 or self.info['holding_state'] == 2:  # is holding
                self.info['holding_state'] = 2
                if env.last_interaction[0] in ['Box', 'Laptop']:
                    print('holding', env.last_interaction[0])
                    if env.last_interaction[0] in ['Laptop']:
                        self.info['holding_obj_with_hole'] = True
                    self.info['holding_box'] = True
            else:
                self.info['holding_state'] = 0  # not holding
                self.info['holding_obj'] = -1
                self.info['holding_box'] = False
                self.info['holding_obj_with_hole'] = False
            
            self.orientation = eval_util.get_orientation(self.last_orientation, self.do)
            print('orientation: ', self.orientation)
            print('info', self.info)
            self.last_orientation = self.orientation
        
        local_map = self.sem_map_module.update_local_map_and_process(obs, self.info)  # 出生local_map
        chessboard = self.sem_map_module.local_map_to_chessboard()
        obj_dict = self.sem_map_module.chessboard_info(self.args.obj_list)
        self.obj_dict = obj_dict
        
        last_pos = obj_dict["Agent"][0]
        print('last_pos', last_pos)
        
        # save sem map
        # if self.args.debug:
        #     map_save_dir = os.path.join(self.args.output_dir, self.trial_dir, 'sem_map')
        #     if not os.path.exists(map_save_dir):
        #         os.makedirs(map_save_dir)
        #     map_save_path = os.path.join(map_save_dir, '{:09d}.png'.format(t_agent))
        #     self.sem_map_module.visualize_w_small(map_save_path)
        
        
        # looking around at first
        if t_agent < 4:     
            action_low =["RotateRight_90"]
            action_low_mask = [None]
            object = [None]
                           
            self.visualize_chessboard(t_agent, last_pos, last_pos)

        else:
            '''Do agent steps'''
            # FIXME: is_arrived()
            if isinstance(self.curr_subgoal, str):
                '''Navigate'''
                # large object and small object could be same!
                # FIXME:
                # 6: 0/5     5: 2/5    4: 2/5   4+2: 2/5
                
                if self.small_target == "Microwave":
                    is_arrived_distance = 3 * self.args.chess_extend_ratio
                elif self.small_target == "Fridge":
                    is_arrived_distance = 3 * self.args.chess_extend_ratio
                elif self.small_target == "Desk Lamp":
                    is_arrived_distance = 6 * self.args.chess_extend_ratio
                elif self.small_target == "Floor Lamp":
                    is_arrived_distance = 6 * self.args.chess_extend_ratio
                elif self.small_target == "Sink Basin":
                    is_arrived_distance = 3 * self.args.chess_extend_ratio
                elif self.small_target == "Pan":
                    is_arrived_distance = 3 * self.args.chess_extend_ratio
                else:
                    is_arrived_distance = 5 * self.args.chess_extend_ratio
                    
                print(f"is_arrived_distance: {is_arrived_distance} small_target: {self.small_target}")
                if eval_util.is_arrived(last_pos, obj_dict, self.small_target, is_arrived_distance):  # camel space format
                    # facing to small target
                    rotate_actions = agent_obj_rel.rotate_to_target(last_pos, self.orientation, obj_dict[self.small_target])
                    if len(rotate_actions) > 0:
                        for action in rotate_actions:
                            mask = None
                            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)
                            if not t_success:
                                print('action failed.', err)
                                num_fails += 1
                            action_count += 1
                    
                    # Update Orientation
                    self.curr_location = gen_util.get_sim_location(env.last_event)
                    dx, dy, self.do = pu.get_rel_pose_change(self.curr_location, self.last_sim_location)
                    # self.last_sim_location = self.curr_location          
                    # self.info['sensor_pose'] = [dx, dy, self.do]
                    
                    self.orientation = eval_util.get_orientation(self.orientation, self.do)
                    print('orientation: ', self.orientation)
                    print('info', self.info)
                    
                    # local_map = self.sem_map_module.update_local_map_and_process(obs, self.info)
                         
                    # FIXME: basin       
                    target_object = self.small_target.replace(' ', '')
                    if 'Basin' in target_object:
                        target_object = target_object.replace('Basin', '')
                    print('target_object', target_object)
                    
                    # visible = True
                    visible = False
                    
                    # if self.small_target == 'Floor Lamp':
                    #     mask = None
                    #     t_success, _, _, err, _ = env.va_interact('LookUp_15', interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)
                    #     if not t_success:
                    #         print('action failed.', err)
                    #         num_fails += 1
                    #     action_count += 1
                        
                    #     t_success, _, _, err, _ = env.va_interact('LookUp_15', interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)
                    #     if not t_success:
                    #         print('action failed.', err)
                    #         num_fails += 1
                    #     action_count += 1
                        
                    
                    # Using GT mask & visible
                    # if self.args.renderInstanceSegmentation:
                    #     for obj in env.last_event.metadata["objects"]:
                    #         if obj["objectId"] in env.last_event.instance_masks.keys() and obj["objectType"] == target_object:
                    #             print(obj["objectId"])
                    #             if obj["visible"]: 
                    #                 visible = True
                    #                 break
                    # else:
                    if self.args.seg == 'DINO':
                        rcnn_pred = self.seg_model.segmentation(env.last_event.cv2img.astype(np.uint8), obj_classes=self.obj_classes)
                        for idx, pred in enumerate(rcnn_pred.class_id):
                            if pred == self.args.obj_list.index(self.small_target):
                                visible = True
                                break
                    else:  # maskrcnn
                        # preds = [pred = types.SimpleNamespace(label=label, box=box, score=score, mask=mask)]
                        rcnn_pred = self.seg_model.predict_objects(Image.fromarray(env.last_event.frame), verbose=self.args.debug, confidence_threshold=0.7)
                        for pred in rcnn_pred:
                            if pred.label == target_object:
                                visible = True
                                break
                    
                    if visible:                        
                        print('goal object {} founded!'.format(self.small_target))
                        self.navigate_finished = True
                        self.searching = False  
                    else:
                        # if self.small_target == 'Floor Lamp':
                        #     mask = None
                        #     t_success, _, _, err, _ = env.va_interact('LookDown_15', interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)
                        #     if not t_success:
                        #         print('action failed.', err)
                        #         num_fails += 1
                        #     action_count += 1
                            
                        #     t_success, _, _, err, _ = env.va_interact('LookDown_15', interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)
                        #     if not t_success:
                        #         print('action failed.', err)
                        #         num_fails += 1
                        #     action_count += 1
                        
                        available_position = eval_util.find_available_position_w_margin(obj_dict, self.sem_map_module.available_pos_boolean, self.sem_map_module.collision)
                        # print('available_position', available_position)
                        # prompt = eval_util.prompt_v9_new(obj_dict, self.curr_subgoal, available_position, traj=self.history_traj)
                        # prompt = eval_util.prompt_v13(obj_dict, self.curr_subgoal, available_position, traj=self.history_traj)
                        prompt = eval_util.prompt_zeroshot_mistral(obj_dict, self.curr_subgoal, available_position, traj=self.history_traj)
                        
                        print(prompt)

                # if find the small target object, change to search small object strategy
                # only for those who has small target object
                elif self.small_target in obj_dict.keys() and not self.searching:
                    print('small goal object {} detected!'.format(self.small_target))
                    self.curr_subgoal = self.curr_subgoal.replace(self.large_target.lower().replace(' ', ''), self.small_target.lower().replace(' ', ''))
                    print('changing navigate subgoal to', self.curr_subgoal)
                    self.searching = True   
                
                # only be executed when small target object is different from large object!!!
                # if arrived large target object, facing to large target to find small object
                elif eval_util.is_arrived(last_pos, obj_dict, self.large_target) and not self.searching:
                    if self.large_target in obj_dict.keys():
                        rotate_actions = agent_obj_rel.rotate_to_target(last_pos, self.orientation, obj_dict[self.large_target])
                        if len(rotate_actions) > 0:
                            action_low = rotate_actions
                            action_low_mask = [None for _ in rotate_actions]
                            object = [None for _ in rotate_actions]
                        else:
                            action_low = []
                            action_low_mask = []
                            object = []
                        
                    print('goal object {} founded!'.format(self.large_target))
                    self.curr_subgoal = self.curr_subgoal.replace(self.large_target.lower().replace(' ', ''), self.small_target.lower().replace(' ', ''))
                    print('changing navigate subgoal to', self.curr_subgoal)
                    self.searching = True  
                     
                # navigate to large target object
                else:
                    available_position = eval_util.find_available_position_w_margin(obj_dict, self.sem_map_module.available_pos_boolean, self.sem_map_module.collision)
                    # print('available_position', available_position)
                    # prompt = eval_util.prompt_v9_new(obj_dict, self.curr_subgoal, available_position, traj=self.history_traj)
                    # prompt = eval_util.prompt_v13(obj_dict, self.curr_subgoal, available_position, traj=self.history_traj)
                    
                    prompt = eval_util.prompt_zeroshot_mistral(obj_dict, self.curr_subgoal, available_position, traj=self.history_traj)
                    
                    print(prompt)
            
            else:
                action_low, action_low_mask, object = self.interaction2act(env)
                # TODO: 强制面向小物体
                interact_step = True 
        
        pred = (action_low, action_low_mask, object)
        print('pred', pred)
        return last_pos, pred, prompt, interact_step, available_position, self.orientation, action_count, num_fails
        
    def interaction2act(self, env):
        # pred={}
        # pred["action_low"] = []
        # pred["action_low_mask"] = []
        # pred["object"] = []
        
        action_low = []
        action_low_mask = []
        object = []
        
        # 当 goto 交互物体被省略时，此时代表应当到达交互物体附近，因此强制通过旋转使得面向小物体
        # target_object = None
        # if len(objects) > 0:
        #     target_object = objects[-1].replace(' ', '')
        # print('target_object', target_object)
        # visible = False
        # while not visible and action_count < 3 and target_object != None:
        #     for obj in env.last_event.metadata["objects"]:
        #         if obj["objectId"] in env.last_event.instance_masks.keys() and obj["objectType"] == target_object:
        #             print(obj["objectId"])
        #             if obj["visible"]: 
        #                 visible = True
        #                 break
                    
        #     print('is visible?', visible)
        #     if not visible:
        #         mask = None
        #         action = 'RotateRight'
        #         _ = env.va_interact(action, interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)
        #         action_count += 1
        
        subgoal_verb, objects = self.curr_subgoal  # [small_object, receptacle_object] lower format, alarmclock
        objects = [gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[obj]) for obj in objects]  # camel space format, Alarm Clock
        
        # if self.sliced and self.subgoal_idx > self.slice_subgoal_idx and len(objects) > 0:
        #     objects[0] += ' Sliced'
        
        if subgoal_verb == "CleanObject":
            action_low = ["PutObject","ToggleObjectOn","ToggleObjectOff","PickupObject"]
            object = ["Sink Basin","Faucet","Faucet",objects[0]]
            
        elif subgoal_verb == "CoolObject":
            action_low = ["OpenObject","PutObject","CloseObject","OpenObject","PickupObject","CloseObject"]
            object = ["Fridge","Fridge","Fridge","Fridge",objects[0],"Fridge"]
            
        elif subgoal_verb == "HeatObject":
            action_low = ["OpenObject","PutObject","CloseObject","ToggleObjectOn","ToggleObjectOff","OpenObject","PickupObject","CloseObject"]
            object = ["Microwave","Microwave","Microwave","Microwave","Microwave","Microwave",objects[0],"Microwave"]
            
        elif subgoal_verb == "NoOp":
            action_low = ["<<stop>>"]
            object = [None]
            
        elif subgoal_verb == "ToggleObject":
            if objects[-1] == 'Floor Lamp':
                action_low = ["LookUp_15", "LookUp_15", "ToggleObjectOn"]
                object = [None, None, "Floor Lamp"]
            else:
                action_low = ["ToggleObjectOn"]
                object = [objects[0]]
        elif subgoal_verb == "PutObject":
            if objects[-1] in self.openable_obj:
                action_low = ["LookDown_15", "LookDown_15", "OpenObject", "PutObject", "CloseObject", "LookUp_15", "LookUp_15"]
                object = [None, None, objects[-1], objects[-1], objects[-1], None, None]
                # action_low = ["OpenObject", "PutObject"]
                # object = ["Cabinet", "Cabinet"]
            elif objects[-1] == 'Fridge':
                action_low = ["OpenObject", "PutObject"]
                object = ['Fridge', 'Fridge']
            elif objects[-1] == 'Microwave':
                action_low = ["OpenObject", "PutObject"]
                object = ['Microwave', 'Microwave']
            # elif objects[-1] == 'Garbage Can':
            #     action_low = ["LookDown_15", "LookDown_15", "PutObject", "LookUp_15", "LookUp_15"]
            #     object = [None, None, objects[-1], None, None]
            else:
                action_low = ["PutObject"]
                object = [objects[-1]]
        else:
            action_low = [subgoal_verb]
            object = [objects[0]]
        
        # if object=None, 则返回None
        # pred["action_low_mask"] = [None for obj in pred["object"]]
        for obj in object:
            if obj == None:
                action_low_mask += [None]
            else:
                # action_low_mask += [eval_util.extract_rcnn_pred(obj, self.args.obj_list.index(obj), self.seg_model, env, self.args.debug, self.args.renderInstanceSegmentation)]
                action_low_mask += [None]
        self.interact_finished = True
        return action_low, action_low_mask, object
    
    
    def move_to_target_v2(self, next_pos):
        forward_action_list = eval_util.move_to_target(next_pos, self.obj_dict, self.small_target, threshold_dis=1)
        backward_action_list = eval_util.move_to_target(next_pos, self.obj_dict, self.small_target, threshold_dis=6)
        return forward_action_list, backward_action_list
 
    
    def facing(self, env, next_pos):
        action_count = 0
        num_fails = 0
        # if at searching state and facing small target object
        if self.small_target in self.obj_dict.keys():
            rotate_actions = agent_obj_rel.rotate_to_target(next_pos, self.orientation, self.obj_dict[self.small_target])
            if len(rotate_actions) > 0:
                for action in rotate_actions:
                    mask = None
                    t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)
                    if not t_success:
                        print('action failed.', err)
                        num_fails += 1
                    action_count += 1
        # self.interact_finished = False
        return action_count, num_fails
        
        
    def move_closer(self, env, next_pos):  
        action_count = 0
        num_fails = 0
        # move closer to the small target object
        if self.curr_subgoal[0] not in ["CleanObject", "CoolObject", "HeatObject"]:
            return action_count, num_fails
        if self.small_target in self.obj_dict.keys():  
            move_actions = eval_util.move_to_target(next_pos, self.obj_dict, self.small_target)
            if len(move_actions) > 0:
                for action in move_actions:
                    mask = None
                    t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)
                    if not t_success:
                        print('action failed.', err)
                        num_fails += 1
                    action_count += 1
        # self.interact_finished = False
        return action_count, num_fails
        

    def facing_copy(self, env, next_pos):
        action_count = 0
        num_fails = 0
        # if at searching state and facing small target object
        if self.small_target in self.obj_dict.keys():
            rotate_actions = agent_obj_rel.rotate_to_target(next_pos, self.orientation, self.obj_dict[self.small_target])
            if len(rotate_actions) > 0:
                for action in rotate_actions:
                    mask = None
                    t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)
                    if not t_success:
                        print('action failed.', err)
                        num_fails += 1
                    action_count += 1
        self.interact_finished = False
        return action_count, num_fails
        
        
    def move_closer_copy(self, env, next_pos):  
        action_count = 0
        num_fails = 0
        # move closer to the small target object     
        if not eval_util.is_arrived(next_pos, self.obj_dict, self.small_target, threshold_dis=5):
            action = 'MoveAhead'
            mask = None
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)
            if not t_success:
                print('action failed.', err)
                num_fails += 1
            action_count += 1
        self.interact_finished = False
        return action_count, num_fails
