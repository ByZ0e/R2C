import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from datetime import datetime
import cv2
import sys

from alfred.utils import eval_util, gen_util
from alfred import constants
import envs.utils.pose as pu
from keyboard_fb.chessboard_vis.infer_vis import ChessVis
from keyboard_fb import agent_obj_rel

# def compute_metrics(subgoal_success, subgoal_idx, reward, task, t_agent):
#     '''
#     compute metrics for subgoal evaluation
#     '''
#     pl = float(t_agent) + 1 # +1 for last action
#     expert_pl = len([ll for ll in task['plan']['low_actions']
#                      if ll['high_idx'] == subgoal_idx])
#     s_spl = (1 if subgoal_success else 0) * min(
#         1., expert_pl / (pl + sys.float_info.epsilon))
#     plw_s_spl = s_spl * expert_pl
#     metrics = {'success_spl': float(s_spl),
#                'subgoal_path_len_weighted_success_spl': float(plw_s_spl),
#                'subgoal_path_len_weight': float(expert_pl),
#                'reward': float(reward),
#                'success': subgoal_success}
#     return metrics


def compute_metrics(success, reward, task, t, pcs):
    '''
    compute metrics for task evaluation
    '''
    # goal_conditions
    goal_condition_success_rate = pcs[0] / float(pcs[1])
    # SPL
    path_len_weight = len(task['plan']['low_actions'])
    s_spl = (1 if success else 0) * min(1., path_len_weight / float(t))
    pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))
    # path length weighted SPL
    plw_s_spl = s_spl * path_len_weight
    plw_pc_spl = pc_spl * path_len_weight
    metrics = {'completed_goal_conditions': int(pcs[0]),
               'total_goal_conditions': int(pcs[1]),
               'goal_condition_success': float(goal_condition_success_rate),
               'success_spl': float(s_spl),
               'path_len_weighted_success_spl': float(plw_s_spl),
               'goal_condition_spl': float(pc_spl),
               'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
               'path_len_weight': int(path_len_weight),
               'reward': float(reward),
               'success': success}
    return metrics


def evaluate_subgoal(env, model, tokenizer, trial_dir, json_dict, args, seg_model, sem_map_module, output_dir = 'evals'):
    output_dir = args.output_dir
    # setup scene
    # scene = json_dict["scene"]["floor_plan"]
    task_type = eval_util.setup_scene(env, json_dict, reward_type='dense')
    
    # TODO: subgoal
    # task_info = eval_util.read_task_data(traj_data, subgoal_idx)  # return task_dict['subgoal_idx'],['subgoal_action']
    gt_subgoals = gen_util.get_subgoals(json_dict)  # ["GotoLocation", ["obj", "r_obj"]]
    # gt_subgoals = [["GotoLocation", ["fridge"]], ["OpenObject", ["fridge"]]]
    if gt_subgoals[-1][0] != "NoOp":
        gt_subgoals.append(["NoOp", []])
    
    # gt_subgoals.insert(2, ["GotoLocation", ["sinkbasin"]])
   
    # init agent state
    # holding_obj (int): object index in obs
    info = {'sensor_pose': [0., 0., 0.], 'holding_state': 0, 'holding_obj': -1, 'holding_box': False}
    init_action = json_dict['scene']['init_action']
    orientation, do = 0, 0
    history_traj = []
    
    # init sem map module
    # FIXME: 
    # sem_map_module.sem_map_builder.agent_height = (0.675 + init_action["y"]) * 100.
    # FILM 使用 155.
    sem_map_module.sem_map_builder.agent_height = 155.
    sem_map_module.reset_view_angle(init_action["horizon"])
    print('init cameraHorizon: ', init_action["horizon"])
    # sem_map_module.reset_view_angle(0)
    last_sim_location = gen_util.get_location(init_action)
    
    # init visualization
    chess_vis = ChessVis(x_down=True, chess_size=int(args.map_size_cm/args.map_resolution/args.map_resolution))
    chess_vis.reset_history_traj()
    
    # init logger
    if not os.path.exists(os.path.join(output_dir, trial_dir)):
        os.makedirs(os.path.join(output_dir, trial_dir))
    logger = eval_util.OutputRedirector(os.path.join(output_dir, trial_dir, 'log.txt'))
    logger.start()
    
    prev_action, episode_end, subgoal_success, action_success, searching = None, False, False, False, False
    # remove_pos = []
    # pickup_one, pickup_obj, recep_obj = False, None, None
    t_agent, num_fails, reward, subgoal_idx = 0, 0, 0, -1   
    
    print('gt_subgoals:', gt_subgoals)
    while t_agent < args.max_steps:
        print('-----------------------------------------')
        print('step: ', t_agent)
    
        # get an observation and do an agent step
        rgb, bgr, depth = eval_util.get_rgbd(env)
        obs, sem_seg_image = gen_util.preprocess_obs(rgb, bgr, depth, args, seg_model, env)

        # save seg image
        if args.debug:
            seg_output_dir = os.path.join(output_dir, trial_dir, 'seg_images')
            if not os.path.exists(seg_output_dir):
                os.makedirs(seg_output_dir)
            seg_save_path = os.path.join(seg_output_dir, '{:09d}.png'.format(t_agent))
            cv2.imwrite(seg_save_path, sem_seg_image)
    
        # update info & orientation
        if t_agent > 0:  # step=0 | do not need to 
            curr_location = gen_util.get_sim_location(env.last_event)
            dx, dy, do = pu.get_rel_pose_change(curr_location, last_sim_location)
            last_sim_location = curr_location          
            info['sensor_pose'] = [dx, dy, do]
            if prev_action == 'PickupObject':  # start holidng
                info['holding_state'] = 1
                holding_obj = env.last_interaction[0]  # space format
                info['holding_obj'] = args.obj_list.index(holding_obj) + 4
                if env.last_interaction[0] == 'Box':
                    info['holding_box'] = True
                print('start holding', env.last_interaction[0])
            elif prev_action == 'PutObject':  # end holding
                info['holding_state'] = 3
                info['holding_obj'] = -1
                info['holding_box'] = False
                print('end holding')
                # if not pickup_one:
                #     pickup_one = True
                #     pickup_obj = gt_subgoals[subgoal_idx-1][1][0]
                #     pickup_obj = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[pickup_obj])
                #     recep_obj = gt_subgoals[subgoal_idx-1][1][1]
                #     recep_obj = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[recep_obj])
            elif info['holding_state'] == 1 or info['holding_state'] == 2:  # is holding
                info['holding_state'] = 2
                if env.last_interaction[0] == 'Box':
                    info['holding_box'] = True
            else:
                info['holding_state'] = 0  # not holding
                info['holding_obj'] = -1
                info['holding_box'] = False
            
            orientation = eval_util.get_orientation(orientation, do)
            print('orientation: ', orientation)
            print('info', info)

        # update sem map & chessboard
        sem_map_module.reset_view_angle(env.last_event.metadata['agent']['cameraHorizon'])
        print('cameraHorizon: ', env.last_event.metadata['agent']['cameraHorizon'])
        local_map = sem_map_module.update_local_map_and_process(obs, info)  # 出生local_map
        chessboard = sem_map_module.local_map_to_chessboard()
        obj_dict = sem_map_module.chessboard_info(args.obj_list)
        # print(obj_dict)
        
        last_pos = obj_dict["Agent"][0]
        print('last_pos', last_pos)
        
        # save sem map
        if args.debug:
            map_save_dir = os.path.join(output_dir, trial_dir, 'sem_map')
            if not os.path.exists(map_save_dir):
                os.makedirs(map_save_dir)
            map_save_path = os.path.join(map_save_dir, '{:09d}.png'.format(t_agent))
            sem_map_module.visualize_w_small(map_save_path)

        # looking around at first
        if t_agent < 4:     
            action = 'RotateRight'
            mask = None
            _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            
            # visualize chessboard
            if args.debug:
                chess_output_dir = os.path.join(output_dir, trial_dir, 'chess')
                chess_vis.infer_chessboard(t_agent, obj_dict, args.obj_list, last_pos, last_pos, orientation, subgoal=None, output_dir=chess_output_dir)
                chess_vis.add_history_traj(last_pos)
            
            t_agent += 1
            continue
        
           
        '''Set subgoal'''
        if subgoal_success or subgoal_idx == -1:
            if subgoal_idx == -1: subgoal_idx += 1
            subgoal = gt_subgoals[subgoal_idx]
            print(subgoal)
            
            if subgoal[0] == 'GotoLocation':
                # FIXME: what is the target object?
                # target_obj = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[subgoal[1][0]])
                
                large_target = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[subgoal[1][-1]])
                # small_target = 'none'
                if subgoal_idx + 1 < len(gt_subgoals) - 1 and prev_action != 'PickupObject':
                    small_target = gt_subgoals[subgoal_idx+1][1][-1]
                    small_target = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[small_target])
                else: 
                    small_target = large_target
                subgoal = eval_util.get_subgoal_str(subgoal)
                print('large_target', large_target)
                print('small_target', small_target)

            subgoal_success = False
        
        action_count = 0
        navigate_finished = False
        if not action_success: next_pos = last_pos
        
        # if pickup_one and task_type=='pick_two_obj_and_place':
        #     # gt_subgoals[subgoal_idx-1]: ('PutObject', ['keychain', 'sofa'])
        #     obj_dict, remove_pos = eval_util.delet_pickup_obj(obj_dict, pickup_obj, recep_obj, remove_pos)  # space format
        
        # if pickup_obj != None:
        #     if pickup_obj in obj_dict.keys():
        #         print(obj_dict[pickup_obj])
        #     else:
        #         print('{} has been removed'.format(pickup_obj))
        
        '''Do agent steps'''
        # FIXME: is_arrived()
        if isinstance(subgoal, str):
            '''Navigate'''
            # if arrived small target object, stop navigate
            # or if nav has no small target, small target is set to be larget target
            if eval_util.is_arrived(next_pos, obj_dict, small_target):
                # facing to small target
                rotate_actions = agent_obj_rel.rotate_to_target(next_pos, orientation, obj_dict[small_target])
                if len(rotate_actions) > 0:
                    for action in rotate_actions:
                        mask = None
                        _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                        t_agent += 1
                
                if not eval_util.is_arrived(next_pos, obj_dict, small_target, threshold_dis=5):
                    action = 'MoveAhead'
                    mask = None
                    _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                    t_agent += 1
                    
                print('goal object {} founded!'.format(small_target))
                navigate_finished = True
                searching = False
            
            # if arrived large target object, change to search small object strategy
            # only for those who has small target object
            # elif eval_util.is_arrived(next_pos, obj_dict, large_target) and small_target != large_target and not searching:
            elif small_target in obj_dict.keys() and small_target != large_target and not searching:
                # facing to large target
                if large_target in obj_dict.keys():
                    rotate_actions = agent_obj_rel.rotate_to_target(next_pos, orientation, obj_dict[large_target])
                    if len(rotate_actions) > 0:
                        for action in rotate_actions:
                            mask = None
                            _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                            t_agent += 1
                    
                # print('goal object {} founded!'.format(large_target))
                print('small goal object {} detected!'.format(small_target))
                # FIXME: lower format!
                subgoal = subgoal.replace(large_target.lower().replace(' ', ''), small_target.lower().replace(' ', ''))
                print('changing navigate subgoal to', subgoal)
                searching = True
                
                # prompt, available_position = eval_util.prompt_v9(obj_dict, subgoal, traj=history_traj)
                available_position = eval_util.find_available_position_w_margin(obj_dict, sem_map_module.available_pos_boolean, sem_map_module.collision)
                prompt = eval_util.prompt_v9_new(obj_dict, subgoal, available_position, traj=history_traj)
                print(prompt)
                
                episode_end, prev_action, next_pos, num_fails, _, _, action_count, action_success = eval_util.agent_nav_step(
                    model, tokenizer, prompt, available_position, orientation, last_pos, env, args, num_fails, seg_model)
                if action_success:
                    history_traj.insert(0, last_pos)
                else:
                    if t_agent > 4:
                        sem_map_module.collision.append(next_pos)
            
            else:  # navigate to large target object
                # prompt, available_position = eval_util.prompt_v9(obj_dict, subgoal, traj=history_traj)
                available_position = eval_util.find_available_position_w_margin(obj_dict, sem_map_module.available_pos_boolean, sem_map_module.collision)
                prompt = eval_util.prompt_v9_new(obj_dict, subgoal, available_position, traj=history_traj)
                print(prompt)
                
                episode_end, prev_action, next_pos, num_fails, _, _, action_count, action_success = eval_util.agent_nav_step(
                    model, tokenizer, prompt, available_position, orientation, last_pos, env, args, num_fails, seg_model)
                if action_success:
                    history_traj.insert(0, last_pos)
                else:
                    if t_agent > 4:
                        sem_map_module.collision.append(list(next_pos))
        
        else:
            '''Interact'''
            episode_end, prev_action, num_fails, _, _, action_count, action_success = eval_util.agent_step_auto(
                subgoal, env, args, num_fails, seg_model, output_dir, trial_dir, t_agent)
        
        # visualize chessboard
        if args.debug:
            chess_output_dir = os.path.join(output_dir, trial_dir, 'chess')
            chess_vis.infer_chessboard(t_agent, obj_dict, args.obj_list, last_pos, next_pos, orientation, output_dir=chess_output_dir)
            chess_vis.add_history_traj(last_pos)
        
        # get rewards
        reward += env.get_transition_reward()[0]  # task.transition_reward() update goal_idx, finished
        print('task goal_idx: ', env.task.goal_idx)
        t_agent += action_count
        
        # update subgoal
        if isinstance(subgoal, str) and navigate_finished:
            subgoal_idx += 1
            # subgoal_success = (env.get_subgoal_idx() == subgoal_idx)  # env.get_subgoal_idx() return env.finished
            subgoal_success = True
        
        # FIXME: what if interaction action failed?
        elif not isinstance(subgoal, str):
            subgoal_idx += 1
            # subgoal_success = (env.get_subgoal_idx() == subgoal_idx)  # env.get_subgoal_idx() return env.finished
            subgoal_success = True
        
        # break if stop is predicted or args.max_fails is reached
        if episode_end:
            break

    
    # compute metrics and dump a video
    success = env.get_goal_satisfied()
    metrics = compute_metrics(success, reward, json_dict, t_agent, env.get_goal_conditions_met())
    print(metrics)
    # sys.stdout.flush()
    logger.stop()
    print('----------------------finished eval----------------------')
    return dict(**metrics)