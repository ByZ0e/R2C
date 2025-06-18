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
    eval_util.setup_scene(env, json_dict, reward_type='dense')
    
    # FIXME: 
    # task_info = eval_util.read_task_data(traj_data, subgoal_idx)  # return task_dict['subgoal_idx'],['subgoal_action']
    gt_subgoals = gen_util.get_subgoals(json_dict)  # ["GotoLocation", ["obj", "r_obj"]]
    # gt_subgoals = [["GotoLocation", ["fridge"]], ["OpenObject", ["fridge"]]]
    if gt_subgoals[-1][0] != "NoOp":
        gt_subgoals.append(["NoOp", []])
   
    # init agent state
    info = {'sensor_pose': [0., 0., 0.], 'holding_state': 0, 'holding_obj': -1, 'holding_box': False}
    init_action = json_dict['scene']['init_action']
    orientation, do = 0, 0
    history_traj = []
    
    # init sem map module
    sem_map_module.sem_map_builder.agent_height = 155.
    sem_map_module.reset_view_angle(init_action["horizon"])
    last_sim_location = gen_util.get_location(init_action)
    
    # init visualization
    chess_vis = ChessVis(x_down=True, chess_size=int(args.map_size_cm/args.map_resolution/args.map_resolution))
    chess_vis.reset_history_traj()
    
    if not os.path.exists(os.path.join(output_dir, trial_dir)):
        os.makedirs(os.path.join(output_dir, trial_dir))
    logger = eval_util.OutputRedirector(os.path.join(output_dir, trial_dir, 'log.txt'))
    logger.start()
    
    prev_action, episode_end, subgoal_success, action_success = None, False, False, False
    t_agent, num_fails, reward, subgoal_idx = 0, 0, 0, 0    
    
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
    
        if t_agent > 0:  # step=0 | do not need to update info & orientation 
            curr_location = gen_util.get_sim_location(env.last_event)
            dx, dy, do = pu.get_rel_pose_change(curr_location, last_sim_location)
            last_sim_location = curr_location          
            info['sensor_pose'] = [dx, dy, do]
            if prev_action == 'PickupObject':  # start holidng
                info['holding_state'] = 1
                holding_obj = env.last_interaction[0]  # space format
                info['holding_obj'] = args.obj_list.index(holding_obj)
                if env.last_interaction[0] == 'Box':
                    info['holding_box'] = True
                print('start holding', env.last_interaction[0])
            elif prev_action == 'PutObject':  # end holding
                info['holding_state'] = 3
                info['holding_obj'] = -1
                info['holding_box'] = False
                print('end holding')
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

        local_map = sem_map_module.update_local_map_and_process(obs, info)  # 出生local_map
        chessboard = sem_map_module.local_map_to_chessboard()
        obj_dict = sem_map_module.chessboard_info(args.obj_list)
        # print(obj_dict)
        
        last_pos = obj_dict["Agent"][0]
        print('last_pos', last_pos)
        
        # set subgoal
        # env.task.goal_idx = subgoal_idx
        # env.task.finished = subgoal_idx - 1
        subgoal = gt_subgoals[subgoal_idx]
        print(subgoal)
        # FIXME: if subgoal_success? next subgoal: try again   
        if subgoal[0] == 'GotoLocation':
            large_target = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[subgoal[1][-1]])
            small_target = 'none'
            # FIXME: what is the target object?
            if subgoal_idx + 1 < len(gt_subgoals) - 1 and prev_action is not 'PickupObject':
                small_target = gt_subgoals[subgoal_idx+1][1][-1]
                small_target = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[small_target])
            # target_obj = gen_util.camel_to_space(constants.OBJECTS_LOWER_TO_UPPER[subgoal[1][0]])
            subgoal = eval_util.get_subgoal_str(subgoal)
            print('large_target', large_target)
            print('small_target', small_target)

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
        
        action_count = 0
        navigate_finished = False
        if not action_success: next_pos = last_pos
        # model excute actions
        # FIXME: is_arrived()
        if isinstance(subgoal, str):
            '''Navigate'''
            # if arrived small target object, stop navigate
            if eval_util.is_arrived(next_pos, obj_dict, small_target):
                rotate_actions = agent_obj_rel.rotate_to_target(next_pos, orientation, obj_dict[small_target])
                if len(rotate_actions) > 0:
                    for action in rotate_actions:
                        mask = None
                        _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                        curr_location = gen_util.get_sim_location(env.last_event)
                        dx, dy, do = pu.get_rel_pose_change(curr_location, last_sim_location)
                        orientation = eval_util.get_orientation(orientation, do)
                        local_map = sem_map_module.update_local_map_and_process(obs, info)
                        chessboard = sem_map_module.local_map_to_chessboard()
                        obj_dict = sem_map_module.chessboard_info(args.obj_list)
                        
                        # visualize chessboard
                        if args.debug:
                            chess_output_dir = os.path.join(output_dir, trial_dir, 'chess')
                            chess_vis.infer_chessboard(t_agent, obj_dict, args.obj_list, next_pos, next_pos, orientation, subgoal=None, output_dir=chess_output_dir)
                        
                        t_agent += 1
                    
                print('goal object {} founded!'.format(small_target))
                navigate_finished = True
            
             
            # if arrived large target object, change to search small object strategy
            # elif eval_util.is_arrived(next_pos, obj_dict, large_target):
            #     if small_target is not 'none':  
            #         # navigate to small target object
            #         subgoal = subgoal.replace(large_target, small_target)
                    
            #         # look down to search small object
            #         eval_util.look_down(env, args, sem_map_module)
            #         eval_util.look_down(env, args, sem_map_module)
                    
            #         # visualize chessboard
            #         if args.debug:
            #             chess_output_dir = os.path.join(output_dir, trial_dir, 'chess')
            #             chess_vis.infer_chessboard(t_agent, obj_dict, args.obj_list, last_pos, last_pos, orientation, subgoal=None, output_dir=chess_output_dir)
            #         print('-----------------------------------------')
            #         print('step: ', t_agent)
            #         print('Look Down twice')
            #         t_agent += 2
                
                # else:  # only need to navigate to large target object
                #     rotate_actions = agent_obj_rel.rotate_to_target(next_pos, orientation, obj_dict[small_target])
                #     if len(rotate_actions) > 0:
                #         for action in rotate_actions:
                #             mask = None
                #             _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                #             curr_location = gen_util.get_sim_location(env.last_event)
                #             dx, dy, do = pu.get_rel_pose_change(curr_location, last_sim_location)
                #             orientation = eval_util.get_orientation(orientation, do)
                #             local_map = sem_map_module.update_local_map_and_process(obs, info)
                #             chessboard = sem_map_module.local_map_to_chessboard()
                #             obj_dict = sem_map_module.chessboard_info(args.obj_list)
                            
                #             # visualize chessboard
                #             if args.debug:
                #                 chess_output_dir = os.path.join(output_dir, trial_dir, 'chess')
                #                 chess_vis.infer_chessboard(t_agent, obj_dict, args.obj_list, last_pos, last_pos, orientation, subgoal=None, output_dir=chess_output_dir)                      
                            
                #             t_agent += 1
                #     print('goal object {} founded!'.format(large_target))
                #     navigate_finished = True  
            
            else:  # navigate to large target object
                # TODO: generate prompt
                prompt, available_position = eval_util.prompt_v9(obj_dict, subgoal, traj=history_traj)
                print(prompt)
                
                # episode_end, prev_action, next_pos, num_fails, _, _, action_count, action_success = eval_util.agent_step(
                #     model, tokenizer, prompt, orientation, last_pos, env, args, num_fails, seg_model)
                episode_end, prev_action, next_pos, num_fails, _, _, action_count, action_success = eval_util.agent_nav_step(
                    model, tokenizer, prompt, available_position, orientation, last_pos, env, args, num_fails, seg_model)
                if action_success:
                    history_traj.insert(0, last_pos)
        
        else:
            '''Interact'''
            episode_end, prev_action, num_fails, _, _, action_count, action_success = eval_util.agent_step_auto(
                subgoal, env, args, num_fails, seg_model)
        
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
            subgoal_success = (env.get_subgoal_idx() == subgoal_idx)  # env.get_subgoal_idx() return env.finished
        
        # FIXME: what if interaction action failed?
        elif not isinstance(subgoal, str):
            subgoal_idx += 1
            subgoal_success = (env.get_subgoal_idx() == subgoal_idx)  # env.get_subgoal_idx() return env.finished
            # print('-----------------------------------------')
            # print('step: ', t_agent)
            # print('Look Up twice')
            # eval_util.look_up(env, args, sem_map_module)
            # eval_util.look_up(env, args, sem_map_module)
            # t_agent += 2
        
        # break if stop is predicted or args.max_fails is reached
        if episode_end or subgoal_success:
            break

    
    # compute metrics and dump a video
    success = env.get_goal_satisfied()
    metrics = compute_metrics(success, reward, json_dict, t_agent, env.get_goal_conditions_met())
    print(metrics)
    # sys.stdout.flush()
    logger.stop()
    print('----------------------finished eval----------------------')
    return dict(**metrics)