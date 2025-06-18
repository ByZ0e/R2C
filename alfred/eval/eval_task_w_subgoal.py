import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from datetime import datetime
import cv2

from alfred.utils import eval_util, gen_util
import envs.utils.pose as pu
from keyboard_fb.chessboard_vis.infer_vis import ChessVis


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

def evaluate_task(env, model, tokenizer, trial_dir, json_dict, args, seg_model, sem_map_module, output_dir = 'evals'):
    output_dir = args.output_dir
    # setup scene
    # scene = json_dict["scene"]["floor_plan"]
    eval_util.setup_scene(env, json_dict, reward_type='dense')
    
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
    
    prev_action = None
    t, num_fails, reward = 0, 0, 0

    if not os.path.exists(os.path.join(output_dir, trial_dir)):
        os.makedirs(os.path.join(output_dir, trial_dir))
    
    # sys.stdout = Logger(filename=os.path.join(output_dir, trial_dir, 'log.txt'), stream=sys.stdout)
    logger = eval_util.OutputRedirector(os.path.join(output_dir, trial_dir, 'log.txt'))
    logger.start()

    '''Look down first'''
    # action = 'LookDown'
    # mask = None
    # _ = env.va_interact(
    #             action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
    # sem_map_module.reset_view_angle(init_action["horizon"]+15)
    # t = 1
    
    subgoal = "(" + json_dict["plan"]["high_pddl"][0]["discrete_action"]["action"] + ", [" + json_dict["plan"]["high_pddl"][0]["discrete_action"]["args"][0] + "])"
    # subgoal = "(GotoLocation, [toilet])"
    print(subgoal)
    
    while t < args.max_steps:
        print('step: ', t)
        # get an observation and do an agent step
        rgb, bgr, depth = eval_util.get_rgbd(env)
        obs, sem_seg_image = gen_util.preprocess_obs(rgb, bgr, depth, args, seg_model, env)

        # save seg image
        if args.debug:
            seg_output_dir = os.path.join(output_dir, trial_dir, 'seg_images')
            if not os.path.exists(seg_output_dir):
                os.makedirs(seg_output_dir)
            seg_save_path = os.path.join(seg_output_dir, '{:09d}.png'.format(t))
            cv2.imwrite(seg_save_path, sem_seg_image)

        if t > 0:
            curr_location = gen_util.get_sim_location(env.last_event)
            dx, dy, do = pu.get_rel_pose_change(curr_location, last_sim_location)
            last_sim_location = curr_location          
            info['sensor_pose'] = [dx, dy, do]
            if prev_action == 'PickupObject':  # start holidng
                info['holding_state'] = 1
                info['holding_obj'] = args.obj_list.index(env.last_interaction[0])
                if env.last_interaction[0] == 'Box':
                    info['holding_box'] = True
                print('start holding', env.last_interaction[0])
            elif prev_action == 'PutObject':  # end holding
                info['holding_state'] = 3
                info['holding_obj'] = -1
                info['holding_box'] = False
                print('end holding')
            elif info['holding_state'] == 1:  # is holding
                info['holding_state'] = 2
                if env.last_interaction[0] == 'Box':
                    info['holding_box'] = True
            else:
                info['holding_state'] = 0  # not holding
                info['holding_obj'] = -1

        local_map = sem_map_module.update_local_map_and_process(obs, info)  # 出生local_map
        chessboard = sem_map_module.local_map_to_chessboard()
        obj_dict = sem_map_module.chessboard_info(args.obj_list)
        # print(obj_dict)
        

        if t < 4:
            orientation = eval_util.get_orientation(orientation, do)
            last_pos = obj_dict["Agent"][0]
            # print('original orientation: ', orientation)
            print('last_pos', last_pos)
            
            action = 'RotateRight'
            mask = None
            _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            
            # visualize chessboard
            if args.debug:
                chess_output_dir = os.path.join(output_dir, trial_dir, 'chess')
                chess_vis.infer_chessboard(t, obj_dict, args.obj_list, last_pos, last_pos, orientation, subgoal, output_dir=chess_output_dir)
                chess_vis.add_history_traj(last_pos)

            t += 1
            continue

        # TODO: generate prompt

        obj_info = eval_util.obj_descriptor(obj_dict)
        task = json_dict["turk_annotations"]["anns"][0]["task_desc"]
        # subgoal = "(GotoLocation, [sidetable])"
        # subgoal = "(" + json_dict["plan"]["high_pddl"][0]["discrete_action"]["action"] + ", [" + json_dict["plan"]["high_pddl"][0]["discrete_action"]["args"][0] + "])"
        # print(subgoal)
        position = str(obj_dict["Agent"][0]).replace("(", "[").replace(")", "]")
        
        # prompt = eval_util.prompt_templete_v2.format(obj_info, task, subgoal, position)
        prompt = eval_util.prompt_v9(obj_dict, subgoal, traj=history_traj)
        # FIXME: PRINT PROMPT
        print(prompt)

        orientation = eval_util.get_orientation(orientation, do)
        last_pos = obj_dict["Agent"][0]
        print('original orientation: ', orientation)
        print('last_pos', last_pos)

        action_count = 0
        episode_end, prev_action, next_pos, num_fails, _, _, action_count, action_success = eval_util.agent_step(
            model, tokenizer, prompt, orientation, last_pos, env, args, num_fails, seg_model)
        if action_success:
            history_traj.insert(0, last_pos)
        
        # visualize chessboard
        if args.debug:
            chess_output_dir = os.path.join(output_dir, trial_dir, 'chess')
            chess_vis.infer_chessboard(t, obj_dict, args.obj_list, last_pos, next_pos[0], orientation, subgoal, output_dir=chess_output_dir)
            chess_vis.add_history_traj(last_pos)
        
        # get rewards
        reward += env.get_transition_reward()[0]
        t += action_count
        # break if stop is predicted or args.max_fails is reached
        if episode_end:
            break

    # compute metrics and dump a video
    success = env.get_goal_satisfied()
    metrics = compute_metrics(success, reward, json_dict, t, env.get_goal_conditions_met())
    print(metrics)
    # sys.stdout.flush()
    logger.stop()
    print('----------------------finished eval----------------------')
    return dict(**metrics)

    