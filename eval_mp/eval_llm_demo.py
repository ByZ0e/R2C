import os
import sys
sys.path.append("..")
# os.environ['ALFRED_ROOT'] = '/home/lm2/projects/Real2Chess/alfred'
# sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
# # sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
# sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import json
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
import torch.multiprocessing as mp
import pprint
from importlib import import_module
import random
import time

from eval_task import EvalTask
from alfred.env.thor_env import ThorEnv
# from models.llm_demo import LLM as LLM_model
from alfred.utils import eval_util, gen_util
from alfred import constants
import envs.utils.pose as pu
from keyboard_fb.chessboard_vis.infer_vis import ChessVis
from keyboard_fb import agent_obj_rel
from vllm import LLM, SamplingParams
import re

PROCESS_NUM = 60
TRY_TIMES = 3

system_prompt = """You are a helpful, respectful and honest assistant. 
Always answer as helpfully as possible, while being safe.  
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information."""


def generate_prompt_with_llama2_format(instruction, plan=None, eos_token="</s>"):
  """
      <s>[INST] <<SYS>>
      {{ system_prompt }}
      <</SYS>>

      {{ user_message }} [/INST]
  """
  prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n{instruction}\n[/INST]"

  return prompt

def nextpos_string2pos(output):
    pos = output.find("# Next position")
    start_pos = output.find("[", pos)
    end_pos = output.find("]", pos)
    next_pos = output[start_pos : end_pos + 1]
    matches = re.findall(r'\[(\d+),\s*(\d+)\]', next_pos)
    next_position = [(int(match[0]), int(match[1])) for match in matches]
    return next_position

def LLM_inference(chessboard_info_queue, next_position_queue_list, num_alive_processes):
    # set gpu=2,3
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    
    # LLM initialization
    model_v9 = "vipl-vrc/real2chess-0421"
    version_1 = "491cfd39d0c3542c76b71f883eadd571c3ad2e8a"
    version_2 = "2689456d404a97cbd9a213c8a8dbb2e0490d1fe5"
    model_v13 = "vipl-vrc/real2chess-0505"
    llama_model = "vipl-vrc/real2chess-0521"
    # test zero-shot
    mistral_model = "mistralai/Mistral-7B-Instruct-v0.2"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1500)
    # llm = LLM(model=model_v9, revision=version_1, tensor_parallel_size=1)
    # llm = LLM(model=model_v13, tensor_parallel_size=1)
    # llm = LLM(model=llama_model, tensor_parallel_size=1)
    llm = LLM(model=mistral_model, tensor_parallel_size=1)
    
    
    # LLM inference
    # inference_num = 0
    while True:
        # wait for chess_info to be updated
        chessboard_info_dict = {}
        num_results_received = 0
        # print("LLM is waiting for chessboard_info")
        process_id_recieve_list = []
        while num_results_received < num_alive_processes.value:
            # print(num_alive_processes.value)
            process_id, chess_info = chessboard_info_queue.get()
            if chess_info == None:
                print(f"process {process_id} has finished")
                num_alive_processes.value -= 1
                continue
            chessboard_info_dict[process_id] = chess_info
            num_results_received += 1
            process_id_recieve_list.append(process_id)
        # print(f"LLM received all chessboard_info of step {inference_num}")
        
        if num_alive_processes.value == 0:
            break
        
        # print(f"process_id_recieve_list: {process_id_recieve_list}")
        
        # LLM inference
        prompts = []
        process_id_list = []
        for process_id, prompt in chessboard_info_dict.items():
            # prompt = generate_prompt_with_llama2_format(prompt)
            prompt = f"[INST] {prompt} [/INST]"
            prompts.append(prompt)
            process_id_list.append(process_id)
        outputs = llm.generate(prompts, sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]
        
        # parse the output
        # next_positions_list = []
        # for output in outputs:
        #     output = output.outputs[0].text.strip()
        #     # find # Next position
        #     pos = output.find("# Next position")
        #     start_pos = output.find("[", pos)
        #     end_pos = output.find("]", pos)
        #     next_pos = output[start_pos : end_pos + 1]
        #     matches = re.findall(r'\[(\d+),\s*(\d+)\]', next_pos)
        #     next_position = [(int(match[0]), int(match[1])) for match in matches]
        #     next_positions_list.append(next_position)
            
        # send next_position to each process
        # for idx, process_id in enumerate(process_id_list):
        #     pos = next_positions_list[idx]
        #     next_position_queue_list[process_id].put(pos)
        #     # print(f"Send {pos} to process {process_id}")
        
        
        for idx, process_id in enumerate(process_id_list):
            msg = outputs[idx]
            next_position_queue_list[process_id].put(msg)
            
        print("Positions have been sent to all processes")


class LLMEval(EvalTask):
    '''
    dump action-sequences for leaderboard eval
    '''
    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager

        # load splits
        with open(self.args.splits) as f:
            self.splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.splits.items()})

        #TODO: load seg model
        # M = import_module(self.args.model)
        # self.model = M
        
        # success and failure lists
        self.create_stats()

        # set random seed for shuffling
        random.seed(int(time.time()))
    
    @classmethod
    def run(cls, task_queue, args, lock, splits, successes, failures, results, process_idx, chessboard_info_queue, next_position_queue, gpu_idx):
        '''
        evaluation loop
        '''
        # set CUDA env
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
        
        # start THOR
        env = ThorEnv(x_display=args.x_display)

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                # load model
                M = import_module(args.model)
                model = M.R2C(args)
                r_idx = task['repeat_idx']
                # TODO: only test first anno
                # FIXME:
                if r_idx != 0:
                    continue
                # if 'pick_cool_then_place' not in task['task']:
                #     continue
                traj = model.start_eval(task, args.eval_split, r_idx)
                # print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, traj, args, lock, splits, successes, failures, results, process_idx, chessboard_info_queue, next_position_queue)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()
        chessboard_info_queue.put((process_idx, None))

    @classmethod
    def evaluate(cls, env, model, r_idx, json_dict, args, lock, splits, successes, failures, results, process_idx, chessboard_info_queue, next_position_queue):
        # reset model

        # setup scene
        eval_util.setup_scene(env, json_dict, reward_type='dense')
        
        # init task eval
        model.get_subgoals(r_idx, process_idx, chessboard_info_queue, next_position_queue)
        model.agent_state_init()
        model.init_chessboard()

        success = False
        
        prev_action, episode_end, subgoal_success, action_success = None, False, False, False
        t_agent, num_fails, reward, subgoal_idx = 0, 0, 0, -1
        
        
        next_pos = None
        loop_step = 0
        
        # if model.heat == True:
        if model.microwave == True:
            _ = env.va_interact('LookUp_15', interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
            # FIXME:
            # t_agent += 1
            print('detect heat task, look up first')
        
        # _ = env.va_interact('LookUp_15', interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
        # _ = env.va_interact('LookUp_15', interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
        # _ = env.va_interact('LookUp_15', interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
        
        while not episode_end:
            interact_tries = 0
            # break if max_steps reached
            if t_agent >= args.max_steps:
                print("Too long to solve this question")
                break
            
            print('-----------------------------------------')
            print('step: ', t_agent)
            
            # set subgoal
            if t_agent != 0:
                subgoal_success = model.update_subgoal_idx()
            if subgoal_success or loop_step == 0:
                model.set_subgoal(prev_action)
            
            # if not action_success: next_pos = last_pos
            if t_agent == 4:
                next_pos = last_pos
            
            # TODO:
            # 获取agent当前位置，棋盘格移动前置动作（开头自转、交互子任务动作、朝向前摇），prompt，是否执行交互子任务flag
            last_pos, (action_low, action_low_mask, object), \
            prompt, interact_step, available_position, orientation, action_count, fails_count \
            = model.step(env, t_agent, prev_action, next_pos)
            
            t_agent += action_count
            num_fails += fails_count
            
            # m_pred包含各种情况，可能是导航需要的前置动作，可能是交互动作
            loop_step += 1
            
            m_pred = {"action_low": action_low, "action_low_mask": action_low_mask, "object": object}
            print('m_pred', m_pred)
            
            if prompt != None:
                # TODO: prompt 压入chessboard_info
                chessboard_info = (process_idx, prompt)
                chessboard_info_queue.put(chessboard_info)
                print(f"Put info into chessboard_info_queue, t_agent: {t_agent}")
            
                # TODO: get next position
                available, tries = False, 0
                while not available and tries < TRY_TIMES:
                    tries += 1
                    valid = True
                    next_pos_output = next_position_queue.get()
                    print(next_pos_output)
                    next_pos = nextpos_string2pos(next_pos_output)
                    print(f"Get {next_pos} from next_position_queue, t_agent: {t_agent}")
                    
                    if len(next_pos) == 0:
                        valid = False
                    else: 
                        next_pos = next_pos[0]
                        
                    if list(next_pos) in available_position: 
                        available = True
                    else:
                        if tries < TRY_TIMES:
                            valid = False
                            
                    if valid == False:
                        chessboard_info_queue.put(chessboard_info)
                        print(f"Put info into chessboard_info_queue, retry: {tries}")
                        
                    
                        
                if tries >= TRY_TIMES:
                    if len(available_position) == 0:
                        print('Got stuck, task failed!')
                        episode_end = True
                        continue
                    else:
                        next_pos = random.choice(available_position)  # random choose one from available positions
                
                print('next_pos', next_pos)
                # vis
                model.visualize_chessboard(t_agent, last_pos, next_pos)
                action_list = eval_util.postion2action(orientation, last_pos, next_pos)
                
                if len(action_list) > 0:
                    print('m_pred["action_low"]', m_pred["action_low"])
                    print('action_list', action_list)
                    m_pred["action_low"] += action_list
                    # pred["action_low_mask"] = np.random.randint(2, size=(1, constants.SCREEN_WIDTH,constants.SCREEN_HEIGHT))
                    m_pred["action_low_mask"] += [None for _ in action_list]
                    m_pred["object"] += [None for _ in action_list]
                
            if len(m_pred['action_low']) == 0:  
                continue
            
            action_success = True
            # 序贯执行m_pred的所有动作
            for i in range(len(m_pred['action_low'])):
                # print(i)
                # print(m_pred)
                if m_pred['action_low'][i] == cls.STOP_TOKEN:
                    print("predicted STOP")
                    episode_end = True
                    break
                
                # get action and mask
                action, mask, obj = m_pred['action_low'][i], m_pred['action_low_mask'][i], m_pred['object'][i]
                if obj != None:
                    if args.seg != 'maskrcnn':
                        mask = eval_util.extract_seg_pred(obj, model.args.obj_list.index(obj), model.seg_model, env, 
                                                        model.args.debug, model.args.renderInstanceSegmentation, model.obj_classes)
                    else:
                        if obj == 'Sink Basin':obj = 'Sink'
                        elif obj == 'Bathtub Basin':obj = 'Bathtub'
                        try:
                            class_idx = model.seg_model.vocab_obj.word2index(gen_util.remove_space(obj))
                        except:
                            print('class label error')
                            class_idx = 0
                        mask = eval_util.extract_rcnn_pred(class_idx, model.seg_model, env, verbose=args.debug)
                t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)  
            
                if interact_step:
                    model.save_sam_seg(env, t_agent)
                
                if not t_success:
                    print('action failed.', err)
                    
                    if not eval_util.has_interaction(action):
                        action_success = False
                        num_fails += 1
                        if num_fails >= args.max_fails:
                            if args.debug:
                                print("Interact API failed {} times; latest error '{}'".format(
                                    num_fails, err))
                            episode_end = True 
                        
                    else: # interaction try strategy
                        # get max_try number
                        
                        # ===================== start retrying =====================
                        # action_count, fails_count = model.facing(env, next_pos)
                        # t_agent += action_count
                        # num_fails += fails_count
                        turn_back_list = ["RotateRight", "RotateRight"]
                        look_success = False
                        
                        # look up first
                        # if obj == "Toilet":
                        #     lookup_action = "LookUp_15"
                        #     look_t_success, _, _, look_err, _ = env.va_interact(lookup_action, interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                        #     mask = eval_util.extract_rcnn_pred(obj, model.args.obj_list.index(obj), model.seg_model, env, model.args.debug, model.args.renderInstanceSegmentation)
                        #     interation_t_success, _, _, interation_err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                        #     t_agent += 2
                        #     model.save_bgr(bgr=env.last_event.cv2img, step=t_agent)
                            
                        #     if interation_t_success:
                        #         look_success = True
                        #         pass
                        #     else:
                        #         num_fails += 1
                                
                        if not look_success:
                            # Move ahead and try
                            # FIXME: try more step
                            MoveAhead_num = 3
                            for ahead_step in range(MoveAhead_num):
                                action_move_ahead = "MoveAhead"
                                move_t_success, _, _, move_err, _ = env.va_interact(action_move_ahead, interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                                
                                if args.seg != 'maskrcnn':
                                    mask = eval_util.extract_seg_pred(obj, model.args.obj_list.index(obj), model.seg_model, env, 
                                                                    model.args.debug, model.args.renderInstanceSegmentation, model.obj_classes)
                                else:
                                    if obj == 'Sink Basin':obj = 'Sink'
                                    elif obj == 'Bathtub Basin':obj = 'Bathtub'
                                    try:
                                        class_idx = model.seg_model.vocab_obj.word2index(gen_util.remove_space(obj))
                                    except:
                                        print('class label error')
                                        class_idx = 0
                                    mask = eval_util.extract_rcnn_pred(class_idx, model.seg_model, env, verbose=args.debug)
                                
                                interation_t_success, _, _, interation_err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                                t_agent += 2
                                model.save_bgr(bgr=env.last_event.cv2img, step=t_agent)
                                model.save_depth(depth=env.last_event.depth_frame, step=t_agent, add_depth_noise=args.add_depth_noise)
                                model.save_sam_seg(env, t_agent)
                                
                                
                                if move_t_success:
                                    turn_back_list.append("MoveAhead")
                                else:
                                    num_fails += 1
                                    break
                                if interation_t_success:
                                    break
                                else:
                                    print('action failed.', interation_err)
                                
                            if interation_t_success:
                                t_success = True
                                env.last_interaction = (obj, mask)
                                pass
                            else:
                                # try move back
                                num_fails += 1
                                if num_fails >= args.max_fails:
                                    episode_end = True
                                    pass
                                else:
                                    do_not_interation =False
                                    turn_back_list += ["MoveAhead", "RotateLeft", "RotateLeft"]
                                    for back_action in turn_back_list:
                                        move_t_success, _, _, move_err, _ = env.va_interact(back_action, interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                                        t_agent += 1
                                        model.save_bgr(bgr=env.last_event.cv2img, step=t_agent)
                                        model.save_depth(depth=env.last_event.depth_frame, step=t_agent, add_depth_noise=args.add_depth_noise)
                                        model.save_sam_seg(env, t_agent)
                                        if not move_t_success:
                                            num_fails += 1
                                            do_not_interation =True
                                            print("Little incredible. We failed to turn back")
                                            break
                                    if not do_not_interation:
                                        if args.seg != 'maskrcnn':
                                            mask = eval_util.extract_seg_pred(obj, model.args.obj_list.index(obj), model.seg_model, env, 
                                                                            model.args.debug, model.args.renderInstanceSegmentation, model.obj_classes)
                                        else:
                                            if obj == 'Sink Basin':obj = 'Sink'
                                            elif obj == 'Bathtub Basin':obj = 'Bathtub'
                                            class_idx = model.seg_model.vocab_obj.word2index(gen_util.remove_space(obj))
                                            mask = eval_util.extract_rcnn_pred(class_idx, model.seg_model, env, verbose=args.debug)
                                        
                                        interation_t_success, _, _, interation_err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                                        t_agent += 1
                                        model.save_bgr(bgr=env.last_event.cv2img, step=t_agent)
                                        model.save_depth(depth=env.last_event.depth_frame, step=t_agent, add_depth_noise=args.add_depth_noise)
                                        model.save_sam_seg(env, t_agent)
                                        if interation_t_success:
                                            t_success = True
                                            env.last_interaction = (obj, mask)
                                        else:
                                            print('action failed.', interation_err)
                                            num_fails += 1
                                            
                                        # go back
                                        move_t_success, _, _, move_err, _ = env.va_interact("MoveAhead", interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                                        t_agent += 1
                                        model.save_sam_seg(env, t_agent)
                                        if not move_t_success:
                                            num_fails += 1   
                                            
                                    # UPDATE: try left and right
                                    if interation_t_success == False:  
                                        # try left
                                        left_action_list = ["RotateLeft", "MoveAhead", "RotateRight"]
                                        move_ahead_success = True
                                        for left_action in left_action_list:
                                            move_t_success, _, _, move_err, _ = env.va_interact(left_action, interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                                            t_agent += 1
                                            model.save_bgr(bgr=env.last_event.cv2img, step=t_agent)
                                            model.save_depth(depth=env.last_event.depth_frame, step=t_agent, add_depth_noise=args.add_depth_noise)
                                            model.save_sam_seg(env, t_agent)
                                            if not move_t_success:
                                                if left_action == "MoveAhead":
                                                    move_ahead_success = False
                                                num_fails += 1
                                                continue
                                        
                                        if move_ahead_success:
                                            if args.seg != 'maskrcnn':
                                                mask = eval_util.extract_seg_pred(obj, model.args.obj_list.index(obj), model.seg_model, env, 
                                                                                model.args.debug, model.args.renderInstanceSegmentation, model.obj_classes)
                                            else:
                                                if obj == 'Sink Basin':obj = 'Sink'
                                                elif obj == 'Bathtub Basin':obj = 'Bathtub'
                                                class_idx = model.seg_model.vocab_obj.word2index(gen_util.remove_space(obj))
                                                mask = eval_util.extract_rcnn_pred(class_idx, model.seg_model, env, verbose=args.debug)
                                            
                                            
                                            interation_t_success, _, _, interation_err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                                            t_agent += 1
                                            model.save_bgr(bgr=env.last_event.cv2img, step=t_agent)
                                            model.save_depth(depth=env.last_event.depth_frame, step=t_agent, add_depth_noise=args.add_depth_noise)
                                            model.save_sam_seg(env, t_agent)
                                            if interation_t_success:
                                                t_success = True
                                                env.last_interaction = (obj, mask)
                                            else:
                                                print('action failed.', interation_err)
                                                num_fails += 1
                                            
                                        if not interation_t_success:
                                            # try right
                                            move_ahead_step_num = 2 if move_ahead_success else 1
                                            right_action_list = ["RotateRight"]
                                            for i in range(move_ahead_step_num):
                                                right_action_list.append("MoveAhead")
                                            right_action_list.append("RotateLeft")
                                            
                                            for right_action in right_action_list:
                                                move_t_success, _, _, move_err, _ = env.va_interact(right_action, interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                                                t_agent += 1
                                                model.save_bgr(bgr=env.last_event.cv2img, step=t_agent)
                                                model.save_depth(depth=env.last_event.depth_frame, step=t_agent, add_depth_noise=args.add_depth_noise)
                                                model.save_sam_seg(env, t_agent)
                                                if not move_t_success:
                                                    num_fails += 1
                                                    continue
                                                
                                            if args.seg != 'maskrcnn':
                                                mask = eval_util.extract_seg_pred(obj, model.args.obj_list.index(obj), model.seg_model, env, 
                                                                                model.args.debug, model.args.renderInstanceSegmentation, model.obj_classes)
                                            else:
                                                if obj == 'Sink Basin':obj = 'Sink'
                                                elif obj == 'Bathtub Basin':obj = 'Bathtub'
                                                class_idx = model.seg_model.vocab_obj.word2index(gen_util.remove_space(obj))
                                                mask = eval_util.extract_rcnn_pred(class_idx, model.seg_model, env, verbose=args.debug)
                                            interation_t_success, _, _, interation_err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                                            t_agent += 1
                                            model.save_bgr(bgr=env.last_event.cv2img, step=t_agent)
                                            model.save_depth(depth=env.last_event.depth_frame, step=t_agent, add_depth_noise=args.add_depth_noise)
                                            model.save_sam_seg(env, t_agent)
                                            if interation_t_success:
                                                t_success = True
                                                env.last_interaction = (obj, mask)
                                            else:
                                                print('action failed.', interation_err)
                                                num_fails += 1
                                
                                
                        # if obj == "Toilet":
                        #     # look down
                        #     lookdown_action = "LookDown_15"
                        #     look_t_success, _, _, look_err, _ = env.va_interact(lookdown_action, interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                        #     t_agent += 1
                                 
                        # ===================== end for retrying =====================
                        

                        
                        # ===========================================================================================================
                        # History Version        
                        # Forward_action_list, Backward_action_list = eval_util.move_to_target_v2(next_pos, )
                        
                        # action_count, fails_count = model.facing(env, next_pos)
                        # t_agent += action_count
                        # num_fails += fails_count
                        
                        # for phase in [Forward_action_list, Backward_action_list]:
                            
                        #     forward_num = 0
                        #     if episode_end == True:
                        #         break
                        #     for try_action in phase:  # [rotateright, rotateright, moveahead, rotateleft, rotateleft]
                        #         print('agent try interaction, try times: {}'.format(interact_tries+1))
 
                        #         # action_count, fails_count = model.move_closer(env, next_pos)
                        #         # t_agent += action_count
                        #         # num_fails += fails_count
                                
                        #         # move
                        #         move_t_success, _, _, move_err, _ = env.va_interact(try_action, interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
                                
                        #         # interaction
                        #         if "Rotate" not in try_action:
                        #             interation_t_success, _, _, interation_err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
                                
                        #         interact_tries += 1
                        
                        #         if num_fails >= args.max_fails:
                        #             episode_end = True
                        #             break
                                
                        #         if move_t_success:
                        #             forward_num += 1
                        #         else:
                        #             break
                                
                        #         if interation_t_success:
                        #             break
                        #         else:
                        #             continue 
                            
                            
                        #     # Return phase
                        #     # turn back
                            
                        #     for i in range(forward_num):
                        #         pass
                        #         # move ahead
                                

                            
                        # # if interact_tries >= TRY_TIMES - 1:
                        # #     action_success = False
                        
                        # # interact_tries = 0  # reset try times!!
                        
                        # ===========================================================================================================
                          
                else:  
                    if interact_step: # interact action success!
                        env.last_interaction = (obj, mask)
                
                # next time-step
                t_reward, t_done = env.get_transition_reward()
                reward += t_reward
                
                # FIXME: update previous action
                if eval_util.has_interaction(action):
                    if t_success:
                        prev_action = str(action)
                else:
                    prev_action = str(action)
                
                t_agent += 1
                
            
            if not interact_step:
                if not action_success:
                    model.update_collision(next_pos)
                elif t_agent >= 4:
                    model.update_history(last_pos)
            # else:
            #     if not action_success and interact_tries < TRY_TIMES - 1:
            #         print('agent try interaction subgoal, try times: {}'.format(interact_tries))
                    
            #         action_count, fails_count = model.facing(env, next_pos)
            #         t_agent += action_count
            #         num_fails += fails_count
                    
            #         action_count, fails_count = model.move_closer(env, next_pos)
            #         t_agent += action_count
            #         num_fails += fails_count
                    
            #         interact_tries += 1
            #         if num_fails >= args.max_fails:
            #             episode_end = True
            #     elif action_success:
            #         continue
            #     else:  # exceed try times
            #         interact_tries = 0
            

        # check if goal was satisfied
        print(f"step token:{t_agent}")
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True


        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(json_dict['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t_agent))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t_agent))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': json_dict['task_id'],
                    #  'type': json_dict['task_type']+'_'+json_dict['pddl_params']['object_target']+'_'+json_dict['pddl_params']['parent_target'],
                     'type': json_dict['task_type'],
                     'repeat_idx': int(r_idx),
                    #  'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward),
                     'step num': int(t_agent)}
        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = cls.get_metrics(successes, failures)

        print("-------------")
        print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")
        
        if args.log:
            model.close_log()

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = cls.get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        lock.release()

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        # for split in self.args.eval_split: 
            # files = self.splits[f'{split}']
        files = self.splits[f'{self.args.eval_split}']

        # add seen trajectories to queue
        for traj in files:
            task_queue.put(traj)

        return task_queue

    # TODO: 分发进程
    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        processes = []
        lock = self.manager.Lock()
        
        # create and start task processes
        for n in range(self.args.num_process):
            print(n)
            # gpu_idx = "3" if n < self.args.num_process/5 else "0"
            # gpu_idx = "3" if n < self.args.num_process/2 else "0"
            # gpu_idx = "1" 
            # gpu_idx = "2" if n < self.args.num_process/2 else "3"
            # gpu_idx = "0"            
            if n < self.args.num_process/3:
                gpu_idx = "0"
            elif n < self.args.num_process*2/3:
                gpu_idx = "2"
            else:
                gpu_idx = "3"
            # if n < self.args.num_process/3:
            #     gpu_idx = "0"
            # else:
            #     gpu_idx = "1"
                
            thread = mp.Process(target=self.run, args=(task_queue, self.args, lock,
                                                       self.splits, self.successes, self.failures, self.results,
                                                       n, self.chessboard_info_queue, self.next_position_queue_list[n],
                                                       gpu_idx))
            thread.start()
            processes.append(thread)

        # create and start LLM process
        LLM = mp.Process(target=LLM_inference, args=(self.chessboard_info_queue, self.next_position_queue_list, self.num_alive_processes))
        LLM.start()

        for p in processes:
            p.join()
            
        LLM.join()

        # save
        self.save_results()

    def create_stats(self):
        '''
        all GLOBAL i/o variants!!!
        storage for success, failure, and results info
        '''
        self.successes, self.failures = self.manager.list(), self.manager.list()
        self.results = self.manager.dict()
        
        # self.chessboard_info_queue = self.manager.list()
        # self.next_position_queue_list = self.manager.list()
        
        # UPDATE: use Queue instead of list
        self.chessboard_info_queue = mp.Queue()
        self.next_position_queue_list = [mp.Queue() for i in range(self.args.num_process)]
        
        # Numer of alive processes
        self.num_alive_processes = self.manager.Value('i', self.args.num_process)

    def save_results(self):
        '''
        save actseqs as JSONs
        '''
        # TODO:成功失败？
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        save_path = self.args.output_dir
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        json_save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(json_save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)
    
            
if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    # settings
    # parser.add_argument('--splits', type=str, default="/home/lm2/projects/Real2Chess_maskrcnn/alfred/data/splits/test_sliced.json")
    # parser.add_argument('--splits', type=str, default="/home/lm2/projects/Real2Chess_maskrcnn/alfred/data/splits/easy_data.json")
    parser.add_argument('--splits', type=str, default="/home/lm2/projects/Real2Chess/alfred/data/splits/oct21.json")
    # parser.add_argument('--splits', type=str, default="/home/lm2/projects/Real2Chess_maskrcnn/alfred/data/splits/sample_data_100_1.json")
    parser.add_argument('--data_dir', type=str, default="/home/lm2/projects/ET/data/json_2.1.0")
    parser.add_argument('--reward_config', default='/home/lm2/projects/alfred/models/config/rewards.json')
    parser.add_argument('--model', type=str, default='models.llm_maskrcnn')
    parser.add_argument('--model_path', type=str, default="vipl-vrc/real2chess-0421")
    parser.add_argument('--revision', type=str, default='2689456d404a97cbd9a213c8a8dbb2e0490d1fe5')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--num_process', type=int, default=PROCESS_NUM)
    parser.add_argument('--num_processes', type=int, default=1)  # sem mapping
    # FIXME:
    parser.add_argument('--eval_split', type=str, default='valid_seen', choices=['train', 'valid_seen', 'valid_unseen'])
    # parser.add_argument('--eval_split', type=str, nargs="+", default=['valid_seen','valid_unseen'], choices=['train', 'valid_seen', 'valid_unseen'])
    parser.add_argument('--output_dir', type=str, default="result_test_msrnn_rebuttal_depth/{}".format(time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(round(time.time())))))
    # parser.add_argument('--x_display', type=str, default="12.0")
    parser.add_argument('--x_display', type=int, default=0)
    parser.add_argument('--log', type=int, default=1)
    parser.add_argument('--det_model', type=str, default='DINO')
    
    # =======================================================================================
    # Mapping
    parser.add_argument('--global_downscaling', type=int, default=1)
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=1.0)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--collision_threshold', type=float, default=0.20)
    
    parser.add_argument('--print_time', type=int, default=0)
    parser.add_argument('--camera_height', type=float, default=1.55,
                        help="agent camera height in metres")
    parser.add_argument('--max_depth', type=float, default=5.0)
    
    parser.add_argument('--seg_model', type=str, default='maskrcnn', choices=['gt', 'dino', 'maskrcnn'])
    parser.add_argument('--add_depth_noise', type=bool, default=True)
    parser.add_argument('--agent_step_length', type=float, default=constants.AGENT_STEP_SIZE * 100)
    parser.add_argument('--chess_extend_ratio', type=float, default=1)
    # parse arguments
    args = parser.parse_args()
    
    if args.seg_model == 'gt':
        args.renderInstanceSegmentation = True
    elif args.seg_model == 'dino':
        args.renderInstanceSegmentation = False
        args.seg = 'dino'
    elif args.seg_model == 'maskrcnn':
        args.renderInstanceSegmentation = False
        args.seg = 'maskrcnn'
        args.maskrcnn_path = '../models/maskrcnn_model.pth'
        print('using maskrcnn')
    else:
        raise ValueError("Invalid choice")

    # fixed settings (DO NOT CHANGE)
    args.max_steps = 120
    args.max_fails = 10
    
    args.debug = True
    args.smooth_nav = False
    
    # args.device = torch.device("cuda:1")
    args.obj_list = ['Sink Basin', 'Arm Chair', 'Bathtub Basin', 'Bed', 'Cabinet', 'Cart', 'Coffee Machine', 'Coffee Table',
                                    'Counter Top', 'Desk', 'Dining Table', 'Drawer', 'Dresser', 'Fridge', 'Garbage Can',
                                    'Microwave', 'Ottoman', 'Safe', 'Shelf', 'Side Table', 'Sofa',
                                    'Stove Burner', 'TV Stand', 'Toilet', 'Faucet', 'Desk Lamp', 'Floor Lamp', 'None']  # 28
    args.num_sem_categories = 28            # Grounding SAM 输出 23+1+1+1 类 - ButterKnife + 'Faucet', 'Desk Lamp', 'Floor Lamp'

    args.env_frame_width = 300              # 仿真环境 frame 大小 [300, 300]
    args.env_frame_height = 300
    args.frame_height = 150                 # 降采样之后的 frame 大小，用于语义建图 [150, 150]
    args.frame_width = 150
    args.hfov = 60                          # env fieldOfView

    args.map_size_cm = 1600                 # global map size 12m * 12m
    args.map_resolution = 5                 # size of map bins 5cm
    args.global_downscaling = 1             # ratio of global over local map，full_map 到 local_map 的降采样系数
    args.vision_range = 100                 # diameter of local map region visible by the agent (in cells)

    args.print_images = 1                   # 语义地图需要
    args.save_pictures = True               # save latest semantic map image to Sem_Map/Sem_Map.png
    
    args.no_straight_obs = True

    # leaderboard dump
    eval = LLMEval(args, manager)

    # start threads
    eval.spawn_threads()