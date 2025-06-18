import os
import torch
import json
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import glob
from tqdm import tqdm, trange

from alfred.utils import gen_util
from alfred.env.thor_env import ThorEnv
import envs.utils.pose as pu
from keyboard_fb.sem_map_helper import SemMapHelper
from segmentation_helper import SegmentationHelper
from arguments import get_args
import alfred.constants as constants


class DataBuilder():
    """build LLM format data"""
    def __init__(self, args, data_dir) -> None:
        self.args = args
        self.data_dir = data_dir
        self.partitions = args.partitions
        self.rgb_dir = 'raw_images'
        self.depth_dir = 'depth_images'
        self.instance_dir = 'instance_masks'
        self.sem_map_dir = 'sem_maps'
        self.data_file = 'data_v2.json'
        self.seg_dir = 'seg_images'
        self.chessboard_dir = 'chessboard'
        self.json_name = 'traj_data.json'
        self.data_info = {}
        
        # FIXME: 
        self.renderInstanceSegmentation = False
        self.startEnv = False
        self.cover = True
        self.max_depth = 5.0
        self.map_save_large_objects = ['Sink Basin', 'Arm Chair', 'Bathtub Basin', 'Bed', 'Cabinet', 'Cart', 'Coffee Machine', 'Coffee Table',
                                  'Counter Top', 'Desk', 'Dining Table', 'Drawer', 'Dresser', 'Fridge', 'Garbage Can',
                                  'Microwave', 'Ottoman', 'Safe', 'Shelf', 'Side Table', 'Sofa',
                                  'Stove Burner', 'TV Stand', 'Toilet', 'Faucet', 'Desk Lamp', 'Floor Lamp', 'None']  # 28
        self.obj_list = self.map_save_large_objects  # task-aware obj list
        self.is_holding = False
        self.holding_obj = -1
        self.hl_actions = []
        self.orientation = 0.0
        # make a list of all the traj_data json files
        self.traj_list = self.read_data()
        
        # start THOR
        if self.startEnv:
            self.env = ThorEnv(x_display=args.x_display)

        if not self.renderInstanceSegmentation:
            self.seg_model = SegmentationHelper(self.args)

        self.traj2map()

    def read_data(self):
        traj_list = []
        print('Indexing images in {}'.format(self.partitions))
        for partition in self.partitions:
            for dir_name in sorted(
                    glob.glob(os.path.join(self.data_dir, partition, '*/*'))):
                if 'trial_' in os.path.basename(dir_name):
                    json_path = os.path.join(dir_name, self.json_name)
                    if not os.path.isfile(json_path):
                        continue
                    traj_list.append('/'.join(json_path.split('/')[-4:]))
        num_files, num_processed_files = len(traj_list), 0
        print('# trajs: ', num_files)
        # print(traj_list.index('train/look_at_obj_in_light-Box-None-FloorLamp-212/trial_T20190908_193427_340509/traj_data.json'))
        return traj_list

    def traj2map(self):  
        for json_path in tqdm(self.traj_list[self.args.start_idx:self.args.end_idx]):
            with open(os.path.join(self.data_dir, json_path)) as json_file:
                json_dict = json.load(json_file)
            trial_dir = json_path.replace(self.json_name, '')
            self.trial_dir = trial_dir
            self.json_dict = json_dict
            num_steps = len(json_dict["images"])  # from step0 to end_step
            print("scene: ", trial_dir)
            print('# steps: ', num_steps)  

            if os.path.isfile(os.path.join(self.data_dir, self.trial_dir, self.data_file)) and not self.cover:
                continue 
            # FIXME:
            if 'agent_poses' not in self.json_dict.keys():
                continue
            
            # setup scene
            if self.startEnv:
                gen_util.setup_scene(self.env, self.json_dict, reward_type='dense')

            # init data info dict
            task_info_dic = gen_util.read_task_info(self.json_dict)
            self.data_info = {"task": task_info_dic, "steps": []}
            
            # init object list
            self.obj_list = gen_util.get_obj_list(json_dict)
            
            self.args.obj_list = self.obj_list
            self.args.num_sem_categories  = len(self.obj_list)
            self.data_info["task"]["obj_list"] = self.obj_list

            # update class in seg model
            if not self.renderInstanceSegmentation:
                self.seg_model.obj_classes = self.obj_list      
            
            # init agent state
            self.is_holding = False
            self.holding_obj = -1
            self.orientation = 0.0
           
            # init semantic mapping module
            self.sem_map_module = SemMapHelper(self.args) 
            # self.seg_model.add_obj_targets(obj_targets)
            # target_objects = self.seg_model.obj_classes
            
            # init agent state
            step = 0
            info = {}
            info['sensor_pose'] = [0., 0., 0.]
            info['holding_state'] = 0  # not holding
            info['holding_obj'] = -1  # holding nothing

            init_action = self.json_dict['scene']['init_action']
            # self.sem_map_module.reset_agent_height(0.75 + init_action["y"])  # 0.675 + agent['position']['y']
            self.sem_map_module.sem_map_builder.agent_height = 155.
            self.sem_map_module.reset_view_angle(init_action["horizon"])
            last_sim_location = gen_util.get_location(init_action)

            # save init rgbd+seg (optional)
            if self.startEnv:
                rgb, bgr, depth = self.save_init_obs()
            else:
                rgb_file = os.path.join(self.data_dir, trial_dir, self.rgb_dir, 'init.png')
                depth_file = os.path.join(self.data_dir, trial_dir, self.depth_dir, 'init.npy')
                rgb = np.array(Image.open(rgb_file))
                bgr = cv2.imread(rgb_file)
                depth = np.load(depth_file)
            
            # get obs
            obs, sem_seg_image = self._preprocess_obs(rgb, bgr, depth, step)

            # save seg image
            self.save_init_seg_image(sem_seg_image)
            
            # get updated local map
            local_map = self.sem_map_module.update_local_map_and_process(obs, info)  # 出生local_map
            
            # save sem map
            self.save_init_sem_map()

            # sem map to chessboard
            chessboard = self.sem_map_module.local_map_to_chessboard()
            self.save_init_chessboard()
            
            obj_dict = self.sem_map_module.chessboard_info(self.obj_list)
            # print(obj_dict)

            # get high-level actions
            self.hl_actions = gen_util.get_subgoals(self.json_dict)

            # 读取当前步对应的输出, step 0 对应 ll_action[0]
            # get next-step action
            ll_action, PickUp, PutDown, high_idx = gen_util.get_ll_action(self.json_dict, step, num_steps)

            # update output
            # agent_pose = info['sensor_pose'] + [init_action["horizon"]]
            agent_pose = self.compute_pose(obj_dict, info, init_action["horizon"])
            self.update_output(high_idx, obj_dict, agent_pose, ll_action)
            
            while step < num_steps:
                print('step ', step)
                # do an agent step and get an observation
                # 0.png是agent.step(ll_action[0])之后的观察
                rgb_file = os.path.join(self.data_dir, trial_dir, self.rgb_dir, '{:09d}.png'.format(step))
                depth_file = os.path.join(self.data_dir, trial_dir, self.depth_dir, '{:09d}.npy'.format(step))
                rgb = np.array(Image.open(rgb_file))
                bgr = cv2.imread(rgb_file)
                depth = np.load(depth_file)
                
                # get obs 
                obs, sem_seg_image = self._preprocess_obs(rgb, bgr, depth, step)
                
                # save seg image
                seg_save_path = os.path.join(self.data_dir, trial_dir, self.seg_dir, '{:09d}.png'.format(step))
                cv2.imwrite(seg_save_path, sem_seg_image)

                # get pose change
                agent_pose_dic = self.json_dict['agent_poses'][step]
                curr_pose = gen_util.read_agent_pose(agent_pose_dic)
                curr_location = gen_util.get_location(curr_pose)
                dx, dy, do = pu.get_rel_pose_change(curr_location, last_sim_location)
                last_sim_location = curr_location          
                info['sensor_pose'] = [dx, dy, do]

                # update holding state
                info = self.update_holding_state(ll_action, PickUp, PutDown, info)

                # get updated local map
                self.sem_map_module.reset_view_angle(curr_pose['horizon'])
                local_map = self.sem_map_module.update_local_map_and_process(obs, info)
                chessboard = self.sem_map_module.local_map_to_chessboard()
                obj_dict = self.sem_map_module.chessboard_info(self.obj_list)
                # print(local_map.shape)
                
                # save sem map(optional)
                map_save_path = os.path.join(self.data_dir, trial_dir, self.sem_map_dir, '{:09d}.png'.format(step))
                self.sem_map_module.visualize_w_small(map_save_path)
                # save chessboard(optional)
                chessboard_save_path = os.path.join(self.data_dir, trial_dir, self.chessboard_dir, '{:09d}.png'.format(step))
                self.sem_map_module.chessboard_visualize(chessboard_save_path)
                
                step += 1
                # get next-step action, step 1 对应 ll_action[1]
                ll_action, PickUp, PutDown, high_idx = gen_util.get_ll_action(self.json_dict, step, num_steps)

                # update output
                # agent_pose = info['sensor_pose'] + [curr_pose['horizon']]
                agent_pose = self.compute_pose(obj_dict, info, curr_pose['horizon'])
                self.update_output(high_idx, obj_dict, agent_pose, ll_action)
            
            # post-process
            self.post_process(num_steps)
            with open(os.path.join(self.data_dir, self.trial_dir, self.data_file), 'w') as f:
                json.dump(self.data_info, f)
    
    def _preprocess_obs(self, rgb, bgr, depth, step):
        
        if self.renderInstanceSegmentation:
            # GT segmentation
            sem_seg_pred, sem_seg_image = self._get_gt_segmentation(step)
        else:
            # Grounded SAM, 似乎要 BGR 图片, seg_seg_pred [300, 300, #Class]
            sem_seg_pred, sem_seg_image = self.seg_model.segmentation_for_map(bgr.astype(np.uint8))

        # TODO: event.depth_frame 单位是 mm，点云地图要 cm
        depth = depth / 1000.
        mask = depth > self.max_depth
        # FIXME: 
        # depth[mask] = self.max_depth
        depth[mask] = 100.
        depth = depth * 100.
        
        ds = self.args.env_frame_width // self.args.frame_width  # Downscaling factor, args.env_frame_width=640, args.frame_width=160
        if ds != 1:
            res = transforms.Compose([transforms.ToPILImage(), transforms.Resize((self.args.frame_height, self.args.frame_width), interpolation=Image.NEAREST)])
            rgb = np.asarray(res(rgb.astype(np.uint8)))
            bgr = np.asarray(res(bgr.astype(np.uint8)))
            # sem_seg_image = np.asarray(res(sem_seg_image.astype(np.uint8)))
            depth = depth[ds//2::ds, ds//2::ds]
            sem_seg_pred = sem_seg_pred[ds//2::ds, ds//2::ds]

        
        depth = np.expand_dims(depth, axis=2)

        obs = np.concatenate((rgb, depth, sem_seg_pred),axis=2).transpose(2, 0, 1) # C H W

        return obs, sem_seg_image

    # FIXME:
    def _get_gt_segmentation(self, step):
        mask_file = os.path.join(self.data_dir, self.trial_dir, self.instance_dir, '{:09d}.png'.format(step))
        sem_seg_image = np.array(Image.open(mask_file))  # 300, 300, 3
        # sem_seg_image = self.event.instance_segmentation_frame
        sem_seg_pred = np.zeros((self.args.env_frame_width, self.args.env_frame_width, self.args.num_sem_categories))

        color_to_object_type = gen_util.get_color_to_object(self.json_dict)
        colors = self.env.last_event.object_id_to_color

        for color, obj in color_to_object_type.items():
            obj_type = gen_util.camel_to_space(obj["objectType"])
            if obj_type in self.obj_list:
                print(obj_type)
                obj_id = self.obj_list.index(obj_type)
                # obj_mask = self.event.instance_masks[obj["objectId"]] * 1.
                # sem_seg_pred[:, :, obj_id] += obj_mask
                # mask = np.where(sem_seg_image == (135, 129, 110))
                obj_mask = np.where(sem_seg_image == color)
                sem_seg_pred[obj_mask, obj_id] = 1.
        return sem_seg_pred, sem_seg_image
    
    def update_holding_state(self, ll_action, PickUp, PutDown, info): 
        if PickUp:
            self.is_holding = True
            info['holding_state'] = 1 # start holding
            info['holding_obj'] = self.obj_list.index(ll_action[1]) + 4
            self.holding_obj = info['holding_obj']
        elif PutDown:
            self.is_holding = False
            info['holding_state'] = 3 # end holding
            self.holding_obj = -1
        else:
            if self.is_holding: 
                info['holding_state'] = 2 # is holding
            else:
                info['holding_state'] = 0 # not holding
        return info
    
    def compute_pose(self, obj_dict, info, horizon):
        dx, dy, do = info['sensor_pose']
        print("compute_pose: ", do)
        self.orientation += do * 57.29577951308232
        self.orientation = np.fmod(self.orientation-180.0, 360.0) + 180.0
        self.orientation = np.fmod(self.orientation+180.0, 360.0) - 180.0
        print("compute_pose: ", self.orientation)
        agent_pose = {
            "xy": obj_dict["Agent"],
            "orientation": self.orientation,
            "cameraHorizon": int(horizon)
        }
        return agent_pose

    def update_output(self, high_idx, obj_dict, agent_pose, ll_action):
        step_info_dict = {}
        if high_idx == -1:  # ending step
            step_info_dict['subgoal'] = 'NoOp'
        else:
            step_info_dict['subgoal'] = self.hl_actions[high_idx]
        step_info_dict['chess_info'] = obj_dict
        step_info_dict['agent_pose'] = agent_pose
        step_info_dict['ll_action'] = ll_action

        self.data_info["steps"].append(step_info_dict)

    def save_init_obs(self):
        rgb = self.env.last_event.frame
        bgr = self.env.last_event.cv2img
        depth = self.env.last_event.depth_frame
        rgb_save_path = os.path.join(self.data_dir, self.trial_dir, self.rgb_dir, 'init.png')
        cv2.imwrite(rgb_save_path, bgr)
        depth_save_path = os.path.join(self.data_dir, self.trial_dir, self.depth_dir, 'init.npy')
        np.save(depth_save_path, depth)
        instance_save_path = os.path.join(self.data_dir, self.trial_dir, self.instance_dir, 'init.png')
        cv2.imwrite(instance_save_path, self.env.last_event.instance_segmentation_frame)

        return rgb, bgr, depth

    def save_init_seg_image(self, sem_seg_image):
        seg_save_path = os.path.join(self.data_dir, self.trial_dir, self.seg_dir)
        if not os.path.exists(seg_save_path):
            os.makedirs(seg_save_path)
        cv2.imwrite(os.path.join(seg_save_path, 'init.png'), sem_seg_image)    
    
    def save_init_sem_map(self):
        map_save_path = os.path.join(self.data_dir, self.trial_dir, self.sem_map_dir)
        if not os.path.exists(map_save_path):
            os.makedirs(map_save_path)
        self.sem_map_module.visualize_w_small(os.path.join(map_save_path, 'init.png'))
    
    def save_init_chessboard(self):
        chessboard_save_path = os.path.join(self.data_dir, self.trial_dir, self.chessboard_dir)
        if not os.path.exists(chessboard_save_path):
            os.makedirs(chessboard_save_path)
        self.sem_map_module.chessboard_visualize(os.path.join(chessboard_save_path, 'init.png'))

    def post_process(self, num_steps):
        for idx in range(num_steps):
            # next-step position
            self.data_info["steps"][idx]["next_position"] = self.data_info["steps"][idx+1]["chess_info"]["Agent"]
            # subgoal finish?
            next_subgoal = self.data_info["steps"][idx+1]["subgoal"]
            current_subgoal = self.data_info["steps"][idx]["subgoal"]
            if current_subgoal == next_subgoal:
                self.data_info["steps"][idx]["subgoal_finish"] = 0  # not finish if SAME
            else:
                self.data_info["steps"][idx]["subgoal_finish"] = 1  # finish if NOT SAME
        # ending step
        self.data_info["steps"][num_steps]["next_position"] = []
        self.data_info["steps"][num_steps]["subgoal_finish"] = 1

            

if __name__ == '__main__':
    args = get_args()
    args.device = torch.device("cuda:0")
    args.obj_list = ['Sink Basin', 'Arm Chair', 'Bathtub Basin', 'Bed', 'Cabinet', 'Cart', 'Coffee Machine', 'Coffee Table',
                                  'Counter Top', 'Desk', 'Dining Table', 'Drawer', 'Dresser', 'Fridge', 'Garbage Can',
                                  'Microwave', 'Ottoman', 'Safe', 'Shelf', 'Side Table', 'Sofa',
                                  'Stove Burner', 'TV Stand', 'Toilet', 'Faucet', 'Desk Lamp', 'Floor Lamp', 'None']  # 28
    args.num_sem_categories = 28            # Grounding SAM 输出 23+1+1+1 类 - ButterKnife + faucet + deskLamp + FloorLamp
    args.num_processes = 1                  # 单线程

    args.env_frame_width = 300              # 仿真环境 frame 大小 [300, 300]
    args.env_frame_height = 300
    args.frame_height = 150                 # 降采样之后的 frame 大小，用于语义建图 [150, 150]
    args.frame_width = 150
    args.hfov = 60                          # env fieldOfView
    
    args.map_size_cm = 1200                 # global map size 12m * 12m
    args.map_resolution = 5                 # size of map bins 5cm
    args.global_downscaling = 1             # ratio of global over local map，full_map 到 local_map 的降采样系数
    args.vision_range = 100                 # diameter of local map region visible by the agent (in cells)
    
    args.print_images = 1                   # 语义地图需要
    args.save_pictures = False              # save latest semantic map image to Sem_Map/Sem_Map.png
    
    args.x_display = 0

    args.partitions = ['train', 'valid_seen', 'valid_unseen']
    args.start_idx = 0
    args.end_idx = 1
    data = DataBuilder(args, data_dir=constants.ET_DATA)