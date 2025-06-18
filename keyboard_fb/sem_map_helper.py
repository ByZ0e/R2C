import os
import numpy as np
import torch
import skimage
import pickle
from PIL import Image, ImageDraw
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import copy

from models.sem_mapping import Semantic_Mapping
import envs.utils.pose as pu


class SemMapHelper():
    def __init__(self,
        args,
        map_x_down=True,
        device=torch.device("cuda:0"),
        ) -> None:
        
        self.args = args
        self.map_x_down = map_x_down
        # self.device = args.device
        
        # 单线程
        self.num_scenes = 1
        self.num_processes = 1
        
        # Initialize map variables
        ### Full map consists of multiple channels containing the following:
        ### 1. Obstacle Map
        ### 2. Exploread Area
        ### 3. Current Agent Location
        ### 4. Past Agent Locations
        ### 5,6,7,.. : Semantic Categories
        # self.nlarge = 28  # num large map save obj classes - ButterKnife + Faucet', 'Desk Lamp', 'Floor Lamp' 无用了
        self.nc = self.args.num_sem_categories + 4  # num channels
        # self.chess_size = int(self.args.map_size_cm / args.agent_step_length)  # chess grid size is equal to agent_step_length
        self.chess_size = self.args.map_size_cm // self.args.map_resolution // 5  # ziyi
        # Calculating full and local map sizes
        # map_size_cm = 1200
        # map_resolution = 5
        # full map [240, 240]
        # local map [240, 240]
        self.map_size = self.args.map_size_cm // self.args.map_resolution # 240
        self.full_w, self.full_h = self.map_size, self.map_size
        self.local_w, self.local_h = int(self.full_w / self.args.global_downscaling), \
                                     int(self.full_h / self.args.global_downscaling)

        # Initializing full and local map
        
        # self.full_map = torch.zeros(self.num_scenes, self.nc, self.full_w, self.full_h).float().to(self.device)
        # self.local_map = torch.zeros(self.num_scenes, self.nc, self.local_w, self.local_h).float().to(self.device)
        self.full_map = torch.zeros(self.num_scenes, self.nc, self.full_w, self.full_h).float().cuda()
        self.local_map = torch.zeros(self.num_scenes, self.nc, self.local_w, self.local_h).float().cuda()

        # Initial full and local pose
        # self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        # self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.device)
        self.full_pose = torch.zeros(self.num_scenes, 3).float().cuda()
        self.local_pose = torch.zeros(self.num_scenes, 3).float().cuda()

        # Origin of local map
        self.origins = np.zeros((self.num_scenes, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((self.num_scenes, 4)).astype(int)

        ### Planner pose inputs has 7 dimensions
        ### 1-3 store continuous global agent location  [x, y, o]
        ### 4-7 store local map boundaries              [gx1, gx2, gy1, gy2]
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7))
        
        # init map
        self.init_map_and_pose()
        
        # self.sem_map_builder = Semantic_Mapping(self.args).to(self.device)
        self.sem_map_builder = Semantic_Mapping(self.args).cuda()
        self.sem_map_builder.eval()
        self.reset_view_angle(45)
        
        # color_palette = d3_40_colors_rgb.flatten()
        color_palette = [1.0, 1.0, 1.0,  # white
                        0.6, 0.6, 0.6,  # obstacle
                        0.95, 0.95, 0.95,  # exp
                        0.96, 0.36, 0.26,  # agent
                        0.12156862745098039, 0.47058823529411764, 0.7058823529411765,  # goal
                        0.9400000000000001, 0.7818, 0.66,  # 5 + 16
                        0.9400000000000001, 0.8868, 0.66,
                        0.8882000000000001, 0.9400000000000001, 0.66,
                        0.7832000000000001, 0.9400000000000001, 0.66,
                        0.6782000000000001, 0.9400000000000001, 0.66,
                        0.66, 0.9400000000000001, 0.7468000000000001,
                        0.66, 0.9400000000000001, 0.9018000000000001,
                        0.66, 0.9232, 0.9400000000000001,
                        0.66, 0.8182, 0.9400000000000001,
                        0.66, 0.7132, 0.9400000000000001,
                        0.7117999999999999, 0.66, 0.9400000000000001,
                        0.8168, 0.66, 0.9400000000000001,
                        0.9218, 0.66, 0.9400000000000001,
                        0.9400000000000001, 0.66, 0.9031999999999998,
                        0.9400000000000001, 0.66, 0.748199999999999]

        color_palette += pickle.load(open("../visualize_sem_map/miscellaneous/flattened.p", "rb")).tolist()

        self.color_palette = [int(x * 255.) for x in color_palette]
        
        # history collision pool
        self.collision = []
        
        self.step = 0


    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose(self):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0 # [6.0, 6.0, 0.0] (m)

        locs = self.full_pose.cpu().numpy()
        ### 1-3 store continuous global agent location  [x, y, o]
        self.planner_pose_inputs[:, :3] = locs
        
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]  # [120, 120] (index)

            self.full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            self.lmb[e] = self.get_local_map_boundaries((loc_r, loc_c),
                                                (self.local_w, self.local_h),
                                                (self.full_w, self.full_h))

            ### 4-7 store local map boundaries              [gx1, gx2, gy1, gy2]
            self.planner_pose_inputs[e, 3:] = self.lmb[e]               # [0, 240, 0, 240] (index)
            ### Origin of local map 原点
            self.origins[e] = [self.lmb[e][2] * self.args.map_resolution / 100.0,
                            self.lmb[e][0] * self.args.map_resolution / 100.0, 0.]  # [0., 0., 0.] (m)

        for e in range(self.num_scenes):
            self.local_map[e] = self.full_map[e, :, self.lmb[e, 0]:self.lmb[e, 1], self.lmb[e, 2]:self.lmb[e, 3]]
            # self.local_pose[e] = self.full_pose[e] - \
            #                 torch.from_numpy(self.origins[e]).to(self.device).float()    # [6.0, 6.0, 0.0] (m)
            self.local_pose[e] = self.full_pose[e] - \
                            torch.from_numpy(self.origins[e]).float().cuda() 

    def reset_view_angle(self, view_angle):
        self.sem_map_builder.set_view_angles([view_angle] * self.num_processes)
    
    def reset_agent_height(self, height):
        # FIXME: for navigation
        self.sem_map_builder.agent_height = height*100.
        # FIXME: for ALFRED
        # self.sem_map_builder.agent_height = 155.
        
        print("#######SemMap:", "agent_height", self.sem_map_builder.agent_height)

    def first_update_local_map(self, obs, info):
        obs = torch.tensor(obs).float().unsqueeze(0).to(self.device)
        
        # Get pose from info
        # 处理后的 2D 世界坐标系下的坐标，智能体面朝 x 轴正向时 rotation 为 0，pose 实际记录 dx, dy, do
        pose = torch.from_numpy(np.asarray(
            [info['sensor_pose']])
            ).float().to(self.device)
        
        _, self.local_map, _, self.local_pose = self.sem_map_builder.forward_origion(obs, pose, self.local_map, self.local_pose, info)
        
        locs = self.local_pose.cpu().numpy()
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            self.local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        
        # FIXME: 这里省略了随机设置的 goal 和 goal_map
        
        # for envs.plan_act_and_preprocess
        self.planner_inputs = [{} for e in range(self.num_scenes)]
        for e, p_input in enumerate(self.planner_inputs):
            # p_input['newly_goal_set'] =newly_goal_set
            p_input['map_pred'] = self.local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = self.local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = self.planner_pose_inputs[e]
            # p_input['goal'] = goal_maps[e] 
            # p_input['new_goal'] = 1
            # p_input['found_goal'] = 0
            # p_input['wait'] = wait_env[e] or finished[e]
            # p_input['list_of_actions'] = list_of_actions_s[e]
            # p_input['list_of_actions_pointer'] = list_of_actions_pointer_s[e]
            # p_input['consecutive_interaction'] = None
            # p_input['consecutive_target'] = None
            if self.args.visualize or self.args.print_images:
                self.local_map[e, -1, :, :] = 1e-5
                # sem_map: [240, 240], values: 0, 1, 2, 3, ..., 25
                p_input['sem_map_pred'] = self.local_map[e, 4:, :, :].argmax(0).cpu().numpy()

    # TODO: 在 update_local_map 之前省略了一些操作，由 planner 推断下一步动作，判断动作失败或成功，并进行相应处理和记录
    def update_local_map(self, obs, info):
        obs = torch.tensor(obs).float().unsqueeze(0).to(self.device)
        
        # [dx, dy, do] 初始化为 [0., 0., 0.]
        pose = torch.from_numpy(np.asarray(
            [info['sensor_pose']])
            ).float().to(self.device)
        
        _, self.local_map, _, self.local_pose = self.sem_map_builder.forward_origion(obs, pose, self.local_map, self.local_pose, info)
        
        locs = self.local_pose.cpu().numpy()
        
        # FIXME: 首次更新 local_map 和之后更新 local_map 在此处有区别，首次更新 local_map 不更新
        # planner_pose_inputs[:, :3] 和 local_map[:, 2, :, :]，但是由于初始化全为 0，统一写法（第一次也更新）不影响实际值
        self.planner_pose_inputs[:, :3] = locs + self.origins
        # local_map[:, 2, :, :]: Current Agent Location
        # local_map[:, 3, :, :]: Past Agent Locations
        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        
        print("sem_map: locs", locs)
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            # FIXME: 首次更新 local_map 时，Agent Location 的半径是 1*1 的，统一和之后更新 local_map 变为 2*2
            self.local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
        
        # TODO: 这里似乎省略了很多
        
        # for envs.plan_act_and_preprocess
        self.planner_inputs = [{} for e in range(self.num_scenes)]
        for e, p_input in enumerate(self.planner_inputs):
            # p_input['newly_goal_set'] =newly_goal_set
            p_input['map_pred'] = self.local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = self.local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = self.planner_pose_inputs[e]
            # p_input['goal'] = goal_maps[e] 
            # p_input['new_goal'] = 1
            # p_input['found_goal'] = 0
            # p_input['wait'] = wait_env[e] or finished[e]
            # p_input['list_of_actions'] = list_of_actions_s[e]
            # p_input['list_of_actions_pointer'] = list_of_actions_pointer_s[e]
            # p_input['consecutive_interaction'] = None
            # p_input['consecutive_target'] = None
            
            if self.args.visualize or self.args.print_images:
                self.local_map[e, -1, :, :] = 1e-5
                # sem_map: [240, 240], values: 0, 1, 2, 3, ..., 25
                p_input['sem_map_pred'] = self.local_map[e, 4:, :, :].argmax(0).cpu().numpy()
    
    def update_local_map_and_process(self, obs, info):
        # print(self.args.obj_list)
        # obs = torch.tensor(obs).float().unsqueeze(0).to(self.device)
        obs = torch.tensor(obs).float().unsqueeze(0).cuda()
        print("#######SemMap:", "obs", obs.shape)
        
        # [dx, dy, do] 初始化为 [0., 0., 0.]
        # pose = torch.from_numpy(np.asarray(
        #     [info['sensor_pose']])
        #     ).float().to(self.device)
        pose = torch.from_numpy(np.asarray(
            [info['sensor_pose']])
            ).float().cuda()
        
        # TODO: 这里需要传入智能体是否开始抓取，以及抓取的物体类别（在obs中的哪一层）
        _, self.local_map, _, self.local_pose = self.sem_map_builder(obs, pose, self.local_map, self.local_pose, info)
        
        locs = self.local_pose.cpu().numpy()
        
        # FIXME: 首次更新 local_map 和之后更新 local_map 在此处有区别，首次更新 local_map 不更新
        # planner_pose_inputs[:, :3] 和 local_map[:, 2, :, :]，但是由于初始化全为 0，统一写法（第一次也更新）不影响实际值
        self.planner_pose_inputs[:, :3] = locs + self.origins
        # local_map[:, 2, :, :]: Current Agent Location
        # local_map[:, 3, :, :]: Past Agent Locations
        self.local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        
        print("#######SemMap:", "locs", locs)
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            # FIXME: 首次更新 local_map 时，Agent Location 的半径是 1*1 的，统一和之后更新 local_map 变为 2*2
            self.local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
        
        # TODO: 这里似乎省略了很多
        
        # for envs.plan_act_and_preprocess
        self.planner_inputs = [{} for e in range(self.num_scenes)]
        for e, p_input in enumerate(self.planner_inputs):
            # p_input['newly_goal_set'] =newly_goal_set
            p_input['map_pred'] = self.local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = self.local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = self.planner_pose_inputs[e]

            # num_obj_targets = self.args.num_sem_categories - self.nlarge
            num_obj_targets = self.args.num_sem_categories - self.args.obj_list.index('None') - 1
            # self.local_map[e, -1, :, :] = 1e-5  # None -> unexplore
            self.local_map[e, -(1+num_obj_targets), :, :] = 1e-5  # None -> unexplore
            # 大物体拍扁
            squeezed_local_map = self.local_map[e, 4:-num_obj_targets, :, :].argmax(0).cpu().numpy()
            # 附着小物体，置信度>0.3
            
            print("#######SemMap:", "num_sem_categories", self.args.num_sem_categories)
            
            print("#######SemMap:", "num_obj_targets", num_obj_targets)
            
            for obj in range(num_obj_targets):
                squeezed_local_map[self.local_map[e, -(1+obj), :, :].cpu().numpy() > 0.3] = self.args.num_sem_categories - (obj+1)
                
            #     mask_rgb = np.where(self.local_map[e, -(1+obj), :, :].cpu().numpy() > 0.3, 255, 0).astype(np.uint8)
            #     mask_rgb = np.stack([mask_rgb]*3, axis=-1)
            #     mask_rgb = np.flipud(mask_rgb)
            #     image = Image.fromarray(mask_rgb, 'RGB')
            #     image.save(f"/home/lm2/projects/Real2Chess_DINO/stupid_bug/{-(1+obj)}_rgb_{self.step}.png")
            
            p_input['sem_map_pred'] = squeezed_local_map
        return squeezed_local_map

    def store_and_record(self, dir="output"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        map_path = os.path.join(dir, "map.png")
        cv2.imwrite(map_path, self.sem_map_image_p)
    
    def store_and_record_4c(self, dir="output"):
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        map_path = os.path.join(dir, "map.png")
        cv2.imwrite(map_path, self.sem_map_image_p_4c)

    def visualize(self):
        inputs = self.planner_inputs[0].copy()
        map_shape = (self.args.map_size_cm // self.args.map_resolution, self.args.map_size_cm // self.args.map_resolution)
        collision_map = np.zeros(map_shape)
        visited = np.zeros(map_shape)

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            inputs['pose_pred']

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        # goal = inputs['goal']
        # steps = 0
        # goal_visualize = None
        # if steps <= 1:
        #     goal = inputs['goal']
        # else:
        #     goal = goal_visualize
        
        # sem_map: [240, 240], values: 0, 1, 2, 3, ..., 25
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        grid = np.rint(map_pred)
        explored = np.rint(exp_pred)

        # values: 5, 6, 7, 8, ..., 29
        sem_map += 5
        # value == 29, category: none
        no_cat_mask = sem_map == 5 + self.args.num_sem_categories - 1

        map_mask = np.rint(map_pred) == 1
        map_mask = np.logical_or(map_mask, collision_map == 1)
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = visited[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1
        
        # TODO: 如果不低头，map_pred 里面就是空的
        # mask_rgb = np.where(no_cat_mask == 1, 255, 0).astype(np.uint8)
        # mask_rgb = np.stack([mask_rgb]*3, axis=-1)
        # image = Image.fromarray(mask_rgb, 'RGB')
        # image.save("no_cat_mask_rgb.png")
        # mask_rgb = np.where(map_mask == 1, 255, 0).astype(np.uint8)
        # mask_rgb = np.stack([mask_rgb]*3, axis=-1)
        # image = Image.fromarray(mask_rgb, 'RGB')
        # image.save("map_mask_rgb.png")

        sem_map[vis_mask] = 3

        curr_mask = np.zeros(vis_mask.shape)
        selem = skimage.morphology.disk(2)
        curr_mask[start[0], start[1]] = 1
        curr_mask = 1 - skimage.morphology.binary_dilation(
            curr_mask, selem) != True
        curr_mask = curr_mask == 1
        sem_map[curr_mask] = 3

        # selem = skimage.morphology.disk(4)
        # goal_mat = 1 - skimage.morphology.binary_dilation(goal, selem) != True
        # # goal_mat = goal
        # goal_mask = goal_mat == 1
        # sem_map[goal_mask] = 4
        # print(sem_map.shape, sem_map.min(), sem_map.max())

        # self.print_log(sem_map.shape, sem_map.min(), sem_map.max())
        # self.print_log(vis_mask.shape)
        # sem_map = self.compress_sem_map(sem_map)

        semantic_img = Image.new("P", (sem_map.shape[1],
                                       sem_map.shape[0]))

        semantic_img.putpalette(self.color_palette)
        # semantic_img.putdata((sem_map.flatten() % 39).astype(np.uint8))
        semantic_img.putdata((sem_map.flatten()).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")

        semantic_img = np.flipud(semantic_img)
        
        # RGBA -> BGR
        self.sem_map_image = semantic_img[:, :, [2, 1, 0]]
        self.sem_map_image_p = self.sem_map_image
        self.sem_map_image_p_4c = semantic_img[:, :, [2, 1, 0, 3]]
        # Downscaling sem_map_image
        # ds = self.map_size // self.args.frame_width  # Downscaling factor
        res = transforms.Compose([transforms.ToPILImage(), transforms.Resize((self.args.frame_height, self.args.frame_width), interpolation=Image.NEAREST)])
        self.sem_map_image = np.asarray(res(self.sem_map_image.astype(np.uint8)))
        # print(self.sem_map_image.shape)

        if self.args.save_pictures:
            print('save semantic map...')
            cv2.imwrite("Sem_Map/" + "Sem_Map.png", semantic_img[:, :, [2, 1, 0, 3]])

    def visualize_w_small(self, save_path):
        inputs = copy.deepcopy(self.planner_inputs[0])
        # inputs = self.planner_inputs[0].copy()
        map_shape = (self.args.map_size_cm // self.args.map_resolution, self.args.map_size_cm // self.args.map_resolution)
        collision_map = np.zeros(map_shape)
        visited = np.zeros(map_shape)

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            inputs['pose_pred']

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        # goal = inputs['goal']
        # steps = 0
        # goal_visualize = None
        # if steps <= 1:
        #     goal = inputs['goal']
        # else:
        #     goal = goal_visualize
        
        # sem_map: [240, 240], values: 0, 1, 2, 3, ..., 25
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        grid = np.rint(map_pred)
        explored = np.rint(exp_pred)

        # values: 5, 6, 7, 8, ..., 29
        sem_map += 5
        # value == 29, category: none
        # no_cat_mask = sem_map == 5 + self.nlarge - 1
        no_cat_mask = sem_map == 5 + self.args.obj_list.index('None')

        map_mask = np.rint(map_pred) == 1
        map_mask = np.logical_or(map_mask, collision_map == 1)
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = visited[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1
        
        # mask_rgb = np.where(no_cat_mask == 1, 255, 0).astype(np.uint8)
        # mask_rgb = np.stack([mask_rgb]*3, axis=-1)
        # mask_rgb = np.flipud(mask_rgb)
        # image = Image.fromarray(mask_rgb, 'RGB')
        # image.save(f"/home/lm2/projects/Real2Chess_DINO/stupid_bug/no_cat_mask_rgb_{self.step}.png")
        mask_rgb = np.where(map_mask == 1, 255, 0).astype(np.uint8)
        mask_rgb = np.stack([mask_rgb]*3, axis=-1)
        mask_rgb = np.flipud(mask_rgb)
        image = Image.fromarray(mask_rgb, 'RGB')
        image.save(f"/home/lm2/projects/Real2Chess_DINO/stupid_bug/map_mask_rgb_{self.step}.png")
        print(self.step)
        self.step += 1

        sem_map[vis_mask] = 3

        curr_mask = np.zeros(vis_mask.shape)
        selem = skimage.morphology.disk(2)
        curr_mask[start[0], start[1]] = 1
        curr_mask = 1 - skimage.morphology.binary_dilation(
            curr_mask, selem) != True
        curr_mask = curr_mask == 1
        sem_map[curr_mask] = 3

        # selem = skimage.morphology.disk(4)
        # goal_mat = 1 - skimage.morphology.binary_dilation(goal, selem) != True
        # # goal_mat = goal
        # goal_mask = goal_mat == 1
        # sem_map[goal_mask] = 4
        # print(sem_map.shape, sem_map.min(), sem_map.max())

        # self.print_log(sem_map.shape, sem_map.min(), sem_map.max())
        # self.print_log(vis_mask.shape)
        # sem_map = self.compress_sem_map(sem_map)

        semantic_img = Image.new("P", (sem_map.shape[1],
                                       sem_map.shape[0]))

        semantic_img.putpalette(self.color_palette)
        # semantic_img.putdata((sem_map.flatten() % 39).astype(np.uint8))
        semantic_img.putdata((sem_map.flatten()).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")

        semantic_img = np.flipud(semantic_img)
        
        # RGBA -> BGR
        self.sem_map_image = semantic_img[:, :, [2, 1, 0]]
        self.sem_map_image_p = self.sem_map_image
        self.sem_map_image_p_4c = semantic_img[:, :, [2, 1, 0, 3]]
        # Downscaling sem_map_image
        # ds = self.map_size // self.args.frame_width  # Downscaling factor
        res = transforms.Compose([transforms.ToPILImage(), transforms.Resize((self.args.frame_height, self.args.frame_width), interpolation=Image.NEAREST)])
        self.sem_map_image = np.asarray(res(self.sem_map_image.astype(np.uint8)))
        # print(self.sem_map_image.shape)

        if self.args.save_pictures:
            # print('save semantic map...')
            cv2.imwrite(save_path, semantic_img[:, :, [2, 1, 0, 3]])

    def local_map_to_chessboard(self):
        ''' 
        generate and save abstract chess map
        
        target_size: the abstract chessboard size
        '''
        inputs = copy.deepcopy(self.planner_inputs[0])
        # inputs = self.planner_inputs[0].copy()
        map_shape = (self.args.map_size_cm // self.args.map_resolution, self.args.map_size_cm // self.args.map_resolution)
        visited = np.zeros(map_shape)
        
        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        sem_map = inputs['sem_map_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        
        # before values: 0, 1, 2, 3, ..., 24
        # values: 5, 6, 7, 8, ..., 29
        sem_map += 5
        # value == 29, category: none
        # no_cat_mask = sem_map == 5 + self.nlarge - 1
        no_cat_mask = sem_map == 5 + self.args.obj_list.index('None')
        
        
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        
        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2
        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1
        
        # 上面的内容是将 local_map 中的 map_pred 和 exp_pred 填入 sem_map 中
        
        chessboard = np.zeros((self.chess_size, self.chess_size))
        # chessboard info
        delta_x = int(sem_map.shape[0] / self.chess_size)
        delta_y = int(sem_map.shape[1] / self.chess_size)
        # print("sem_map.shape = ", sem_map.shape)
        # print("chessboard.shape = ", [self.chess_size, self.chess_size])
        # print("delta_x = ", delta_x)
        # print("delta_y = ", delta_y)
        
        for i in range(self.chess_size):
            for j in range(self.chess_size):

                # 得到image_array的相应区域
                sem_map_region = sem_map[i * delta_x:(i + 1) * delta_x, j * delta_y:(j + 1) * delta_y]

                # 找到最多的RGB颜色
                flattened_region = sem_map_region.reshape(-1)
                unique_idx, counts = np.unique(flattened_region, return_counts=True)
                
                # 使用 index 时，直接使用 np.where(condition) 即可
                # 如只有全空才记录白色或灰色
                if (len(unique_idx) == 1 and unique_idx[0] == 0) or\
                (len(unique_idx) == 1 and unique_idx[0] == 2) or\
                (len(unique_idx) == 2 and unique_idx[0] == 0 and unique_idx[1] == 2) or\
                (len(unique_idx) == 2 and unique_idx[0] == 2 and unique_idx[1] == 0):
                    most_common_index = np.argmax(counts)
                    most_common_color = unique_idx[most_common_index]
                # 否则移除白色和灰色
                else:
                    index_white = np.where(unique_idx == 0)
                    unique_idx = np.delete(unique_idx, index_white)
                    counts = np.delete(counts, index_white)

                    index_gray = np.where(unique_idx == 2)
                    unique_idx = np.delete(unique_idx, index_gray)
                    counts = np.delete(counts, index_gray)
            
                    # 从其他语义物体中选择最多的颜色
                    most_common_index = np.argmax(counts)
                    most_common_color = unique_idx[most_common_index]
                
                # 只要有小物体的像素，整个区块就都算作小物体
                tmp_category = None
                # tmp_count = 0
                # for k in range(len(unique_idx)):
                #     if unique_idx[k] > 5 + self.nlarge - 1 and counts[k] > tmp_count:
                #         tmp_category = unique_idx[k]
                #         tmp_count = counts[k]
                
                for k in range(len(unique_idx)):
                    if unique_idx[k] > 5 + self.args.obj_list.index('None'):
                        if tmp_category == None:
                            tmp_category = unique_idx[k]
                        elif unique_idx[k] < tmp_category:
                            tmp_category = unique_idx[k]
                
                if tmp_category is not None:
                    chessboard[i, j] = tmp_category
                else:
                    chessboard[i, j] = most_common_color

        chessboard[start[0] // delta_x, start[1] // delta_y] = 3
        
        self.chessboard_available_pos(chessboard, sem_map, start)
        
        # 在此之前的 sem_map 和 chessboard 都是 x 轴向上，y 轴向右
        # 最终的 chessboard 是 x 轴向下，y 轴向右，符合 Image 的坐标系
        if self.map_x_down:
            self.chessboard = np.flipud(chessboard).astype(np.int32)
        else:
            self.chessboard = chessboard.astype(np.int32)
        
        self.anti_collision_from_history()
        
        return self.chessboard
    
    def chessboard_visualize(self, save_path):
        chessboard_img = Image.new("P", (self.chessboard.shape[1],
                                       self.chessboard.shape[0]))

        chessboard_img.putpalette(self.color_palette)
        chessboard_img.putdata((self.chessboard.flatten()).astype(np.uint8))
        chessboard_img = np.array(chessboard_img.convert("RGBA"))

        # FIXME: x up/down
        if not self.map_x_down:
            chessboard_img = np.flipud(chessboard_img)

        # 将chessboard_img转换为PIL图像
        abstract_image = Image.fromarray(chessboard_img.astype(np.uint8))

        # 存储一份高清版
        resized_image = abstract_image.resize((240, 240), Image.NEAREST)

        draw = ImageDraw.Draw(resized_image)

        # 绘制网格线
        # 绘制垂直线
        for x in range(0, self.map_size+1, 5):
            line = ((x, 0), (x, self.map_size))
            draw.line(line, fill=128, width=1)
        # 绘制水平线
        for y in range(0, self.map_size+1, 5):
            line = ((0, y), (self.map_size, y))
            draw.line(line, fill=128, width=1)
        
        if self.args.save_pictures:
            # print('save chessboard_img...')
            # cv2.imwrite(save_path, chessboard_img[:, :, [2, 1, 0, 3]])
            resized_image.save(save_path)

    def chessboard_info(self, obj_list):
        """
        对有颜色区域划出bounding box，将其定位范围
        在范围内用字典记录所有颜色及其对应的坐标列表
        读取color2obj.json将颜色转换为物体名
        """
        obj_list = ['Unexplore', 'Obstacle', 'Explore', 'Agent', 'Goal'] + obj_list
        # 找出存在非白像素的最大矩形区域
        # 1. 找出不是白色像素的坐标
        white_index = np.where(self.chessboard == 0)

        # 创建一个全集
        blank_image = np.zeros((self.chess_size, self.chess_size),dtype=int)
        for idx, i in enumerate(white_index[0]):
            blank_image[i, white_index[1][idx]] = 1

        # 2. 用全集减去白色坐标得到非白色坐标
        non_white_index = np.where(blank_image == 0)
        
        # 4. 找出非白色坐标的最大矩形区域
        max_x = max(non_white_index[0])
        min_x = min(non_white_index[0])
        max_y = max(non_white_index[1])
        min_y = min(non_white_index[1])
        # print("max_x = ", max_x)
        # print("min_x = ", min_x)
        # print("max_y = ", max_y)
        # print("min_y = ", min_y)

        # 在该区域内统计所有颜色的坐标，表示为字典
        obj_dict = {}
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                # 将i, j转化为原图中的坐标
                # pos_i = min_x + i - 1 
                # pos_j = min_y + j - 1
                pos_i = i
                pos_j = j
                pos_i_record = pos_i + 1
                pos_j_record = pos_j + 1
                
                # FIXME: ()/[]?
                obj = obj_list[self.chessboard[pos_i, pos_j]]
                if obj in obj_dict:
                    obj_dict[obj].append([pos_i_record, pos_j_record])
                else:
                    obj_dict[obj] = [[pos_i_record, pos_j_record]]
                # FIXME: 翻转 xy for 汉炫，用于开源模型测试
                # if obj in obj_dict:
                #     obj_dict[obj].append((pos_j_record, pos_i_record))
                # else:
                #     obj_dict[obj] = [(pos_j_record, pos_i_record)]

        """
        先行后列，以左上角为原点
        """
        return obj_dict

    def chessboard_available_pos(self, chessboard, sem_map, start):
        delta_x = int(sem_map.shape[0] / self.chess_size)
        delta_y = int(sem_map.shape[1] / self.chess_size)
        
        sem_map_x = start[0]
        sem_map_y = start[1]
        chess_x = sem_map_x // delta_x
        chess_y = sem_map_y // delta_y
        
        available_pos = [True, True, True, True]
        
        forward_margin = 8 # 8cells = 40cm，前方空间
        side_margin = 2 # 2cells = 10cm，侧方空间，左右各10cm，自身所在位置占5cm
        margin = 3 # 2cells = 10cm
        
        # print(sem_map[sem_map_x - forward_margin:sem_map_x + forward_margin + 1, sem_map_y - forward_margin:sem_map_y + forward_margin + 1])
        # sem_map 和 chessboard 都是 x 轴向上，y 轴向右
        # 0
        if chess_y + 2 < self.chess_size and chessboard[chess_x, chess_y + 1] in [0, 2]:
            # sem_map_region = sem_map[chess_x * delta_x:(chess_x + 1) * delta_x, (chess_y + 1) * delta_y:(chess_y + 2) * delta_y + margin]
            sem_map_region = sem_map[sem_map_x - side_margin:sem_map_x + side_margin + 1, sem_map_y + 1:sem_map_y + forward_margin + 1]
            flattened_region = sem_map_region.reshape(-1)
            unique_idx, counts = np.unique(flattened_region, return_counts=True)
            map_content = [unique_idx[i] for i in range(len(unique_idx)) if unique_idx[i] not in [0, 2]]
            if len(map_content) > 0:
                available_pos[0] = False
        else:
            available_pos[0] = False
        
        # 90
        if chess_x + 2 < self.chess_size and chessboard[chess_x + 1, chess_y] in [0, 2]:
            # sem_map_region = sem_map[(chess_x + 1) * delta_x:(chess_x + 2) * delta_x + margin, chess_y * delta_y:(chess_y + 1) * delta_y]
            sem_map_region = sem_map[sem_map_x + 1:sem_map_x + forward_margin + 1, sem_map_y - side_margin:sem_map_y + side_margin + 1]
            flattened_region = sem_map_region.reshape(-1)
            unique_idx, counts = np.unique(flattened_region, return_counts=True)
            map_content = [unique_idx[i] for i in range(len(unique_idx)) if unique_idx[i] not in [0, 2]]
            if len(map_content) > 0:
                available_pos[1] = False
        else:
            available_pos[1] = False
        
        # 180
        if chess_y - 2 >= 0 and chessboard[chess_x, chess_y - 1] in [0, 2]:
            # sem_map_region = sem_map[chess_x * delta_x:(chess_x + 1) * delta_x, (chess_y - 1) * delta_y - margin:chess_y * delta_y]
            sem_map_region = sem_map[sem_map_x - side_margin:sem_map_x + side_margin + 1, sem_map_y - forward_margin:sem_map_y]
            flattened_region = sem_map_region.reshape(-1)
            unique_idx, counts = np.unique(flattened_region, return_counts=True)
            map_content = [unique_idx[i] for i in range(len(unique_idx)) if unique_idx[i] not in [0, 2]]
            if len(map_content) > 0:
                available_pos[2] = False
        else:
            available_pos[2] = False
        
        # 270
        if chess_x - 2 >= 0 and chessboard[chess_x - 1, chess_y] in [0, 2]:
            # sem_map_region = sem_map[(chess_x - 1) * delta_x - margin:chess_x * delta_x, chess_y * delta_y:(chess_y + 1) * delta_y]
            sem_map_region = sem_map[sem_map_x - forward_margin:sem_map_x, sem_map_y - side_margin:sem_map_y + side_margin + 1]
            flattened_region = sem_map_region.reshape(-1)
            unique_idx, counts = np.unique(flattened_region, return_counts=True)
            map_content = [unique_idx[i] for i in range(len(unique_idx)) if unique_idx[i] not in [0, 2]]
            if len(map_content) > 0:
                available_pos[3] = False
        else:
            available_pos[3] = False
        
        # 在 [0, 90, 180, 270] 度的方向上，是否有可行的位置
        self.available_pos_boolean = available_pos
        print('available_pos_boolean', self.available_pos_boolean)
    
    def anti_collision_from_history(self):
        for coll in self.collision:
            if self.chessboard[coll[0]-1,coll[1]-1] == 0 or self.chessboard[coll[0]-1,coll[1]-1] == 2:
                self.chessboard[coll[0]-1,coll[1]-1] = 1
    
    # def find_available_position_w_anti_collision_from_history(self, obj_dict):
    #     for coll in self.collision:
    #         if self.chessboard[coll[0],coll[1]] == 0 or self.chessboard[coll[0],coll[1]] == 2:
    #             self.chessboard[coll[0],coll[1]] = 1
        
    #     agent_pos = obj_dict['Agent'][0]
    #     agent_x, agent_y = agent_pos
        
    #     available_boolean = self.available_pos_boolean
    #     # from [0, 90, 180, 270] to [90, 270, 180, 0] 上下左右
    #     new_available_boolean = [available_boolean[1], available_boolean[3], available_boolean[2], available_boolean[0]]
    #     direction_candidate = [[agent_x-1, agent_y], [agent_x+1, agent_y], [agent_x, agent_y-1], [agent_x, agent_y+1]]
        
    #     selection_info = []
    #     for i in range(4):
    #         if new_available_boolean[i] and direction_candidate[i] not in self.collision:
    #             selection_info.append(direction_candidate[i])
    #     return selection_info
