
"""
智能体视野范围，边长为 args.vision_range * args.map_resolution 长的正方形，单位为 cm，智能体在底边中点
      _ _ _ _ _
    |           |
    |           |
    |           |
    |           |
    | _ _ ^ _ _ |
          ^
"""

import skimage
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np

from utils.model import get_grid, ChannelPool, Flatten, NNBase
import envs.utils.depth_utils as du


class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = torch.device("cuda:0")
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm//args.global_downscaling
        self.n_channels = 3 # TODO: add argument
        self.vision_range = args.vision_range   # diameter of local map region visible by the agent (in cells), 100
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale           # frame downscaling before projecting to point cloud, 1
        self.print_time = args.print_time
        self.cat_pred_threshold = args.cat_pred_threshold   # number of depth points to be in bin to classify it as a certain semantic category, 5
        self.exp_pred_threshold = args.exp_pred_threshold   # number of depth points to be in bin to consider it as explored, 1
        self.map_pred_threshold = args.map_pred_threshold   # number of depth points to be in bin to consider it as obstacle, 1
        self.num_sem_categories = args.num_sem_categories

        self.no_straight_obs = args.no_straight_obs
        self.view_angles = [0.0]*args.num_processes

        self.max_height = int(360 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height*100.
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi/2.0]
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        # self.init_grid = torch.zeros(args.num_processes, 1 + self.num_sem_categories, vr, vr,
        #                         self.max_height - self.min_height).float().to(self.device)
        # self.feat = torch.ones(args.num_processes, 1 + self.num_sem_categories,
        #                   self.screen_h//self.du_scale * self.screen_w//self.du_scale
        #                  ).float().to(self.device)
        self.init_grid = torch.zeros(args.num_processes, 1 + self.num_sem_categories, vr, vr,
                                self.max_height - self.min_height).float().cuda()
        self.feat = torch.ones(args.num_processes, 1 + self.num_sem_categories,
                          self.screen_h//self.du_scale * self.screen_w//self.du_scale
                         ).float().cuda()
        
        # self.mask: [bs, 1, 240, 240]
        # self.mask = torch.ones(args.num_processes, 1, 
        #                         self.map_size_cm//self.resolution,
        #                         self.map_size_cm//self.resolution
        #                         ).float().to(self.device)
        self.mask = torch.ones(args.num_processes, 1, 
                                self.map_size_cm//self.resolution,
                                self.map_size_cm//self.resolution
                                ).float().cuda()
        x1 = self.map_size_cm//(self.resolution * 2) - self.vision_range//2 # 70
        x2 = x1 + self.vision_range # 170
        y1 = self.map_size_cm//(self.resolution * 2) # 120
        y2 = y1 + self.vision_range # 220
        
        # FIXME: update the mask strategy
        self.mask[:, :, y1:y2, x1:x2] = 0.0
        # for x in range(x1, x2):
        #     for y in range(y1, y2):
        #         if y - y1 >= int(np.tan(np.deg2rad((180 - self.fov) / 2.)) * abs(x - (x1 + x2 - 1) / 2.)):
        #             self.mask[:, :, y, x] = 0.0
        # print("size:", self.map_size_cm//self.resolution)
        # print("fov:", self.fov)
        # print(x1, x2, y1, y2)
        
        # torch.set_printoptions(threshold=np.inf)
        # torch.set_printoptions(linewidth=1000)
        # print(self.mask)
        
    def set_view_angles(self, view_angles):
        self.view_angles = [-view_angle for view_angle in view_angles]

    def forward(self, obs, pose_obs, maps_last, poses_last, info, build_maps=True, no_update=False):
        # TODO: 
        # input:
        # holding_state (int): 0, not holding; 1, start holding; 2, holding; 3, end holding
        # holding_obj (int): object index in obs
        holding_state = info['holding_state']
        holding_obj = info['holding_obj']
        holding_large_obj = info['holding_box']
        holding_obj_with_hole = info['holding_obj_with_hole']
        
        # obs: [bs, RGBD + categories, args.frame_height, args.frame_width]
        bs, c, h, w = obs.size()
        # print("Semantic Mapping ### obs size", obs.size())
        depth = obs[:,3,:,:]
        
        # FIXME:
        if holding_state in [1, 2]:
            mask_err_below = depth < 50.0
            if holding_large_obj:
                mask_err_below = depth < 70.0
        else:
            mask_err_below = depth < 0.0
        depth[mask_err_below] = 10000.0
        
        # TODO: 如果后续 mask 错误非常离谱，扩大 mask 就无法覆盖孔洞了
        # 部分物体有孔洞，其中的 depth 不准确，如果持有物体的 mask 同样不准确，需要将 mask 扩大一点
        if holding_state in [1, 2] and holding_obj_with_hole:
            obs_device = obs.device
            obj_seg = obs[0, holding_obj, :, :].cpu().numpy()
            selem = skimage.morphology.disk(2)
            obj_seg = skimage.morphology.binary_dilation(obj_seg, selem) * 0.9
            obj_seg = torch.tensor(obj_seg, device=obs_device)
            obs[0, holding_obj, :, :] = obj_seg
        
        # camera_matrix
        # Namespace(xc=74.5, zc=74.5, f=75.00000000000001)
        
        # [bs, args.frame_height, args.frame_width, 3]
        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        
        # [bs, args.frame_height, args.frame_width, 3]
        agent_view_t = du.transform_camera_view_t_multiple(point_cloud_t, self.agent_height, self.view_angles, self.device)
        # Z 轴 + 智能体高度
        
        # [bs, args.frame_height, args.frame_width, 3]
        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)
        # X 轴平移 vision_range / 2，便于与 Y 轴的归一化同时进行，如果不平移需要单独归一化
        
        # 线性归一化，此时还未进行超出 vision_range 范围的阶段
        # X is positive going right，以智能体为中心，左右各能看到 vision_range / 2 个 cell
        # Y is positive into the image，智能体向前能看到 vision_range 个 cell
        # Z is positive up in the image，在高度上，智能体能看到 min_h 到 max_h 之间的 cell
        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        # [bs, args.frame_height, args.frame_width, 3]
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] / xy_resolution)
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] - vision_range//2.)/vision_range*2.
        XYZ_cm_std[...,2] = XYZ_cm_std[...,2] / z_resolution
        XYZ_cm_std[...,2] = (XYZ_cm_std[...,2] -
                             (max_h+min_h)//2.)/(max_h-min_h)*2.
        
        feat = self.feat.clone()
        # print("Semantic Mapping ### self.feat size", feat.size())
        # [bs, categories+1, frame_height * frame_width]
        feat[:,1:,:] = nn.AvgPool2d(self.du_scale)(obs[:,4:,:,:]
                        ).view(bs, c-4, h//self.du_scale * w//self.du_scale)
        # feat[:,1:,:] = feat[:,1:,:] * 0.0

        XYZ_cm_std = XYZ_cm_std.permute(0,3,1,2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])
        # print("Semantic Mapping ### XYZ_cm_std size", XYZ_cm_std.size())
        # [bs, 3, args.frame_height * args.frame_width]
        
        if holding_state in [1, 2]:
            # self.feat 和 XYZ_cm_std 都要在最后的 frame_height * frame_width 维度上删减 小物体部分点云
            # obj_mask = np.zeros((h * w), dtype=int)
            # print("Semantic Mapping ### Holding!!! ", holding_obj)
            obj_mask = obs[0, holding_obj, :, :].cpu().detach().numpy()
            obj_mask = (obj_mask > 0.0).astype(int).flatten()
            indices_to_keep = np.where(obj_mask == 0)[0]
            
            # final dimension of self.feat and XYZ_cm_std: [:, :, frame_height * frame_width - len(holding_obj_mask)]
            feat = feat[:, :, indices_to_keep]
            XYZ_cm_std = XYZ_cm_std[:, :, indices_to_keep]
            # print("Semantic Mapping ### self.feat size after remove holding obj", feat.size())
            # print("Semantic Mapping ### XYZ_cm_std size after remove holding obj", XYZ_cm_std.size())
        
        # Input:
        # init_grid: [bs, categories+1, 100, 100, 80]
        # feat: [bs, categories+1, frame_height * frame_width]
        # XYZ_cm_std: [bs, 3, frame_height * frame_width]
        # Output:
        # [bs, categories+1, 100, 100, 80] 5m*5m*4m
        voxels = du.splat_feat_nd(self.init_grid*0., feat, XYZ_cm_std).transpose(2,3)
        
        # 整个 voxels 在 Z 轴上范围是 [-8, 72]（单位：cell）
        # FIXME: ALFRED 5, Navigation 10
        min_z = int(15 / z_resolution - min_h)
        # 地面上一个 cell 的高度是 5cm，避免把地面上的点也算进去
        max_z = int((self.agent_height + 1 + 50)/z_resolution - min_h)
        # FIXME: 智能体高度约 1.575m，可以加上一定的余量
        # min_z 到 max_z 之间的区间可以看作智能体在空间中的高度范围，需要关注这一区间内的障碍物
        # 语义地图也使用这一区间内的数据进行构建，可以看作智能体的可见范围，如果高度超过智能体 50cm，可以认为不可操作

        # 沿 Z 轴压缩成 2D map，[bs, categories+1, 100, 100]
        # print('zzzzzzz', min_z, max_z)
        agent_height_proj = voxels[...,min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        # fp_map_pred and fp_exp_pred: [bs, 1, 100, 100]
        fp_map_pred = agent_height_proj[:,0:1,:,:]
        fp_exp_pred = all_height_proj[:,0:1,:,:]
        fp_map_pred = fp_map_pred/self.map_pred_threshold
        fp_exp_pred = fp_exp_pred/self.exp_pred_threshold
        # torch.clamp：将输入张量每个元素的值压缩到区间 [min,max]
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        # c = RGBD + categories
        # agent_view: [bs, 4 + categories, 240, 240]
        # agent_view = torch.zeros(bs, c,
        #                          self.map_size_cm//self.resolution,
        #                          self.map_size_cm//self.resolution
        #                          ).to(self.device)
        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm//self.resolution,
                                 self.map_size_cm//self.resolution
                                 ).cuda()

        x1 = self.map_size_cm//(self.resolution * 2) - self.vision_range//2
        x2 = x1 + self.vision_range
        # FIXME:
        y1 = self.map_size_cm//(self.resolution * 2)
        # 对于大房间，向前看5m，边界处点云效果较差
        look_forward = self.vision_range - 4 # 96 cells = 480 cm
        # look_forward = self.vision_range # 100 cells = 500 cm
        y2 = y1 + look_forward
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred[:, :, 0:look_forward, :]
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred[:, :, 0:look_forward, :]
        agent_view[:, 4:, y1:y2, x1:x2] =  torch.clamp(
                        agent_height_proj[:,1:,:,:]/self.cat_pred_threshold,
                        min=0.0, max=1.0)[:, :, 0:look_forward, :]

        if self.cat_pred_threshold > 5.0:
            agent_view[:, 4:, y1:y2, x1:x2][np.where(agent_view[:, 4:, y1:y2, x1:x2].cpu().detach().numpy()< 0.5) ] = 0.0
        
        if no_update:
            agent_view = torch.zeros(agent_view.shape).to(device = self.device)
        corrected_pose = pose_obs


        def get_new_pose_batch(pose, rel_pose_change):

            pose[:,1] += rel_pose_change[:,0] * \
                            torch.sin(pose[:,2]/57.29577951308232) \
                        + rel_pose_change[:,1] * \
                            torch.cos(pose[:,2]/57.29577951308232)
            pose[:,0] += rel_pose_change[:,0] * \
                            torch.cos(pose[:,2]/57.29577951308232) \
                        - rel_pose_change[:,1] * \
                            torch.sin(pose[:,2]/57.29577951308232)
            pose[:,2] += rel_pose_change[:,2]*57.29577951308232

            pose[:,2] = torch.fmod(pose[:,2]-180.0, 360.0)+180.0
            pose[:,2] = torch.fmod(pose[:,2]+180.0, 360.0)-180.0

            return pose
        ###################################################################
        # 当前 map 与历史 map 合并
        ###################################################################
        
        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()
        st_pose[:, :2] = - (st_pose[:, :2]
                                * 100.0/self.resolution
                                - self.map_size_cm//(self.resolution*2)) /\
                                (self.map_size_cm//(self.resolution*2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])
        
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                        self.device)

        if holding_state == 1:
            # agent_view: [bs, 4 + categories, 240, 240]
            history_view_mask = torch.ones_like(agent_view)
            history_view_mask[:, holding_obj, :, :] = self.mask
            history_view_mask = F.grid_sample(history_view_mask, rot_mat, align_corners=True)
            history_view_mask = F.grid_sample(history_view_mask, trans_mat, align_corners=True)
            # print(maps_last[:, 0, :, :].int())
            # print(history_view_mask[:, holding_obj, :, :].int())
            # input()
            
            maps_last = maps_last * history_view_mask
        
        # size: agent_view = rotated = translated
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2,1)

        # fp_map_pred: [bs, 1, 100, 100]
        # map_pred: [bs, 4 + categories, 240, 240], values: [0, 1]
        return fp_map_pred, map_pred, pose_pred, current_poses
    
    def forward_origion(self, obs, pose_obs, maps_last, poses_last, build_maps=True, no_update = False):
        # obs: [bs, RGBD + categories, args.frame_height, args.frame_width]
        bs, c, h, w = obs.size()
        depth = obs[:,3,:,:]
        
        # camera_matrix
        # Namespace(xc=74.5, zc=74.5, f=75.00000000000001)
        
        # [bs, args.frame_height, args.frame_width, 3]
        point_cloud_t = du.get_point_cloud_from_z_t(depth, self.camera_matrix, self.device, scale=self.du_scale)
        
        # [bs, args.frame_height, args.frame_width, 3]
        agent_view_t = du.transform_camera_view_t_multiple(point_cloud_t, self.agent_height, self.view_angles, self.device)
        # Z 轴 + 智能体高度
        
        # [bs, args.frame_height, args.frame_width, 3]
        agent_view_centered_t = du.transform_pose_t(agent_view_t, self.shift_loc, self.device)
        # X 轴平移 vision_range / 2，便于与 Y 轴的归一化同时进行，如果不平移需要单独归一化
        
        # 线性归一化，此时还未进行超出 vision_range 范围的阶段
        # X is positive going right，以智能体为中心，左右各能看到 vision_range / 2 个 cell
        # Y is positive into the image，智能体向前能看到 vision_range 个 cell
        # Z is positive up in the image，在高度上，智能体能看到 min_h 到 max_h 之间的 cell
        max_h = self.max_height
        min_h = self.min_height
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        vision_range = self.vision_range
        # [bs, args.frame_height, args.frame_width, 3]
        XYZ_cm_std = agent_view_centered_t.float()
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] / xy_resolution)
        XYZ_cm_std[...,:2] = (XYZ_cm_std[...,:2] - vision_range//2.)/vision_range*2.
        XYZ_cm_std[...,2] = XYZ_cm_std[...,2] / z_resolution
        XYZ_cm_std[...,2] = (XYZ_cm_std[...,2] -
                             (max_h+min_h)//2.)/(max_h-min_h)*2.
        
        # [bs, categories+1, frame_height * frame_width]
        self.feat[:,1:,:] = nn.AvgPool2d(self.du_scale)(obs[:,4:,:,:]
                        ).view(bs, c-4, h//self.du_scale * w//self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0,3,1,2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3])
        # [bs, 3, args.frame_height * args.frame_width]

        # Input:
        # init_grid: [bs, categories+1, 100, 100, 80]
        # feat: [bs, categories+1, frame_height * frame_width]
        # XYZ_cm_std: [bs, 3, frame_height * frame_width]
        # Output:
        # [bs, categories+1, 100, 100, 80] 5m*5m*4m
        voxels = du.splat_feat_nd(self.init_grid*0., self.feat, XYZ_cm_std).transpose(2,3)
        
        # 整个 voxels 在 Z 轴上范围是 [-8, 72]（单位：cell）
        min_z = int(10 / z_resolution - min_h)
        # 地面上一个 cell 的高度是 5cm，避免把地面上的点也算进去
        max_z = int((self.agent_height + 1 + 50)/z_resolution - min_h)
        print("agent height projection: ", min_z, max_z)
        # FIXME: 智能体高度约 1.575m，可以加上一定的余量
        # min_z 到 max_z 之间的区间可以看作智能体在空间中的高度范围，需要关注这一区间内的障碍物
        # 语义地图也使用这一区间内的数据进行构建，可以看作智能体的可见范围，如果高度超过智能体 50cm，可以认为不可操作

        # 沿 Z 轴压缩成 2D map，[bs, categories+1, 100, 100]
        agent_height_proj = voxels[...,min_z:max_z].sum(4)
        all_height_proj = voxels.sum(4)

        # fp_map_pred and fp_exp_pred: [bs, 1, 100, 100]
        fp_map_pred = agent_height_proj[:,0:1,:,:]
        fp_exp_pred = all_height_proj[:,0:1,:,:]
        fp_map_pred = fp_map_pred/self.map_pred_threshold
        fp_exp_pred = fp_exp_pred/self.exp_pred_threshold
        # torch.clamp：将输入张量每个元素的值压缩到区间 [min,max]
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)
        
        # from PIL import Image
        # map_pred = fp_map_pred.squeeze().squeeze().cpu().detach().numpy()
        # mask_rgb = np.where(map_pred > 0, 255, 0).astype(np.uint8)
        # mask_rgb = np.stack([mask_rgb]*3, axis=-1)
        # image = Image.fromarray(mask_rgb, 'RGB')
        # image.save("fp_map_pred.png")
        # exp_pred = fp_exp_pred.squeeze().squeeze().cpu().detach().numpy()
        # mask_rgb = np.where(exp_pred > 0, 255, 0).astype(np.uint8)
        # mask_rgb = np.stack([mask_rgb]*3, axis=-1)
        # image = Image.fromarray(mask_rgb, 'RGB')
        # image.save("fp_exp_pred.png")
        
        # # FIXME: 为什么平视状态，map_pred 要设置为 0.0
        # # Bug: 注释掉该行代码，否则会导致平视状态下，无法建出墙等障碍物
        # if self.no_straight_obs:
        #     print("@" * 20)
        #     for vi, va in enumerate(self.view_angles):
        #         print("view angle: ", va)
        #         if abs(va - 0) <= 5:
        #             fp_map_pred[vi, :, :, :] = 0.0

        pose_pred = poses_last

        # c = RGBD + categories
        # agent_view: [bs, 4 + categories, 240, 240]
        # agent_view = torch.zeros(bs, c,
        #                          self.map_size_cm//self.resolution,
        #                          self.map_size_cm//self.resolution
        #                          ).to(self.device)
        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm//self.resolution,
                                 self.map_size_cm//self.resolution
                                 ).cuda()

        x1 = self.map_size_cm//(self.resolution * 2) - self.vision_range//2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm//(self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, 4:, y1:y2, x1:x2] =  torch.clamp(
                        agent_height_proj[:,1:,:,:]/self.cat_pred_threshold,
                        min=0.0, max=1.0)

        if self.cat_pred_threshold > 5.0:
            agent_view[:, 4:, y1:y2, x1:x2][np.where(agent_view[:, 4:, y1:y2, x1:x2].cpu().detach().numpy()< 0.5) ] = 0.0
        
        if no_update:
            agent_view = torch.zeros(agent_view.shape).to(device = self.device)
        corrected_pose = pose_obs
        
        """
        语义地图坐标系
        观看语义地图图片时， x 轴正向代表图片中的向右， y 轴正向代表图片中的向上
        智能体初始化坐标为地图的中心，旋转角度为 0 ，面朝 x 轴正向，左侧为 y 轴正向，即 x' 和 x 轴方向一致， y' 和 y 轴方向一致
        在 AI2THOR 中的 2D 世界坐标系，和处理后的 2D 世界坐标系中，智能体初始旋转角度不一定为 0 ，受仿真环境自身坐标系影响，
        处理后的 2D 世界坐标系只是保证各个坐标轴和旋转方向和语义地图坐标系一致，
        智能体在语义地图坐标系中的坐标是通过将处理后的 2D 世界坐标系中的坐标变换 dx, dy, do 累加到语义地图坐标系初始坐标得到的。
            ^ y
            |
            |
            |
            |
            |       ^ y'
            |       |
            |       |
            |       A - -> x'
            |
            |
            0 - - - - - - - - - - - - - -> x
        """

        def get_new_pose_batch(pose, rel_pose_change):

            pose[:,1] += rel_pose_change[:,0] * \
                            torch.sin(pose[:,2]/57.29577951308232) \
                        + rel_pose_change[:,1] * \
                            torch.cos(pose[:,2]/57.29577951308232)
            pose[:,0] += rel_pose_change[:,0] * \
                            torch.cos(pose[:,2]/57.29577951308232) \
                        - rel_pose_change[:,1] * \
                            torch.sin(pose[:,2]/57.29577951308232)
            pose[:,2] += rel_pose_change[:,2]*57.29577951308232

            pose[:,2] = torch.fmod(pose[:,2]-180.0, 360.0)+180.0
            pose[:,2] = torch.fmod(pose[:,2]+180.0, 360.0)-180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        
        # 当前 map 与历史 map 合并
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                                * 100.0/self.resolution
                                - self.map_size_cm//(self.resolution*2)) /\
                                (self.map_size_cm//(self.resolution*2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                        self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2,1)

        # fp_map_pred: [bs, 1, 100, 100]
        # map_pred: [bs, 4 + categories, 240, 240], values: [0, 1] 函数调用实际只接收 map_pred
        return fp_map_pred, map_pred, pose_pred, current_poses


"""
TODO: 如何解决可移动物体，如何实现实例化
思路一：构建当前视角 mask ，可移动物体所在2D地图，利用 mask 将其置为 0，只显示当前视角可见的小物体
    问题 
        1. 物体会一直跟随智能体
        2. 可能会将视野中的第二个同类别小物体一起挖掉
        3. 只删除小物体类别层的语义不行，还要删除障碍物地图 map_pred 和 exp_pred 层的 mask 区域，会丢失较多信息

思路二：直接在 obs 中根据物体类别，根据正在交互的小物体类别的语义位置，直接去除小物体的点云，挖掉小物体
    问题
        1. 挖点云的过程中，可能会将视野中的第二个同类别小物体一起挖掉
        2. 智能体刚开始抓取物体的时候，只会挖掉手里的物体，原位置上的物体会一直保留在地图中
"""