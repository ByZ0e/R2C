import os
import json
import random
import time
import torch
import filelock
import torch.multiprocessing as mp
import glob
from tqdm import tqdm, trange
from termcolor import colored
from random import sample
# from alfred.data import AlfredDataset
from alfred import constants
from alfred.utils import eval_util, gen_util
from alfred.env.thor_env import ThorEnv

from keyboard_fb.sem_map_helper import SemMapHelper
# from segmentation_helper import SegmentationHelper
from arguments import get_args
from keyboard_fb.chessboard_vis.infer_vis import ChessVis
from alfred.eval.eval_task import evaluate_task
from alfred.eval.eval_subgoal import evaluate_subgoal

from transformers import AutoModelForCausalLM, AutoTokenizer


class EvalMaster(object):
    def __init__(self, args, data_dir, model_path, revision="main"):
        self.args = args
        self.data_dir = data_dir
        self.partitions = args.partitions
        self.json_name = 'traj_data.json'
        self.traj_list = self.read_data()
        self.revision = revision

        self.env = ThorEnv(x_display=args.x_display)
        if not self.args.renderInstanceSegmentation:
            pass
            # self.seg_model = SegmentationHelper(self.args)
        else:
            self.seg_model = None

        # TODO: load model
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        

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
        # print(traj_list.index('train/look_at_obj_in_light-BaseballBat-None-DeskLamp-303/trial_T20190907_060429_471715/traj_data.json'))
        # print(traj_list.index('train/look_at_obj_in_light-Book-None-DeskLamp-302/trial_T20190909_085137_911990/traj_data.json'))
        # print(traj_list.index('train/pick_and_place_simple-DishSponge-None-Toilet-403/trial_T20190907_192722_315071/traj_data.json'))
        # print(traj_list.index('train/pick_two_obj_and_place-ToiletPaper-None-CounterTop-417/trial_T20190906_224516_486826/traj_data.json'))
        # print(traj_list.index('train/look_at_obj_in_light-Box-None-FloorLamp-205/trial_T20190906_211850_157561/traj_data.json'))
        # print(traj_list.index('train/pick_clean_then_place_in_recep-TomatoSliced-None-SideTable-3/trial_T20190908_232637_947235/traj_data.json'))
        # print(traj_list.index('train/pick_cool_then_place_in_recep-Bowl-None-Cabinet-25/trial_T20190908_073356_031859/traj_data.json'))
        # print(traj_list.index('train/look_at_obj_in_light-WateringCan-None-FloorLamp-223/trial_T20190906_205726_057580/traj_data.json'))
        # print(traj_list.index('train/pick_clean_then_place_in_recep-Fork-None-DiningTable-24/trial_T20190906_205216_940941/traj_data.json'))
        # print(traj_list.index('train/pick_heat_then_place_in_recep-Potato-None-SinkBasin-30/trial_T20190908_080346_586851/traj_data.json'))
        # print(traj_list.index('train/pick_and_place_simple-Cloth-None-Toilet-417/trial_T20190908_163202_959027/traj_data.json'))
        # print(traj_list.index('train/pick_clean_then_place_in_recep-Ladle-None-Drawer-16/trial_T20190909_024333_182094/traj_data.json'))
        # print(traj_list.index('valid_seen/pick_two_obj_and_place-KeyChain-None-Sofa-225/trial_T20190907_143000_904569/traj_data.json'))
        # print(traj_list.index('valid_seen/pick_two_obj_and_place-RemoteControl-None-Ottoman-203/trial_T20190906_182952_858723/traj_data.json'))
        # print(traj_list.index('valid_unseen/look_at_obj_in_light-CellPhone-None-FloorLamp-219/trial_T20190908_044113_026049/traj_data.json'))
        # print(traj_list.index('valid_unseen/pick_two_obj_and_place-SoapBar-None-Cabinet-424/trial_T20190909_081746_857594/traj_data.json'))
        # print(traj_list.index('valid_unseen/look_at_obj_in_light-CellPhone-None-FloorLamp-219/trial_T20190908_044139_261907/traj_data.json'))
        # print(traj_list.index('valid_seen/pick_and_place_simple-Vase-None-CoffeeTable-207/trial_T20190909_091246_807206/traj_data.json'))
        return traj_list

    def eval(self):
        # FIXME: random or index?
        if self.args.randomEval:
            eval_list = sample(self.traj_list, 100)
        else:
            eval_list = self.traj_list[self.args.start_idx:self.args.end_idx]
        for json_path in tqdm(eval_list):
            with open(os.path.join(self.data_dir, json_path)) as json_file:
                json_dict = json.load(json_file)
            trial_dir = json_path.replace(self.json_name, '')
            print(trial_dir)
            self.trial_dir = trial_dir
            self.json_dict = json_dict

            obj_list = gen_util.get_obj_list(self.json_dict)
            if not self.args.renderInstanceSegmentation:
                self.seg_model.obj_classes = obj_list
                print(self.seg_model.obj_classes)
            self.args.obj_list = obj_list
            self.args.num_sem_categories = len(obj_list)
            print('input obj_list', self.args.obj_list)
            
            self.sem_map_module = SemMapHelper(self.args, map_x_down=True)
            if self.args.eval == "task":
                metrics = evaluate_task(self.env, self.model, self.tokenizer, self.trial_dir, self.json_dict, self.args, self.seg_model, self.sem_map_module)
            elif self.args.eval == "subgoal":
                metrics = evaluate_subgoal(self.env, self.model, self.tokenizer, self.trial_dir, self.json_dict, self.args, self.seg_model, self.sem_map_module)
            else:
                print('eval function not implemented')

if __name__ == '__main__':
    args = get_args()
    args.device = torch.device("cuda:1")
    args.obj_list = ['Sink Basin', 'Arm Chair', 'Bathtub Basin', 'Bed', 'Cabinet', 'Cart', 'Coffee Machine', 'Coffee Table',
                                    'Counter Top', 'Desk', 'Dining Table', 'Drawer', 'Dresser', 'Fridge', 'Garbage Can',
                                    'Microwave', 'Ottoman', 'Safe', 'Shelf', 'Side Table', 'Sofa',
                                    'Stove Burner', 'TV Stand', 'Toilet', 'Faucet', 'Desk Lamp', 'Floor Lamp', 'None']  # 28
    args.num_sem_categories = 28            # Grounding SAM 输出 23+1+1+1 类 - ButterKnife + 'Faucet', 'Desk Lamp', 'Floor Lamp'
    args.num_processes = 1                  # 单线程

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

    args.x_display = 0
    args.partitions = ['train', 'valid_seen', 'valid_unseen']
    # args.start_idx = 7000
    # args.end_idx = 7080
    args.start_idx = 0
    args.end_idx = 1
    
    args.max_steps = 1000
    args.debug = True
    args.smooth_nav = True
    data = EvalMaster(args, data_dir=constants.ET_DATA, model_path="vipl-vrc/real2chess-0411")
