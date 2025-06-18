import os
import re
import json
import numpy as np
import io
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')


class ChessVis():
    def __init__(self,
                chess_size=48,
                large_chessboard_size=640,
                x_down=True):
        self.chess_size = chess_size
        self.large_chessboard_size = large_chessboard_size
        color_palette = [1.0, 1.0, 1.0,  # white
                        0.6, 0.6, 0.6,  # obstacle
                        0.95, 0.95, 0.95,  # exp
                        0.96, 0.36, 0.26,  # agent
                        0.3647, 0.6784, 0.8863, # [93, 173, 226, 255] next position
                        # 0.12156862745098039, 0.47058823529411764, 0.7058823529411765,  # goal
                        0.2, 0.2, 0.2,  # goal
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
        color_palette_RGB = np.array(color_palette).reshape(-1, 3)
        self.color_palette = [int(x * 255.) for x in color_palette]
        self.color_palette_RGB = [[int(i * 255.) for i in x] for x in color_palette_RGB]
        
        self.history_traj = []
        
        self.vis_orientation = True
        self.vis_next_pos = True
        self.vis_history = True
        self.vis_subgoal = True
        
        self.x_down = x_down
    
    def reset_history_traj(self):
        self.history_traj = []
    
    def add_history_traj(self, pos):
        self.history_traj.append(pos)

    def agent_loc(self, x, y, ori):
        scale = self.large_chessboard_size // self.chess_size
        # FIXME: x up/down
        if self.x_down:
            left_top_x = (x - 1) * scale
        else:
            left_top_x = (self.chess_size - x) * scale
        left_top_y = (y - 1) * scale
        
        ori = int(ori)
        while ori < 0:
            ori += 360
        while ori >= 360:
            ori -= 360

        # for scale = 5
        # if ori == 0:
        #     return [[left_top_x + scale // 2, left_top_y + scale - 2], [left_top_x + scale // 2, left_top_y + scale - 1]]
        # elif ori == 90:
        #     return [[left_top_x + 0, left_top_y + scale // 2], [left_top_x + 1, left_top_y + scale // 2]]
        # elif ori == 180:
        #     return [[left_top_x + scale // 2, left_top_y + 0], [left_top_x + scale // 2, left_top_y + 1]]
        # elif ori == 270:
        #     return [[left_top_x + scale - 2, left_top_y + scale // 2], [left_top_x + scale - 1, left_top_y + scale // 2]]
        # for scale = 11
        if ori == 0:
            return [[left_top_x + scale // 2, left_top_y + scale - 2]]
        elif ori == 90:
            return [[left_top_x + 1, left_top_y + scale // 2]]
        elif ori == 180:
            return [[left_top_x + scale // 2, left_top_y + 1]]
        elif ori == 270:
            return [[left_top_x + scale - 2, left_top_y + scale // 2]]

    def postion2action(self, ori, last_pos, next_pos):
        """
        Input:
            last_pos: (x, y)
            next_pos: (x, y)
            ori: (0/90/180/270, int)
        Output:
            action_list(list):[action1, action2, ...]
        x-y coodinate: x is down, y is right
        """
        ori = int(ori)
        while ori < 0:
            ori += 360
        while ori >= 360:
            ori -= 360
        x, y = last_pos
        x_next, y_next = next_pos
        
        if ori == 0:
            if x_next == x and y_next > y:  # Modified here
                return ["MoveAhead"]
            elif x_next == x and y_next < y:  # Modified here
                return ["RotateRight", "RotateRight", "MoveAhead"]
            elif x_next > x and y_next == y:  # Modified here
                return ["RotateRight", "MoveAhead"]
            elif x_next < x and y_next == y:  # Modified here
                return ["RotateLeft", "MoveAhead"]
        elif ori == 90:
            if x_next == x and y_next > y:  # Modified here
                return ["RotateRight", "MoveAhead"]
            elif x_next == x and y_next < y:  # Modified here
                return ["RotateLeft", "MoveAhead"]
            elif x_next > x and y_next == y:  # Modified here
                return ["RotateRight", "RotateRight", "MoveAhead"]
            elif x_next < x and y_next == y:  # Modified here
                return ["MoveAhead"]
        elif ori == 180:
            if x_next == x and y_next > y:  # Modified here
                return ["RotateRight", "RotateRight", "MoveAhead"]
            elif x_next == x and y_next < y:  # Modified here
                return ["MoveAhead"]
            elif x_next > x and y_next == y:  # Modified here
                return ["RotateLeft", "MoveAhead"]
            elif x_next < x and y_next == y:  # Modified here
                return ["RotateRight", "MoveAhead"]
        elif ori == 270:
            if x_next == x and y_next > y:  # Modified here
                return ["RotateLeft", "MoveAhead"]
            elif x_next == x and y_next < y:  # Modified here
                return ["RotateRight", "MoveAhead"]
            elif x_next > x and y_next == y:  # Modified here
                return ["MoveAhead"]
            elif x_next < x and y_next == y:  # Modified here
                return ["RotateRight", "RotateRight", "MoveAhead"]
        else:
            print("Error: ori should be 0/90/180/270")
            return None
    
    def postion2action_x_up(self, ori, last_pos, next_pos):
        """
        Input:
            last_pos: (x, y)
            next_pos: (x, y)
            ori: (0/90/180/270, int)
        Output:
            action_list(list):[action1, action2, ...]
        x-y coodinate: x is up, y is right
        """
        ori = int(ori)
        while ori < 0:
            ori += 360
        while ori >= 360:
            ori -= 360
        x, y = last_pos
        x_next, y_next = next_pos
        
        if ori == 0:
            if x_next == x and y_next > y:  # Modified here
                return ["MoveAhead"]
            elif x_next == x and y_next < y:  # Modified here
                return ["RotateRight", "RotateRight", "MoveAhead"]
            elif x_next < x and y_next == y:  # Modified here
                return ["RotateRight", "MoveAhead"]
            elif x_next > x and y_next == y:  # Modified here
                return ["RotateLeft", "MoveAhead"]
        elif ori == 90:
            if x_next == x and y_next > y:  # Modified here
                return ["RotateRight", "MoveAhead"]
            elif x_next == x and y_next < y:  # Modified here
                return ["RotateLeft", "MoveAhead"]
            elif x_next < x and y_next == y:  # Modified here
                return ["RotateRight", "RotateRight", "MoveAhead"]
            elif x_next > x and y_next == y:  # Modified here
                return ["MoveAhead"]
        elif ori == 180:
            if x_next == x and y_next > y:  # Modified here
                return ["RotateRight", "RotateRight", "MoveAhead"]
            elif x_next == x and y_next < y:  # Modified here
                return ["MoveAhead"]
            elif x_next < x and y_next == y:  # Modified here
                return ["RotateLeft", "MoveAhead"]
            elif x_next > x and y_next == y:  # Modified here
                return ["RotateRight", "MoveAhead"]
        elif ori == 270:
            if x_next == x and y_next > y:  # Modified here
                return ["RotateLeft", "MoveAhead"]
            elif x_next == x and y_next < y:  # Modified here
                return ["RotateRight", "MoveAhead"]
            elif x_next < x and y_next == y:  # Modified here
                return ["MoveAhead"]
            elif x_next > x and y_next == y:  # Modified here
                return ["RotateRight", "RotateRight", "MoveAhead"]
        else:
            print("Error: ori should be 0/90/180/270")
            return None
    
    def visualize_color(self, color_values, labels):
        color_values = color_values[:len(labels)]
        # 创建调色盘的可视化
        fig, ax = plt.subplots(figsize=(6, 2))  # 调整图像大小
        for i, color in enumerate(color_values):
            color_float = [x / 255.0 for x in color]  # 转换颜色元组为浮点数列表
            # print(color_float)
            ax.add_patch(patches.Rectangle((i, 0), 1, 1, color=color_float))  # 添加矩形块
        x = [i for i in range(len(color_values))]
        
        plt.xticks(x, labels, rotation='vertical')
        ax.set_xlim(0, len(labels))  # 设置x轴范围
        ax.set_ylim(0, 1)  # 设置y轴范围
        ax.axis('on')  # 隐藏坐标轴
        plt.yticks([])
        plt.subplots_adjust(bottom=0.6)
        # plt.savefig('./visualize_sem_map/color_w_small.png', dpi=100)
        # plt.show()  # 显示调色盘
        buffer_ = io.BytesIO()
        plt.savefig(buffer_, format = 'png')
        dataPIL = Image.open(buffer_)
        # dataPIL = dataPIL.resize((250, 85), Image.NEAREST)
        color = np.asarray(dataPIL)
        color = color.transpose(1, 0, 2)[:, ::-1, :]
        
        plt.close()
        buffer_.close()
        return color
    
    def chessboard_visualize(self,
                            chessboard,
                            color,
                            pos_now,
                            pos_next=None,
                            orientation_now=None,
                            filename="chessboard.png"):
        #######################################################################
        #       600 * 800 图片大小，左侧为调色盘，右侧为 chessboard
        #       同时可视化 当前位置 和 预测的下一位置，位置中心用黑色标记
        #######################################################################
        chessboard_img = Image.new("P", (chessboard.shape[1],
                                        chessboard.shape[0]))
        chessboard_img.putpalette(self.color_palette)
        chessboard_img.putdata((chessboard.flatten()).astype(np.uint8))
        chessboard_img = np.array(chessboard_img.convert("RGBA"))
        # 这里不需要 图片上下翻转
        # FIXME: x up/down
        if not self.x_down:
            chessboard_img = np.flipud(chessboard_img)

        # 将chessboard_img转换为PIL图像
        abstract_image = Image.fromarray(chessboard_img.astype(np.uint8))
        # 存储一份高清版
        resized_image = abstract_image.resize((self.large_chessboard_size, self.large_chessboard_size), Image.NEAREST)
        resized_chessboard = np.array(resized_image)
        
        scale = self.large_chessboard_size // self.chess_size
        if self.vis_history:
            for i in range(len(self.history_traj)):
                old_x, old_y = self.history_traj[i]
                # for scale = 5
                # resized_chessboard[(48 - old_x) * scale + scale // 2, (old_y - 1) * scale + scale // 2] = [93, 173, 226, 255]
                # for scale = 11
                # FIXME: x up/down
                if self.x_down:
                    new_x = (old_x - 1) * scale + scale // 2
                else:
                    new_x = (self.chess_size - old_x) * scale + scale // 2
                new_y = (old_y - 1) * scale + scale // 2
                for m in [-1, 0, 1]:
                    for n in [-1, 0, 1]:
                        resized_chessboard[new_x + m, new_y + n] = [93, 173, 226, 255]
        
        # 当前位置
        x, y = pos_now
        # FIXME: x up/down
        if self.x_down:
            mid_x = (x - 1) * scale + scale // 2
        else:
            mid_x = (self.chess_size - x) * scale + scale // 2
        mid_y = (y - 1) * scale + scale // 2
        for m in [-1, 0, 1]:
            for n in [-1, 0, 1]:
                resized_chessboard[mid_x + m, mid_y + n] = [0, 0, 0, 255]
        
        if self.vis_orientation and orientation_now is not None:
            # 智能体朝向
            x, y = pos_now
            orientation = orientation_now
            agent_loc_list = self.agent_loc(x, y, orientation)
            for loc in agent_loc_list:
                # for scale = 5
                # resized_chessboard[loc[0], loc[1]] = [0, 0, 0, 255]
                # for scale = 11
                for m in [-1, 0, 1]:
                    for n in [-1, 0, 1]:
                        resized_chessboard[loc[0] + m, loc[1] + n] = [0, 0, 0, 255]
        
        if self.vis_next_pos and pos_next is not None and pos_next != pos_now:
            x, y = pos_next
            # FIXME: x up/down
            if self.x_down:
                mid_x = (x - 1) * scale + scale // 2
            else:
                mid_x = (self.chess_size - x) * scale + scale // 2
            mid_y = (y - 1) * scale + scale // 2
            for m in [-1, 0, 1]:
                for n in [-1, 0, 1]:
                    resized_chessboard[mid_x + m, mid_y + n] = [0, 0, 0, 255]

        # 调色盘
        # new_background = np.ones((600, 800, 4)) * 255
        # new_background[36:564, 236:764, :] = resized_chessboard
        # new_background[:, 30:230, :] = color
        height = max(600, self.large_chessboard_size)
        width = 230 + self.large_chessboard_size
        color_top = (height - 600) // 2
        chess_top = (height - self.large_chessboard_size) // 2
        new_background = np.ones((height, width, 4)) * 255
        new_background[chess_top:chess_top+self.large_chessboard_size, 230:width, :] = resized_chessboard
        new_background[color_top:color_top+600, 30:230, :] = color
        
        resized_image = Image.fromarray(new_background.astype(np.uint8))
        resized_image.save(filename)

    def infer_chessboard(self, count, chess_info, obj_list, pos_now, pos_next=None, orientation_now=None, subgoal=None, output_dir="output/chess"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        chess_list = ['Unexplore', 'Obstacle', 'Explore', 'Agent', 'NextPos', 'Goal'] + obj_list
        # subgoal_list = [obj.lower().replace(' ', '') for obj in chess_list]
        color = self.visualize_color(self.color_palette_RGB, chess_list)
        if self.vis_subgoal and subgoal is not None:
            subgoal = re.findall(r'\[(.*?)\]', subgoal)[0]
            # print(subgoal)
        
        # chessboard x 轴向下，y 轴向右，符合 Image 的坐标系
        # 同样假定 pos_now 和 pos_next 也是 x 轴向下，y 轴向右
        chessboard = np.zeros((self.chess_size, self.chess_size))
        for obj in chess_info.keys():
            for pos in chess_info[obj]:
                x, y = pos
                chessboard[x-1][y-1] = chess_list.index(obj)
                # 可视化 subgoal
                if self.vis_subgoal and subgoal is not None:
                    if obj.lower().replace(' ', '') == subgoal:
                        chessboard[x-1][y-1] = 5
        if self.vis_next_pos and pos_next is not None and pos_next != pos_now:
            chessboard[pos_next[0]-1][pos_next[1]-1] = 4

        output_file = os.path.join(output_dir, str(count)+".png")
        self.chessboard_visualize(chessboard, color, pos_now, pos_next, orientation_now, output_file)


if __name__ == "__main__":
    chess_vis = ChessVis(x_down=False)
    
    json_file = "/home/lm2/projects/LLaMA-Factory/data/inst_answer_test_data_v1_subset.jsonl"
    output_dir = "output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 单步调用
    with open(json_file, 'r') as file:
        # 逐行读取
        count = 0
        for line in file:
            # 解析每行的JSON内容
            data = json.loads(line.strip())
            
            chess_info = data["Chess_info"]
            obj_list = data["obj_list"]
            pos_now = data["Agent_Position"]
            pos_next = data["Next_Position"]
            orientation_now = 0
            print(data["Subgoal"])
            chess_vis.infer_chessboard(count, chess_info, obj_list, pos_now, pos_next, orientation_now, data["Subgoal"])

            count += 1
            # break
