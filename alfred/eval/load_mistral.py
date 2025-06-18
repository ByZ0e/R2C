from transformers import AutoModelForCausalLM, AutoTokenizer
import json 
from tqdm import tqdm
import os
import torch

prompt_templete = """On a 48*48 block chessboard, the rules of the game are as follows:
In the chessboard, there are the following explored objects: Obstacle: {}
The movement is forbidden on the object block.
You can only move 1 block at a time.
You are an agent at {}.
Your Task is {}.

First, analyze the current position and your target position.
Second, Please select your next position from {} and analyze all the above four options one by one.
Finally, use ### Next Positon: (x,y) to answer me which coordinate is your next position.
"""

def obj_descriptor(obj_dict):
    obj_info = ""
    # 遍历obj_dict字典，描述每个物体的位置
    for obj_name, obj_blocks in obj_dict.items():
        # 排除unexplore\explore\agent
        if obj_name not in ['unexplore', 'explore', 'agent', 'grid']:
            info = obj_name + ": " + str(obj_blocks) + "\n"
            obj_info += info

    return obj_info


def prompte_genertor(obj_dict, task):
    """
    Args:
        obj_dict (dict): chessboard object dictionary
        task (str): eg. "go to a desk"

    Returns:
        str: generated prompt
    """
    agent_pos = obj_dict['Agent'][0]
    agent_x, agent_y = agent_pos
    unexplore = obj_dict['Unexplore']
    explore = obj_dict['Explore']
    available_block = unexplore + explore
    # agent四周的位置
    direction_candidate = [(agent_x-1, agent_y), (agent_x+1, agent_y), (agent_x, agent_y-1), (agent_x, agent_y+1)]
    selection_info = []
    for i in direction_candidate:
        if i in available_block:
            selection_info.append(i)
            
    chess_info = obj_descriptor(obj_dict)
    prompt = prompt_templete.format(chess_info, agent_pos, task, selection_info)
    return prompt


def model_init(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def model_generate(model, tokenizer, prompt):
    input_prompt = f"<s>[INST] {prompt} [/INST]"
    model_inputs = tokenizer([input_prompt], return_tensors="pt").to("cuda")
    model.to("cuda")

    # 只输出[\INST]之后的内容
    generated_ids = model.generate(**model_inputs, 
                                   max_new_tokens=1024, 
                                   do_sample=True, 
                                   temperature=0.7,
                                   top_p=1)
    result = tokenizer.batch_decode(generated_ids)[0]
    result = result.split(prompt)[-1]
    result = result.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    return result


def get_next_postion(result):
    # 找到### Next Positon:内容，返回之后的坐标
    # 如果找不到，返回None
    next_position = None
    position_idx = result.find("### Next Position:")
    # 在position_idx之后找到()中的内容
    start_idx = result.find("(", position_idx)
    end_idx = result.find(")", position_idx)
    if start_idx != -1 and end_idx != -1:
        next_position = result[start_idx+1:end_idx]
        # 将字符串坐标转换为元组
        next_position = tuple(map(int, next_position.split(",")))
    return next_position


def predict_next_position(prompt, model, tokenizer):
    """
    Args:
        prompt (str): input prompt
        model (model): msitral model
        tokenizer (tokenizer): mistral tokenizer

    Returns:
        tuple: eg. (25, 26)
    """
    times = 0
    while True:
        result = model_generate(model, tokenizer, prompt)
        next_position = get_next_postion(result)
        if next_position != None:
            return next_position
        else:
            times += 1
            print(f"Can't find the next position, retrying {times} times...")


if __name__ == "__main__":
    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    model, tokenizer = model_init(model_path)
    # 循环测试
    while True:
        prompt = input("Please input the prompt: ")
        prompt = """On a 48*48 block chessboard, the rules of the game are as follows:
In the chessboard, there are the following explored objects: Obstacle: [(5, 18), (6, 18), (7, 18), (8, 18), (9, 18), (10, 18), (11, 18), (12, 18), (13, 18), (24, 18), (28, 18), (29, 18), (23, 19), (33, 19), (21, 20), (22, 20), (23, 20), (30, 20), (31, 20), (32, 20), (33, 20), (22, 21), (23, 21), (33, 21), (23, 22), (33, 22), (32, 23), (32, 26), (23, 27), (24, 27), (32, 27), (22, 28), (23, 28), (33, 28), (30, 29), (31, 29), (32, 29), (33, 29), (29, 31), (30, 31), (29, 32), (30, 32), (33, 38), (33, 39), (33, 40), (33, 41), (15, 42), (16, 42), (17, 42), (18, 42), (19, 42), (20, 42), (21, 42), (23, 42), (24, 42), (25, 42), (26, 42), (27, 42), (28, 42), (33, 42)]
Bathtub Basin: [(20, 18), (21, 18), (22, 18), (23, 18), (20, 19), (21, 19), (22, 19)]
Desk: [(25, 18), (24, 19), (25, 19), (26, 19), (27, 19), (28, 19), (24, 20), (25, 20), (26, 20), (27, 20), (28, 20), (25, 21), (26, 21), (27, 21)]
Floor Lamp: [(26, 18), (27, 18)]
Bed: [(15, 19), (16, 19), (17, 19), (15, 20), (16, 20), (17, 20), (18, 20), (19, 20), (15, 21), (16, 21), (17, 21), (18, 21), (19, 21), (20, 21), (21, 21), (16, 22), (17, 22), (18, 22), (19, 22), (20, 22), (21, 22), (22, 22), (15, 23), (16, 23), (17, 23), (18, 23), (19, 23), (20, 23), (21, 23), (22, 23), (23, 23), (15, 24), (16, 24), (17, 24), (18, 24), (19, 24), (20, 24), (21, 24), (22, 24), (23, 24), (16, 25), (17, 25), (18, 25), (19, 25), (20, 25), (21, 25), (22, 25), (23, 25), (15, 26), (16, 26), (17, 26), (18, 26), (19, 26), (20, 26), (21, 26), (22, 26), (23, 26), (15, 27), (16, 27), (17, 27), (18, 27), (19, 27), (20, 27), (21, 27), (22, 27), (15, 28), (16, 28), (17, 28), (18, 28), (19, 28), (20, 28), (21, 28)]
Cabinet: [(33, 23), (33, 24), (33, 25), (33, 26), (33, 27)]
Counter Top: [(5, 26), (6, 26), (4, 27), (5, 27), (6, 27), (4, 28), (5, 28), (6, 28), (4, 29), (5, 29), (6, 29), (4, 30), (5, 30), (6, 30), (4, 31), (5, 31), (6, 31), (4, 32), (6, 32), (4, 33), (6, 33), (4, 34), (6, 34), (4, 35), (6, 35)]
Alarm Clock: [(5, 32), (5, 33)]
Arm Chair: [(5, 34), (7, 34), (30, 34), (31, 34), (5, 35), (7, 35), (29, 35), (30, 35), (31, 35), (29, 36), (30, 36), (31, 36), (32, 36), (30, 37), (31, 37), (32, 37), (29, 38), (30, 38), (31, 38), (29, 39), (30, 39), (31, 39), (32, 39), (28, 40), (29, 40), (30, 40), (31, 40), (32, 40), (28, 41), (29, 41), (30, 41), (31, 41), (32, 41), (29, 42), (30, 42), (31, 42)]

The movement is forbidden on the object block.
You can only move 1 block at a time.

You are an agent at (25,25)
Your Task is to go to a desk.

First, analyze the current position and your target position.
Second, Please select your next position from [(25, 26), (25, 24), (24, 25), (26, 25)] and analyze all the above four options one by one.
Finally, use ### Next Positon: (x,y) to answer me which coordinate is your next position."""
        result = model_generate(model, tokenizer, prompt)
        print(f"\nAnswer: {result}")
        print("--------------------------------------------------------")
        print("Extract the next position:")
        next_position = get_next_postion(result)
        if next_position != None:
            print(f"Next Position: {next_position}")
        else:
            print("Next Position: None")
        print("--------------------------------------------------------")