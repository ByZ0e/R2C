from collections import Counter
import numpy as np

def agent_obj_rel(agent_loc, obj_loc):
    """
    chessboard coordinate system: x is down, y is right
    
    Args:
        agent_loc [x, y]: agent location in chessboard
        obj_loc_list [x1, y1]: one of target object location
    """
    x, y = agent_loc
    x_obj, y_obj = obj_loc
    dx = x_obj - x
    dy = y_obj - y
    if dx == 0 and dy == 0:
        return -1
    if dy > dx and dy >= -dx:
        return 0
    if dy < -dx and dy >= dx:
        return 90
    if dy < dx and dy <= -dx:
        return 180
    if dy > -dx and dy <= dx:
        return 270

def count_obj_pixels_around_agent(agent_loc, obj_loc_list):
    """
    chessboard coordinate system: x is down, y is right

    Args:
        agent_loc [x, y]: agent location in chessboard
        obj_loc_list [[x1, y1], [x2, y2], ]: target object locations
    """
    result = []
    for obj_loc in obj_loc_list:
        rel_ori = agent_obj_rel(agent_loc, obj_loc)
        result.append(rel_ori)
    around_pixels = Counter(result)
    print(around_pixels.most_common(1))
    return around_pixels.most_common(1)[0][0]

def get_nearest_obj_pixel_around_agent(agent_loc, obj_loc_list):
    """
    chessboard coordinate system: x is down, y is right

    Args:
        agent_loc [x, y]: agent location in chessboard
        obj_loc_list [[x1, y1], [x2, y2], ]: target object locations
    """
    x, y = agent_loc
    min = np.inf
    nearest_obj_pixel = None
    for obj_loc in obj_loc_list:
        x_obj, y_obj = obj_loc
        l1_dis = abs(x_obj - x) + abs(y_obj - y)
        if l1_dis < min:
            min = l1_dis
            nearest_obj_pixel = obj_loc
        
    rel_ori = agent_obj_rel(agent_loc, nearest_obj_pixel)
    return rel_ori

# agent_loc = [0, 0]
# obj_loc_list = [[-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1], [-1, -1], [0, -1], [1, -1], [0, -3]]
# result = count_obj_pixels_around_agent(agent_loc, obj_loc_list)
# print(result)

def rotate_to_target(agent_loc, agent_ori, obj_loc_list):
    """
    Args:
        agent_loc [x, y]: agent location in chessboard
        agent_ori: agent orientation in chessboard
        obj_loc_list [[x1, y1], [x2, y2], ]: target object locations
    """
    ori = int(agent_ori)
    while ori < 0:
        ori += 360
    while ori >= 360:
        ori -= 360
    
    # target_ori = count_obj_pixels_around_agent(agent_loc, obj_loc_list)
    target_ori = get_nearest_obj_pixel_around_agent(agent_loc, obj_loc_list)
    
    if target_ori == -1 or ori % 360 == target_ori:
        return []
    elif (ori + 90) % 360 == target_ori:
        return ["RotateLeft"]
    elif (ori - 90) % 360 == target_ori:
        return ["RotateRight"]
    elif (ori + 180) % 360 == target_ori:
        return ["RotateLeft", "RotateLeft"]
    else:
        print("Error: agent_ori and target_ori are not matched!!!")
        print(f"agent_ori: {agent_ori}, target_ori: {target_ori}")

# agent_loc = [0, 0]
# obj_loc_list = [[-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1], [-1, -1], [0, -1], [1, -1], [0, -3]]
# print(rotate_to_target(agent_loc, 0, obj_loc_list))
# print(rotate_to_target(agent_loc, 90, obj_loc_list))
# print(rotate_to_target(agent_loc, 180, obj_loc_list))
# print(rotate_to_target(agent_loc, 270, obj_loc_list))