import os
import json
import numpy as np
import pandas as pd

points_dict = {
    0: "C2 centroid",
    1: "C2 bottom left",
    2: "C2 bottom right",
    3: "C3 top left",
    4: "C3 top right",
    5: "C3 bottom left",
    6: "C3 bottom right",
    7: "C4 top left",
    8: "C4 top right",
    9: "C4 bottom left",
    10: "C4 bottom right",
    11: "C5 top left",
    12: "C5 top right",
    13: "C5 bottom left", 
    14: "C5 bottom right",
    15: "C6 top left",
    16: "C6 top right",
    17: "C6 bottom left",
    18: "C6 bottom right",
    19: "C7 top left",
    20: "C7 top right",
    21: "C7 bottom left",
    22: "C7 bottom right",
}

# CAL FUNCTIONS 1
def cal_dist_adj_row(points):
    # points shape: (23, 2)
    # Except 0 position, 1-2, 3-4, ... calculation.
    candidate = points[1:]
    dists = []
    for i in range(0, len(candidate), 2):
        p1 = candidate[i]
        p2 = candidate[i+1]
        dist = np.linalg.norm(p1-p2)
        dists.append(dist)

    return {
        'row2': dists[0],
        'row3s': dists[1],
        'row3x': dists[2],
        'row4s': dists[3],
        'row4x': dists[4],
        'row5s': dists[5],
        'row5x': dists[6],
        'row6s': dists[7],
        'row6x': dists[8],
        'row7s': dists[9],
        'row7x': dists[10]
    }

# CAL FUNCTIONS 2 TODO
def cal_dis_adj_col(points):
    # points shape: (23, 2)
    # Except 0 position, 1-3, 2-4, ... calculation.

    p1 = points[1]
    p2 = points[3]
    line23q = np.linalg.norm(p1-p2)

    p1 = points[3]
    p2 = points[5]
    line33q = np.linalg.norm(p1-p2)

    p1 = points[5]
    p2 = points[7]
    line34q = np.linalg.norm(p1-p2)

    p1 = points[7]
    p2 = points[9]
    line44q = np.linalg.norm(p1-p2)

    p1 = points[9]
    p2 = points[11]
    line45q = np.linalg.norm(p1-p2)

    p1 = points[11]
    p2 = points[13]
    line55q = np.linalg.norm(p1-p2)

    p1 = points[13]
    p2 = points[15]
    line56q = np.linalg.norm(p1-p2)

    p1 = points[15]
    p2 = points[17]
    line66q = np.linalg.norm(p1-p2)

    p1 = points[17]
    p2 = points[19]
    line67q = np.linalg.norm(p1-p2)

    p1 = points[19]
    p2 = points[21]
    line77q = np.linalg.norm(p1-p2)

    p1 = points[2]
    p2 = points[4]
    line23h = np.linalg.norm(p1-p2)

    p1 = points[4]
    p2 = points[6]
    line33h = np.linalg.norm(p1-p2)

    p1 = points[6]
    p2 = points[8]
    line34h = np.linalg.norm(p1-p2)

    p1 = points[8]
    p2 = points[10]
    line44h = np.linalg.norm(p1-p2)

    p1 = points[10]
    p2 = points[12]
    line45h = np.linalg.norm(p1-p2)

    p1 = points[12]
    p2 = points[14]
    line55h = np.linalg.norm(p1-p2)

    p1 = points[14]
    p2 = points[16]
    line56h = np.linalg.norm(p1-p2)

    p1 = points[16]
    p2 = points[18]
    line66h = np.linalg.norm(p1-p2)

    p1 = points[18]
    p2 = points[20]
    line67h = np.linalg.norm(p1-p2)

    p1 = points[20]
    p2 = points[22]
    line77h = np.linalg.norm(p1-p2)

    return {
        'line23q': line23q,
        'line33q': line33q,
        'line34q': line34q,
        'line44q': line44q,
        'line45q': line45q,
        'line55q': line55q,
        'line56q': line56q,
        'line66q': line66q,
        'line67q': line67q,
        'line77q': line77q,
        'line23h': line23h,
        'line33h': line33h,
        'line34h': line34h,
        'line44h': line44h,
        'line45h': line45h,
        'line55h': line55h,
        'line56h': line56h,
        'line66h': line66h,
        'line67h': line67h,
        'line77h': line77h
    }


def cal_angle(points):
    a = np.concatenate([points[0], [0]])
    b = np.concatenate([points[1], [0]])
    c = np.concatenate([points[2], [0]])
    d = np.concatenate([points[3], [0]])
    ab = b - a
    cd = d - c

    cross_product = np.cross(ab, cd)
    dot_product = np.dot(ab[:2], cd[:2])
    cos_angle = dot_product / (np.linalg.norm(ab[:2]) * np.linalg.norm(cd[:2]))
    sin_angle = cross_product[2] / \
        (np.linalg.norm(ab[:2]) * np.linalg.norm(cd[:2]))
    angle = np.arctan2(sin_angle, cos_angle)
    return -np.degrees(angle)

# CAL FUNCTIONS 3
def cal_angle_adj(points):
    # points shape: (23, 2)
    # Except 0, 21, 22 position, 1-2$3-4, 5-6$7-8, ... calculation.
    candidate = points[1:21]
    angles = []
    for i in range(0, len(candidate), 4):
        angle = cal_angle(candidate[i:i+4])
        angles.append(angle)

    return {
        'SA2-3': angles[0],
        'SA3-4': angles[1],
        'SA4-5': angles[2],
        'SA5-6': angles[3],
        'SA6-7': angles[4],
    }

# CAL FUNCTIONS 4
def cal_angle_not_adj(points):
    # 3-4 7-8 11-12 15-16
    # 9-10 13-14 17-18 21-22
    pair1 = [[3, 4], [7, 8], [11, 12], [15, 16]]
    pair2 = [[9, 10], [13, 14], [17, 18], [21, 22]]
    angles = []
    for p1_idx in range(len(pair1)):
        for p2_idx in range(p1_idx, len(pair2)):
            pp1 = points[pair1[p1_idx][0]]
            pp2 = points[pair1[p1_idx][1]]
            pp3 = points[pair2[p2_idx][0]]
            pp4 = points[pair2[p2_idx][1]]
            p4 = np.array([pp1, pp2, pp3, pp4])
            angle = cal_angle(p4)
            angles.append(angle)

    return {
        'FSU3-4': angles[0],
        'FSU3-5': angles[1],
        'FSU3-6': angles[2],
        'FSU3-7': angles[3],
        'FSU4-5': angles[4],
        'FSU4-6': angles[5],
        'FSU4-7': angles[6],
        'FSU5-6': angles[7],
        'FSU5-7': angles[8],
        'FSU6-7': angles[9]
    }

# CAL FUNCTIONS 5
def cal_sva(points):
    # 1-3 5-7 9-11 13-15 17-19
    # 2-4 6-8 10-12 14-16 18-20

    pair = [[1, 3], [5, 7], [9, 11], [13, 15], [17, 19],
            [2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    svas = []
    for p in pair:
        p1 = points[p[0]]
        p2 = points[p[1]]
        sva = p1[0] - p2[0]
        svas.append(sva)

    return {
        'SVA-2F': svas[0],
        'SVA-3F': svas[1],
        'SVA-4F': svas[2],
        'SVA-5F': svas[3],
        'SVA-6F': svas[4],
        'SVA-2B': svas[5],
        'SVA-3B': svas[6],
        'SVA-4B': svas[7],
        'SVA-5B': svas[8],
        'SVA-6B': svas[9]
    }

# CAL FUNCTIONS 6
def cal_c_type(points):
    # C2-7Cobb
    p1 = points[1]
    p2 = points[2]
    p3 = points[21]
    p4 = points[22]
    c2_7cobb = cal_angle(np.array([p1, p2, p3, p4]))

    # C2-6Cobb
    p1 = points[1]
    p2 = points[2]
    p3 = points[17]
    p4 = points[18]
    c2_6cobb = cal_angle(np.array([p1, p2, p3, p4]))

    # sva
    p1 = points[0]
    p2 = points[20]
    sva = p2[0] - p1[0]

    # cci
    p1 = points[2]
    p2 = points[22]
    A = np.linalg.norm(p1-p2)
    a1 = cal_dist_point_line(points[6], p1, p2)
    a2 = cal_dist_point_line(points[10], p1, p2)
    a3 = cal_dist_point_line(points[14], p1, p2)
    a4 = cal_dist_point_line(points[18], p1, p2)
    cci = (a1+a2+a3+a4) / (A) * 100

    # ccl
    mid_2 = np.array([points[1], points[2]]).mean(axis=0)  # 计算点points[1]和points[2]的中点
    mid_3 = cal_intersection_points(points[3:7])  # 计算点points[3]到points[6]的交点
    mid_6 = cal_intersection_points(points[15:19])  # 计算点points[15]到points[18]的交点
    mid_7 = np.array([points[21], points[22]]).mean(axis=0)  # 计算点points[21]和points[22]的中点
    ccl_angle = abs(cal_angle(np.array([mid_3, mid_2, mid_6, mid_7])))  # 计算由四个中点形成的角度，确保这个角度是正数

    # 新的逻辑来决定position的值
    line_direction = mid_7 - mid_6  # 从mid_6到mid_7的向量，代表无限延长的线
    mid23_vector = mid_3 - mid_2  # 从mid_2到mid_3的向量
    cross_product = np.cross(np.append(line_direction, 0), np.append(mid23_vector, 0))

    # 使用叉积结果判断位置
    if cross_product[-1] > 0:
        position = 1  # mid_2和mid_3的连线在mid_6和mid_7的延长线的右侧
    elif cross_product[-1] < 0:
        position = -1  # mid_2和mid_3的连线在mid_6和mid_7的延长线的左侧
    else:
        position = 0  # 三点共线，这意味着mid_2和mid_3的连线与mid_6和mid_7的延长线重合

    # 根据position的值计算CCL
    if position == 1:
        ccl = 180 - ccl_angle  # 如果mid_2和mid_3的连线在mid_6和mid_7的延长线的右侧，ccl是180度减去角度
    elif position == -1:
        ccl = (180 + ccl_angle) % 360  # 如果mid_2和mid_3的连线在mid_6和mid_7的延长线的左侧，ccl是180度加上角度再取360度的模
    else:
        ccl = 0  # 如果mid_2和mid_3的连线与mid_6和mid_7的延长线重合，ccl为0



    return {
        'C2-7Cobb': c2_7cobb,
        'C2-6Cobb': c2_6cobb,
        'SVA': sva,
        'CCI': cci,
        'CCL': ccl
    }

# CAL FUNCTIONS 7
def cal_cns(points):
    # c2s
    p1 = points[1]
    p2 = points[2]
    p3 = points[1]
    p4 = np.array([p2[0], p1[1]])
    c2s = cal_angle(np.array([p1, p2, p3, p4]))

    # c3s
    p1 = points[3]
    p2 = points[4]
    p3 = points[3]
    p4 = np.array([p2[0], p1[1]])
    c3s = cal_angle(np.array([p1, p2, p3, p4]))

    # c4s
    p1 = points[7]
    p2 = points[8]
    p3 = points[7]
    p4 = np.array([p2[0], p1[1]])
    c4s = cal_angle(np.array([p1, p2, p3, p4]))

    # c5s
    p1 = points[11]
    p2 = points[12]
    p3 = points[11]
    p4 = np.array([p2[0], p1[1]])
    c5s = cal_angle(np.array([p1, p2, p3, p4]))

    # c6s
    p1 = points[15]
    p2 = points[16]
    p3 = points[15]
    p4 = np.array([p2[0], p1[1]])
    c6s = cal_angle(np.array([p1, p2, p3, p4]))

    # c7s
    p1 = points[19]
    p2 = points[20]
    p3 = points[19]
    p4 = np.array([p2[0], p1[1]])
    c7s = cal_angle(np.array([p1, p2, p3, p4]))

    return {
        'C2S': -c2s,
        'C3S': -c3s,
        'C4S': -c4s,
        'C5S': -c5s,
        'C6S': -c6s,
        'C7S': -c7s
    }

# CAL FUNCTIONS 8
def cal_cn(points):
    # c3
    p1 = points[3]
    p2 = points[4]
    p3 = points[5]
    p4 = points[6]
    c3 = cal_angle(np.array([p1, p2, p3, p4]))

    # c4
    p1 = points[7]
    p2 = points[8]
    p3 = points[9]
    p4 = points[10]
    c4 = cal_angle(np.array([p1, p2, p3, p4]))

    # c5
    p1 = points[11]
    p2 = points[12]
    p3 = points[13]
    p4 = points[14]
    c5 = cal_angle(np.array([p1, p2, p3, p4]))

    # c6
    p1 = points[15]
    p2 = points[16]
    p3 = points[17]
    p4 = points[18]
    c6 = cal_angle(np.array([p1, p2, p3, p4]))

    # c7
    p1 = points[19]
    p2 = points[20]
    p3 = points[21]
    p4 = points[22]
    c7 = cal_angle(np.array([p1, p2, p3, p4]))

    return {
        'VBA3': c3,
        'VBA4': c4,
        'VBA5': c5,
        'VBA6': c6,
        'VBA7': c7
    }

def cal_intersection_points(points):
    a = points[0]
    b = points[1]
    c = points[2]
    d = points[3]

    matrix = np.array([d - a, b - c]).T
    vector = b - a
    t, s = np.linalg.solve(matrix, vector)
    intersection = a + t*(d-a)
    return intersection

def point_position_relative_to_line(a, b, c):
    cross_product = (a[0] - b[0]) * (c[1] - b[1]) - (a[1] - b[1]) * (c[0] - b[0])

    if cross_product > 0:
        return 1 # right
    elif cross_product < 0:
        return -1 # left
    else:
        return 0 # on the line


def cal_dist_point_line(a, b, c):
    bc_y = b[1] - c[1]
    cb_x = c[0] - b[0]
    bccb = b[0]*c[1] - c[0]*b[1]

    dist = np.abs(bc_y*a[0] + cb_x*a[1] + bccb) / np.sqrt(bc_y**2 + cb_x**2)
    cross_product = (a[0] - b[0]) * (c[1] - b[1]) - (a[1] - b[1]) * (c[0] - b[0])
    if cross_product > 0:
        dist = -dist
    return dist

# CAL FUNCTIONS 9
def cal_toyama(points):
    mid_2 = np.array([points[1], points[2]]).mean(axis=0)
    mid_7 = np.array([points[21], points[22]]).mean(axis=0)

    intersection3 = cal_intersection_points(points[3:7])
    toyama3 = cal_dist_point_line(intersection3, mid_2, mid_7)

    intersection4 = cal_intersection_points(points[7:11])
    toyama4 = cal_dist_point_line(intersection4, mid_2, mid_7)

    intersection5 = cal_intersection_points(points[11:15])
    toyama5 = cal_dist_point_line(intersection5, mid_2, mid_7)

    intersection6 = cal_intersection_points(points[15:19])
    toyama6 = cal_dist_point_line(intersection6, mid_2, mid_7)

    intersection7 = cal_intersection_points(points[19:23])
    toyama7 = cal_dist_point_line(intersection7, mid_2, mid_7)

    return {
        'Toyama3': toyama3,
        'Toyama4': toyama4,
        'Toyama5': toyama5,
        'Toyama6': toyama6,
        'Toyama7': toyama7
    }


def read_json_file(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    points_data = json_data['shapes']
    points = {}
    for point in points_data:
        points[point['label']] = point['points'][0]

    points = np.array([points[key] for key in points_dict.values()])

    pic_id = json_path.split(os.sep)[-1].split('.')[0]
    return pic_id, points


def read_json_folder(json_path):
    json_files = os.listdir(json_path)
    json_files = [os.path.join(json_path, json_file)
                  for json_file in json_files]
    json_files = [
        json_file for json_file in json_files if json_file.endswith('.json')]
    
    pic_ids = []
    points_list = []
    for json_file in json_files:
        pic_id, points = read_json_file(json_file)
        pic_ids.append(pic_id)
        points_list.append(points)

    return pic_ids, points_list


def cal_all_points(points):
    cal_result = {}
    cal_1 = cal_dist_adj_row(points)
    cal_result.update(cal_1)
    cal_2 = cal_dis_adj_col(points)
    cal_result.update(cal_2)
    cal_3 = cal_angle_adj(points)
    cal_result.update(cal_3)
    cal_4 = cal_angle_not_adj(points)
    cal_result.update(cal_4)
    cal_5 = cal_sva(points)
    cal_result.update(cal_5)
    cal_6 = cal_c_type(points)
    cal_result.update(cal_6)
    cal_7 = cal_cns(points)
    cal_result.update(cal_7)
    cal_8 = cal_cn(points)
    cal_result.update(cal_8)
    cal_9 = cal_toyama(points)
    cal_result.update(cal_9)

    return cal_result

def dict2excel(dict_data, excel_path):
    import pandas as pd
    df = pd.DataFrame(dict_data)
    df.to_excel(excel_path, index=False)

# 修改后的读取文件夹函数
def read_json_folder(json_path):
    json_files = os.listdir(json_path)
    json_files = [os.path.join(json_path, json_file) for json_file in json_files if json_file.endswith('.json')]

    # 对文件进行排序，保持原始顺序
    json_files.sort()

    pic_ids = []
    points_list = []
    for json_file in json_files:
        pic_id, points = read_json_file(json_file)
        pic_ids.append(pic_id)
        points_list.append(points)

    return pic_ids, points_list

# def cal_all_points(points):
    cal_result = {}
    cal_1 = cal_dist_adj_row(points)
    cal_result.update(cal_1)
    cal_2 = cal_dis_adj_col(points)
    cal_result.update(cal_2)
    cal_3 = cal_angle_adj(points)
    cal_result.update(cal_3)
    cal_4 = cal_angle_not_adj(points)
    cal_result.update(cal_4)
    cal_5 = cal_sva(points)
    cal_result.update(cal_5)
    cal_6 = cal_c_type(points)
    cal_result.update(cal_6)
    cal_7 = cal_cns(points)
    cal_result.update(cal_7)
    cal_8 = cal_cn(points)
    cal_result.update(cal_8)
    cal_9 = cal_toyama(points)
    cal_result.update(cal_9)

    return cal_result

def dict2excel(dict_data, excel_path):
    import pandas as pd
    df = pd.DataFrame(dict_data)
    df.to_excel(excel_path, index=False)

def cal_json_folder(json_folder, excel_path='result.xlsx'):
    pic_ids, points_list = read_json_folder(json_folder)
    cal_results = []
    for pic_id, points in zip(pic_ids, points_list):
        result = {}
        result['pic_id'] = pic_id
        cal_result = cal_all_points(points)
        result.update(cal_result)
        cal_results.append(result)

    dict2excel(cal_results, excel_path)

if __name__ == '__main__':
    p1 = np.array([0, 0])
    p2 = np.array([0, 1])
    p3 = np.array([1, 1])
    position = point_position_relative_to_line(p3, p1, p2)
    print(position)
    json_folder_path = 'C:\\Users\\Yu Ran\\Desktop\\KeyPointsDetection\\EvalData'  # 您的JSON文件夹路径
    output_excel_path = 'C:\\Users\\Yu Ran\\Desktop\\KeyPointsDetection\\result.xlsx'  # 您想要保存结果的Excel文件路径
    cal_json_folder(json_folder_path, output_excel_path)

# 主程序入口
if __name__ == '__main__':
    json_folder_path = 'C:\\Users\\Yu Ran\\Desktop\\KeyPointsDetection\\EvalData'  # 您的JSON文件夹路径
    output_excel_path = 'C:\\Users\\Yu Ran\\Desktop\\KeyPointsDetection\\result.xlsx'  # 您想要保存结果的Excel文件路径
    cal_json_folder(json_folder_path, output_excel_path)
