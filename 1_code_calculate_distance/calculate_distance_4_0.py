import csv
import os
import math

def euclidean_distance(A, B):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B)]))
    return distance


sub_path = 'E:/1_keypoint_observe_8/4_keypoints_data_wrist/Veronica_130322/'
shoulder_file = 'shoulder.csv'
twist_file = 'two_wrist.csv'
shoulder_distance = []

f = open(sub_path + shoulder_file, 'r')
for row in f:
    name = row.split(',')[0]

    if name == 'frame':
        continue
    A = (float(row.split(',')[1]), float(row.split(',')[2]))
    B = (float(row.split(',')[3]), float(row.split(',')[4]))
    distance = euclidean_distance(A, B)
    shoulder_distance.append(distance)

# with open(sub_path + 'shoulder_distance.csv', 'wb') as f:
#     ft = csv.writer(f)
#     ft.writerows(shoulder_distance)

left_wrist = []
right_wrist = []
t = open(sub_path + twist_file, 'r')
for row in t:
    name = row.split(',')[0]
    if name == 'frame':
        continue
    left_wrist.append([row.split(',')[1], row.split(',')[2]])
    right_wrist.append([row.split(',')[3], row.split(',')[4].split('\n')[0]])

movements = []
movements.append(['frame', 'shoulder_distance', 'Normalized_left_wrist', 'Normalized_right_wrist'])
for i in range(1,len(shoulder_distance)):
    # left wrist
    pre_left_x0 = float(left_wrist[i-1][0])
    pre_left_y0 = float(left_wrist[i-1][1])
    pre_point = (pre_left_x0, pre_left_y0)

    left_x0 = float(left_wrist[i][0])
    left_y0 = float(left_wrist[i][1])
    point = (left_x0, left_x0)

    left_wrist_move = euclidean_distance(pre_point, point) / shoulder_distance[i]

    # right wrist
    pre_right_x0 = float(right_wrist[i - 1][0])
    pre_right_y0 = float(right_wrist[i - 1][1])
    pre_pointy = (pre_right_x0, pre_right_y0)

    right_x0 = float(right_wrist[i][0])
    right_y0 = float(right_wrist[i][1])
    pointy = (right_x0, right_y0)

    right_wrist_move = euclidean_distance(pre_pointy, pointy) / shoulder_distance[i]

    movements.append([str(i), str(shoulder_distance[i]), str(left_wrist_move), str(right_wrist_move)])

with open(sub_path + 'normalized_distance.csv', 'wb') as f:
    ft = csv.writer(f)
    ft.writerows(movements)
