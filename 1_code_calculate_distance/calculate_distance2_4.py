import csv
import os
import math

def euclidean_distance(A, B):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B)]))
    return distance


path = 'J:/1_keypoint_observe_8/4_keypoints_data_wrist_shorten/Version3_with_valid_score/'
list = os.listdir(path)
for n in range(len(list)):
    subject = list[n]
    print subject
    # if subject != 'Jacob_150917':
    #     continue

    sub_path = path + subject + '/'
    shoulder_file = 'shoulder.csv'
    twist_file = 'two_wrist.csv'
    shoulder_distance = []

    f = open(sub_path + shoulder_file, 'r')
    for row in f:
        name = row.split(',')[0]

        if name == 'frame':
            continue
        left_score = float(row.split(',')[3])
        right_score = float(row.split(',')[6])
        if left_score == 0 or right_score == 0:
            distance = -1.0
        else:
            A = (float(row.split(',')[1]), float(row.split(',')[2]))
            B = (float(row.split(',')[4]), float(row.split(',')[5]))
            distance = euclidean_distance(A, B)
        shoulder_distance.append([distance])
    with open(sub_path + 'shoulder_distance.csv', 'wb') as fw:
        ftw = csv.writer(fw)
        ftw.writerows(shoulder_distance)


    left_wrist = []
    right_wrist = []
    left_wrist_scores = []
    right_wrist_scores = []
    t = open(sub_path + twist_file, 'r')
    for r in t:
        name = r.split(',')[0]
        if name == 'frame':
            continue
        left_wrist_score = float(r.split(',')[3])
        left_wrist_scores.append([left_wrist_score])
        right_wrist_score = float(r.split(',')[6].split('\n')[0])
        right_wrist_scores.append([right_wrist_score])

        left_wrist.append([float(r.split(',')[1]), float(r.split(',')[2])])
        right_wrist.append([float(r.split(',')[4]), float(r.split(',')[5])])

    with open(sub_path + 'left_wrist.csv', 'wb') as fw1:
        ftw1 = csv.writer(fw1)
        ftw1.writerows(left_wrist)
    with open(sub_path + 'right_wrist.csv', 'wb') as fw2:
        ftw2 = csv.writer(fw2)
        ftw2.writerows(right_wrist)
    with open(sub_path + 'left_wrist_scores.csv', 'wb') as fw3:
        ftw3 = csv.writer(fw3)
        ftw3.writerows(left_wrist_scores)
    with open(sub_path + 'right_wrist_scores.csv', 'wb') as fw4:
        ftw4 = csv.writer(fw4)
        ftw4.writerows(right_wrist_scores)

    movements = []
    movements.append(['frame', 'shoulder_distance', 'left_distance', 'Normalized_left_distance','right_distance', 'Normalized_right_wrist'])
    for i in range(1, len(shoulder_distance)):
        # left wrist
        pre_left_score = float(left_wrist_scores[i-1][0])
        left_score = float(left_wrist_scores[i][0])
        if pre_left_score == 0 or left_score == 0 or float(shoulder_distance[i][0]) == -1.0:
            left_distance = -1.0
            left_wrist_move = -1.0
        else:
            pre_left_x0 = float(left_wrist[i-1][0])
            pre_left_y0 = float(left_wrist[i-1][1])
            pre_point = (pre_left_x0, pre_left_y0)

            left_x0 = float(left_wrist[i][0])
            left_y0 = float(left_wrist[i][1])
            point = (left_x0, left_y0)

            left_distance = euclidean_distance(pre_point, point)
            left_wrist_move = left_distance / shoulder_distance[i][0]


        # right wrist
        pre_right_score = float(right_wrist_scores[i-1][0])
        right_score = float(right_wrist_scores[i][0])
        if pre_right_score == 0 or right_score == 0 or shoulder_distance[i][0] == -1.0:
            right_distance = -1.0
            right_wrist_move = -1.0
        else:
            pre_right_x0 = float(right_wrist[i - 1][0])
            pre_right_y0 = float(right_wrist[i - 1][1])
            pre_pointy = (pre_right_x0, pre_right_y0)

            right_x0 = float(right_wrist[i][0])
            right_y0 = float(right_wrist[i][1])
            pointy = (right_x0, right_y0)

            right_distance = euclidean_distance(pre_pointy, pointy)
            right_wrist_move = right_distance / shoulder_distance[i][0]

        movements.append([str(i), str(shoulder_distance[i][0]), str(left_distance), str(left_wrist_move), str(right_distance), str(right_wrist_move)])

    with open(sub_path + 'normalized_distance.csv', 'wb') as f:
        ft = csv.writer(f)
        ft.writerows(movements)
