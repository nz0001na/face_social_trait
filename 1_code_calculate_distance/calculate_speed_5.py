import csv
import os
import math

path = 'E:/1_keypoint_observe_8/4_keypoints_data_wrist_shorten/'
list = os.listdir(path)
for n in range(len(list)):
    subject = list[n]
    print subject

    sub_path = path + subject + '/'
    shoulder_file = 'normalized_distance.csv'

    left_wrist_move = []
    right_wrist_move = []
    f = open(sub_path + shoulder_file, 'r')
    for row in f:
        name = row.split(',')[0]

        if name == 'frame':
            continue
        left_wrist_move.append(float(row.split(',')[3]))
        right_wrist_move.append(float(row.split(',')[5]))

    # move sum per second
    count = len(left_wrist_move)
    iter_num = count/30
    left_speed = []
    right_speed = []
    for i in range(iter_num):
        start = i*30
        end = i*30+29

        y = sum(left_wrist_move[start:end+1])
        left_speed.append([str(y)])

        z = sum(right_wrist_move[start:end+1])
        right_speed.append([str(z)])
        print 'd'


    with open(sub_path + 'left_speed.csv', 'wb') as f:
        ft = csv.writer(f)
        ft.writerows(left_speed)

    with open(sub_path + 'right_speed.csv', 'wb') as f1:
        ft1 = csv.writer(f1)
        ft1.writerows(right_speed)