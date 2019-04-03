import csv
import os
import math

path = 'J:/1_keypoint_observe_8/4_keypoints_data_wrist_shorten/Version3_with_valid_score/'
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

        l_sum = 0
        l_count = 0
        r_sum = 0
        r_count = 0
        for k in range(start, end+1):
            l_value = float(left_wrist_move[k])
            if(l_value != -1.0):
                l_count = l_count + 1
                l_sum = l_sum + l_value

            r_value = float(right_wrist_move[k])
            if(r_value != -1.0):
                r_count = r_count + 1
                r_sum = r_sum + r_value

        if l_count == 0:
            l_speed = -1.0
        else:
            l_speed = l_sum / l_count
        left_speed.append([str(l_speed)])

        if r_count == 0:
            r_speed = -1.0
        else:
            r_speed = r_sum / r_count
        right_speed.append([str(r_speed)])



    with open(sub_path + 'left_speed_frame.csv', 'wb') as f:
        ft = csv.writer(f)
        ft.writerows(left_speed)

    with open(sub_path + 'right_speed_frame.csv', 'wb') as f1:
        ft1 = csv.writer(f1)
        ft1.writerows(right_speed)