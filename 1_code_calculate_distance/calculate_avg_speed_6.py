import csv
import os
import math
import numpy as np

path = 'J:/1_keypoint_observe_8/4_keypoints_data_wrist_shorten/Version3_with_valid_score/'
list = os.listdir(path)
avg_list = []
avg_list.append(['name', 'avg'])
for i in range(len(list)):
    name = list[i]
    # if name == 'left_avg.csv':
    #     continue
    speed = []
    with open(path + name + '/left_speed_frame.csv', 'rb') as f:
        for row in f:
            value = float(row.split('\r')[0])
            if value == -1.0:
                continue
            speed.append(value)
    avg = np.mean(speed)
    avg_list.append([name, str(avg)])


with open('J:/1_keypoint_observe_8/5_report_shorten/version3/left_frame_avg.csv', 'wb') as f1:
    ft1 = csv.writer(f1)
    ft1.writerows(avg_list)