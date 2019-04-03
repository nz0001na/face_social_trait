import csv
import os

des_path = 'J:/1_keypoint_observe_8/4_keypoints_data_wrist_shorten/Version3_with_valid_score/'
if os.path.exists(des_path) is False:
    os.makedirs(des_path)


path = 'J:/1_keypoint_observe_8/3_keypoints_data_shorten/'
list = os.listdir(path)
count = len(list)
for n in range(count):
    name = list[n]
    print name

    des_subject = des_path + name + '/'
    if os.path.exists(des_subject) is False:
        os.makedirs(des_subject)

    shoulder_list = []
    shoulder_list.append(['frame', 'left_x0', 'left_y0', 'left_score', 'right_x0', 'right_y0','right_score'])

    wrist = []
    wrist.append(['frame', 'left_x0', 'left_y0', 'left_score', 'right_x0', 'right_y0', 'right_score'])

    frame_path = path + name + '/'
    frame_list = sorted(os.listdir(frame_path))
    for i in range(len(frame_list)):
        file_name = frame_list[i]
        if file_name == 'box.csv':
            continue

        f = csv.reader(open(frame_path + file_name, 'rb'))
        for row in f:
            if row[0] == 'X':
                left_x0 = row[10]
                right_x0 = row[11]
                left_shoulder_x0 = row[6]
                right_shoulder_x0 = row[7]
            if row[0] == 'Y':
                left_y0 = row[10]
                right_y0 = row[11]
                left_shoulder_y0 = row[6]
                right_shoulder_y0 = row[7]
            if row[0] == 'Score':
                left_score = row[10]
                if float(left_score) < 3.0:
                    left_score = '0'
                right_score = row[11]
                if float(right_score) < 3.0:
                    right_score = '0'
                left_shoulder_score = row[6]
                if float(left_shoulder_score) < 3.0:
                    left_shoulder_score = '0'
                right_shoulder_score = row[7]
                if float(right_shoulder_score) < 3.0:
                    right_shoulder_score = '0'
        shouder_point = [file_name.split('_')[0], left_shoulder_x0, left_shoulder_y0, left_shoulder_score, right_shoulder_x0, right_shoulder_y0, right_shoulder_score]
        pre_point = [file_name.split('_')[0], left_x0, left_y0, left_score, right_x0, right_y0, right_score]

        shoulder_list.append(shouder_point)
        wrist.append(pre_point)

    with open(des_subject + 'shoulder.csv', 'wb') as f:
        ft = csv.writer(f)
        ft.writerows(shoulder_list)

    with open(des_subject + 'two_wrist.csv', 'wb') as f:
        ft = csv.writer(f)
        ft.writerows(wrist)



