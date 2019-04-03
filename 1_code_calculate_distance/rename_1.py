import os
import csv
import shutil

path = 'E:/1_keypoint_observe_8/2_npy_scenario8_shorten_raw/'

des_path = 'E:/1_keypoint_observe_8/2_npy_scenario8_shorten_rename/'

list = os.listdir(path)
for i in range(len(list)):
    # subject_id = list[i]
    # subject = subject_id[0:len(subject_id)-2]
    subject = list[i]
    print subject

    des_subject = des_path + subject + '/'
    if os.path.exists(des_subject) is False:
        os.makedirs(des_subject)

    frame_list = sorted(os.listdir(path + subject + '/'))
    for n in range(len(frame_list)):
        src_frame = path + subject + '/' + frame_list[n]
        des_frame = des_subject + format(n+1, '06d') + '_bkps.npz'
        shutil.copy(src_frame, des_frame)

        if (n%100 == 0):
            print '     ' + str(n)
        # 008010_bkps.npz

    print 'd'