import os
import csv
import shutil
import math

def euclidean_distance(A, B):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B)]))
    return distance

a = euclidean_distance((3,4),(4,5))



path = 'E:/1_keypoint_observe_8/2_npy_feature/Matthew_130920_2/'

# 'Breanna_151009_2'   1125
# 'Michael_160122_3'

des_path = 'E:/1_keypoint_observe_8/Matthew_130920/'
if os.path.exists(des_path) is False:
    os.makedirs(des_path)

frame_list = sorted(os.listdir(path))
for n in range(len(frame_list)):
    src_frame = path + frame_list[n]
    des_frame = des_path + format(n+1, '06d') + '_bkps.npz'
    shutil.copy(src_frame, des_frame)

    if (n%100 == 0):
        print '     ' + str(n)
    # 008010_bkps.npz




# list = os.listdir(path)
# for i in range(len(list)):
#     subject_id = list[i]
#     subject = subject_id[0:len(subject_id)-2]
#
#     print subject
#
#     des_subject = des_path + subject + '/'
#     if os.path.exists(des_subject) is False:
#         os.makedirs(des_subject)
#
#     frame_list = sorted(os.listdir(path + subject_id + '/'))
#     for n in range(len(frame_list)):
#         src_frame = path + subject_id + '/' + frame_list[n]
#         des_frame = des_subject + format(n+1, '06d') + '_bkps.npz'
#         shutil.copy(src_frame, des_frame)
#
#         if (n%100 == 0):
#             print '     ' + str(n)
#         # 008010_bkps.npz
#
#     print 'd'