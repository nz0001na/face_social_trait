import os
import numpy as np
import csv
# import the necessary packages
from collections import namedtuple

# calculate IoU
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou



gt_file = 'crop_size.csv'
f = csv.reader(open(gt_file, 'rb'))
for row in f:
    name = row[0]
    if name == 'subject':
        continue
    print name

    gt = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]

    des_path = 'E:/1_keypoint_observe_8/3_keypoints_data_shorten/' + name + '/'
    if os.path.exists(des_path) is False:
        os.makedirs(des_path)

    src_path = 'E:/1_keypoint_observe_8/2_npy_scenario8_shorten_rename/' + name + '/'
    npy_list = os.listdir(src_path)

    box_list = []
    box_list.append(['name', 'index', 'x0', 'y0', 'x1', 'y1', 'score'])
    for n in range(len(npy_list)): # len(npy_list)
        file = src_path + npy_list[n]
        kps = np.load(file)

        kps_box = kps['kps_box']
        kps_mat = kps['kps_mat']
        iou_max_index = 0
        iou_max_value = 0
        for i in range(len(kps_box)):
            bbox = [kps_box[i][0], kps_box[i][1], kps_box[i][2], kps_box[i][3]]
            # gt = [254,160,478,384]
            iou_value = bb_intersection_over_union(bbox, gt)
            if iou_value > iou_max_value:
                iou_max_value = iou_value
                iou_max_index = i

        index = iou_max_index
        item_box = kps_box[index]
        box_list.append([npy_list[n], index, item_box[0], item_box[1], item_box[2], item_box[3], item_box[4]])

        item_mat = kps_mat[index]
        mat_list = []
        mat_list.append(['title', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19'])
        mat_list.append(['X',item_mat[0][0],item_mat[0][1],item_mat[0][2],item_mat[0][3],item_mat[0][4],
                         item_mat[0][5],item_mat[0][6],item_mat[0][7],item_mat[0][8],item_mat[0][9],
                         item_mat[0][10],item_mat[0][11],item_mat[0][12],item_mat[0][13],item_mat[0][14],
                         item_mat[0][15],item_mat[0][16],item_mat[0][17],item_mat[0][18]])
        mat_list.append(['Y',item_mat[1][0],item_mat[1][1],item_mat[1][2],item_mat[1][3],item_mat[1][4],
                         item_mat[1][5],item_mat[1][6],item_mat[1][7],item_mat[1][8],item_mat[1][9],
                         item_mat[1][10],item_mat[1][11],item_mat[1][12],item_mat[1][13],item_mat[1][14],
                         item_mat[1][15],item_mat[1][16],item_mat[1][17],item_mat[1][18]])
        mat_list.append(['Score',item_mat[2][0],item_mat[2][1],item_mat[2][2],item_mat[2][3],item_mat[2][4],
                         item_mat[2][5],item_mat[2][6],item_mat[2][7],item_mat[2][8],item_mat[2][9],
                         item_mat[2][10],item_mat[2][11],item_mat[2][12],item_mat[2][13],item_mat[2][14],
                         item_mat[2][15],item_mat[2][16],item_mat[2][17],item_mat[2][18]])

        with open(des_path + npy_list[n].split('.')[0] + '.csv', 'wb') as p:
            pt = csv.writer(p)
            pt.writerows(mat_list)


    with open(des_path + 'box.csv', 'wb') as f:
        ft = csv.writer(f)
        ft.writerows(box_list)


