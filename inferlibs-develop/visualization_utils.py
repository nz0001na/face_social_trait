#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Scripts to save and visualize Detectron outouts. 
Adopted from facebook's detectron package. 

https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
'''


# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Detection output visualization module."""

import cv2
import numpy as np
import os,sys


# https://github.com/cocodataset/cocoapi
# pycocotools should be installed to run detectron. 
# if this script is used only for visualization of detectron outputs, 
# then no need to install pycocotools.
try:
    import pycocotools.mask as mask_util
    found = True
except ImportError:
    found = False



import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pdb

#plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator


def get_keypoint_names():
    keypoint_names = [ 'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'mid_shoulder', 'mid_hip']
    return keypoint_names


def get_keypoint_names_openpose():
    keypoint_names = [ 'nose', 'mid_shoulder', 'right_shoulder', 'right_elbow', 'right_wrist', 
                      'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee', 'right_ankle',
                      'left_hip', 'left_knee', 'left_ankle', 'right_eye', 'left_eye', 'right_ear', 'left_ear']
    return keypoint_names



def colormap(rgb=False):
    color_list = np.array(
        [   0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000 ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb: color_list = color_list[:, ::-1]
    
    return color_list


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')




def vis_detectron(im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9,
                  kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False):
    """Visual debugging of detections."""
    
    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    dataset_keypoints = get_keypoint_names()

    if segms is not None:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    # So boxes: [x0, y0, x1, y1] where the origin is the left-top corner. 
    # Notice that matrix dims of image are like transpose of those numbers 
    # (but we don't need to alter that). 
    #areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    #sorted_inds = np.argsort(-areas)
    person_indx = 1 # keypoint_utils.get_person_class_index()

    save_box = []
    save_mask= []
    save_kps = []
    pdb.set_trace()

    mask_color_id = 0
    #for i in sorted_inds: # try without sorting w.r.t. area.
    for i in range(boxes.shape[0]):
        
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        
        # This sets visualization to only person. 
        # Comment if objects are also considered. 
        if classes[i] != person_indx: # specific to MCOCO 
            continue 

        save_box.append(boxes[i,:])

        # show box (off by default)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset), # 'person {:0.2f}'.format(score),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if segms is not None and len(segms) > i:
            
            save_mask.append(masks[:,:,i])
            
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    plt.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

                if kps[2, i2] > kp_thresh:
                    plt.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            # position of mid shoulder
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            # score of mid shoulder
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            # position of mid hip
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            # score of mid hip
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            
            # save these additional values as well.
            save_kps.append(np.hstack(( kps[:3,:], np.append(mid_shoulder, sc_mid_shoulder).reshape(-1,1),
                                       np.append(mid_hip, sc_mid_hip).reshape(-1,1) )) )
            
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = plt.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)

 
    if segms is not None: pre_name = 'mask_'
    if keypoints is not None: pre_name = 'kps_'
    
    output_name = pre_name + os.path.basename(im_name)
    fig.savefig(os.path.join(output_dir, output_name), dpi=dpi)
    plt.close('all')
    
    
    bname = os.path.splitext(os.path.basename(im_name))[0]
    if len(save_box)>0 and len(save_mask)>0:
        if not os.path.exists(os.path.join(output_dir,'bmasks')): os.makedirs(os.path.join(output_dir,'bmasks'))
        np.savez_compressed(os.path.join(output_dir,'bmasks',bname+'_bmasks'),masks_box=np.array(save_box),masks_mat=np.array(save_mask))


    if len(save_box)>0 and len(save_kps)>0:
        if not os.path.exists(os.path.join(output_dir,'bkps')): os.makedirs(os.path.join(output_dir,'bkps'))
        np.savez_compressed(os.path.join(output_dir,'bkps',bname+'_bkps'),kps_box=np.array(save_box),kps_mat=np.array(save_kps))
 

       
#

def saveonly_detectron(im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2):

    
    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)


    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    dataset_keypoints = get_keypoint_names()

    if segms is not None:
        masks = mask_util.decode(segms)


    # Display in largest to smallest order to reduce occlusion
    #areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    #sorted_inds = np.argsort(-areas)
    person_indx = 1

    save_box = []
    save_mask= []
    save_kps = []
    
    #for i in sorted_inds: # try without sorting w.r.t. area.
    for i in range(boxes.shape[0]):
        
        score = boxes[i, -1]
        if score < thresh:
            continue
        
        if classes[i] != person_indx: # specific to MCOCO 
            continue 

        save_box.append(boxes[i,:])

        # mask
        if segms is not None and len(segms) > i:
            save_mask.append(masks[:,:,i])
            
        # keypoints
        if keypoints is not None and len(keypoints) > i:

            kps = keypoints[i]

            # add mid shoulder / mid hip for better visualization
            # position of mid shoulder
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            # score of mid shoulder
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            # position of mid hip
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            # score of mid hip
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            
            # save these additional values as well.
            save_kps.append(np.hstack(( kps[:3,:], np.append(mid_shoulder, sc_mid_shoulder).reshape(-1,1),
                                       np.append(mid_hip, sc_mid_hip).reshape(-1,1) )) )
            
 
    
    bname = os.path.splitext(os.path.basename(im_name))[0]
    if len(save_box)>0 and len(save_mask)>0:
        if not os.path.exists(os.path.join(output_dir,'bmasks')): os.makedirs(os.path.join(output_dir,'bmasks'))
        np.savez_compressed(os.path.join(output_dir,'bmasks',bname+'_bmasks'),masks_box=np.array(save_box),masks_mat=np.array(save_mask))


    if len(save_box)>0 and len(save_kps)>0:
        if not os.path.exists(os.path.join(output_dir,'bkps')): os.makedirs(os.path.join(output_dir,'bkps'))
        np.savez_compressed(os.path.join(output_dir,'bkps',bname+'_bkps'),kps_box=np.array(save_box),kps_mat=np.array(save_kps))
 
    
#
def visonly_detectron(im,im_name,output_dir,
                      kps_boxes=None,keypoints=None,masks_boxes=None,masks=None,
                      thresh=0.8,kp_thresh=2,dpi=200,box_alpha=0.0,show_class=False):
    """Visual debugging of detections."""

    plot_keypoints = True
    if kps_boxes is None or kps_boxes.shape[0]==0 or max(kps_boxes[:, 4])<thresh:
        plot_keypoints = False
   
    plot_masks = True
    if masks_boxes is None or masks_boxes.shape[0]==0 or max(masks_boxes[:, 4])<thresh:
        plot_masks = False
             
    if not (plot_keypoints or plot_masks):
        return
    
    dataset_keypoints = get_keypoint_names()

    color_list = colormap(rgb=True) / 255

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    if plot_keypoints:
        for i in range(kps_boxes.shape[0]):
            
            bbox = kps_boxes[i, :4]
            score = kps_boxes[i, -1]
            if score < thresh:
                continue
            
            # show box (off by default)
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                       bbox[2]-bbox[0],bbox[3]-bbox[1],
                                       fill=False, edgecolor='g',
                                       linewidth=0.5, alpha=box_alpha))
    
            if show_class:
                ax.text(bbox[0], bbox[1]-2,'person {:0.2f}'.format(score),
                    fontsize=3,family='serif',color='white',
                    bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'))

            # show keypoints
            if keypoints is not None and len(keypoints) > i:
                kps = keypoints[i]
                plt.autoscale(False)
                for l in range(len(kp_lines)):
                    i1 = kp_lines[l][0]
                    i2 = kp_lines[l][1]
                    if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                        x = [kps[0, i1], kps[0, i2]]
                        y = [kps[1, i1], kps[1, i2]]
                        line = plt.plot(x, y)
                        plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                    if kps[2, i1] > kp_thresh:
                        plt.plot(
                            kps[0, i1], kps[1, i1], '.', color=colors[l],
                            markersize=3.0, alpha=0.7)
    
                    if kps[2, i2] > kp_thresh:
                        plt.plot(
                            kps[0, i2], kps[1, i2], '.', color=colors[l],
                            markersize=3.0, alpha=0.7)
    
                # add mid shoulder / mid hip for better visualization
                # position of mid shoulder
                mid_shoulder =  kps[:2, dataset_keypoints.index('mid_shoulder')] 
                # score of mid shoulder
                sc_mid_shoulder = kps[2, dataset_keypoints.index('mid_shoulder')]
                
                # position of mid hip
                mid_hip = kps[:2, dataset_keypoints.index('mid_hip')] 
                # score of mid hip
                sc_mid_hip = kps[2, dataset_keypoints.index('mid_hip')] 
                
                if (sc_mid_shoulder > kp_thresh and
                        kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                    x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                    y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
                if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                    x = [mid_shoulder[0], mid_hip[0]]
                    y = [mid_shoulder[1], mid_hip[1]]
                    line = plt.plot(x, y)
                    plt.setp(line, color=colors[len(kp_lines) + 1], linewidth=1.0, alpha=0.7)


    if plot_masks:
        mask_color_id = 0
        for i in range(masks_boxes.shape[0]):
            
            bbox = masks_boxes[i, :4]
            score = masks_boxes[i, -1]
            if score < thresh:
                continue
            
            # show box (off by default)
            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                       bbox[2]-bbox[0],bbox[3]-bbox[1],
                                       fill=False, edgecolor='g',
                                       linewidth=0.5, alpha=box_alpha))
    
            if show_class:
                ax.text(bbox[0], bbox[1]-2,'person {:0.2f}'.format(score),
                    fontsize=3,family='serif',color='white',
                    bbox=dict(facecolor='g', alpha=0.4, pad=0, edgecolor='none'))
                
            # show mask
            if masks is not None and len(masks) > i:
                
                
                img = np.ones(im.shape)
                color_mask = color_list[mask_color_id % len(color_list), 0:3]
                mask_color_id += 1
    
                w_ratio = .4
                for c in range(3):
                    color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
                for c in range(3):
                    img[:, :, c] = color_mask[c]
                e = masks[i, :, :]
    
                _, contour, hier = cv2.findContours(
                    e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
                for c in contour:
                    polygon = Polygon(
                        c.reshape((-1, 2)),
                        fill=True, facecolor=color_mask,
                        edgecolor='w', linewidth=1.2,
                        alpha=0.5)
                    ax.add_patch(polygon)
    

    if plot_keypoints and plot_masks: pre_name = 'joint_'        
    elif plot_keypoints: pre_name = 'kps_'
    elif plot_masks: pre_name = 'mask_'
    
    output_name = pre_name + os.path.basename(im_name)
    fig.savefig(os.path.join(output_dir, output_name), dpi=dpi)
    plt.close('all')
    
    

# generate the visualized images with keypoints
def visonly_openpose(im,im_name,output_dir,keypoints=None,
                     kp_thresh=0.4,dpi=1000,box_alpha=0.0):
    """Visual debugging of detections."""

    if keypoints is None or len(keypoints)==0 or max(np.hstack(keypoints)[2,:])<kp_thresh:
        return

    dataset_keypoints = get_keypoint_names_openpose()

    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    # ax.imshow(im)


    for i in range(len(keypoints)):
        
        # show keypoints
        kps = keypoints[i]

        score = max(kps[-1,:])

        if score < kp_thresh:
            continue

        plt.autoscale(False)
        for l in range(len(kp_lines)):
            i1 = kp_lines[l][0]
            i2 = kp_lines[l][1]
            if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                x = [kps[0, i1], kps[0, i2]]
                y = [kps[1, i1], kps[1, i2]]
                line = plt.plot(x, y)
                plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
            if kps[2, i1] > kp_thresh:
                plt.plot(
                    kps[0, i1], kps[1, i1], '.', color=colors[l],
                    markersize=3.0, alpha=0.7)

            if kps[2, i2] > kp_thresh:
                plt.plot(
                    kps[0, i2], kps[1, i2], '.', color=colors[l],
                    markersize=3.0, alpha=0.7)

        # add mid shoulder / mid hip for better visualization
        # position of mid shoulder
        mid_shoulder =  kps[:2, dataset_keypoints.index('mid_shoulder')] 
        # score of mid shoulder
        sc_mid_shoulder = kps[2, dataset_keypoints.index('mid_shoulder')]
        
        # position of mid hip
        mid_hip = (
            kps[:2, dataset_keypoints.index('right_hip')] +
            kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
        # score of mid hip
        sc_mid_hip = np.minimum(
            kps[2, dataset_keypoints.index('right_hip')],
            kps[2, dataset_keypoints.index('left_hip')])
        
        if (sc_mid_shoulder > kp_thresh and kps[2, dataset_keypoints.index('nose')] > kp_thresh):
            x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
            y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
            line = plt.plot(x, y)
            plt.setp(line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
        if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
            x = [mid_shoulder[0], mid_hip[0]]
            y = [mid_shoulder[1], mid_hip[1]]
            line = plt.plot(x, y)
            plt.setp(line, color=colors[len(kp_lines) + 1], linewidth=1.0, alpha=0.7)


    pre_name = 'opose_'        
    
    output_name = pre_name + os.path.basename(im_name)
    fig.savefig(os.path.join(output_dir, output_name), dpi=dpi)
    plt.close('all')
    
    
