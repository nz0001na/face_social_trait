#!/usr/bin/env python2
# -*- coding: utf-8 -*-

##############################################################################
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import os, sys
import numpy as np

caffe2_lib = '/home/umit/Downloads/progs/caffe2/caffe2/build'
detectron_lib = '/home/umit/Downloads/progs/detectron/detectron/lib'
if not caffe2_lib in sys.path: sys.path.insert(0,caffe2_lib)
if not detectron_lib in sys.path: sys.path.insert(1, detectron_lib)
# sys.path.append('/home/me/mypy') 

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from decutils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import decutils.c2 as c2_utils
import decutils.logging
#import decutils.vis as vis_utils
import visualization_utils 
import timeit
from data_utils import count_frames_opencv_manual,printProgressBar

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
decutils.logging.setup_logging(__name__)

#arg_cfg = '/home/umit/Documents/Research_DNN/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'
#arg_weights = '/home/umit/Downloads/detectron_wts/e2e_mask_rcnn_R-101-FPN_2x_model_final.pkl'

arg_cfg = '/home/umit/Documents/Research_DNN/configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml'
arg_weights = '/home/umit/Downloads/detectron_wts/e2e_keypoint_rcnn_R-101-FPN_s1x_model_final.pkl'

# ---- RAM Problem in laptop!
#arg_cfg = '/home/umit/Documents/Research_DNN/configs/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml'
#arg_weights = '/home/umit/Downloads/detectron_wts/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x_model_final.pkl'


video_file = '/home/umit/Documents/Research_ADOS/videos/Jb_140228_2.MPG'
nframes = count_frames_opencv_manual(video_file)
arg_output_dir ='/home/umit/Documents/Research_ADOS/videos/Jb_140228_2'

save_and_visualize = False
save_only = True

cap = cv2.VideoCapture(video_file)
# Check if video opened successfully
if not cap.isOpened(): 
  sys.exit("Error in opening video stream or file")

# Get and convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))   
  
# Detectron settings:
logger = logging.getLogger(__name__)
merge_cfg_from_file(arg_cfg)
cfg.TEST.WEIGHTS = arg_weights
cfg.NUM_GPUS = 1
assert_and_infer_cfg()
model = infer_engine.initialize_model_from_cfg()
dummy_coco_dataset = dummy_datasets.get_coco_dataset()


# Set the output folder:
if not os.path.exists(arg_output_dir):
    os.makedirs(arg_output_dir)
else:
    sys.exit('\n\t Might overwrite files\n')
    #print('\n\t Might overwrite files\n')


tic=timeit.default_timer()
cnt_ii=0
printProgressBar(0,nframes)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cnt_ii+=1
        
        im_name = '{:06d}.png'.format(cnt_ii)
        #print('Processing {}'.format(im_name))
       
        timers = defaultdict(Timer)
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(model, frame, None, timers=timers) # make timers none.
        
        # Plot and save the outputs.
        # BGR -> RGB for visualization (if opencv is used for visualization then keep in BGR).
        if save_and_visualize: visualization_utils.vis_detectron(frame[:, :, ::-1],im_name, arg_output_dir,
                                          cls_boxes,segms=cls_segms,keypoints=cls_keyps,dataset=dummy_coco_dataset,
                                          box_alpha=0.3,show_class=True,
                                          thresh=0.7,kp_thresh=2)
        
        elif save_only: visualization_utils.saveonly_detectron(frame[:, :, ::-1],im_name, arg_output_dir,
                                          cls_boxes,segms=cls_segms,keypoints=cls_keyps,
                                          thresh=0.7,kp_thresh=2) # keep thresholds low for saveonly case. 
    else:
        break
    
    printProgressBar(cnt_ii,nframes)


toc=timeit.default_timer() 
print('\nTotal elapsed time: %d min, %d sec'%np.divmod(toc-tic,60))

# When everything done, release the video capture object
cap.release()
# Closes all windows
cv2.destroyAllWindows()
