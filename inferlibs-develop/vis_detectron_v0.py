#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
    Visulize the keypoints in video frame-by-frame
"""

import numpy as np
import os,re,sys
import cv2
from data_utils import count_frames_opencv_manual,printProgressBar
from visualization_utils import visonly_detectron
import timeit


# read Video file.
video_file = 'J:/2_ADOS/Jacob_140228/Jacob_140228_2.MPG'
nframes = count_frames_opencv_manual(video_file)

# read Keypoints (.npz format): Location of detectron outputs.
kps_folder = 'C:/0_NaZhang_backup/6 Autism/4_code_from_umit/Jb_140228_2/bkps'
if os.path.exists(kps_folder):
    kps_files = sorted(os.listdir(kps_folder))
    kps_files = [ f for f in kps_files if re.match(r'.*\.(npy|npz)', f, flags=re.I) and not '~' in f ]
    # A simple check for whether detectron outputs match with the video.  
    if nframes != len(kps_files): sys.exit('Inconsistency between kps files and video frames!')
else:
    kps_files, kps_box, kps_mat = None, None, None 

# get mask file
masks_folder = 'C:/0_NaZhang_backup/6 Autism/4_code_from_umit/Jb_140228_2/bmasks'
if os.path.exists(masks_folder):
    masks_files = sorted(os.listdir(masks_folder))
    masks_files = [ f for f in masks_files if re.match(r'.*\.(npy|npz)', f, flags=re.I) and not '~' in f ]
    # A simple check for whether detectron outputs match with the video.  
    if nframes != len(masks_files): sys.exit('Inconsistency between mask files and video frames!')
else:
    masks_files, masks_box, masks_mat = None, None, None
    
# Where to save outputs. 
output_dir = 'J:/4_keypoint_visual/vis_output/Jb_140228_2'


# Set the output folder:
if not os.path.exists(output_dir): os.makedirs(output_dir)
else: print('\n\t Might overwrite files\n')
    #sys.exit('Might overwrite files')
    

#video_file_base = os.path.splitext(os.path.basename(video_file))[0]
 
# open video file with openCV.
cap = cv2.VideoCapture(video_file)
# Get and convert the resolutions from float to integer.
frame_width, frame_height = int(cap.get(3)), int(cap.get(4)) 

# Check if video opened successfully
if (cap.isOpened()== False): sys.exit('Error in opening video file')

# set time counter
tic=timeit.default_timer()
cnt_ii=0
printProgressBar(0,nframes)
while(cap.isOpened()):
    success, frame = cap.read()
    if not success: break
    cnt_ii+=1
    
    if kps_files is not None: 
        if len(kps_files[0].split('_')[0]) == 5: frame_name = '{:05d}'.format(cnt_ii)
        elif len(kps_files[0].split('_')[0]) == 6: frame_name = '{:06d}'.format(cnt_ii)

        if frame_name != kps_files[cnt_ii-1].split('_')[0]: sys.exit('Error in loading kps files')
        frame_name_ext = '{:06}.png'.format(cnt_ii) 
     
        # Load respective keypoints file.
        kps = np.load(os.path.join(kps_folder,'%s_%s'%(frame_name,kps_files[0].split('_')[1])))
        # print  kps.files
        kps_mat  = kps['kps_mat']
        kps_box = kps['kps_box']

    if masks_files is not None: 
        if len(masks_files[0].split('_')[0]) == 5: frame_name = '{:05d}'.format(cnt_ii)
        elif len(masks_files[0].split('_')[0]) == 6: frame_name = '{:06d}'.format(cnt_ii)

        if frame_name != masks_files[cnt_ii-1].split('_')[0]: sys.exit('Error in loading kps files')
        frame_name_ext = '{:06}.png'.format(cnt_ii) 
     
        # Load respective keypoints file.
        masks = np.load(os.path.join(masks_folder,'%s_%s'%(frame_name,masks_files[0].split('_')[1])))
        # print  masks.files
        masks_mat = masks['masks_mat']
        masks_box  = masks['masks_box']


    if (kps_files is not None) or (masks_files is not None): 
        visonly_detectron(frame[:,:,::-1],frame_name_ext,output_dir,
                          kps_boxes=kps_box,keypoints=kps_mat,
                          masks_boxes=masks_box,masks=masks_mat,thresh=0.85,kp_thresh=3) # (0.85,3)
    
    printProgressBar(cnt_ii,nframes)
    

toc=timeit.default_timer() 
print('\nTotal elapsed time: %d min, %d sec'%np.divmod(toc-tic,60))

# When everything done, release the video capture object
cap.release()
# Closes all windows
cv2.destroyAllWindows()
