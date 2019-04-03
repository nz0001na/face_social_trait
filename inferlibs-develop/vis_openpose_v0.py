#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import os,re,sys
import cv2
from data_utils import count_frames_opencv_manual,printProgressBar
from visualization_utils import visonly_openpose
import timeit
import json


# Video file. 
video_file = '/home/umit/Documents/Research_ET/iTwins_Stimuli/social_vid_02contrVideo for Windows_socialvids_higherQual[0].avi'
nframes = count_frames_opencv_manual(video_file)

# Location of detectron outputs.
openpose_outs = '/home/umit/Documents/Research_ET/iTwins_Stimuli/itwins_clip0'
if os.path.exists(openpose_outs):
    OP_files = sorted(os.listdir(openpose_outs))
    OP_files = [ f for f in OP_files if re.match(r'.*\.(json)', f, flags=re.I) and not '~' in f ]
    # A simple check for whether detectron outputs match with the video.  
    if nframes != len(OP_files): sys.exit('Inconsistency between openpose files and video frames!')

    
# Where to save outputs. 
output_dir = '/home/umit/Documents/Research_ET/iTwins_Stimuli/out2_op'


# Set the output folder:
if not os.path.exists(output_dir): os.makedirs(output_dir)
else: print('\n\t Might overwrite files\n')
    #sys.exit('Might overwrite files')
    

# open video file with openCV.
cap = cv2.VideoCapture(video_file)
# Get and convert the resolutions from float to integer.
frame_width, frame_height = int(cap.get(3)), int(cap.get(4)) 

# Check if video opened successfully
if (cap.isOpened()== False): sys.exit('Error in opening video file')


tic=timeit.default_timer()
cnt_ii=0
printProgressBar(0,nframes)
while(cap.isOpened()):
    success, frame = cap.read()
    if not success: break
    cnt_ii+=1
    
    ext_name = '{:012d}'.format(cnt_ii-1)
    frame_name_ext = '{:05d}.png'.format(cnt_ii) 
    
    openpose_file = os.path.join(openpose_outs,'social_vid_02contrVideo[0]_%s_keypoints.json'%ext_name)
       

    with open(openpose_file) as data_file:    
        data_persons = json.load(data_file)
    
    npersons_frame = len(data_persons['people'])    
    
    pose_points=[]
    face_points=[]
    for person in data_persons['people']:
        
        if len(person['pose_keypoints_2d']): this_pose = np.array(person['pose_keypoints_2d']).reshape(-1,3).T # (x,y,score)
        if len(person['face_keypoints_2d']): this_face = np.array(person['face_keypoints_2d']).reshape(-1,3).T # (x,y,score)
            
        pose_points.append(this_pose)
        face_points.append(this_face)

    visonly_openpose(frame[:,:,::-1],frame_name_ext,output_dir,
                     keypoints=pose_points,kp_thresh=0.4)
 
    printProgressBar(cnt_ii,nframes)

toc=timeit.default_timer() 
print('\nTotal elapsed time: %d min, %d sec'%np.divmod(toc-tic,60))

# When everything done, release the video capture object
cap.release()
# Closes all windows
cv2.destroyAllWindows()
