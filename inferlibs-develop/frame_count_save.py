#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
To compute number of frames in a video file. 

ffmpeg and openCV count different number of frames. 
This script shows how large the difference is. 
For feature extraction, we rely on openCV. 
"""


import numpy as np
import cv2,sys
import timeit
import matplotlib.pyplot as plt


video_file = '/home/umit/Documents/Research_ADOS/videos/Jacob_140228_2.MPG'


cap = cv2.VideoCapture(video_file)

save_frames = False
frames_out = '/home/umit/Documents/Research_ADOS/videos/some_name'


# Make a copy of this driver script for future usage.
import os  
if save_frames:
    if not os.path.exists(frames_out):
        os.makedirs(frames_out)
    else:
        sys.exit('Output folder already exists!!!')



# Check if camera opened successfully.
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
# Get and convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 

tic=timeit.default_timer()
cnt_ii=0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cnt_ii+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if save_frames:
            img_name = '{:06d}.png'.format(cnt_ii)
            #print('Processing {}'.format(img_name))
            cv2.imwrite('%s/%s'%(frames_out,img_name),frame)
    else:
        break

toc=timeit.default_timer() 
#print '\nTotal elapsed time: %d min, %d sec'% np.divmod(toc-tic,60)

print 'Total number of frames with openCV: %d'%cnt_ii
    
    
# When everything done, release the video capture object.
cap.release()
# Closes all open windows.
cv2.destroyAllWindows()


import imageio
vid = imageio.get_reader(video_file,'ffmpeg')
# number of frames in video
print 'Total number of frames with ffmpeg: %d'%vid._meta['nframes']



import subprocess

def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x]

print getLength(video_file)[0].strip()


cap = cv2.VideoCapture(video_file)

if not cap.isOpened(): sys.exit("Could not open: ", video_file)

nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

print 'Total number of frames with cap.get: %d'%nframe


