#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Supplementary functions.
"""


import numpy as np
import sys,timeit
import cv2


# read count of frames of video file
def count_frames_opencv_manual(video_file):
    vidcap = cv2.VideoCapture(video_file)
    count = 0
    while True:
        success,image = vidcap.read()
        if not success:
            break
        count += 1

    vidcap.release()
    
    return count


def printProgressBar(iteration, totalIter, decimals=1, length=50, fill='+'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    if not 'prog_time' in globals():
        global prog_time
        prog_time = timeit.default_timer()
        
    percent = ("{0:.1f}").format(100 * (iteration / float(totalIter)))
    filledLength = int(length * iteration // totalIter)
    bar = fill * filledLength + '-' * (length - filledLength)
    print_time = '%d min, %d sec'%np.divmod(timeit.default_timer()-prog_time,60)
    if iteration>0: sys.stdout.write('\r%s |%s| %s%s %s. Time elapsed: %s.' % ('Progress:', bar, percent, '%', 'Complete', print_time))
    if iteration == totalIter: 
        sys.stdout.write('\n')
        del prog_time
    sys.stdout.flush()
    
'''
import time
toolbar_width = 20
printprogressbar(0, toolbar_width)
for i in range(toolbar_width):
    time.sleep(2) # do real work here
    # update the bar
    printprogressbar(i+1, toolbar_width)
'''
