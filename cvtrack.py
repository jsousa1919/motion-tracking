#!/usr/bin/python

import argparse
import cv2
import imutils
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('-F', '--file', help='Path to video, defaults to camera')
parser.add_argument('-l', '--background-buffer-length', type=int, default=64, help='Number of frames to use as a background averaging buffer')
parser.add_argument('-w', '--width', type=int, default=500, help='Width to resize video frames to, 0 for no resize')
parser.add_argument('-t', '--threshold', type=int, default=5, help='Threshhold pixel diff for changes')
parser.add_argument('-b', '--blur', type=int, default=21, help='Diameter of gaussian blur')
args = vars(parser.parse_args())

if args.get('file'):
    cam = cv2.VideoCapture(args['file'])
else:
    cam = cv2.VideoCapture(0)
    time.sleep(0.5)

backgroundBuffer = None
one_behind = None
two_behind = None
#cv2.namedWindow('background')
#cv2.namedWindow('frame')
#cv2.namedWindow('diff')
#cv2.namedWindow('bgdiff')
#cv2.namedWindow('gray')
cv2.namedWindow('foreground')
background = None

def velocity(one, two, thresh, dilate_iterations=2):
    diff = cv2.absdiff(one, two)
    diff = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, None, iterations=dilate_iterations)
    return diff

def accumulate_background(backgroundBuffer, bg, two, one):
    ret = bg
    bg_filter = None
    diff = velocity(one, two, args['threshold'])
    if backgroundBuffer is None:
        backgroundBuffer = [diff]
    else:
        backgroundBuffer = np.append(backgroundBuffer, [diff], 0)
    print len(backgroundBuffer)
    print int(args.get('background_buffer_length'))
    if len(backgroundBuffer) == args.get('background_buffer_length'):
        backgroundBuffer = np.delete(backgroundBuffer, 0, axis=0)
    collected = np.bitwise_or.reduce(backgroundBuffer, axis=0)
    bg_filter = collected.copy()
    bg_filter[collected == 0] = 255
    bg_filter[collected != 0] = 0
    #cv2.imshow('bgdiff', collected)
    ret = one.copy()
    ret[bg_filter == 0] = 0
    return (backgroundBuffer, ret, bg_filter, diff)

while True:
    (grabbed, frame) = cam.read()

    if not grabbed:
        break

    if args.get('width'):
        frame = imutils.resize(frame, width=args['width'])

    gray = frame
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (args['blur'], args['blur']), 0)
    #gray = cv2.medianBlur(gray, 5)
    if background is None:
        background = gray.copy()
    if one_behind is None:
        one_behind = gray
    else:
        two_behind = one_behind
        one_behind = gray
    if one_behind is not None and two_behind is not None:
        (backgroundBuffer, background, bg_filter, prev_diff) = accumulate_background(backgroundBuffer, background, two_behind, one_behind)
        if np.any(background):
            if bg_filter is not None:
                foreground = frame.copy()
                foreground[bg_filter == 255, :] = 0
                cv2.imshow('foreground', foreground)
            #cv2.imshow('background', background)
            #cv2.imshow('frame', frame)
            #cv2.imshow('gray', gray)
            #cv2.imshow('diff', prev_diff)
    if cv2.waitKey(10) % 256 == ord('q'):
        break

cv2.destroyAllWindows()


