#!/usr/bin/python

import argparse
import cv2
import imutils
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='Path to video, defaults to camera')
parser.add_argument('-b', '--background-buffer-length', type=int, default=100, help='Number of frames to use as a background averaging buffer')
parser.add_argument('-w', '--width', type=int, default=0, help='Width to resize video frames to, defaults to no resize')
parser.add_argument('-t', '--threshold', type=int, default=5, help='Threshhold pixel diff for changes')
args = vars(parser.parse_args())

if args.get('file'):
    cam = cv2.VideoCapture(args['file'])
else:
    cam = cv2.VideoCapture(0)
    time.sleep(0.5)

backgroundBuffer = None
one_behind = None
two_behind = None
cv2.namedWindow('background')
cv2.namedWindow('angie')
cv2.namedWindow('diff')
background = None

def accumulate_background(backgroundBuffer, bg, two, one):
    ret = bg
    diff = cv2.absdiff(two, one)
    diff = cv2.threshold(diff, args.get('threshold'), 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, None, iterations=2)
    cv2.imshow('diff', diff)
    if backgroundBuffer == None:
        backgroundBuffer = [diff]
    else:
        backgroundBuffer = np.append(backgroundBuffer, [diff], 0)
    print len(backgroundBuffer)
    print int(args.get('background_buffer_length'))
    if len(backgroundBuffer) == args.get('background_buffer_length'):
        backgroundBuffer = np.delete(backgroundBuffer, 1, axis=0)
        diff = np.sum(backgroundBuffer, axis=0)
        print np.any(diff)
        ret = bg.copy()
        tmp = bg.copy()
        tmp[diff == 0] = 0
        ret = cv2.add(ret, tmp)
        tmp = one.copy()
        tmp[diff > 0] = 0
        ret = cv2.add(ret, tmp)
        if np.any(bg):
            ret = ret / 2
    return (backgroundBuffer, ret)

while True:
    (grabbed, frame) = cam.read()

    if not grabbed:
        break

    if args.get('width'):
        frame = imutils.resize(frame, width=args['width'])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if background == None:
        background = gray.copy()
        background[:][:] = 0
    if one_behind == None:
        one_behind = gray
    else:
        two_behind = one_behind
        one_behind = gray
    if one_behind != None and two_behind != None:
        (backgroundBuffer, background) = accumulate_background(backgroundBuffer, background, two_behind, one_behind)
    if np.any(background):
        diff = cv2.absdiff(background, gray)
        diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.dilate(diff, None, iterations=2)
        cv2.imshow('background', background)
        cv2.imshow('angie', frame)
        if cv2.waitKey(50) % 256 == ord('q'):
            break

cv2.destroyAllWindows()


