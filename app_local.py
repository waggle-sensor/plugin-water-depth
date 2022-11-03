import time
import argparse

import cv2
import torch
import numpy as np

from unet_module import Unet_Main

import glob

def run(args):
    unet_main = Unet_Main()
    y1, y2, x1, x2 = args.cropping.strip().split(' ')

    match = {}
    a = args.mapping.strip().split(' ')
    for i in a:
        b = i.split(',')
        match[int(b[0])] = b[1]

    # files = glob.glob(args.image+'*.jpg')
    files = [args.image]
    for i in files:
        print(i)
        image = cv2.imread(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[int(y1):int(y2), int(x1):int(x2)]
        depth = unet_main.run(image, out_threshold=args.threshold)

        if depth != None:
            if depth not in match:
                print('out of range')
            else:
                print(match[depth])
        else:
            print('no detection')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-image', dest='image',
        action='store', required=True,
        help='Image path')
    parser.add_argument(
        '-threshold', dest='threshold',
        action='store', default=175, type=int,
        help='Measuring sick segmentation threshold')

    parser.add_argument(
        '-cropping', dest='cropping',
        action='store', default="200 700 600 800", type=str,
        help='Points for cropping as string, put the order of "y1 y2 x1 x2"')
    parser.add_argument(
        '-mapping', dest='mapping',
        action='store', default="469,6 467,7 465,8 463,9 461,10 459,11 457,12 455,13 453,14 450,15 448,16 446,17 444,18 442,19 440,20",
        type=str, help='Points for mapping result to water depth as string, put the order of "pixel_height,depth_in_cm pixel_height,depth_in_cm ..."')

    run(parser.parse_args())
