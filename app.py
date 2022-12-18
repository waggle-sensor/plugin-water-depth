import time
import argparse

import cv2
import torch
import numpy as np

from unet_module import Unet_Main

from waggle.plugin import Plugin
from waggle.data.vision import Camera

TOPIC_WATERDEPTH = "env.depth.water"
TOPIC_WATERDEPTHLOG = "log.depth.water"


def parse_mapping_table(lines):
    xp = []
    fp = []
    for line in lines.split(" "):
        pixel_height, centimeter = line.split(",")
        xp.append(float(pixel_height))
        fp.append(float(centimeter))
    return np.array(xp), np.array(fp)


def get_water_depth_to_cm(pixel_height, xp, fp):
    unknown_height = -999
    # check if xp is increasing. np.interp works only with increasing xp
    if np.all(np.diff(xp) > 0):
        return np.interp(pixel_height, xp, fp, left=unknown_height, right=unknown_height)
    else:
        return np.interp(pixel_height, xp[::-1], fp[::-1], left=unknown_height, right=unknown_height)


def run(args):
    print("Water depth estimation starts...")
    with Plugin() as plugin, Camera(args.stream) as camera:
        timestamp = time.time()
        print(f"Loading Model at time: {timestamp}")
        with plugin.timeit("plugin.duration.loadmodel"):
            unet_main = Unet_Main()

        xp, fp = parse_mapping_table(args.mapping.strip())
        sampling_countdown = -1
        if args.sampling_interval >= 0:
            print(f"Sampling enabled -- occurs every {args.sampling_interval}th inferencing")
            sampling_countdown = args.sampling_interval

        while True:
            with plugin.timeit("plugin.duration.input"):
                sample = camera.snapshot()
                image = sample.data
                imagetimestamp = sample.timestamp
                y1, y2, x1, x2 = args.cropping.strip().split(' ')
                image = image[int(y1):int(y2), int(x1):int(x2)]
            if args.debug:
                s = time.time()
            with plugin.timeit("plugin.duration.inferencing"):
                depth = unet_main.run(image, out_threshold=args.threshold)
            if args.debug:
                e = time.time()
                print(f'Time elapsed for inferencing: {e-s} seconds')

            if depth != None:
                depth_in_cm = get_water_depth_to_cm(depth, xp, fp)
                if depth_in_cm < 0:
                    plugin.publish(TOPIC_WATERDEPTHLOG, 'out of range', timestamp=imagetimestamp)
                else:
                    plugin.publish(TOPIC_WATERDEPTH, depth_in_cm, timestamp=imagetimestamp)
            else:
                plugin.publish(TOPIC_WATERDEPTHLOG, 'no detection', timestamp=imagetimestamp)

            if sampling_countdown > 0:
                sampling_countdown -= 1
            elif sampling_countdown == 0:
                sample.save('sample.jpg')
                plugin.upload_file('sample.jpg')
                print("A sample is published")
                # Reset the count
                sampling_countdown = args.sampling_interval

            if args.continuous:
                if args.interval > 0:
                    time.sleep(args.interval)
            else:
                exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='Debug flag')
    parser.add_argument(
        '-threshold', dest='threshold',
        action='store', default=175, type=int,
        help='Measuring sick segmentation threshold')

    parser.add_argument(
        '-continuous', dest='continuous',
        action='store_true', default=False,
        help='Flag to run indefinitely')
    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="camera",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0, type=int,
        help='Inference interval in seconds')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')

    parser.add_argument(
        '-cropping', dest='cropping',
        action='store', default="200 700 600 800", type=str,
        help='Points for cropping as string, put the order of "y1 y2 x1 x2"')
    parser.add_argument(
        '-mapping', dest='mapping',
        action='store', default="469,6 467,7 465,8 463,9 461,10 459,11 457,12 455,13 453,14 450,15 448,16 446,17 444,18 442,19 440,20",
        type=str, help='Points for mapping result to water depth as string, put the order of "pixel_height,depth_in_cm pixel_height,depth_in_cm ..."')

    run(parser.parse_args())
