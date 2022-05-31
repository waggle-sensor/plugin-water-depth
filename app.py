import time
import argparse

import cv2
import torch
import numpy as np

from unet_module import Unet_Main

from waggle.plugin import Plugin
from waggle.data.vision import Camera

TOPIC_WATERDEPTH = "env.water.depth"

def run(args):
    print("Water depth estimation starts...")
    with Plugin() as plugin, Camera(args.stream) as camera:
        timestamp = time.time()
        print(f"Loading Model at time: {timestamp}")
        with plugin.timeit("plugin.duration.loadmodel"):
            unet_main = Unet_Main()
            match = {}
            with open('mapping.txt', 'r') as f:
                for line in f:
                    a = line.strip().split(',')
                    match[int(a[0])] = float(a[1])

        sampling_countdown = -1
        if args.sampling_interval >= 0:
            print(f"Sampling enabled -- occurs every {args.sampling_interval}th inferencing")
            sampling_countdown = args.sampling_interval

        while True:
            with plugin.timeit("plugin.duration.input"):
                sample = camera.snapshot()
                image = sample.data
                imagetimestamp = sample.timestamp
                #image = cv2.imread('test_image.jpg')
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #imagetimestamp = time.time()
                image = image[100:1060, 400:600]
            if args.debug:
                s = time.time()
            with plugin.timeit("plugin.duration.inferencing"):
                depth = unet_main.run(image, out_threshold=args.threshold)
            if args.debug:
                e = time.time()
                print(f'Time elapsed for inferencing: {e-s} seconds')


            if depth != None:
                if depth not in match:
                    plugin.publish(TOPIC_WATERDEPTH, 'out of range', timestamp=timestamp)
                else:
                    plugin.publish(TOPIC_WATERDEPTH, match[depth], timestamp=timestamp)
            else:
                plugin.publish(TOPIC_WATERDEPTH, 'no detection', timestamp=timestamp)

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
        action='store', default=0.9, type=float,
        help='Cloud pixel determination threshold')

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

    run(parser.parse_args())
