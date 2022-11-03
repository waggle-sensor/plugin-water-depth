import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from unet import UNet
from color_map import color_map_viz

import math
import sys

class Unet_Main:
    def __init__(self):
        self.net = UNet(n_channels=3, n_classes=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device=self.device)
        self.net.load_state_dict(torch.load('model.pth', map_location=self.device))
        self.net.eval()

    def preprocess(self, img_nd, img_size=(300,300)):
        img_nd = cv2.resize(img_nd, img_size)
        if len(img_nd.shape) == 2:
            # mask target image
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans.astype(float)


    def run(self, full_img, out_threshold):
        h, w, _ = full_img.shape

        img = torch.from_numpy(self.preprocess(full_img))

        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img)

            probs = torch.sigmoid(output)

            probs = probs.squeeze(0)
            scores = probs.detach().cpu().numpy().reshape(-1)
            scores5 = scores * int(255)
            as5 = np.reshape(np.array(scores5), (300,300))

        ##### mask
        for i in range(len(as5)):
            for j in range(len(as5[0])):
                if as5[i][j] > out_threshold:
                    as5[i][j] = 255
                else:
                    as5[i][j] = 0
        #### end
        scores_resized = cv2.resize(as5, (w,h))

        vertical_indicies = np.where(np.any(scores_resized, axis=1))[0]
        if vertical_indicies != []:
            return vertical_indicies[-1]
        else:
            return None
