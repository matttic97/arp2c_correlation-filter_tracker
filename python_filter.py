import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.tracker import Tracker
from utils.ex3_utils import create_gauss_peak, create_cosine_window


class FilterTracker(Tracker):

    def __init__(self):
        super().__init__()
        self.parameters = FCParams()

    def name(self):
        return 'simplified_mosse_tracker'

    def preprocess(self, img, shape):
        cut = img[shape[0]:shape[1], shape[2]:shape[3]]
        cut = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        cut = np.log(np.float32(cut)+1.0)
        cut = (cut-cut.mean()) / (cut.std()+self.parameters.lmbd)
        return np.fft.fft2(cut * self.cos)

    def initialize(self, image, region):
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        left = int(max(region[0], 0))
        top = int(max(region[1], 0))
        right = int(min(region[0] + region[2], image.shape[1] - 1))
        bottom = int(min(region[1] + region[3], image.shape[0] - 1))
        self.template = cv2.cvtColor(image[int(top):int(bottom), int(left):int(right)], cv2.COLOR_BGR2GRAY)
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

        left = int(max(round(self.position[0] - float(self.window) / 2), 0))
        top = int(max(round(self.position[1] - float(self.window) / 2), 0))
        right = int(min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1))
        bottom = int(min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1))

        if (right-left) % 2 == 0:
            right += 1
        if (bottom-top) % 2 == 0:
            bottom += 1
        shape = (right-left, bottom-top)

        self.G = np.fft.fft2(create_gauss_peak(shape, self.parameters.sigma))
        self.cos = create_cosine_window(shape)

        P = self.preprocess(image, (top, bottom, left, right))
        Pconj = np.conj(P)

        self.H = (self.G * Pconj) / (P * Pconj + self.parameters.lmbd)
        self.shapeL = shape

    def track(self, image):
        left = int(max(round(self.position[0] - float(self.window) / 2), 0))
        top = int(max(round(self.position[1] - float(self.window) / 2), 0))
        right = left + self.shapeL[0]
        bottom = top + self.shapeL[1]
        dx, dy = (right - image.shape[1] + 1, bottom - image.shape[0] + 1)
        if dx > 0:
            left -= dx
            right -= dx
        if dy > 0:
            bottom -= dy
            top -= dy

        L = self.preprocess(image, (top, bottom, left, right))
        R = np.fft.ifft2(self.H * L)
        y, x = np.unravel_index(R.argmax(), R.shape)
        if x > self.shapeL[0] / 2:
            x = x - self.shapeL[0]
        if y > self.shapeL[1] / 2:
            y = y - self.shapeL[1]

        self.position = (self.position[0]+x, self.position[1]+y)

        left = int(max(round(self.position[0] - float(self.window) / 2), 0))
        top = int(max(round(self.position[1] - float(self.window) / 2), 0))
        right = left + self.shapeL[0]
        bottom = top + self.shapeL[1]
        dx, dy = (right - image.shape[1] + 1, bottom - image.shape[0] + 1)
        if dx > 0:
            left -= dx
            right -= dx
        if dy > 0:
            bottom -= dy
            top -= dy

        L = self.preprocess(image, (top, bottom, left, right))
        Lconj = np.conj(L)
        H = (self.G * Lconj) / (L * Lconj + self.parameters.lmbd)
        self.H = (1 - self.parameters.alpha) * self.H + self.parameters.alpha * H

        return [self.position[0] - self.size[0]/2, self.position[1]-self.size[1]/2, self.size[0], self.size[1]]


class FCParams:
    def __init__(self):
        self.enlarge_factor = 1
        self.alpha = 0.1
        self.sigma = 2
        self.lmbd = 0.001
