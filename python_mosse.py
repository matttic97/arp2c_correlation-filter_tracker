import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.tracker import Tracker
from utils.ex3_utils import create_gauss_peak, create_cosine_window


class MOSSETracker(Tracker):

    def __init__(self):
        super().__init__()
        self.parameters = MOSSEParams()

    def name(self):
        return 'mosse_tracker'

    def preprocess(self, img, shape):
        cut = img[shape[0]:shape[1], shape[2]:shape[3]]
        cut = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        cut = np.log(np.float32(cut)+1.0)
        cut = (cut-cut.mean()) / (cut.std()+self.parameters.lmbd)
        return np.fft.fft2(cut * self.cos)

    def psr(self, window, peak):
        x1, x2, y1, y2 = peak[0] - 5, peak[0] + 5, peak[1] - 5, peak[1] + 5
        g = window[x1:x2, y1:y2]
        return np.real((window[peak[0], peak[1]] - g.mean()) / g.std())

    def update_filter(self, image):
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

        F = self.preprocess(image, (top, bottom, left, right))
        Fconj = np.conj(F)
        self.A = (1 - self.parameters.lr) * self.A + self.parameters.lr * (self.G * Fconj)
        self.B = (1 - self.parameters.lr) * self.B + self.parameters.lr * (F * Fconj)
        self.H = self.A / self.B

    def initialize(self, image, region):
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

        left = int(max(round(self.position[0] - float(self.window) / 2), 0))
        top = int(max(round(self.position[1] - float(self.window) / 2), 0))
        right = int(min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1))
        bottom = int(min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1))

        if (right - left) % 2 == 0:
            right += 1
        if (bottom - top) % 2 == 0:
            bottom += 1
        self.shapeL = (right - left, bottom - top)
        self.G = np.fft.fft2(create_gauss_peak(self.shapeL, self.parameters.sigma))
        self.cos = create_cosine_window(self.shapeL)

        F = self.preprocess(image, (top, bottom, left, right))
        Fconj = np.conj(F)
        self.A = self.G * Fconj
        self.B = (F * Fconj) + self.parameters.lmbd
        self.H = self.A / self.B

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

        F = self.preprocess(image, (top, bottom, left, right))
        R = np.fft.ifft2(self.H * F)
        y, x = np.unravel_index(R.argmax(), R.shape)
        if x > self.shapeL[0] / 2:
            x = x - self.shapeL[0]
        if y > self.shapeL[1] / 2:
            y = y - self.shapeL[1]

        self.position = (self.position[0] + x, self.position[1] + y)

        #psr = self.psr(R, (x, y))
        self.update_filter(image)

        return [self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2, self.size[0], self.size[1]]


class MOSSEParams:
    def __init__(self):
        self.enlarge_factor = 1
        self.sigma = 3.3
        self.lmbd = 0.001
        self.lr = 0.06
