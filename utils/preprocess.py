import numpy as np
import cv2


def flip(img):
    opt = np.random.randint(-1, 1)
    return cv2.flip(img, opt)


def rotation(img):
    rows, cols = img.shape[:2]
    angle = np.random.randint(1, 360)
    M = cv2.getRotationMatrix2D(((cols-1) / 2, (rows-1) / 2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def noise(img, mean=0, var=0.001):
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    return out


