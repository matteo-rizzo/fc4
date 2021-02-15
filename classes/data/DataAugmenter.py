import math
import random

import cv2
import numpy as np

from config import *
from utils import rotate_and_crop


class DataAugmenter:

    @staticmethod
    def create_lut(f: callable, num_samples: int):
        """ Returns a function that takes array(int, 0,..resolution - 1) """
        lut = np.array([f(x) for x in np.linspace(0, 1, num_samples)], dtype=np.float32)
        return lambda x: np.take(lut, x.astype('int32'))

    def augment(self, ldr, illuminant):
        angle = (random.random() - 0.5) * AUGMENTATION_ANGLE
        scale = math.exp(random.random() * math.log(AUGMENTATION_SCALE[1] / AUGMENTATION_SCALE[0])) * \
                AUGMENTATION_SCALE[0]
        s = min(max(int(round(min(ldr.shape[:2]) * scale)), 10), min(ldr.shape[:2]))
        start_x, start_y = random.randrange(0, ldr.shape[0] - s + 1), random.randrange(0, ldr.shape[1] - s + 1)
        flip_lr, flip_td = random.randint(0, 1), random.randint(0, 1)
        color_aug = np.zeros(shape=(3, 3))
        for i in range(3):
            color_aug[i, i] = 1 + random.random() * AUGMENTATION_COLOR - 0.5 * AUGMENTATION_COLOR
            for j in range(3):
                if i != j:
                    color_aug[i, j] = (random.random() - 0.5) * AUGMENTATION_COLOR_OFFDIAG

        def crop(img, illumination):
            if img is None:
                return None
            img = img[start_x:start_x + s, start_y:start_y + s]
            img = rotate_and_crop(img, angle)
            img = cv2.resize(img, (FCN_INPUT_SIZE, FCN_INPUT_SIZE))
            if AUGMENTATION_FLIP_LEFTRIGHT and flip_lr:
                img = img[:, ::-1]
            if AUGMENTATION_FLIP_TOPDOWN and flip_td:
                img = img[::-1, :]

            img = img.astype(np.float32)
            new_illuminant = np.zeros_like(illumination)
            illumination = illumination[::-1]
            for i in range(3):
                for j in range(3):
                    new_illuminant[i] += illumination[j] * color_aug[i, j]

            if AUGMENTATION_COLOR_OFFDIAG > 0:
                # Matrix mul, slower
                new_image = np.zeros_like(img)
                for i in range(3):
                    for j in range(3):
                        new_image[:, :, i] += img[:, :, j] * color_aug[i, j]
            else:
                img *= np.array([[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]], dtype=np.float32)
                new_image = img

            new_image = np.clip(new_image, 0, 65535)

            def apply_non_linearity(image):
                if AUGMENTATION_GAMMA != 0:
                    res = 1024
                    image = np.clip(image * (res * 1.0 / 65536), 0, res - 1)
                    gamma = 1.0 + (random.random() - 0.5) * AUGMENTATION_GAMMA
                    mapping = self.create_lut(lambda x: x ** gamma * 65535.0, res)
                    return mapping(image)
                else:
                    return image

            if SPATIALLY_VARIANT:
                split = new_image.shape[1] / 2
                new_image[:, :split] = apply_non_linearity(new_image[:, :split])
                new_image[:, split:] = apply_non_linearity(new_image[:, split:])
            else:
                new_image = apply_non_linearity(new_image)

            new_illuminant = np.clip(new_illuminant, 0.01, 100)
            return new_image, new_illuminant[::-1]

        return crop(ldr, illuminant)
