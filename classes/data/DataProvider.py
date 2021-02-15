import random

import numpy as np

from auxiliary.utils import load_data
from classes.data.DataAugmenter import DataAugmenter
from scripts.condition import AsyncTaskManager
from config import *


class DataProvider:

    def __init__(self, is_training, folds):
        self.is_training = is_training
        self.cursor = 0
        self.data_augmenter = DataAugmenter()
        self.records = load_data(folds)
        random.shuffle(self.records)
        self.data_count = len(self.records)
        print("#records: {} preprocessing...".format(self.data_count))
        self.preprocess()
        self.images, self.n_rgb, self.illuminants, self.batch_size, self.async_task = [], [], [], None, None

    def preprocess(self):
        for i in range(len(self.records)):
            self.images.append(self.records[i].img)
            self.n_rgb.append(None)
            self.illuminants.append(self.records[i].illuminant)
        self.illuminants = np.vstack(self.illuminants)

    def set_batch_size(self, batch_size: int):
        assert self.batch_size is None
        self.batch_size = batch_size

    def shuffle(self):
        indices = range(self.data_count)
        random.shuffle(indices)
        self.images = [self.images[i] for i in indices]
        self.n_rgb = [self.n_rgb[i] for i in indices]
        self.illuminants = [self.illuminants[i] for i in indices]

    def get_batch_(self):
        batch_size = self.batch_size
        indices = []
        while len(indices) < batch_size:
            s = min(self.data_count - self.cursor, batch_size - len(indices))
            indices += range(self.cursor, self.cursor + s)
            if self.cursor + s >= self.data_count:
                if self.is_training and DATA_SHUFFLE:
                    self.shuffle()
            self.cursor = (self.cursor + s) % self.data_count

        next_batch = [[], [], []]
        for i in indices:
            ldr, n_rgb, illuminant = self.images[i], self.n_rgb[i], self.illuminants[i]
            if self.is_training and AUGMENTATION:
                ldr, illuminant = self.data_augmenter.augment(ldr, illuminant)
            else:
                ldr = ldr[:FCN_INPUT_SIZE, :FCN_INPUT_SIZE]
            n_rgb = None
            next_batch[0].append(ldr)
            next_batch[1].append(n_rgb)
            next_batch[2].append(illuminant)

        return np.stack(next_batch[0]), np.stack(next_batch[1]), np.vstack(next_batch[2])

    def get_batch(self):
        return AsyncTaskManager(self.get_batch_) if self.async_task is None else self.async_task.get_next()

    def stop(self):
        self.async_task.stop()
