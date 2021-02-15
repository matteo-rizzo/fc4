import cv2
import numpy as np

from classes.data.DataProvider import DataProvider


def show_patches():
    dp = DataProvider(True, ['g0'])
    dp.set_batch_size(10)
    while True:
        batch = dp.get_batch()
        for img in batch[0]:
            img = img / img.max()
            cv2.imshow("Input", np.power(img, 1 / 2.2))
            cv2.waitKey(0)


if __name__ == '__main__':
    show_patches()
