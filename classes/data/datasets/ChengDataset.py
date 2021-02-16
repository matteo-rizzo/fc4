import os
import random

import cv2
import numpy as np
import scipy

from auxiliary.settings import DATA_FRAGMENT, BOARD_FILL_COLOR
from auxiliary.utils import slice_list
from classes.data.ImageRecord import ImageRecord
from classes.data.datasets.Dataset import Dataset


class ChengDataset(Dataset):

    def __init__(self, camera_id: int):
        camera_names = ["Canon1DsMkIII", "Canon600D", "FujifilmXM1", "NikonD5200",
                        "OlympusEPL6", "PanasonicGX1", "SamsungNX2000", "SonyA57"]
        self.camera_name = camera_names[camera_id]

    def get_subset_name(self) -> str:
        return self.camera_name + '-'

    def get_name(self) -> str:
        return "cheng"

    def regenerate_meta_data(self):
        meta_data = []
        ground_truth = scipy.io.loadmat(self.get_directory() + 'ground_truth/' + self.camera_name + '_gt.mat')
        illuminants = ground_truth['groundtruth_illuminants']
        cc_coords = ground_truth['CC_coords']
        illuminants /= np.linalg.norm(illuminants, axis=1)[..., np.newaxis]
        file_names = sorted(os.listdir(os.path.join(self.get_directory(), "images")))
        file_names = filter(lambda f: f.startswith(self.camera_name), file_names)
        extras = {"darkness_level": ground_truth['darkness_level'],
                  "saturation_level": ground_truth['saturation_level']}

        for i, file_name in enumerate(file_names):
            y1, y2, x1, x2 = cc_coords[i]
            mcc_coord = np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            meta_data.append(ImageRecord(self.get_name(), file_name, illuminants[i], mcc_coord, extras=extras))

        random.shuffle(meta_data)

        if DATA_FRAGMENT != -1:
            meta_data = meta_data[:DATA_FRAGMENT]
            print("WARN: using only first {:d} images...".format(len(meta_data)))

        meta_data = slice_list(meta_data, [1] * self.get_folds())
        self.dump_meta_data(meta_data)

    def load_image(self, file_name, darkness_level, saturation_level):
        file_path = self.get_directory() + '/images/' + file_name
        raw = np.array(cv2.imread(file_path, -1), dtype='float32')
        raw = np.maximum(raw - darkness_level, [0, 0, 0])
        raw *= 1.0 / saturation_level
        return raw

    def load_image_without_mcc(self, r) -> np.array:
        img = (np.clip(self.load_image(r.file_name, r.extras['darkness_level'],
                                       r.extras['saturation_level']), 0, 1) * 65535.0).astype(np.uint16)
        polygon = r.mcc_coord
        polygon = polygon.astype(np.int32)
        cv2.fillPoly(img, [polygon], (BOARD_FILL_COLOR,) * 3)
        return img
